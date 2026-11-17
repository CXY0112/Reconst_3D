"""Inference-only NoKSR adapter; official ScanNet Serial weights and network code."""
from pathlib import Path
import argparse,importlib.util,json,sys,time,types,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def package(name,path):
    m=types.ModuleType(name); m.__path__=[str(path)]; sys.modules[name]=m


def load(name,path,rewrites=()):
    source=path.read_text()
    for old,new in rewrites:
        if old not in source: raise RuntimeError('Upstream changed: '+str(path))
        source=source.replace(old,new)
    module=types.ModuleType(name); module.__file__=str(path); module.__package__=name.rpartition('.')[0]
    sys.modules[name]=module
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module


def cpu_sparse_convolution(module,sparse):
    """Exact sparse cross-correlation using PyTorch; CPU spconv native is not trusted."""
    import torch,itertools
    cache={}
    def forward(self,inputs):
        if inputs.features.is_cuda: return self._original_forward(inputs)
        indices=inputs.indices; shape=tuple(inputs.spatial_shape); kernel=tuple(self.kernel_size); dilation=tuple(self.dilation)
        key=(indices.data_ptr(),len(indices),shape,kernel,dilation)
        if key not in cache:
            dims=torch.tensor(shape,dtype=torch.int64)
            coords=indices.long()
            def encode(c): return ((c[:,0]*shape[0]+c[:,1])*shape[1]+c[:,2])*shape[2]+c[:,3]
            codes=encode(coords); sorted_codes,sorted_rows=torch.sort(codes)
            if len(torch.unique_consecutive(sorted_codes))!=len(codes): raise RuntimeError('Sparse grid indices must be unique')
            pairs=[]
            for offset in itertools.product(*(range(k) for k in kernel)):
                delta=torch.tensor([(offset[i]-kernel[i]//2)*dilation[i] for i in range(3)])
                near=coords.clone(); near[:,1:]+=delta
                valid=((near[:,1:]>=0)&(near[:,1:]<dims)).all(dim=1)
                rows=torch.nonzero(valid).flatten(); lookup=encode(near[rows]); pos=torch.searchsorted(sorted_codes,lookup)
                inside=pos<len(sorted_codes); rows,pos,lookup=rows[inside],pos[inside],lookup[inside]
                hit=sorted_codes[pos]==lookup
                pairs.append((offset,rows[hit],sorted_rows[pos[hit]]))
            cache[key]=(indices,pairs)
        out=inputs.features.new_zeros((len(indices),self.out_channels))
        for (x,y,z),destination,source in cache[key][1]:
            if len(source): out[destination]+=inputs.features[source]@self.weight[:,x,y,z,:].T
        if self.bias is not None: out+=self.bias
        return inputs.replace_feature(out)
    for layer in module.modules():
        if isinstance(layer,sparse.SubMConv3d):
            layer._original_forward=layer.forward; layer.forward=types.MethodType(forward,layer)
    # Validate the actual fallback repeatedly against dense PyTorch, including bias.
    for seed in range(8):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            kernel=3 if seed<4 else 5
            layer=sparse.SubMConv3d(6,32,kernel,padding=kernel//2,bias=True).eval()
            layer._original_forward=layer.forward; layer.forward=types.MethodType(forward,layer)
            xyz=torch.cartesian_prod(torch.arange(3),torch.arange(3),torch.arange(3)).int()
            indices=torch.cat((torch.zeros((len(xyz),1),dtype=torch.int32),xyz),dim=1)
            features=torch.randn(len(xyz),6)
            inputs=sparse.SparseConvTensor(features,indices,[5,5,5],1)
            with torch.no_grad(): actual=layer(inputs).features
            dense=torch.zeros(1,6,5,5,5); dense[0,:,xyz[:,0],xyz[:,1],xyz[:,2]]=features.T
            expected=torch.nn.functional.conv3d(dense,layer.weight.permute(0,4,1,2,3).contiguous(),layer.bias,padding=kernel//2)
            expected=expected[0,:,xyz[:,0],xyz[:,1],xyz[:,2]].T
            if not torch.allclose(actual,expected,atol=1e-5,rtol=1e-5): raise RuntimeError('CPU sparse operator equivalence failed')
    cache.clear()


def main():
    from geometry_methods import NOKSR_DEFAULTS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--device',default='cpu'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--attention',choices=('auto','standard','flash'),default='auto')
    p.add_argument('--probe',action='store_true')
    p.add_argument('--query-batch',type=int,default=32768)
    p.add_argument('--input-points',type=int,default=-1,help='-1: full input; 10000: official ScanNet sampling count')
    p.add_argument('--seed',type=int,default=20261019)
    p.add_argument('--k-neighbors',type=int,default=NOKSR_DEFAULTS['k_neighbors'])
    p.add_argument('--last-n-layers',type=int,default=NOKSR_DEFAULTS['last_n_layers'])
    p.add_argument('--mask-threshold-mm',type=float,default=NOKSR_DEFAULTS['mask_threshold_mm'])
    p.add_argument('--voxel-mm',type=float,default=NOKSR_DEFAULTS['voxel_mm'])
    p.add_argument('--meshing-voxel-mm',type=float,default=NOKSR_DEFAULTS['meshing_voxel_mm'],help='Extraction grid in mm, default15; decoder scale is unchanged')
    p.add_argument('--save-mask-diagnostics',action='store_true',help='Also save the mesh before learned trimming and per-vertex mask distances')
    args=p.parse_args()
    if args.query_batch<=0 or (args.input_points!=-1 and args.input_points<=0): p.error('Use a positive query batch and input-points=-1 or a positive count')
    if args.k_neighbors<=0 or args.k_neighbors%2: p.error('Serial k-neighbors must be a positive even number')
    import math
    if any(not math.isfinite(v) or v<=0 for v in (args.mask_threshold_mm,args.voxel_mm)):
        p.error('Voxel size and mask threshold must be finite positive millimetres')
    if args.meshing_voxel_mm is not None and (not math.isfinite(args.meshing_voxel_mm) or args.meshing_voxel_mm<=0):
        p.error('Meshing voxel size must be finite positive millimetres')
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from geometry_comparison import load_source
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    control=ROOT/'vendor/noksr_linux'; upstream=control/'source'
    # Import only inference modules. Official training, KNN, and GT paths are not used.
    for name,rel in [('noksr','noksr'),('noksr.model','noksr/model'),('noksr.model.module','noksr/model/module')]:
        package(name,upstream/rel)
    site=Path(__import__('sysconfig').get_paths()['purelib'])
    for name,rel in [('timm',''),('timm.layers','layers'),('timm.models','models')]: package(name,site/'timm'/rel)
    drop=importlib.import_module('timm.layers.drop')
    layers=types.ModuleType('timm.models.layers'); layers.DropPath=drop.DropPath; sys.modules[layers.__name__]=layers
    backbone=load('noksr.model.module.point_transformer',upstream/'noksr/model/module/point_transformer.py')
    rewrites=[('import pytorch_lightning as pl','pl = __import__("types").SimpleNamespace(LightningModule=torch.nn.Module)'),
              ('from pytorch3d.ops import knn_points','def knn_points(*a, **k):\n    raise RuntimeError("KNN/GT path is outside the Serial inference adapter")')]
    dec=load('noksr.model.module.decoder',upstream/'noksr/model/module/decoder.py',rewrites)
    gen=load('noksr.model.module.generation',upstream/'noksr/model/module/generation.py',
             rewrites+[('from noksr.utils.samples import BatchedSampler',''),
                       ('from sklearn.neighbors import NearestNeighbors','def NearestNeighbors(*a, **k):\n    raise RuntimeError("Ground-truth mask is not used for inference")')])
    cfg=OmegaConf.create({'model':OmegaConf.load(upstream/'config/model/scannet_model.yaml'),
                         'data':OmegaConf.merge(OmegaConf.load(upstream/'config/data/base.yaml'),OmegaConf.load(upstream/'config/data/scannet.yaml'))})
    pretrained_layers=int(cfg.model.network.default_decoder.last_n_layers)
    if args.last_n_layers!=pretrained_layers:
        raise ValueError(f'last_n_layers={args.last_n_layers} is incompatible with the pretrained ScanNet Serial architecture: '
                         f'{pretrained_layers} trained decoder scales, feature_dim={list(cfg.model.network.default_decoder.feature_dim)}, '
                         f'dec_channels={list(cfg.model.network.point_transformerv3.dec_channels)}. No automatic fallback applied.')
    cfg.model.network.default_decoder.k_neighbors=args.k_neighbors
    cfg.data.voxel_size=args.voxel_mm/1000.
    cfg.data.reconstruction.mask_threshold=args.mask_threshold_mm/1000.
    device=torch.device(args.device)
    flash_available=importlib.util.find_spec('flash_attn') is not None
    use_flash=args.attention=='flash' or (args.attention=='auto' and device.type=='cuda' and flash_available)
    if use_flash and (device.type!='cuda' or not flash_available): raise RuntimeError('FlashAttention requires CUDA and flash_attn')
    cfg.model.network.point_transformerv3.enable_flash=use_flash
    net=backbone.PointTransformerV3(backbone_cfg=cfg.model.network.point_transformerv3).to(device).eval()
    if device.type=='cpu':
        import spconv.pytorch as sparse
        cpu_sparse_convolution(net,sparse)
    def decoder(kind):
        c=cfg.model.network.sdf_decoder if kind=='SDF' else cfg.model.network.mask_decoder
        return dec.Decoder(decoder_cfg=c,supervision=kind,latent_dim=cfg.model.network.latent_dim,
            feature_dim=c.feature_dim,hidden_dim=c.hidden_dim,out_dim=1,voxel_size=cfg.data.voxel_size,activation=c.activation).to(device).eval()
    sdf,mask=decoder('SDF'),decoder('Distance')
    if args.probe:
        print('NoKSR inference modules and ScanNet architecture constructed successfully.'); return
    print('Loading official checkpoint...',flush=True)
    if device.type=='cuda': torch.cuda.synchronize(device)
    stamp=time.perf_counter()
    checkpoint=torch.load(control/'ScanNet_Serial_best.ckpt',map_location='cpu',weights_only=False)
    state=checkpoint['state_dict']; loaded=set()
    for prefix,module in [('point_transformer.',net),('sdf_decoder.',sdf),('mask_decoder.',mask)]:
        selected={key[len(prefix):]:v for key,v in state.items() if key.startswith(prefix)}
        module.load_state_dict(selected,strict=True); loaded.update(key for key in state if key.startswith(prefix))
    if set(state)-loaded: raise RuntimeError('Unaccounted checkpoint keys: '+str(list(set(state)-loaded)[:10]))
    del checkpoint,state
    points,normals,hashes=load_source(args.source)
    source_points=len(points)
    if args.smoke:
        # Local crop is a separate logic test, never reported as the full scene.
        center=np.median(points,axis=0); near=np.argsort(np.linalg.norm(points-center,axis=1))[:4096]
        points,normals=points[near],normals[near]
    elif args.input_points>0:
        sample=np.random.RandomState(args.seed).choice(len(points),args.input_points,replace=True)
        points,normals=points[sample],normals[sample]
    requested_points=len(points)
    # Sparse convolution requires one feature row per occupied 1cm grid cell.
    origin=points.astype(np.float32).min(axis=0)
    grid=np.trunc((points.astype(np.float32)-origin)/.01).astype(np.int32)
    _,unique=np.unique(grid,axis=0,return_index=True); unique=np.sort(unique)
    points,normals,grid=points[unique],normals[unique],grid[unique]
    xyz=torch.tensor(points,dtype=torch.float32,device=device)
    features=torch.cat((torch.tensor(normals,dtype=torch.float32,device=device),xyz),dim=1)
    metadata={'device':str(device),'input_points':len(points),'source_points':source_points,'seed':args.seed,
              'sampled_points_before_grid_uniqueness':requested_points,'grid_uniqueness_mm':10,
              'sampling':'uniform random with replacement' if args.input_points>0 else 'all points',
              'input_sha256':hashes,'smoke':args.smoke,
              'official_checkpoint_strict_load':True,'flash_attention':use_flash,'network_weights_changed':False,
              'voxel_m':float(cfg.data.voxel_size),'query_batch':args.query_batch,
              'parameters':{'k_neighbors':args.k_neighbors,'last_n_layers':args.last_n_layers,
                  'mask_threshold_mm':args.mask_threshold_mm,'voxel_mm':args.voxel_mm,
                  'neighboring':'Serial','serial_orders':['hilbert'],'dist_factor':[4,3,2,2],
                  'input_grid_mm':10,'mise_iter':0,'grid_upsample':1,'trim':True},
              'cpu_compatibility':['training-only Lightning base replaced by nn.Module for inference',
                  'unused KNN/ground-truth imports guarded, Serial path retained',
                  'official attention branch', 'query chunks preserve global serialization depth; exact neighbor and decoder value equivalence checked']}
    if device.type=='cpu': metadata['cpu_compatibility'].append('Pure PyTorch sparse cross-correlation, eight dense-reference checks; original CPU spconv native rejected')
    print('Encoding',len(points),'points...',flush=True)
    with torch.no_grad():
        encoded=net(dict(feat=features,coord=xyz,grid_coord=torch.tensor(grid,device=device),grid_size=.01,offset=torch.tensor([len(xyz)],device=device)))
    if device.type=='cuda': torch.cuda.synchronize(device)
    metadata['encoding_seconds']=time.perf_counter()-stamp
    print('Encoding complete; extracting mesh...',flush=True)
    from noksr.utils.serialization import encode
    def cached_neighbors(self,points,queries):
        depth=self._serial_depth; all_indices=[]
        for order in self.serial_orders:
            key=(points.data_ptr(),len(points),depth,order)
            if key not in self._serial_cache:
                coords=torch.floor(points/.01).int()
                codes=encode(coords,torch.zeros(len(coords),dtype=torch.int64,device=device),depth,order=order)
                self._serial_cache[key]=torch.sort(codes)
            codes,source_indices=self._serial_cache[key]
            qcoords=torch.floor(queries/.01).int()
            qcodes=encode(qcoords,torch.zeros(len(queries),dtype=torch.int64,device=device),depth,order=order)
            positions=torch.searchsorted(codes,qcodes,right=True).unsqueeze(1); k=self.k_neighbors//2
            ids=torch.cat((positions-torch.arange(1,k+1,device=device),positions+torch.arange(k,device=device)),dim=1)
            ids=torch.where((ids>=0)&(ids<len(codes)),ids,-1)
            all_indices.append(torch.where(ids!=-1,source_indices[ids],-1))
        return torch.cat(all_indices,dim=-1)
    class Bounded(torch.nn.Module):
        def __init__(self,module): super().__init__(); self.module=module
        def forward(self,encoder,queries):
            m=self.module; m._serial_depth=int(torch.abs(torch.floor(queries/.01)).max()).bit_length(); m._serial_cache={}
            original=m.serial_neighbor
            # Check exact indices with the full-query depth, including extreme coordinates.
            sample=torch.cat((queries[:127],queries[torch.argmax(torch.abs(torch.floor(queries/.01)).amax(dim=1))].unsqueeze(0)))
            cloud=encoder.coord[:min(len(encoder.coord),4096)]
            expected=original(cloud,sample)
            expected_values=m(encoder,sample)[0]
            m.serial_neighbor=types.MethodType(cached_neighbors,m)
            actual=m.serial_neighbor(cloud,sample)
            if not torch.equal(expected,actual): raise RuntimeError('Serial neighbor equivalence failed')
            actual_values=m(encoder,sample)[0]
            if not torch.allclose(expected_values,actual_values,atol=1e-5,rtol=1e-5): raise RuntimeError('Decoder value equivalence failed')
            values=[]; knn=aggregation=0.
            try:
                for start in range(0,len(queries),args.query_batch):
                    v,k,a=m(encoder,queries[start:start+args.query_batch]); values.append(v); knn+=k; aggregation+=a
                return torch.cat(values),knn,aggregation
            finally: m.serial_neighbor=original; m._serial_cache.clear()
    meshing_voxel=(args.meshing_voxel_mm/1000. if args.meshing_voxel_mm is not None else cfg.data.voxel_size)
    metadata['parameters']['meshing_voxel_mm']=meshing_voxel*1000
    if args.save_mask_diagnostics: cfg.data.reconstruction.trim=False
    generator=gen.Generator(Bounded(sdf),Bounded(mask),meshing_voxel,args.k_neighbors,args.last_n_layers,cfg.data.reconstruction)
    with torch.no_grad(): mesh,timing=generator.generate_dual_mc_mesh({'xyz':xyz},encoded,device)
    if args.save_mask_diagnostics:
        diagnostic_vertices=np.asarray(mesh.vertices).copy();diagnostic_faces=np.asarray(mesh.triangles).copy()
        vertices_tensor=torch.tensor(diagnostic_vertices,dtype=torch.float32,device=device)
        with torch.no_grad(): mask_distances=Bounded(mask)(encoded,vertices_tensor)[0]
        diagnostic_path=args.packet.with_name(args.packet.stem+'_before_mask.npz')
        args.packet.parent.mkdir(parents=True,exist_ok=True)
        np.savez(diagnostic_path,vertices=diagnostic_vertices,triangles=diagnostic_faces,
                 mask_distances=mask_distances.cpu().numpy())
        from nksr import utils
        v,f=utils.apply_vertex_mask(vertices_tensor,torch.tensor(diagnostic_faces,dtype=torch.long,device=device),
                                    mask_distances<cfg.data.reconstruction.mask_threshold)
        from pycg import vis
        mesh=vis.mesh(v,f)
        metadata['mask_diagnostics']=str(diagnostic_path)
        metadata['timing_note']='Includes diagnostic pre-mask export and mask prediction; not a timing benchmark'
    vertices=np.asarray(mesh.vertices); faces=np.asarray(mesh.triangles)
    if not len(faces) or not np.isfinite(vertices).all(): raise RuntimeError('Invalid NoKSR mesh')
    if device.type=='cuda': torch.cuda.synchronize(device)
    seconds=time.perf_counter()-stamp
    from reconstruction_pipeline import file_hash
    if hashes!={key:file_hash(args.source/key) for key in hashes}: raise RuntimeError('Input changed')
    metadata.update(seconds=seconds,vertices=len(vertices),triangles=len(faces),timing=timing,input_unchanged=True)
    args.packet.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.packet,vertices=vertices,triangles=faces,seconds=seconds)
    args.packet.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__': main()
