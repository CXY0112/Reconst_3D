"""Small local voxel submaps; rigid corrections move anchors, export stays exact.

Native RTAB-Map poses, per-frame clouds and stage-two inputs are unchanged.
Preview uses locally fused cells. Final output fuses ORIGINAL retained frame
centroids in the latest world poses, avoiding the extra local quantization.
"""
from collections import OrderedDict
from pathlib import Path
import os,time
import numpy as np
from rtab_client import RtabClient
from rtabmap_backend import RtabmapBackend
from file_snapshots import AtomicJsonWriter


def reduce_cells(keys,sums,counts):
    """Exact cell identities, vectorized accumulation, no Python/SQL point rows."""
    keys=np.asarray(keys,np.int64);sums=np.asarray(sums,np.float64);counts=np.asarray(counts,np.int64)
    if not len(keys):return keys.reshape(0,3),sums.reshape(0,3),counts
    # Three signed 21-bit coordinates fit in positive int64. Beyond ~3km at
    # 3mm resolution use tuple sorting instead: no coordinate wrap or aliasing.
    if keys.min()>=-(1<<20) and keys.max()<(1<<20):
        shifted=keys+(1<<20)
        packed=(shifted[:,0]<<42)|(shifted[:,1]<<21)|shifted[:,2]
        _,first,inverse=np.unique(packed,return_index=True,return_inverse=True)
        unique=keys[first]
    else:
        unique,inverse=np.unique(keys,axis=0,return_inverse=True)
    n=len(unique)
    total=np.column_stack([np.bincount(inverse,weights=sums[:,i],minlength=n) for i in range(3)])
    weight=np.bincount(inverse,weights=counts,minlength=n).astype(np.int64)
    return unique,total,weight


def cells_for(points,size,weights=None):
    points=np.asarray(points,np.float64)
    if weights is None:weights=np.ones(len(points),np.int64)
    return reduce_cells(np.floor(points/size).astype(np.int64),points*np.asarray(weights)[:,None],weights)


def merge(parts):
    parts=[part for part in parts if len(part[0])]
    if not parts:return np.empty((0,3),np.int64),np.empty((0,3)),np.empty(0,np.int64)
    if len(parts)==1:return parts[0]
    return reduce_cells(*(np.concatenate([part[i] for part in parts]) for i in range(3)))


def displacement(a,b,radius):
    return float(np.linalg.norm(a[:3,3]-b[:3,3])+radius*np.linalg.norm(a[:3,:3]-b[:3,:3],ord=2))


class SubmapStore:
    def __init__(self,path,size=.003,check_same_thread=False,max_frames=8,cache_mib=64):
        self.size=size;self.folder=Path(path).parent/'submaps';self.folder.mkdir()
        self.max_frames=max_frames;self.cache_limit=cache_mib*1024**2
        self.groups=[];self.members={};self.poses={};self.cache=OrderedDict();self.cache_bytes=0
        self.sequence=0;self.revision=0;self.snapshot=None;self.snapshot_revision=-1
        self.last_stats={};self.finalized=False
        self.manifest_writer=AtomicJsonWriter(self.folder/'manifest.json')

    def _get(self,g):
        key=g['id']
        if key in self.cache:
            value=self.cache.pop(key);self.cache[key]=value;return value
        with np.load(self.folder/f'{key:06d}.npz',allow_pickle=False) as data:
            value=data['points'],data['weights']
        self._cache(key,value);return value

    def _cache(self,key,value):
        old=self.cache.pop(key,None)
        if old is not None:self.cache_bytes-=sum(a.nbytes for a in old)
        size=sum(a.nbytes for a in value)
        while self.cache and self.cache_bytes+size>self.cache_limit:
            _,old=self.cache.popitem(last=False);self.cache_bytes-=sum(a.nbytes for a in old)
        if size<=self.cache_limit:self.cache[key]=value;self.cache_bytes+=size

    def _assign(self,latest):
        for node,pose in latest.items():
            if node in self.members:continue
            g=self.groups[-1] if self.groups else None
            if g is not None:
                anchor=latest[g['anchor']]
                angle=np.arccos(np.clip((np.trace(anchor[:3,:3].T@pose[:3,:3])-1)/2,-1,1))
            if g is None or len(g['nodes'])>=self.max_frames or np.linalg.norm(anchor[:3,3]-pose[:3,3])>.4 or angle>np.deg2rad(15):
                g=dict(id=len(self.groups)+1,anchor=node,nodes=[],world=pose.copy(),local={},sequence=0)
                self.groups.append(g)
            g['nodes'].append(node);self.members[node]=g['id']

    def _rebuild(self,g,latest,clouds):
        anchor=latest[g['anchor']];inverse=np.linalg.inv(anchor);parts=[];relative={}
        for node in g['nodes']:
            local=inverse@latest[node];relative[node]=local
            points=np.asarray(clouds[node]);points=points@local[:3,:3].T+local[:3,3]
            parts.append(cells_for(points,self.size))
        keys,sums,counts=merge(parts);value=sums/counts[:,None],counts
        key=g['id'];temporary=self.folder/f'.pending_{key:06d}.npz'
        try:
            np.savez(temporary,points=value[0],weights=value[1]);os.replace(temporary,self.folder/f'{key:06d}.npz')
        finally:temporary.unlink(missing_ok=True)
        self._cache(key,value);g['world']=anchor.copy();g['local']=relative
        self.sequence+=1;g['sequence']=self.sequence;self.revision+=1
        for node,local in relative.items():self.poses[node]=anchor@local

    def sync(self,latest,clouds,radii,tolerance=.00075,max_rebuilds=1):
        self._assign(latest);rebuild=[];rigid=[];deferred=0;updated=0;pending=set()
        for g in self.groups:
            missing=any(node not in g['local'] for node in g['nodes'])
            moved=missing or any(displacement(latest[n],self.poses[n],radii[n])>tolerance for n in g['nodes'])
            if not moved:deferred+=len(g['nodes']);continue
            anchor=latest[g['anchor']];inverse=np.linalg.inv(anchor)
            coherent=not missing and all(displacement(inverse@latest[n],g['local'][n],radii[n])<=tolerance for n in g['nodes'])
            if coherent:rigid.append(g)
            else:rebuild.append((missing,g))
        for g in rigid:
            g['world']=latest[g['anchor']].copy();self.revision+=1
            for n in g['nodes']:self.poses[n]=g['world']@g['local'][n]
            updated+=len(g['nodes'])
        # New observations always progress. Changed historical submaps have a
        # fair bounded rebuild budget and coalesce to the latest pose snapshot.
        rebuild.sort(key=lambda item:(not item[0],item[1]['sequence']))
        history=0;rebuilt=0
        for missing,g in rebuild:
            if not missing and history>=max_rebuilds:pending.update(g['nodes']);continue
            self._rebuild(g,latest,clouds);rebuilt+=1;updated+=len(g['nodes'])
            history+=not missing
        self.finalized=False
        self.last_stats=dict(updated_nodes=updated,rigid_submap_updates=len(rigid),rebuilt_submaps=rebuilt,
            pending_pose_nodes=len(pending),deferred_small_pose_nodes=deferred,submaps=len(self.groups),
            cache_bytes=self.cache_bytes,cache_limit_bytes=self.cache_limit)
        if updated:self.save_manifest()
        return pending

    def cloud(self):
        if self.snapshot_revision!=self.revision:
            parts=[]
            for g in self.groups:
                if not g['local']:continue
                points,weights=self._get(g);pose=g['world']
                world=points@pose[:3,:3].T+pose[:3,3]
                parts.append(cells_for(world,self.size,weights))
            _,sums,counts=merge(parts)
            self.snapshot=sums/counts[:,None];self.snapshot.flags.writeable=False
            self.snapshot_revision=self.revision
        return self.snapshot if self.snapshot is not None else np.empty((0,3))

    def exact_cloud(self,latest,clouds):
        # Balanced reduction bounds the number of concurrent intermediate
        # aggregates and avoids merging the whole map once per historical frame.
        levels=[]
        for node,pose in latest.items():
            points=np.asarray(clouds[node]);part=cells_for(points@pose[:3,:3].T+pose[:3,3],self.size)
            level=0
            while level<len(levels) and levels[level] is not None:
                part=merge([levels[level],part]);levels[level]=None;level+=1
            if level==len(levels):levels.append(part)
            else:levels[level]=part
        _,sums,counts=merge([p for p in levels if p is not None]);cloud=sums/counts[:,None]
        cloud.flags.writeable=False;self.finalized=True;self.save_manifest()
        return cloud

    def save_manifest(self):
        data=dict(schema='local_voxel_submaps_v1',voxel_m=self.size,max_frames=self.max_frames,
            max_anchor_translation_m=.4,max_anchor_rotation_degrees=15,preview_relative_tolerance_m=.00075,
            preview_global_voxel_dedup=True,finalized=self.finalized,
            final_method='latest per-frame poses, original retained centroids, exact global voxel fusion',
            groups=[dict(id=g['id'],anchor=g['anchor'],nodes=g['nodes'],world=g['world'].tolist(),
                local_poses={str(n):p.tolist() for n,p in g['local'].items()}) for g in self.groups])
        self.manifest_writer.update(data)

    def close(self):
        try:self.manifest_writer.flush()
        finally:self.cache.clear();self.cache_bytes=0


class SubmapBackend(RtabmapBackend):
    dense_backend='local_voxel_submaps_v1'

    def __init__(self,folder,client_factory=RtabClient,optimizer='default'):
        super().__init__(folder,client_factory=client_factory,optimizer=optimizer,dense_factory=SubmapStore)
        self.exact_poses=None

    def _preview(self,force=False):
        with self.lock:
            latest={n:self.poses[n].copy() for n in self.node_ids}
        nodes=np.asarray([[-p[1,3],-p[2,3],p[0,3]] for p in latest.values()]).reshape(-1,3)
        started=time.perf_counter();store=self.voxel_map
        same=self.exact_poses is not None and latest.keys()==self.exact_poses.keys() and all(np.array_equal(p,self.exact_poses[n]) for n,p in latest.items())
        if same:
            with self.lock:
                for n,p in latest.items():
                    if np.array_equal(self.poses[n],p):self.dirty_nodes.discard(n)
            self.preview_stats=dict(updated_nodes=0,pending_pose_nodes=len(self.dirty_nodes),fusion_ms=0.,cloud_read_ms=0.,force=force,exact=True,submaps=len(store.groups),retained_nodes=len(latest))
            return self.cloud_cache,nodes
        if force:
            world=store.exact_cloud(latest,self.clouds);self.exact_poses=latest
            stats=dict(updated_nodes=len(latest),rigid_submap_updates=0,rebuilt_submaps=0,pending_pose_nodes=0,
                deferred_small_pose_nodes=0,submaps=len(store.groups),cache_bytes=store.cache_bytes,cache_limit_bytes=store.cache_limit)
            fusion_ms=(time.perf_counter()-started)*1000;started=time.perf_counter();pending=set()
        else:
            self.exact_poses=None
            pending=store.sync(latest,self.clouds,self.radii,tolerance=self.preview_tolerance_m)
            stats=store.last_stats.copy();fusion_ms=(time.perf_counter()-started)*1000;started=time.perf_counter()
            world=store.cloud()
        changed=force or stats['updated_nodes'] or self.cloud_cache is None
        if changed:
            self.cloud_cache=np.column_stack((-world[:,1],-world[:,2],world[:,0]));self.cloud_cache.flags.writeable=False
        with self.lock:
            for n,p in latest.items():
                if n not in pending and np.array_equal(self.poses[n],p):self.dirty_nodes.discard(n)
            pending_count=len(self.dirty_nodes)
        self.preview_stats=dict(**stats,fusion_ms=fusion_ms,cloud_read_ms=(time.perf_counter()-started)*1000,
            force=force,exact=force,tolerance_m=self.preview_tolerance_m,retained_nodes=len(latest))
        self.preview_stats['pending_pose_nodes']=pending_count
        return self.cloud_cache,nodes
