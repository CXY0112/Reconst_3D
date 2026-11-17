"""Persistent, cancellable MobileSAM worker. JSON-lines in/out; bounded embeddings."""
from pathlib import Path
from collections import OrderedDict
import argparse, hashlib, json, os, sys, threading, time

ROOT=Path(__file__).resolve().parents[1]
COMMIT='f706ad9c4eb7f219c00d9050e46328518ffb65d2'
WEIGHT_SHA256='6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,required=True)
    args=parser.parse_args()
    def watch():
        while True:
            if (args.control/'stop').exists() or (args.control/'cancel').exists():os._exit(130)
            time.sleep(.1)
    threading.Thread(target=watch,daemon=True).start()
    source=ROOT/'vendor/mobilesam/source/MobileSAM-master'
    if not source.exists():source=ROOT/'vendor/mobilesam/source'/('MobileSAM-'+COMMIT)
    checkpoint=source/'weights/mobile_sam.pt'
    if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=WEIGHT_SHA256:
        raise RuntimeError('MobileSAM 官方权重缺失或校验失败，请参照 docs/MOBILESAM.md 安装')
    sys.path.insert(0,str(source))
    import numpy as np
    import torch
    from PIL import Image
    from mobile_sam import sam_model_registry,SamPredictor
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=sam_model_registry['vit_t'](checkpoint=str(checkpoint)).to(device).eval()
    predictor=SamPredictor(model);cache=OrderedDict()
    print(json.dumps(dict(ready=True,device=device)),flush=True)
    for line in sys.stdin:
        request=json.loads(line);ident=request['id']
        try:
            started=time.perf_counter();path=Path(request['image'])
            rgb=np.asarray(Image.open(path).convert('RGB'));hit=False
            with torch.inference_mode():
                if request['mode']=='box':
                    key=(str(path),path.stat().st_mtime_ns)
                    if key in cache:
                        predictor.features,predictor.original_size,predictor.input_size=cache.pop(key)
                        predictor.is_image_set=True;hit=True
                    else:predictor.set_image(rgb)
                    cache[key]=(predictor.features,predictor.original_size,predictor.input_size)
                    while len(cache)>2:cache.popitem(last=False)
                    box=np.array(request['box'],float)
                    h,w=rgb.shape[:2]
                    if box.shape!=(4,) or not np.isfinite(box).all() or not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
                        raise ValueError('无效RGB框选范围')
                    masks,scores,_=predictor.predict(box=box,multimask_output=False)
                else:raise ValueError('未知MobileSAM操作')
            np.savez_compressed(request['output'],masks=masks,scores=scores)
            print(json.dumps(dict(id=ident,ok=True,device=device,seconds=time.perf_counter()-started,
                embedding_cache_hit=hit,source_commit=COMMIT,checkpoint_sha256=WEIGHT_SHA256)),flush=True)
        except Exception as exc:print(json.dumps(dict(id=ident,ok=False,error=str(exc))),flush=True)


if __name__=='__main__':main()
