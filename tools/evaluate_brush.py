"""Compare Brush training budgets on identical held-out RGB views (PSNR)."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np


def evaluate(root,steps):
    results={};names=None
    for step in steps:
        rows=[]
        images=sorted((root/f'train{step}'/f'eval_{step}').glob('*.png'))
        if not images:raise ValueError(f'No evaluation images for {step}')
        current=[p.name for p in images]
        if names is not None and names!=current:raise ValueError('Different evaluation views')
        names=current
        for path in images:
            predicted=cv2.imread(str(path));original=cv2.imread(str(root/'dataset/images'/path.name))
            original=cv2.resize(original,(predicted.shape[1],predicted.shape[0]),interpolation=cv2.INTER_AREA)
            mse=float(np.mean((predicted.astype(float)-original.astype(float))**2))
            rows.append(dict(frame=path.name,psnr_db=float(10*np.log10(255**2/max(mse,1e-10)))))
        run=json.loads((root/f'train{step}'/'run.json').read_text())
        with (root/f'train{step}'/f'export_{step}.ply').open('rb') as stream:
            count=None
            for _ in range(256):
                line=stream.readline().decode('ascii').strip()
                if line.startswith('element vertex '):count=int(line.split()[-1])
                if line=='end_header':break
        results[str(step)]=dict(frames=rows,mean_psnr_db=float(np.mean([r['psnr_db'] for r in rows])),splats=count,wall_s=run['wall_s'],process_peak_working_set_MiB=run['process_peak_working_set_MiB'])
    report=dict(results=results,evaluation_view_count=len(names),scope=f'Same {len(names)} interleaved held-out views, mean per-image 8-bit PSNR at recorded evaluation resolution; no independent scene/geometry ground truth.',
                caveat='Each budget starts from the same initial dataset and default seed; total steps also affects optimizer schedules, not a resumed checkpoint comparison.')
    filename='comparison_'+'_'.join(map(str,steps))
    (root/(filename+'.json')).write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[names[0],names[len(names)//2],names[-1]]
    fig,axes=plt.subplots(3,len(steps)+1,figsize=(4*(len(steps)+1),8),squeeze=False)
    for row,name in enumerate(selected):
        for col,step in enumerate([None]+steps):
            path=root/'dataset/images'/name if step is None else root/f'train{step}'/f'eval_{step}'/name
            rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
            axes[row,col].imshow(rgb);axes[row,col].axis('off')
            if row==0:axes[row,col].set_title('Input' if step is None else f'{step:,} steps')
    fig.suptitle('Brush on Radeon 780M | fixed held-out views | 320px | 30k splat cap')
    fig.tight_layout();fig.savefig(root/(filename+'.png'),dpi=140);plt.close(fig)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',type=Path,required=True);parser.add_argument('--steps',type=int,nargs='+',required=True)
    args=parser.parse_args();report=evaluate(args.experiment,args.steps)
    print(json.dumps({step:{k:v for k,v in result.items() if k!='frames'} for step,result in report['results'].items()},indent=2))
