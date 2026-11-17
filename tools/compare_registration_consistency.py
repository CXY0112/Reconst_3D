"""Compare geometric consistency of identical saved frame clouds; no ground truth.

Each saved pose maps camera optical coordinates to world coordinates. Adjacent
source-to-target = inverse(previous world pose) @ current world pose. Evaluate
ALL 3 mm frame centroids bidirectionally with nearest-neighbor queries. Report
RMSE within 30 mm alongside coverage so threshold exclusion stays visible.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def compare(root,candidate='bufferx'):
    outputs={};inputs={};all_distances={}
    for kind in ('rgbd',candidate):
        folder=root/kind/'setp1';graph=json.loads((folder/'geometry_poses.json').read_text())
        clouds=[np.load(folder/'map_frames'/f'{node}.npy',allow_pickle=False) for node in graph['node_ids']]
        inputs[kind]=clouds;poses=[np.array(p) for p in graph['poses']];rows=[];all_distances[kind]=[]
        for j in range(1,len(clouds)):
            source,target=clouds[j],clouds[j-1]
            relative=np.linalg.inv(poses[j-1])@poses[j]
            moved=source@relative[:3,:3].T+relative[:3,3]
            forward=cKDTree(target).query(moved,workers=2)[0]
            reverse=cKDTree(moved).query(target,workers=2)[0]
            distances=np.r_[forward,reverse];accepted=distances[distances<.03]
            all_distances[kind].append(distances)
            rows.append(dict(pair=f'{j+1} to {j}',points=[len(source),len(target)],
                overlap_10mm=float(np.mean(distances<.01)),overlap_20mm=float(np.mean(distances<.02)),
                overlap_30mm=float(np.mean(distances<.03)),trimmed_rmse_30mm=float(np.sqrt(np.mean(accepted**2))*1000),
                median_mm=float(np.median(distances)*1000),p90_mm=float(np.quantile(distances,.9)*1000)))
        outputs[kind]=rows
    if len(inputs['rgbd'])!=len(inputs[candidate]) or not all(np.array_equal(a,b) for a,b in zip(inputs['rgbd'],inputs[candidate])):
        raise ValueError('Different frame clouds; comparison refused')
    changes=[]
    for a,b in zip(outputs['rgbd'],outputs[candidate]):
        changes.append(dict(pair=a['pair'],rmse_reduction_percent=100*(1-b['trimmed_rmse_30mm']/a['trimmed_rmse_30mm']),
            coverage_10mm_change_pp=100*(b['overlap_10mm']-a['overlap_10mm'])))
    return dict(source=str(root.resolve()),candidate=candidate,same_input_clouds=True,frames=len(inputs['rgbd']),
        scope='Final optimized poses, adjacent full 3mm centroid clouds; symmetric NN; 30mm truncated RMSE',
        limitation='Geometric consistency only; two pairs share a frame; no independent pose/surface ground truth; not an accuracy rate',
        results=outputs,changes=changes)


def plot(report,output):
    os.environ.setdefault('MPLCONFIGDIR',str(output/'plot_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    fig,ax=plt.subplots(figsize=(7.2,3.7),layout='constrained')
    x=np.arange(len(report['results']['rgbd']));width=.33
    candidate=report.get('candidate','bufferx')
    for kind,label,shift,color in [('rgbd','默认 RGB-D',-.5,'#627086'),(candidate,candidate.upper(),.5,'#237f96')]:
        values=[r['trimmed_rmse_30mm'] for r in report['results'][kind]]
        bars=ax.bar(x+width*shift,values,width,color=color,label=label)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=11)
    ax.set_xticks(x,['第 2 帧 → 第 1 帧','第 3 帧 → 第 2 帧'],fontproperties=font)
    ax.set_ylabel('截断 RMSE（mm），越低越好',fontproperties=font)
    ax.set_title('当前三帧的几何对齐残差',fontproperties=font,pad=15)
    ax.set_ylim(0,15);ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True);ax.yaxis.grid(True,alpha=.18)
    ax.legend(prop=font,frameon=False)
    fig.text(.5,-.04,'双向最近邻，距离 <30 mm；内部一致性指标，不是绝对精度',ha='center',fontproperties=font,fontsize=9)
    fig.savefig(output/'consistency.png',dpi=160,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--plot',action='store_true')
    parser.add_argument('--candidate',default='bufferx',choices=('bufferx','gcvo_cpu','gcvo_cuda'))
    args=parser.parse_args();report=compare(args.source,args.candidate)
    (args.source/'consistency.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.plot:plot(report,args.source)
    print(json.dumps(report,ensure_ascii=False,indent=2))
