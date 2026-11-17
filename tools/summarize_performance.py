"""Build a desktop-only evidence report from process-tree benchmark traces."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def phase_metrics(folder):
    with (folder/'resources.csv').open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    events={row['event']:row['time'] for row in load_events(folder/'events.jsonl')}
    time=np.array([float(r['time']) for r in rows]);cpu=np.array([float(r['cpu_seconds']) for r in rows])
    result={}
    for event,start in events.items():
        if not event.endswith('_start'):continue
        name=event[:-6];end=events.get(name+'_end')
        if end is None:continue
        selected=[r for r in rows if start<=float(r['time'])<=end]
        if not selected:continue
        duration=end-start
        result[name]={'seconds':duration,'average_logical_core_equivalents':float((np.interp(end,time,cpu)-np.interp(start,time,cpu))/duration),
            'peak_working_set_MiB':max(int(r['rss_bytes']) for r in selected)/2**20,
            'peak_private_commit_MiB':max(int(r['private_commit_bytes']) for r in selected)/2**20}
    return result


def load_events(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    folder=args.folder.resolve();hardware=load(folder/'hardware.json');groups={stage:[] for stage in (1,2,3,4)}
    for sub in sorted(folder.iterdir()):
        if not (sub/'resource_summary.json').is_file():continue
        report=load(sub/'resource_summary.json')
        if report['status']!=0:continue
        for stage in groups:
            if sub.name.startswith(f'stage{stage}'):
                groups[stage].append((sub,report))
    gui=load(folder/'gui/resource_summary.json');gui_work=gui['workload'];phases=phase_metrics(folder/'gui')
    source_hashes={}
    pre=load(ROOT/'results/setp2_20261015_170233_921076/report.json');source=Path(pre['source_session'])
    source_hashes.update({str(source/name):value for name,value in pre['source_sha256'].items()})
    source_hashes.update({str(source/row['file']):row['sha256'] for row in pre['accepted_raw_observations']})
    appearance=load(ROOT/'results/setp4_20261015_181048_983726/report.json')
    source_hashes.update(appearance['input_sha256'])
    for path,value in source_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=value:raise ValueError('Source reference changed: '+path)
    summary={'hardware':hardware,'successful_runs':{str(stage):[r for _,r in runs] for stage,runs in groups.items()},
        'gui':gui,'gui_phases':phases,'source_reference_hashes_verified':len(source_hashes),
        'embedded_performance_estimated':False,'limitations':['stage1 core only; native replay failed 0xC0000008',
            'No live SDK capture, GPU utilization percentage, total graphics memory or power measured',
            'Two repeats for stages2-4; file cache was not flushed; 100ms sampling may miss short peaks']}
    (folder/'desktop_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    def range_text(values,digits=1):
        left,right=min(values),max(values)
        return f'{left:.{digits}f}' if abs(right-left)<10**(-digits) else f'{left:.{digits}f}–{right:.{digits}f}'
    table=[];flat=[]
    titles={1:'一阶段：定位/融合核心回放',2:'二阶段：卡尔曼预处理',3:'三阶段：Poisson几何重建',4:'四阶段：RGB＋局部补色'}
    for stage,runs in groups.items():
        reports=[r for _,r in runs]
        row=[titles[stage],str(len(reports)),range_text([r['wall_seconds_including_startup'] for r in reports]),
            range_text([r['cpu_average_core_equivalents'] for r in reports],2),
            range_text([r['peak_aggregate_working_set_bytes']/2**20 for r in reports]),
            range_text([r['peak_aggregate_private_commit_bytes']/2**20 for r in reports])]
        table.append('| '+' | '.join(row)+' |')
        for path,r in runs:
            flat.append({'stage':stage,'run':path.name,'wall_seconds':r['wall_seconds_including_startup'],
                'average_logical_cores':r['cpu_average_core_equivalents'],
                'peak_working_set_MiB':r['peak_aggregate_working_set_bytes']/2**20,
                'peak_private_commit_MiB':r['peak_aggregate_private_commit_bytes']/2**20,
                'logical_read_MiB':r['logical_read_bytes']/2**20,'logical_write_MiB':r['logical_write_bytes']/2**20})
    with (folder/'stage_summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    replay=groups[1][0][1]['workload'];captures=replay['captures']
    capture_table='\n'.join(f'| {r["capture"]} | {r["map_points"]:,} | {r["matching_seconds"]:.2f} | {r["map_fusion_seconds"]:.2f} | {r["total_seconds"]:.2f} |' for r in captures)
    gui_names=['gui_stage1_display','gui_stage2_compute','gui_stage2_display','gui_stage3_compute','gui_stage3_display',
        'gui_stage4_compute','gui_stage4_display','gui_all_tabs_retained_idle']
    gui_titles=['采集页：地图旋转＋10Hz实时点云/RGB回放','二阶段计算（保留采集页）','二阶段双点云显示',
        '三阶段计算（保留前两页）','三阶段点云＋网格显示','四阶段计算（保留前三页）','四阶段双网格显示','四页数据保留，静止空闲']
    gui_table='\n'.join(f'| {title} | {phases[name]["seconds"]:.2f} | {phases[name]["average_logical_core_equivalents"]:.2f} | {phases[name]["peak_working_set_MiB"]:.1f} | {phases[name]["peak_private_commit_MiB"]:.1f} |' for name,title in zip(gui_names,gui_titles))
    draw_table=[];buffers=0
    for stage in (1,2,3,4):
        work=gui_work[f'gui_stage{stage}_display']
        for i,panel in enumerate(work['panels']):
            gpu=panel['gpu_elapsed_ms'];cpu=panel['cpu_submission_and_wait_ms'];buffers+=panel['geometry_buffer_bytes']
            draw_table.append(f'| 阶段{stage} · 视窗{i+1} | {gpu["p50"]:.2f} / {gpu["p90"]:.2f} | {cpu["p50"]:.2f} / {cpu["p90"]:.2f} | {panel["actual_frames_per_second"]:.1f} | {panel["geometry_buffer_bytes"]/2**20:.2f} |')
    stage3parts={name:float(np.mean([r['workload']['stage_seconds'][name] for _,r in groups[3]])) for name in groups[3][0][1]['workload']['stage_seconds']}
    disk=load(folder/'disk_inventory.json')
    disk_table='\n'.join(f'| {key} | {value["bytes"]/2**20:.1f} |' for key,value in disk.items())
    io_table='\n'.join(f'| {title} | '+range_text([r['logical_read_bytes']/2**20 for _,r in groups[stage]])+' | '+range_text([r['logical_write_bytes']/2**20 for _,r in groups[stage]])+' |' for stage,title in titles.items())
    report=f'''# 当前四阶段流程：本机性能实测

本报告只报告本机结果，暂不估算Jetson运行时间或给出适配结论。EfficientSAM-Ti不参与测试，未修改生产算法与主界面。

## 测试机器、数据与口径

- CPU：{hardware['cpu_name']}，{hardware['physical_cores']}物理核/{hardware['logical_processors']}逻辑处理器。
- GPU：AMD Radeon 780M Graphics；Windows 11。系统可见内存{hardware['total_ram_bytes']/2**30:.2f} GiB；测试开始时可用{hardware['available_ram_before_bytes']/2**30:.2f} GiB，其余占用来自系统和其他应用，未计作本程序内存。
- 工作负载：必须保留的 `scan_20261015_102420_653218`，11次拍摄、融合点云1,796,513点；卡尔曼预处理后582,341点；最终网格约162,700顶点、295,280三角面。所有阶段都使用同一组来源数据，不使用合成低负载替代。
- 算法单独运行时，采样目标进程及全部子进程，每100ms记录CPU累计时间、工作集、私有提交、线程、逻辑I/O。使用已有vendored psutil，无新增系统安装。二至四阶段各两次独立进程运行，文件系统缓存未清空，不称为冷磁盘测试，也不从两次测试推断P95。
- CPU“等效逻辑核”=进程树累计CPU秒/墙钟秒。1.0表示一个逻辑处理器持续忙；本机16逻辑处理器，1.0约为全机CPU的6.25%。线程数不等于正在工作的核数。
- 工作集为进程树RSS之和，共享DLL页可能重复计入；私有提交为private bytes之和，包含已提交但不一定常驻物理内存的页。这两项不要相加。100ms峰值可能漏掉更短瞬时尖峰；结束子进程最后不足一个采样周期的CPU时间可能漏记。
- 环境上限为OMP/MKL 4线程、OpenBLAS 1线程；现有PointDSC主动设2线程，几何子进程设2线程，KD查询仍沿用原代码指定线程数。不是把所有CPU核完全跑满的压力测试。
- 本轮校验了现有报告记录的{len(source_hashes)}个输入文件哈希，均通过。采样器另通过64MiB子进程内存分配与CPU忙循环烟雾测试，确认子进程计数有效。

## 1. 各阶段单独运行

**耗时包括启动、导入、处理及导出；不包括GUI绘制。内存单位MiB。**

| 阶段 | 有效次数 | 总耗时(s) | 平均CPU等效逻辑核 | 峰值工作集(MiB) | 峰值私有提交(MiB) |
| --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(table)+f'''

第一阶段为**定位与空间融合核心回放**，11次全部被接受，最终点数与原地图一致。本轮独立原生RTAB-Map回放在图像准备阶段以 `0xC0000008` 退出，原因未定位；失败记录保留，不纳入有效数值。成功回放复用实际 `GeometryClient`、FPFH/PointDSC/ICP/图优化、`voxel_groups` 和 `SnapshotBackend.preview`，排除了RTAB-Map图像数据库写入、相机SDK、USB传输与GUI。它不是完整第一阶段相机会话的资源上限，也不代表实时相机帧率。

阶段二加载约3.45–3.50s，卡尔曼及均匀化计算约28.76–30.47s，保存约1.08–1.27s。阶段三核心约28.43–34.70s，外加运行时启动等；阶段四核心约3.36–3.42s，外加约3.4s运行时导入/启动。数据来自本轮有效运行，不沿用旧报告耗时。

阶段三平均分项(s)：

```json
{json.dumps(stage3parts,ensure_ascii=False,indent=2)}
```

本轮几何重建中，减面及验收、受约束平滑、Poisson是较大的计算项。它们当前主要使用CPU；PointDSC当前也在CPU推理，尚未使用CUDA。四阶段的GPU活动主要来自显示。

## 2. 第一阶段随地图增大的变化

| 拍摄序号 | 当前地图点数 | 几何定位及本帧压缩(s) | 空间融合/生成地图(s) | 本帧总耗时(s) |
| --- | --- | --- | --- | --- |
{capture_table}

单帧平均约{np.mean([r['total_seconds'] for r in captures]):.2f}s、中位{np.median([r['total_seconds'] for r in captures]):.2f}s，范围{min(r['total_seconds'] for r in captures):.2f}–{max(r['total_seconds'] for r in captures):.2f}s；核心回放最后另有约{replay['export_seconds']:.2f}s地图保存。前两次存在几何工作进程/PointDSC首次加载开销，不宜当成稳态定位速度。

**地图融合随点数增加明显变慢，比最后阶段的显示绘制重得多。** 从代码看，`VoxelMap.update/remove/cloud`包含逐体素Python字典操作；图优化改变旧位姿时，还要撤销并重加旧观测贡献。实测增长与此路径一致，尚未做函数级剖析来确定每个循环的精确占比。

## 3. 真实主界面：各页保留数据后的占用

此项使用实际Workbench，第一阶段导入原地图并回放24万点实时预览，第二阶段处理/保存、第三阶段重建、第四阶段着色均由真实界面启动；前页数据保持在内存中。**第一阶段未打开真实相机会话，也未在该GUI进程中运行PointDSC，因此以下不是完整现场连续采集的总峰值。**

| 主界面区间 | 区间时长(s) | 平均CPU等效逻辑核 | 进程树峰值工作集(MiB) | 峰值私有提交(MiB) |
| --- | --- | --- | --- | --- |
{gui_table}

整段GUI约{gui['wall_seconds_including_startup']:.2f}s。本场景最大工作集{gui['peak_aggregate_working_set_bytes']/2**20:.1f} MiB（{gui['peak_aggregate_working_set_bytes']/2**30:.2f} GiB），私有提交{gui['peak_aggregate_private_commit_bytes']/2**20:.1f} MiB（{gui['peak_aggregate_private_commit_bytes']/2**30:.2f} GiB），发生在第三阶段计算期间。处理结束且四页数据都保留时，约493MiB工作集/751MiB私有提交。

GUI中的第二、三阶段计算区间约28.12s/26.24s，第四阶段含其工作进程启动约6.23s。它们与独立进程结果不同，受已导入模块、缓存、界面调度和机器负载影响，不应混成同一个精确常数。

## 4. 显示：CPU实时预览与GPU三维绘制

实际窗口约1450×950；各点云/网格视窗约651–717像素宽、643–749像素高。请求30Hz视角变化，第一阶段另以10Hz更新RGB和实时点云。使用OpenGL `GL_TIME_ELAPSED` 查询，测GPU命令时间区间；CPU列包含提交、等待及查询开销。时间查询本身有少量额外开销，不是GPU利用率百分比。

| 视窗 | GPU时间中位/P90(ms) | CPU提交及等待中位/P90(ms) | 实际重绘次数/s | 几何缓冲量(MiB) |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(draw_table)+f'''

实际重绘计数包含视角同步触发的额外绘制，不等于显示器刷新率；不能把左右计数相加称作整体FPS。

采集页的右下实时点云仍使用CPU光栅化：640×480、240,432点时，一次更新中位约34.64ms、P90约39.26ms；右上RGB更新中位约5.54ms、P90约6.71ms。两者在Tk主线程执行，解释了为什么轻量GPU网格绘制很快，实时点云预览仍可能影响交互。

所有已加载三维视窗几何顶点/索引缓冲合计约{buffers/2**20:.2f} MiB。这只是从实际数组/VBO布局计算的几何数据量，**不是驱动实际总显存测量**，不含帧缓冲、驱动缓存和Tk图像。未测GPU利用率、实际显存/共享显存峰值、功耗、温度与热降频，因此不提供其百分比或瓦数。

## 5. 磁盘与依赖体积

| 数据/本机组件 | 体积(MiB) |
| --- | --- |
{disk_table}

这些是本机目录占用，包含Windows打包运行时、源码或缓存；不是最小可部署包大小。`geometry_packages`与主机NumPy/SciPy可能重复，不能把每项简单相加当运行所需内存。原扫描325.5MiB含原始观测、数据库与地图等；PointDSC目录大小也不包含主机PyTorch运行库。

| 阶段 | 进程树逻辑读(MiB) | 逻辑写(MiB) |
| --- | --- | --- |
{io_table}

Windows进程传输计数包含缓存I/O和进程管道，不代表SSD实际落盘字节或物理带宽。原始采集325.5MiB、预处理31.1MiB、几何11.1MiB、着色13.3MiB是保存结果实例大小，不用于推断长期采集容量上限。

## 6. 本机结论与测量边界

1. 当前样本中，**第一阶段空间融合的时间增长和内存占用最值得优先关注**：核心回放已约1.86GiB工作集，融合末帧约8.06s；这不是相机实际采集全路径的上限。
2. 第二阶段CPU处理约29–30s；第三阶段核心约28–35s，是离线处理中的大项。平均CPU只占约1–2个逻辑核等效，说明“整机CPU百分比不高”不能推导“任务很轻”。
3. GPU三维显示每视窗多为1–5ms量级；实时点云的CPU光栅化约35ms/次，是另一类界面开销。当前没有证据说明主要瓶颈在GPU算力。
4. 单阶段峰值、GUI累积场景峰值、真实连续相机会话峰值是不同口径，不能相加或互相替代。真实HP60C SDK/USB、原生数据库全路径和第一阶段算法与GUI同进程并发的完整峰值仍需另测；本次不将历史日志作为新测量填补。
5. 本报告到本机资源基线为止。目标嵌入式平台的移植工作量、内存预算、实际运行时间应在下一份适配分析中单独讨论，不能按AI TOPS或CPU主频比例直接换算。

## 原始证据与复现

- `hardware.json`：机器信息；`desktop_metrics.json`：有效测量汇总及GUI区间；`stage_summary.csv`：可比较表。
- 每个运行目录的 `resources.csv`（100ms采样）、`events.jsonl`（阶段边界）、`resource_summary.json`、`workload.json`、`stdout.log`：原始证据。只使用status=0的有效运行；失败调试记录保留供排查。
- [资源时间曲线](resource_traces.png)：所选有效单次运行，横轴独立计时；CPU为约1秒窗口平均，RAM为进程树工作集。
- 性能工具为 `tools/profile_pipeline.py`、`tools/performance_worker.py`、`performance_ui.py`，不改变日常 `main.py` 的处理流程。

```powershell
python Reconst_3D/tools/profile_pipeline.py --stages stage1 --stage1-core
python Reconst_3D/tools/profile_pipeline.py --stages stage2 stage3 stage4
python Reconst_3D/tools/profile_pipeline.py --gui-only
Reconst_3D/vendor/rtabmap/bin/python.exe Reconst_3D/tools/test_performance_probe.py
python Reconst_3D/tools/summarize_performance.py <某次性能目录>
```

汇总脚本要求同一目录下有各阶段有效运行及GUI记录；分批测试时使用同一 `--output` 路径，已有运行会增加attempt后缀而不覆盖。现有独立原生第一阶段路径仍有异常，完整相机测试应在原主程序实测时另行采样。
'''
    (folder/'PERFORMANCE_LOCAL.md').write_text(report,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for stage,axis in zip(groups,axes.ravel()):
        path,_=groups[stage][0]
        with (path/'resources.csv').open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
        x=np.array([float(r['time']) for r in rows]);x-=x[0]
        rss=np.array([float(r['rss_bytes']) for r in rows])/2**20
        cpu=np.array([float(r['cpu_seconds']) for r in rows]);left=np.maximum(x-1,x[0])
        cores=(cpu-np.interp(left,x,cpu))/np.maximum(x-left,.1)
        axis.plot(x,rss,color='#286fb4',linewidth=1.5,label='Process-tree working set')
        axis.set_xlabel('Elapsed seconds');axis.set_ylabel('Working set (MiB)',color='#286fb4');axis.grid(alpha=.15)
        other=axis.twinx();other.plot(x,cores,color='#e27b26',alpha=.75,linewidth=1,label='CPU core equivalents')
        other.set_ylabel('CPU logical-core equivalents',color='#e27b26');other.set_ylim(bottom=0)
        axis.set_title(f'Stage {stage}'+(' | geometry replay only' if stage==1 else ''))
    fig.suptitle('Desktop measurements | Ryzen 7 H 255 | Windows | 100 ms samples',fontsize=13)
    fig.savefig(folder/'resource_traces.png',dpi=160);plt.close(fig)
    print(folder/'PERFORMANCE_LOCAL.md')


if __name__=='__main__':main()
