# BUFFER-X 配准前端实验

采集页新增“配准前端”：`RGB-D（现有方案）`、`BUFFER-X（实验／CPU）`。普通与 Beta 界面都能选择；开始扫描后锁定，下一次扫描可以切换。原有 RGB-D 仍为默认，不替换其原生二进制。

BUFFER-X 完整替换连续跟踪中的 GFTT/ORB 提取、视觉匹配、PnP 位姿求解。输入为米制 RGB 光学坐标点云；每次将当前帧配准到最近一个原生关键帧，返回 current→reference 变换。以连续里程计坐标累计，再送入 RTAB-Map；优化位姿只用于稠密融合，避免回环修正被重复施加。

本实验保留 **RTAB-Map 视觉回环检索与几何验证、iSAM2、3 mm 子地图、关键帧筛选、二阶段处理与保存格式**。因此 RGB 仍用于回环和显示。本次没有实现纯几何回环或局部点云地图跟踪；BUFFER-X 当前对比对象为上一个原生关键帧，不等同于原有 F2M 特征地图。BUFFER-X 是配准器，不是完整 SLAM 系统。

## 使用

1. 重新启动 `python main.py`（或 `python main.py beta`），在采集页选择 BUFFER-X 后开始新扫描。
2. 当前本机使用 CPU；完整三尺度、1500 个 FPS 点与官方室内权重，单对推理约分钟级。相机预览独立，但处理繁忙只保留最新输入，连续移动可能造成重叠不足。建议先做下面的离线对比。
3. 用不同数据名称保存两种方案，不覆盖原扫描。保存结果继续用于预处理、几何重建和成果展示。

同一批已录制观测对比（在 Reconst_3D 目录运行，output 必须是新目录）：

```powershell
python tools/replay_registration.py --source results/room2/setp1 --output results/experiments/my_bufferx_ab --frontend both --limit 10
```

`--frontend rgbd` 或 `bufferx` 可单独运行；`--limit 0` 回放全部已保存帧。输出为 `rgbd/setp1`、`bufferx/setp1`，都能导入工作台。每帧状态与耗时记录在对应 `*_frames.jsonl`，汇总在 `summary.json`。

回放以同一批原始 RGB、深度与标定重建两种前端输入，不读取原扫描位姿作为猜测。旧扫描只保存了保留帧，因此它不是原始 10 Hz 序列；回放点云来自标定深度而非原始 SDK XYZ。离线逐帧等待稠密任务完成，避免队列负载差异混入比较。报告重叠/RMSE 是内部一致性指标，不是真值精度。

## 环境与来源

官方：[BUFFER-X](https://github.com/MIT-SPARK/BUFFER-X)，固定提交 `07dd9be5a58c9c02213584535592766c9023cebc`，室内 `threedmatch/Desc` 与 `Pose` 官方权重。许可证保存在 `vendor/bufferx/source/LICENSE`。

官方完整推理要求 Linux/NVIDIA CUDA。本项目在独立进程增加 CPU 推理适配：CPU FPS、球邻域查询、gather/group、KNN、PyTorch SVD；球网格改为跟随输入设备；描述子按 24 个局部块分批执行以降低内存。网络层与权重不变，三尺度与特征数量不降低。CPU 与 CUDA 的数值、采样并列点处理可能有差异；未进行 CPU/CUDA 等价性实测，不将其标称为官方原生 CPU 支持。

本机独立 venv 位于 `vendor/bufferx/env`，复用已有 `vendor/nksr_windows/env` 的 PyTorch/Open3D，不修改基础环境。重建：

```powershell
python tools/setup_bufferx.py
.\tools\build_bufferx_bridge.cmd
```

安装需要 Git/联网；基础 Python 应已具备 PyTorch、Open3D、SciPy、Matplotlib，可用 `--base-python` 指定。原生构建使用本工作区已有 `../experiments/isam2_build` 的 0.23.8 头文件和导入库、Eigen、MSVC，因此不是独立源码包的一键移植脚本。

`vendor/bufferx/runtime.json` 保存解释器、源码、权重、设备、超时与几何验收门槛。可通过 `RECONST_BUFFERX_CONFIG` 指定另一个配置。配准失败不会回退视觉前端；运行环境/进程错误会明确终止并保留恢复目录。

## 验收与诊断

- `bufferx_metrics.jsonl`：接受/拒绝、对应数、内点数、三尺度、降采样尺度、几何验证指标与耗时。
- `bufferx_worker.log`：官方网络推理日志；`bufferx_runtime.json`：该次运行固定的参数快照。
- `optimizer.json`：原生 iSAM2 与外部位姿接口握手。BUFFER-X 使用独立的 `vendor/online_bufferx`，RGB-D 使用原有 `vendor/online_geometry`。
- `geometry_poses.json` 标记 `backend=bufferx`、`loop_closure=rtabmap_visual`。完整原生约束仍在 `map.db`。
- 前端验收包含刚体矩阵、内点数、双向采样重叠、最近邻 RMSE 与运动跳变门槛；重复结构仍可能产生误匹配。原生图采用保守平移方差与固定旋转方差，尚未标定为 BUFFER-X 的统计协方差。

已知变换的权重推理检查：

```powershell
python tools/validate_bufferx.py --source results/room2/setp1/pointcloud_local/1792738933.968000.npz --output results/experiments/my_bufferx_known_pose
```

最终默认配置保留官方室内 `pose_refine=True`：在匹配对应上迭代筛选内点、用加权 SVD 精化刚体变换，不增加 ICP。结果见 `results/experiments/bufferx_refined_known_20261027/report.json`：同一录制点云施加已知刚体变换，平移误差 **0.717 mm**、旋转误差 **0.0279°**、单对 **105.1 s**。这是同支持区域的受控测试，不代表真实相邻帧、低重叠场景或长期轨迹精度。未连接相机验证。

最终三帧真实观测对比位于 `results/experiments/bufferx_refined_ab_20261027`；`rgbd` 和 `bufferx` 两个子目录可分别导入工作台。未开启匹配点精化的首轮诊断另外保留在 `bufferx_known_transform_20261027` 和 `bufferx_ab_20261027`，不作为最终默认结果。
