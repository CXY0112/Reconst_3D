# G-CVO CPU 适配与保留的 CUDA 路径

采集页新增 **G-CVO（CPU）**、**G-CVO（CUDA）**。普通与 Beta 共用该选项；RGB-D 保持默认。重启工作台后选择 G-CVO（CPU）即可测试，原有 BUFFER-X 仍可切换。未增加 MemReg；MAC CPU 恢复分支已接入，见 `docs/MAC.md`。

相机每次开启或重新连接后，从首个同步 RGB-D 帧到达开始预热 1 秒。这一秒内持续接收并丢弃启动帧，不输出给预览、定位、关键帧或保存链路；满 1 秒后交付新到达的同步帧。该策略位于共用相机层，普通/Beta 及各配准前端均适用，离线回放不增加等待。它是固定预热时间，不是自动曝光稳定性检测。

## 两个实现的关系

- **CPU**：本项目独立实现的 float64 / Eigen / OpenMP 原生后端，位于 `gcvo_native/gcvo_cpu.cpp`。不是 GICP 替代，也不声称是作者发布的官方 CPU 实现。
- **CUDA**：保留未修改的官方 `ToyotaResearchInstitute/gcvo`，分支 `release/v2`，提交 `d1268382faf8853499e6f72abe1096ff396a1ae0`，位于 `vendor/gcvo/source`；`gcvo_native/gcvo_cuda.cpp` 仅将官方求解器接到统一 ABI。
- 当前机器已实测 CPU。CUDA 包装器和构建路径已准备，**未在 NVIDIA GPU 上编译/运行或核验 CPU/CUDA 数值一致性**。选择 CUDA 时缺库或无可用设备会明确报错，不会静默改用 CPU。

官方源码的许可证保留在 `vendor/gcvo/source/LICENSE.txt`（CC BY-NC-ND 4.0，部分文件另有来源通知）；本次未执行发布或源码包导出。

## CPU 实现范围与差异

覆盖当前官方默认二阶 GN 路径：

1. 输入米制点云，按体素质心降采样；在有界输入上估计 12 近邻协方差，使用样本协方差分母 k−1。
2. RGB 经标定投影，使用已对齐深度进行遮挡/可见性检查；颜色归一化到 [0,1]。
3. SCALAR、DENSE、RESCALED 三种核，几何核与可选 RGB 核相乘，进行相同稀疏阈值门控。
4. 构造软核关联，保留**原始输入索引顺序的前 K 个合格项**，与该官方版本一致；没有改成最近 K 点 ICP。
5. 累积默认近似 GN 曲率和梯度、SE(3) connection 修正、可选对称化、LDLT 求解与 SE(3) 指数更新；支持 EMA/队列长度尺度衰减和邻居数量衰减。
6. 用精确半径查询裁剪不可能通过门控的候选。半径使用协方差迹的保守上界，候选排序后保持输入顺序；并行逐行计算、固定顺序归约。

CPU 采用 float64，官方 CUDA 主路径采用 float32；最近邻并列点和阈值边界也可能产生差异。当前统一接口没有暴露上游实验性 H1/H2/H3 曲率项和一阶求解路径。RESCALED 的处理遵循该版本实际代码；协方差函数中的 rescaled 标志在上游当前实现中未改变协方差计算。

CPU 增加了明确的稀疏关联、数值失败和迭代上限状态。上游某些提前退出仍返回 0；CUDA 分支不能据此单独认定配准可靠，统一外部几何检查仍需执行。CUDA 包装器未提供的目标函数/Hessian 扩展诊断记录为 null。

## 接入与状态

公开统一约定：`reference_point = T @ current_point`。CPU/CUDA 都返回当前帧到参考帧的变换；适配器内部处理官方容易混淆的 source/target 命名。

跟踪以最近一个原生关键帧为参考，参考未变时用上一可信结果热启动。新关键帧复用已计算的当前局部包，不重复拟合；失败候选不推进可信参考或热启动位姿。外部验收包含刚体矩阵、几何支持点、运动跳变、双向重叠和残差。

通过后将**连续里程计坐标下的位姿**送入已有原生 RTAB-Map/iSAM2，再用优化位姿更新原有子地图。RGB 仍用于已有视觉回环；没有将优化地图位姿重复累积进里程计。RECOVERY 现在并行尝试局部 G-CVO 与后台 FPFH+MAC 初值搜索，仍以最后可信参考帧为目标，尚未加入全地图关键帧检索。

## 稳定基线、置信度与恢复状态

2026-10-27 的稳定版本冻结在 `results/baselines/gcvo_stable_20261027`：`code/` 是加入状态机前的源码/配置/本地 CPU 和外部位姿运行库，`observed/` 为完整现场日志，`dataset/` 为 8 帧固定回放输入，`manifest.json` 记录 SHA256、Python/依赖版本和现场统计。该快照复用项目现有第三方运行时，不是独立安装包；未包含凭据。不要覆盖这个目录。

`gcvo_tracking.py` 实现可解释的质量余量评分，**不是正确概率或经过真值标定的 confidence**。先通过原有收敛、SE(3)、支持点、运动和双向几何验收，再按以下分量计算：

- 重叠余量：从原有最低重叠率 0.35 线性映射至 `good_overlap=0.75`，截断到 [0,1]。
- 残差余量：从原有 RMSE 上限 35 mm 线性映射至 `good_rmse_m=0.015`（15 mm），截断到 [0,1]。
- 运动余量：`1 - max(相对参考帧平移/平移上限, 旋转/旋转上限)`，截断到 [0,1]。
- 总分为 `0.45*重叠余量 + 0.35*残差余量 + 0.20*运动余量`。续算后通过原验收的帧照常评分；仅触发续算不会被判低置信度。求解失败或几何验收失败记 0 分。

参数位于 `runtime.json` 的 `tracking` 对象，CPU/CUDA 共用 Python 策略，CUDA 尚未实机验证。`enabled=false` 可停用新增置信度提交门槛和状态切换，仍保留原几何与收敛验收。

1. 首帧只建立参考，分数为 null，不伪造 1.0；未初始化前失败不会进入没有参考帧的恢复状态。
2. NORMAL 中分数低于 `low_threshold=0.55` 的帧或明确配准失败帧暂停提交。连续 `low_frames=3` 次才进入 RECOVERY；中间一次可信且成功提交会重置计数。
3. RECOVERY 保留最后可信关键帧、里程计锚点与可信初值，继续 G-CVO 局部探测；不提交未确认位姿，不启动新的坐标原点。已有异步稠密任务可能继续完成。
4. 连续 `recovery_good_frames=3` 次达到 `recovery_threshold=0.75`，第三次候选再由原生后端接受，才返回 NORMAL。中间低分/失败或原生拒绝会清空连续确认计数。
5. 恢复使用旧参考下的相对 T 与连续里程计锚点相乘，复用既有提交与关键帧刷新流程。MAC 在不依赖旧初值的情况下搜索相对位姿，仍需与最后可信参考具有足够重叠；完全不重叠时不能承诺恢复。

界面会显示 NORMAL / RECOVERY、质量评分和连续确认进度。`gcvo_cpu_metrics.jsonl` / `gcvo_cuda_metrics.jsonl` 与 `registrations.jsonl` 包含逐帧 `tracking`；`gcvo_tracking_config.json` 保存有效策略；`gcvo_tracking_events.jsonl` 只在发生状态切换时产生。

### 可信参考图（2026-10-28）

选择 G-CVO 后，实时 RGB 旁显示“G-CVO 可信参考关键帧”：缩略图、参考节点号和拍摄时间（北京时间）。它对应当前配准实际使用的原生参考关键帧；成功跟踪但未形成新关键帧时保持原图，因此可能早于最近一次成功的视频帧。

RECOVERY 中图像固定，显示连续确认进度；请对照实时 RGB 返回与图中内容充分重叠的视角。连续确认通过并由原生后端接受新关键帧后才更新参考图。若恢复结果只形成普通跟踪帧而未产生新关键帧，参考图仍与算法使用的旧关键帧保持一致。

仅在新可信关键帧时生成最长 240×135 的独立只读缩略图，通过单独的 UI 事件传递。原图不会进入逐帧 JSON；`tracking.reference_view` 记录节点号、时间与版本号。即使该原生关键帧未进入稠密保存队列，界面仍显示它的参考图。新会话、切换前端或导入点云会清除旧图。普通版与 Beta 共用此功能，无须重新编译原生库。

本次自动测试、A/B 结果和相机验证步骤见 `results/experiments/gcvo_tracking_validation_20261027/REPORT.md`。历史日志的策略模拟并不重新计算参考位姿，不能当作真实恢复成功率。

`gcvo_cpu_metrics.jsonl` / `gcvo_cuda_metrics.jsonl` 保存求解时间、迭代次数、核关联数和几何验收。这里的 `inliers` 是采样几何近邻距离通过门槛的点数，不是 GFTT/ORB 特征内点数；它不是正确概率。`gcvo_runtime.json` 是每次扫描的参数快照。

## 本机配置

参数在 `vendor/gcvo/runtime.json`，也可通过 `RECONST_GCVO_CONFIG` 指向独立 JSON。当前室内试验：上限 3,000 采样点、输入上限 60,000、初始体素 30 mm、12 近邻协方差、RGB+DENSE 核、4 线程。实际保留点数可能低于上限，受体素和 RGB-D 可见性过滤影响。

### 迭代触顶的有限续算

首轮 `max_iterations=80`；仅在触顶时，从该轮候选位姿热启动一次最多 `retry_iterations=80` 的续算，总预算最多 160 次。将 `retry_iterations` 设为 0 可关闭续算。正常收敛帧只运行一次。续算仍须满足原始 `tolerance=0.000012`，随后通过原有双向重叠、RMSE 和运动幅度验收；不会仅凭几何重叠接受未收敛解，也不更新失败候选的可信热启动/参考帧。

CPU 续算保持首轮最终核长度尺度；EMA/窗口统计因 ABI 未导出而重新开始，因此它不是一次连续 160 次迭代的完全等价实现。CUDA 同样支持一次有界热启动，但因没有最终尺度诊断，重新使用配置中的尺度调度；该分支尚未在 GPU 实测。原生 CPU/CUDA 库及官方源码未因续算改动，无须重新编译 DLL。

失败日志也保存 `frontend.attempts`、总迭代次数、求解耗时和最终步长；每轮的迭代计数沿用原生 ABI（提前收敛时为零起始索引）。触顶后仍未收敛会明确显示本帧未提交。续算增加困难帧的延迟，总迭代预算不是墙钟时间上限。

可对保存关键帧作独立复测：

```powershell
python tools/validate_gcvo_continuation.py --source results/.pending_scan_ui_n2qmc --output results/experiments/my_continuation/report.json
```

这些保存帧已经历线上筛选，不能还原未保存的失败输入；测试从单位位姿初始化，只用于验证续算路径，不等同于实际连续跟踪的恢复率。结果保存在 `results/experiments/gcvo_continuation_20261027/report.json`。早先 release_validation / final_ab 报告仍对应加入续算之前的配置与流程。

这是室内 HP60C 尺度配置，不能把 KITTI 米级参数直接用于毫米级近距离建模。配准采样分辨率与最终地图 3 mm 融合分辨率相互独立。

## 构建 CPU

本机已有编译好的 `vendor/gcvo/bin/gcvo_cpu.dll`。重建（Reconst_3D 目录）：

```powershell
python tools/build_gcvo.py --device cpu
```

需要 CMake、C++17 编译器、Eigen。默认复用本项目已有 Eigen；在其他机器可传 `--eigen /path/to/eigen3`。不需要 CUDA、PCL 或 PyTorch。Windows 也保留 `tools/build_gcvo_cpu.cmd`。

## 在之后的 CUDA 电脑构建

保留 `vendor/gcvo/source` 和 `gcvo_native`，在具备兼容 NVIDIA 驱动、CUDA Toolkit、C++17、Eigen3、yaml-cpp、PCL common/io/filters 的环境运行：

```bash
python tools/build_gcvo.py --device cuda --cuda-architectures native
```

依赖位于非标准目录时用重复的 `--prefix /path/to/package-prefix` 提供 CMake 搜索前缀，或 `--eigen` 指定头文件。CUDA 编译器与主机编译器必须兼容；本机的无 GPU 检查不能证明其他机器的完整构建已经通过。

构建将分别放置 `gcvo_cpu.dll`、`gcvo_cuda.dll`（Linux 为 `libgcvo_cpu.so`、`libgcvo_cuda.so`）到 `vendor/gcvo/bin`，CPU 与 CUDA 版本并存。本 Windows 工作台的相机/显示链路尚不因此自动获得 Linux 支持，但离线验证脚本可在依赖满足的 Linux CUDA 机器运行。

## 离线验证与对比

回放相同已保存原始 RGB-D，不连接相机、不读取原位姿作为初值：

```powershell
python tools/replay_registration.py --source results/room2/setp1 --output results/experiments/my_gcvo_ab --frontend rgbd_gcvo_cpu --limit 8
```

也可用 `--frontend gcvo_cpu` 或 `gcvo_cuda` 单独回放。输出为可导入的数据集；两种前端可能形成不同数量的关键帧，不能简单按节点序号强行配对。回放只包含历史保存的帧，并非原始连续 10 Hz 视频；回放会等待稠密任务，不能把每帧回放总耗时当作前端耗时。

2026-10-29 起，新采集的每帧会记录 `rgbd_coordinates_v2` 坐标声明，回放自动使用 `raw_depth_frame`。旧录制没有声明时需要先核验网格，再显式传入 `--depth-frame ir` 或 `--depth-frame rgb`；工具不会凭 `sdk_xyz` 标签猜测。已核验的 `test10281` 使用 RGB 网格，例如：

```powershell
python tools/replay_registration.py --source results/test10281/setp1 --output results/experiments/my_corrected_gcvo --frontend gcvo_cpu --depth-frame rgb
```

坐标契约、修复范围及重新采集要求见 [RGBD_COORDINATES.md](RGBD_COORDINATES.md)。这次修复不会自动修复旧数据的历史轨迹。

独立于关键帧筛选的数值/配对测试：

```powershell
python tools/validate_gcvo.py --source results/experiments/my_gcvo_ab/rgbd/setp1 --output results/experiments/my_gcvo_numeric --device cpu --limit 8
```

在 CUDA 电脑保持同一输入和参数，改为 `--device cuda`。脚本保存三组已知变换误差、每个相邻帧的全云几何一致性、求解/预处理分项时间，并生成 `known_pair.npz` 作为后续跨设备核对输入。实际相邻帧没有独立真值，因此这些残差不等于真实精度。

CPU/CUDA 库都构建后，可对同一个已准备输入执行差分核对：

```bash
python tools/compare_gcvo_devices.py --packet results/experiments/my_gcvo_numeric/known_pair.npz --output device_comparison.json
```

运行时应使用与生成该输入包相同的 `RECONST_GCVO_CONFIG`；保持核函数和求解参数一致。它报告两个后端的位姿差异，不会用 CPU 结果替代 CUDA，也不把单个输入通过当作全部输入等价。

本机最终测试说明与文件位于 `results/experiments/gcvo_cpu_release_validation_20261027`，最终原生回放目录为 `results/experiments/gcvo_cpu_final_ab_20261027`。CPU 是可用试验前端，是否满足严格 10 Hz 必须看包含预处理与几何验收的实际跟踪耗时，不能只看 GN 求解耗时。


## 暂停后的续接（2026-10-30）

暂停停止采样，预览继续显示；已排队的帧仍可能完成处理。因此续接目标是界面显示的最后可信关键帧，不一定是按下暂停前最后看到的预览帧。

继续后，首个实际进入跟踪线程的新采样帧会触发恢复：清除旧位姿预测与旧 MAC 任务，进入 RECOVERY。新帧与最后可信 RGB-D 参考执行 ORB 匹配、带深度的 PnP RANSAC，生成当前帧的参考相对位姿初值。PnP 使用参考深度和两帧内参，输出方向反转为 reference <- current。至少 20 个内点且内点比例 >=30%、重投影中位误差 <=3px 才用于初始化。

图像初值交给现有 G-CVO 精化；仍要求求解收敛、双向几何重叠/RMSE、运动范围、质量评分、连续三帧确认及原生后端接受。图像种子失败时使用既有局部配准/MAC 恢复。新初值也用于普通 RECOVERY 状态；正常跟踪不运行 ORB/PnP。

frames.jsonl 新增 tracking_epoch 与 resumed 标记，暂停后首帧即使因 latest-only 队列丢弃，后续帧仍携带相同新 epoch，恢复通知不会丢失。通知在处理线程执行，避免 UI 线程直接改跟踪器。

验证：22 项采集/跟踪/续接测试通过；生成的 12cm RGB-D 平移经真实 ORB/PnP + G-CVO CPU 求解通过，平移误差约 1.52mm。这是受控生成测试，不代表真实相机续接成功率。真实手持暂停、移动和回位仍需采集验证；建议回到参考缩略图对应位置，镜头停稳 2–3 秒。记录见 results/experiments/gcvo_resume_20261030/REPORT.md。


## 独立替代配准（2026-10-30 更新）

G-CVO 在恢复阶段无法收敛时，替代算法现在可以独立完成恢复：**RGB-D ORB/PnP + 双帧深度验证**，或 **MAC 初值 + 当前帧多尺度 ICP**。两者通过全云几何、质量评分及连续三帧确认后即可提交原生后端，不再要求再次通过 G-CVO。原有地图与里程计参考保留，后续正常帧继续用 G-CVO。

图像路径要求至少20个且>=70% PnP内点同时通过4cm双帧深度一致性；MAC路径始终针对当前帧精化，保留MAC任务编号/参考/时间有效性检查。恢复位姿之间跳变会重置连续确认。界面显示实际替代方法及MAC后台状态，日志保存失败原因。详见 results/experiments/gcvo_independent_recovery_20261030/REPORT.md。本更新替代上节“所有初值都必须由 G-CVO 精化”的限制；采集暂停、epoch传递和既有安全门限继续适用。

32 项离线测试通过，包括“G-CVO持续失败仍能恢复”及“恢复后下一帧真实 G-CVO CPU 成功”。真实手持续接效果需要重新采集验证，建议保存为 test1030_fallback。
