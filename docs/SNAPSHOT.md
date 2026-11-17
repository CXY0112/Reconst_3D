# 历史手动几何图网络拍摄融合（V1–V4）

本文记录历史手动后端，已不作为普通版与Beta默认流程。当前V5采用[原生RTAB-Map连续RGB-D建图](RTABMAP_V5.md)，请以该文为准。

```powershell
python Reconst_3D/main.py
```

默认采用手动拍摄：开始会话 → 拍摄当前位置 → 移动后拍摄有重叠的区域 → 完成 → 选择保存。右侧持续预览；只有定位可靠的照片加入左侧地图。模型冷启动可能需要数秒，计算在后台进行。

## 导入已有点云

无需连接相机，点击顶部“导入点云”选择文件，后台读取后在左侧显示。支持 ASCII/二进制 PLY、PCD、NPY、NPZ、XYZ、TXT、CSV。查看保留全部有效点，自动过滤 NaN/Inf；不会改写源文件，也不会把导入数据作为后续拍摄的融合基准。扫描或保存过程中暂不可导入。

点击左侧获取焦点后，W/S 前后、A/D 左右、Q/E 上下移动视点，按住鼠标左键转向，R 或双击复位。导入成功自动适配视野；导入失败保留原显示。

导入前可选择单位（米/毫米）及坐标（世界 X前Y左Z上 / 相机 X右Y下Z前）。自动模式识别本项目 PLY 注释、`full_geometry_world.npy`、`pointcloud_raw` 文件夹内的原始毫米点云，以及 NPZ 的 `points_mm`、`points_world` 或 `unit`/`coordinate_frame` 元数据。无标注文件默认**米、世界坐标**，外部数据请按其实际约定选择。NumPy 支持 N×3 或 H×W×3 数值数组；NPZ 支持 `points_mm`、`points_world`、`points`、`point_cloud`、`xyz`、`arr_0`；文本前三列为 XYZ，CSV 可含 x,y,z 表头。当前查看统一用蓝色显示几何，不还原文件的 RGB 属性。

启动后在“一阶段 · 点云采集与查看”选项卡点击“导入点云”，选择保存目录中的 `map_cloud.ply`。顶部可随时切换至“二阶段 · 点云预处理”，不需要启动参数。

## 当前默认定位流程

1. 优先读取与深度帧 ID、时间戳一致的 SDK XYZ（IR 坐标、毫米）。缺失时用 RAW16 和出厂标定反投影；完整几何通过 IR→RGB 出厂 R/T 转到统一相机坐标。
2. Open3D 清理几何描述用的离群点，计算 5cm 采样的 FPFH 几何特征。关键帧的特征、搜索树、法线及多尺度点云缓存复用。
3. 使用官方 **PointDSC 室内预训练网络**，按空间一致性筛选 FPFH 候选对应并估计初始刚体变换。PyTorch CPU 推理，每个候选最多 512 对，每次最多尝试 3 个参考关键帧。这些限制仅用于求位姿，不限制显示或保存的地图点数。
4. 以 3cm、1.5cm、7.5mm 三个尺度执行带 Tukey 鲁棒核的 point-to-plane ICP，抵抗部分离群测量。最近邻阈值依次为 8cm、4.5cm、2.5cm，每层最多 20 次迭代。
5. 检查双向重叠、几何残差、精修幅度和近似平面退化；多个参考位置给出明显冲突、但得分接近的结果时拒绝。当前门槛为源点重叠至少 25%、目标至少 15%、残差最多 15mm，ICP 相对粗位姿最多修正 20cm/20°。这些是工程门槛，不代表测量精度保证。
6. 新拍摄与最多三个一致参考位置建立几何边，固定首帧，对全部拍摄位姿进行 Open3D 鲁棒位姿图联合优化。低可信闭环边可被剪除；不把所有误差仅施加到新帧上。
7. 将联合优化位姿和几何约束同步到 RTAB-Map 数据库。默认几何模式关闭其额外 RGB 闭环/近邻精修，避免重新被视觉结果覆盖。

当前 RGB 用于原始记录和 RGB 对齐数据导出，**不作为几何定位必须通过的门槛，也尚未用于辅助重定位**。旧视觉快照前端已移除；默认直接使用 `geometry_snapshot_backend.SnapshotBackend`。

## 融合规则

最终地图采用 **3mm 体素，每个体素一个融合质心**，同格的新旧测量参与均值融合。位姿图优化移动旧节点时，先移除该节点原贡献，再按新位姿重新融合。因此既不是点云直接叠加，也不是只调整新帧而永久冻结旧地图。

不设地图全局总点数上限。新空间会增加体素数量，重复扫描在同一体素中不会无限增加输出点数；传感器噪声仍可能占据邻近体素而产生一定表面厚度。当前为点云质心融合，尚非 TSDF 或网格表面重建。原始拍摄和各关键帧几何仍保留，以支持重新优化，历史内存和磁盘不保证恒定。

配准用的采样不会替换最终的完整 SDK 几何。3mm 是输出空间分辨率，不是硬件精度或绝对定位精度。

## 显示与视角

左侧显示融合地图与扫描节点，拍摄成功后更新；右侧采用旧“三模式图像显示测试.py”的 640×480 黑底针孔投影、2×2 点放大、中心光轴和固定 0.45–4m 深度色阶（近红远蓝）。距离色仅用于显示，不作为网络输入。

右侧点击获得焦点（蓝框），拖动使虚拟摄像机原地转头；W/S 前后、A/D 左右、Q 上、E 下。滚轮前后移动，双击或 R 复位。按键松开或失焦停止移动，新实时帧保留虚拟视点。该操作不修改测量坐标与融合位姿。

左侧也采用独立虚拟摄像机：点击左侧获得蓝框焦点，按住左键拖动使视点原地转头；W/S 沿观察方向前后移动、A/D 左右、Q 上/E 下，滚轮前后移动。双击或 R 将视点重新放到可以看到全图的位置。新融合地图更新时保留已有视点，不绕点云中心旋转，不修改世界坐标。左右框各自独立，按键仅控制获得焦点的一侧。

左侧已替换 Matplotlib，使用 Windows 原生 OpenGL/WGL 和 GPU 顶点缓冲。完整点云仅在地图变化时上传显存，移动视点只更新相机矩阵；按键/鼠标事件合并重绘，取消逐点 CPU 三维重排，不限制显示总点数。AMD Radeon 780M 上，1,796,513 点的 GPU 完成绘制中位约 4.05ms、95分位约5.27ms（独立绘图测试，不含地图配准与体素融合耗时）。GPU 不可用时使用后台 CPU 投影，只显示最新完成帧，避免在界面线程投影百万点。

右侧实时帧只保留最新一帧，目标每 100ms 更新。两侧绘图互相独立；配准和体素融合的计算开销未被此次显示修改消除。底栏继续报告配准、图优化、融合和左侧绘图耗时。

## 保存内容

- `map.db`、`map_poses.txt`：RTAB-Map 数据及同步后的几何优化位姿/约束。
- `map_cloud.ply`：完整 SDK 几何经优化和体素融合的主地图，米制 RTAB-Map 世界坐标（X 前、Y 左、Z 上），仅含 XYZ。
- `geometry_pose_graph.json`：Open3D 联合优化图，保留几何信息矩阵。
- `geometry_poses.json`：节点绑定、光学相机到世界的优化位姿及几何边。
- `full_geometry_world.npy`：最终融合世界点云；`map_cloud_rgb_registered.ply`：RGB 对齐深度的次要导出。
- `pointcloud_raw/`：每次拍摄的 IR XYZ（毫米）；`depth_raw/`、`rgb_raw/`：原始深度和 RGB；`depth/`、`rgb/`：RGB 对齐副本。
- `frames.jsonl`：帧 ID、时间戳、点云来源；`registrations.jsonl`：接受/拒绝及各候选原因；各阶段耗时日志。

`geometry_input.npy`、`geometry_correspondences.npz` 是几何工作进程重复使用的临时交换文件，不代表完整扫描历史。

现阶段支持同一会话内不同时间拍摄；关闭程序后恢复旧地图继续建图尚未实现。已有旧版错位地图不会自动修复。

## 运行环境与来源

主进程使用现有 Python、NumPy、OpenCV、Tkinter 和已安装的 PyTorch CPU，当前显示无需 Matplotlib。Open3D 0.19.0、NumPy 1.26.4、SciPy 1.14.1 安装在项目 `vendor/geometry_packages`，由 `vendor/rtabmap/bin/python.exe` 的独立 Python 3.12 工作进程运行，避免更改系统 Python 和二进制版本冲突。

几何模式使用单独的 `vendor/online_geometry/online_bridge.dll`，重新构建：

```powershell
python Reconst_3D/tools/build_online.py --geometry
```

PointDSC正式推理权重和配置位于 `models/pointdsc`，完整原仓库已外置归档。推理代码适配与来源见 [pointdsc_impl/PROVENANCE.md](../pointdsc_impl/PROVENANCE.md)。权重约 4.4MB，不代表完整运行环境或进程内存大小；未对 HP60C 重新训练。其室内权重配置标记使用 FCGF 训练，官方示例支持替换为 FPFH；本版使用 FPFH 以避免 MinkowskiEngine/CUDA 依赖。

来源：[PointDSC 官方仓库](https://github.com/XuyangBai/PointDSC)、[Open3D 全局配准](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html)、[Open3D 多帧配准](https://www.open3d.org/docs/release/tutorial/pipelines/multiway_registration.html)。

## 验证

当前实测：将 `scan_20261015_093625_137777` 的 RGB 全部置黑回放，10 张均接受，生成 10 节点、24 条几何约束；残差约 7.4–8.5mm。多数稳定匹配约 1–1.5s，最大样本约 3.2s；冷启动另需模型/运行库加载。这些不是新一轮移动实拍或有真值轨迹的精度评估。接受率提升不能替代人工检查和真值验证。

导出的 `map_poses.txt` 与几何优化矩阵逐项比较，最大差约 2.9×10⁻⁶（受文本精度影响）。

当前回归验证入口（不连接相机）：

```powershell
python Reconst_3D/tools/run_tests.py
Reconst_3D/vendor/rtabmap/bin/python.exe Reconst_3D/tests/geometry_engine_check.py
```

历史一次性实机测试和回放脚本已在目录整理中移除；原有采集结果和质量分析报告保留在 `results/`。
