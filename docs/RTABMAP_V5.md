# V5：原生 RTAB-Map 连续 RGB-D 建图

当前为V5.1增量稠密融合版，地图更新与保存优化见[MAP_SCALING.md](MAP_SCALING.md)。原生定位部分不变；本页下方保存内容中的RGB对齐副地图不再自动额外导出。

启动仍是 `python Reconst_3D/main.py`，演示版加 `beta`。两个界面均采用本流程，点击开始才访问相机。当前针对已有 Windows x64 运行库；Linux/Jetson 原生主机、相机与显示适配仍需完成。

## 当前流程

1. 相机持续输出 RGB、深度及 SDK XYZ，右侧独立预览。默认目标采样 10 Hz，普通版可调；待跟踪输入只保留最新一帧，繁忙时替换过期帧。
2. 用出厂内外参将深度对齐到 RGB，送入独立 rtab_host.exe。此进程调用真正的 RTAB-Map 0.23.8，避免将其嵌入式 Python 运行库加载到 UI Python 中。
3. 原生 RGB-D 视觉里程计采用 F2M（当前帧对局部视觉特征地图）。RGB 特征产生对应，深度提供三维约束，持续估计相机位姿。常规跟踪不再运行 FPFH、PointDSC 或稠密 ICP。
4. 最多每 0.5 秒交给 RTAB-Map 建图；原生移动门槛为 3 cm、0.03 rad（约 1.72°），特征上限 1000。不把因运动不足而未保留在优化图中的临时节点当成关键帧。
5. 原生地图使用自身默认回环、几何验证与位姿图优化；不再运行自定义 PointDSC 异步回环，也不通过 geometry_sync 覆盖原生图。原生建图/回环仍在原生处理线程运行，尚未异步化。
6. 新关键帧进入独立稠密线程：局部清洗、表面/法线估计、RGB-D与局部观测保存、3 mm体素地图更新、左侧显示。最多一项运行、一项等待；队满时本节点仅用于原生定位，不新增独立稠密观测。已接收任务不会被替换。
7. 优化移动历史节点时，撤销旧体素贡献并按新位姿重新融合。完成采集后排空任务，再保存数据。二阶段继续默认卡尔曼表面融合与均匀采样，后续几何、RGB、展示入口不变。

视觉里程计需要纹理、稳定曝光与连续重叠；白墙、快速转向或模糊仍可能丢失跟踪。失败帧不强行入图，界面提示缓慢返回已观察视野。目前没有自动回退到 PointDSC。

## 保存和内存边界

registrations.jsonl 将采样编号绑定到保留的原生节点；geometry_poses.json 保存稠密节点的优化位姿。其 edges 为空，不代表原生地图没有约束；原生图保存在 map.db。

pointcloud_local、rgb、depth、rgb_raw、depth_raw 仅为接收的稠密关键帧创建。map.db 还可能包含其他原生定位节点及 RGB-D，不能宣称未用于稠密地图的帧完全不占磁盘。稠密观测保留完整 SDK 视野，对齐 RGB 的裁剪不替代几何来源。

继续使用 map_frames 磁盘缓存及 voxel_map.sqlite，SQLite 页缓存目标16 MiB；输入、稠密队列有界。原生 RTAB-Map WM 时间/数量阈值本版未配置，总内存不是固定上限。原生图、位姿传输、全图显示及导出仍会随地图增长。跨会话、长期真实回环/漂移、原生节点合并后的历史稠密位姿更新仍需验证。

## 验证和部署

本次为已有 RGB-D 合成横移的离线测试，未连接相机，报告在 results/rtab_vo_v5/REPORT.md。Windows受限沙箱仍复现0xc0000008无效句柄；沙箱外独立主机可以运行、保存与退出。

online_metrics.csv 记录前端及原生里程计/建图耗时；dense_metrics.jsonl 记录后台稠密处理；preview_metrics.csv 记录融合和地图读取；video_metrics.json 记录输入丢帧及稠密跳过数；rtab_native.log 为原生日志。绘图耗时不包含这些计算。

Windows包需包含 vendor/online_geometry 下的 online_bridge.dll、rtab_host.exe、msvcr110.dll、msvcp110.dll，及已有 vendor/rtabmap。主机源码/构建脚本在 tools/rtab_host.cpp、tools/build_rtab_host.cmd；构建需要Visual Studio C++ x64。重新构建 online_bridge 另需RTAB-Map/OpenCV开发包。

主机在 online_close 确认数据库写入、发送确认并刷新日志后终止独立进程，避开运行库卸载句柄问题。关闭超时不视为保存成功。源码包不含vendor；来源及哈希在vendor/online_geometry/RTAB_HOST_PROVENANCE.json。

参考：[RTAB-Map官方参数](https://introlab.github.io/rtabmap/api/latest/parameters.html)、[官方实现](https://github.com/introlab/rtabmap)。实际配置以tools/online_bridge.cpp的online_create及已验证二进制为准，source.ini是离线导入配置，不控制当前原生会话。
