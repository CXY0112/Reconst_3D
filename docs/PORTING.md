# 平台适配边界

本次交付为 Windows移植.zip、Linux移植.zip、GitHub开源.zip，均不带 Python 环境。当前行为以 README 和 INSTALLATION 为准，历史专题文档中的旧步骤/实验路径只供参考。

Windows 包带已有专用二进制与模型；安装目标 Python 3.12 和依赖，运行 install_windows.cmd 后启动。不能假定目标机的 GPU、驱动、USB、相机固件与本机相同。

Linux 包带可移植 Python/C++ 源码、固定算法资源、Linux 相机 SDK（含多架构库、udev规则和厂商示例），仍需完成：

1. 将 camera_sdk.py/rtabmap_scan.py 的 Windows 绑定适配 Linux SDK 的 ABI 和同步回调，实测 RGB-D 网格与标定。
2. 将 tools/rtab_host.cpp 的 Windows 管道/加载器实现改为 POSIX，构建 GTSAM iSAM2 版 RTAB-Map 与项目 C ABI 桥接，修改 rtab_client.py 的可执行路径。
3. 将 map_cloud_view.py、mesh_view.py、gaussian_renderer.py 的 WGL 改为 GLX/EGL 或相应跨平台窗口系统。
4. G-CVO CPU 用 Eigen/OpenMP/CMake 在目标平台构建；CUDA 需 NVIDIA 驱动/Toolkit并单独验证。ARM64 不可加载 x86_64 动态库。
5. 先离线构造/导入测试，再相机采集、暂停恢复、三维显示、保存/重载和性能验收。未在本次打包中宣称已完成 Linux 实机适配。

GitHub 包为源码发布，不带相机SDK/闭源或许可不明资源、权重、私密配置和历史数据。
