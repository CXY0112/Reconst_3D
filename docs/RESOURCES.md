# 资源位置与补齐方式

| 资源 | 目标位置 | 获取方式 |
|---|---|---|
| HP60C Windows SDK v1.2.20 | vendor/camera_win64/ | Windows 迁移包已含；GitHub 使用厂商已授权 SDK，对应加密配置文件不能省略 |
| HP60C Linux SDK | vendor/camera_linux/ | Linux 迁移包已含 libs、scripts、configurationfiles、linux 示例；按系统架构选择 |
| RTAB-Map + iSAM2 原生 host | vendor/rtabmap/bin、vendor/online_geometry、vendor/online_bufferx | Windows 迁移包已含；其他情况安装 RTAB/GTSAM/OpenCV，参考 tools/CMakeLists.txt 与 tools/rtab_host.cpp 构建。现有 host 为 Windows API，Linux 必须适配 |
| G-CVO CPU | vendor/gcvo/bin/gcvo_cpu.dll 或 libgcvo_cpu.so | Windows 包已含；其他平台 python tools/build_gcvo.py --device cpu，需 CMake/C++17/Eigen/OpenMP |
| G-CVO CUDA 上游 | vendor/gcvo/source | 私人迁移包已含固定快照；GitHub 自行按上游条款获取 d1268382faf8853499e6f72abe1096ff396a1ae0，再设置构建前缀，不承诺当前机器已验证 CUDA |
| MobileSAM | vendor/mobilesam/source/MobileSAM-master/weights/mobile_sam.pt | 迁移包已含；源码包 python tools/setup_mobilesam.py 获取固定官方提交，脚本校验 SHA |
| BUFFER-X | vendor/bufferx/source/snapshot/threedmatch/{Desc,Pose}/best.pth | 迁移包已含；GitHub 参考 vendor/bufferx/download_weights.py 或 https://huggingface.co/Hyungtae-Lim/BUFFER-X ，再安装独立 CPU 环境 |
| PointDSC | pointdsc_impl、models/pointdsc | 迁移包已含；开源包不带，需从上游确认授权后按 docs 位置安装 |
| NKSR | vendor/nksr_linux/ks.pth | 迁移包已含；其他情况参考 tools/setup_nksr_linux.py 与 https://github.com/nv-tlabs/NKSR 按许可部署 |
| Brush v0.3.0 Windows | vendor/brush/v0.3.0/bin/brush_app.exe | Windows 包已含；源码包从 https://github.com/ArthurBrussee/brush/releases/tag/v0.3.0 获取，预期 SHA256: 37e46cbf808b9983dd15a5f9a25328dbe43e7e06d53c4f59fbeaeb10e3a5b34a |

没有复制本机 Python、site-packages、CUDA Toolkit、WSL/Conda/venv 环境。模型来源和本地文件 SHA 另见 RESOURCE_MANIFEST.json 与第三方许可说明。SuperCluster 自动分割链已移除；历史标签仍可读取。NoKSR 独立实验保留，未包含其大型权重。
