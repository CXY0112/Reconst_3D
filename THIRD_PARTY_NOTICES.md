# 第三方材料与发布范围

本项目原生 CPU G-CVO GN 实现标注为独立编写，保留在源码包；上游库不在 GitHub 包重新分发。项目的 Apache License 2.0 不替代任何第三方许可证。

| 组件 | 来源与本地许可 | 包含范围 |
|---|---|---|
| G-CVO 上游 | https://github.com/ToyotaResearchInstitute/gcvo ，CC BY-NC-ND 4.0，见 third_party/licenses/GCVO-CC-BY-NC-ND-4.0.txt | 两个私人迁移包保留未改动源码；GitHub 仅接口、独立 CPU 实现与获取说明 |
| MAC | https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques ，MIT | 保留来源源码和许可证；项目 CPU 适配见 mac_solver.py |
| BUFFER-X | https://github.com/MIT-SPARK/BUFFER-X ，MIT | 保留源码与 CPU 适配；预训练权重不在 GitHub 包 |
| MobileSAM | https://github.com/ChaoningZhang/MobileSAM ，Apache-2.0 | 保留推理代码；预训练权重不在 GitHub 包 |
| Brush | https://github.com/ArthurBrussee/brush ，Apache-2.0 | Windows 包带原版 v0.3.0 程序，其余包按官方渠道获取 |
| NKSR | https://github.com/nv-tlabs/NKSR ，NVIDIA Source Code License（研究/评估用途限制） | 私人迁移包保留源码和权重，完整许可随包；GitHub 只保留调用接口与获取说明 |
| PointDSC | https://github.com/XuyangBai/PointDSC ，本地快照未附 LICENSE | 只在私人迁移包保留原有推理资源；GitHub 排除推理实现与权重，待作者授权后自行获取 |
| RTAB-Map / GTSAM / OpenCV 等 | https://github.com/introlab/rtabmap 、https://github.com/borglab/gtsam 、https://opencv.org/license/ | Windows 私人迁移包带已有原生运行库；GitHub 不带二进制，源码自行构建 |
| HP60C SDK / 文档 | 厂商资源，来自现有 SDK 与 camera_win64/CONTENTS.json 所记来源 | 仅私人迁移包，按厂商条款使用；GitHub 不重新分发 |

迁移包用于已有工程在授权设备迁移，不是对厂商 SDK、非商业组件或模型权重授予新的再分发权。GitHub 包中的许可证副本仅用于标注许可边界，并不表示对应受限资源已包含。未自动发布远端仓库。
