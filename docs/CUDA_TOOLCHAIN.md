# NKSR 的独立 CUDA 编译环境

本机为 AMD Radeon 780M，没有 NVIDIA 显卡。CUDA 编译器、头文件及库可以安装并编译程序，但不能因此获得 CUDA GPU 运行能力。NKSR 的官方文档支持 CPU 推理；其当前公开扩展仍需 CUDA 工具链构建，因此两项验证分开记录。

## 位置与隔离

- 系统：现有 WSL2 Ubuntu-22.04，Linux x86-64。
- 默认环境：WSL 内 `/opt/reconst_3d/nksr/env`。原生 Linux 文件系统用于避免 Windows 挂载目录的大量小文件 I/O 开销。
- 项目中 `vendor/nksr_linux` 仅作为配置及日志目录。`toolkit_ready.json` 记录编译验证；`nksr_ready.json` 仅在扩展导入成功后生成，另行标记真实推理是否验证。
- Windows 主 Python、已有几何后台和 ROS 环境不替换；不安装 NVIDIA 驱动。
- 环境不进入源码包或 Windows 运行库包。换机器需要重新部署，不能直接搬运这个 Conda 前缀。

## 重建环境

在 Linux / WSL 中，以具有默认目标目录写入权限的账户，定位项目后运行：

```sh
python3 Reconst_3D/tools/setup_nksr_linux.py --phase toolkit
python3 Reconst_3D/tools/setup_nksr_linux.py --phase nksr
python3 Reconst_3D/tools/setup_nksr_linux.py --phase smoke
```

可以通过 `RECONST_NKSR_HOME` 指定另一个**原生 Linux**安装目录；构建扩展时需保留该环境变量。默认 CUDA 12.8、Python 3.10、PyTorch 2.7.0+cu128、匹配的 torch-scatter；采用 micromamba 独立前缀。`--phase verify` 只重新编译验证程序，不重新安装工具链。

验证程序包含实际 CUDA 内核，先通过 nvcc 编译和链接，再查询设备。没有受支持的 NVIDIA 驱动/设备时，运行日志会出现设备数0及错误状态；这只确认编译器可用，不能宣称 GPU 内核执行通过。有可用设备时，程序会执行内核并校验返回值7。

## 源码与构建

Git smart HTTP 在本机发生 TLS 超时，因此使用上游官方 HTTPS 档案：NKSR 解析 public 分支到具体提交，OpenVDB 使用官方 setup.py 指定提交，Eigen 使用官方3.4分支。所有档案记录 URL 和 SHA256，清单位于 Linux 安装目录的 `NKSR/archive_sources.json`。

构建适配在 `setup.py` 中跳过对已记录档案依赖的自动 Git checkout，保留 `setup.py.upstream` 和前后校验和。Conda CUDA 的 targets 头文件及库目录通过 CPATH/LIBRARY_PATH 提供。默认指定 sm_86 编译目标以允许无 GPU 的机器交叉编译；该二进制的 GPU 兼容性尚未实机验证，部署到其他显卡时需按目标架构重建。

为运行无 NVIDIA 驱动的 CPU 路径，安装脚本对上游做了明确的兼容修补：CPU 设备不查询 CUDA 当前设备、权重先载入 CPU、CPU 稀疏网格不创建无用 CUDA 网格、HostBuffer 构造器不查询 CUDA。原文件备份及修改前后 SHA256 记录在 cpu_compat_patches.json；未改变网络结构或几何方程。修改 C++ 头文件后必须重新运行 `--phase build`，再运行 smoke，不能仅凭导入成功声明推理可用。模型调用使用 no_grad，保留 NKSR 内部启用空间梯度计算的能力。

## 工作台调用

本次用户明确返回 WSL 路线后，只有扩展导入、真实推理验证、选用标记全部通过才会默认调用 WSL；仅存在配置文件不会切换。显式 `RECONST_NKSR_PYTHON` 优先级更高，也可指定 `wsl:Ubuntu-22.04`。Windows路径转换成对应 `/mnt/盘符/...`，含空格路径按独立参数传递。工作台与 Poisson 仍在 Windows，只有 NKSR 推理进入 WSL。

只有 `comparison.json` 的两种方法均成功且输入校验不变，才代表完整几何对照完成。工具链编译成功、扩展导入成功、小规模 CPU 推理成功、真实点云对照成功，是四个不同状态，不应混为一谈。

[上游安装环境](https://github.com/nv-tlabs/NKSR/blob/public/environment.yml) · [CPU 推理说明](https://github.com/nv-tlabs/NKSR/blob/public/NKSR-USAGE.md)

## 本次部署状态（2026-10-16）

CUDA12.8.61最小程序编译链接通过；设备检测为0、状态35，未验证GPU内核运行。Linux独立环境现已安装 PyTorch2.7.0+cu128、匹配的 torch-scatter、NKSR1.0.3，扩展已编译导入成功。实际 CPU 球面测试：4,096 输入点、7,486 顶点、14,968 三角面，模型重建及提取约3.15秒，径向误差P50/P95约0.165/0.487mm。这只是链路验证，不代表真实扫描精度。

官方 ks.pth 权重经官方 HuggingFace LFS SHA256 校验后缓存在 Linux TORCH_HOME 中；SHA256 为 c2edb6437d6485c2ed48a756523849628f0c34e232c601113ff657db8714e955。Windows下载副本和校验记录位于 vendor/nksr_linux，不进入发布包。网络受限机器需要另行准备该缓存。

Windows原生测试记录保留在[NKSR_WINDOWS.md](NKSR_WINDOWS.md)。真实扫描的完整质量和耗时以独立 geometry_comparison 目录为准，不能由球面测试推断。
