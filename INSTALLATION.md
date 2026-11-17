# GitHub 开源源码包：安装与验收

提供项目源码、测试、构建脚本、依赖清单和配置模板。独立编写的 G-CVO CPU GN 实现与 Python 接口保留。

## Windows

1. 解压到可写目录，使用 Python 3.12 x64（专用 RTAB 库链接 Python 3.12 ABI），勾选 Tcl/Tk 与 Launcher。
2. 双击 `install_windows.cmd`。在目标电脑创建 `.venv` 并安装依赖；将目标电脑 Python 的两个运行 DLL 复制到 RTAB 运行库目录，以满足其动态链接。压缩包本身不包含 Python。
3. `start_main.cmd` / `start_beta.cmd` 启动。无需在主环境混入本机原有几何包；几何任务默认使用当前解释器，可用 `RECONST_GEOMETRY_PYTHON` 指定其他解释器。
4. 相机使用 `vendor/camera_win64/configurationfiles/hp60c_v2_00_20230704_configEncrypt.json`。Windows UVC 驱动由系统提供；包内没有发现单独的相机 INF/SYS 安装包，未使用无关的 CP210x 串口驱动替代。详见相机部署文档。
5. 首次实拍前核验 SDK、原生 iSAM2 初始化、RGB-D 同步、暂停恢复、保存/重新导入。未在目标新电脑执行实拍验收。


## Linux

```sh
sudo apt install python3-venv python3-tk build-essential cmake libgl1 libusb-1.0-0-dev libudev-dev
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-models.txt -r requirements-geometry.txt
```

不要直接使用 Windows DLL/EXE。Linux 包的 `vendor/camera_linux` 提供厂商 SDK 和 udev 规则；首先选择与系统架构一致的库，查看其 `linux` 示例，再适配 Python 绑定。还需为 RTAB C ABI host 实现 POSIX 进程/管道入口，将显示从 WGL 适配为 GLX/EGL。Qt/OpenCV、GTSAM、RTAB-Map 及编译工具链需在目标系统安装或构建。

## 可选算法与模型

迁移包带 MobileSAM、BUFFER-X、PointDSC、NKSR 的现有推理权重，不带训练数据。GitHub 包仅提供获取方式。BUFFER-X 环境可单独创建，`tools/setup_bufferx.py --base-python <已配置解释器>`；NKSR 另见安装脚本，原本环境、WSL 标记和 CUDA Toolkit 已剔除。

Brush Windows 程序仅在 Windows 迁移包；Linux 或 GitHub 按官方发行渠道获取适用平台程序。内嵌场景显示目前仍是 Windows WGL，Linux 需要适配，不能仅替换 Brush 二进制就宣称可用。

## 验证与完整性

`python tools/check_release.py` 验证包内文件 SHA、排除项和演示链接，不连接相机、不调用云服务；`RELEASE_MANIFEST.json` 列出所有发布文件。密钥在目标电脑配置；旧数据由用户独立迁移。
