"""Build audited Windows, Linux and public-source archives without Python envs."""
from pathlib import Path
import argparse, ast, hashlib, io, json, os, re, shutil, tarfile, tempfile, zipfile

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
VIDEO='https://www.bilibili.com/video/BV1Crnt66E6A'
SKIP={'.git','.venv','venv','env','mamba','__pycache__','.pytest_cache','.cache',
      'build','build-local','build_cpu','build_cpu_final','build_cuda','target','temp',
      'node_modules','results','demo_data','odometry_eval','rkhs_ba','snapshot',
      'wandb','tensorboard','dist','*.egg-info'}
TEXT={'.py','.md','.txt','.json','.cpp','.cc','.c','.cxx','.h','.hh','.hpp','.ipp','.tpp','.cuh','.cu','.cmake',
      '.cmd','.bat','.sh','.yaml','.yml','.toml','.cfg','.in','.rst','.rules'}
PROFILES={'windows':'Windows移植.zip','linux':'Linux移植.zip','github':'GitHub开源.zip'}


def clean_text(text):
    # Paths are replaced only in staged copies; live project files stay intact.
    text=re.sub(r'C:(?:\\{1,2}|/)Users(?:\\{1,2}|/)Starry',lambda m:m[0].replace('Starry','USER'),text,flags=re.I)
    text=text.replace('/mnt/c/Users/Starry','/mnt/c/Users/USER')
    return text


def walk(folder):
    for directory,dirs,names in os.walk(folder):
        dirs[:]=sorted(d for d in dirs if d not in SKIP and not d.endswith(('.egg-info','.dist-info')))
        for name in sorted(names):
            p=Path(directory)/name
            if p.is_symlink() or name.startswith(('.env','.git')) or name in ('nksr_ready.json','runtime.json'):continue
            if p.suffix.lower() in {'.pyc','.pyo','.log','.obj','.pdb','.exp','.bak','.db','.sqlite','.pem','.key'}:continue
            yield p


def put(source,stage,relative=None):
    target=stage/(relative or source.relative_to(ROOT));target.parent.mkdir(parents=True,exist_ok=True)
    if source.suffix.lower() in TEXT or source.name in {'LICENSE','COPYING','NOTICE','CMakeLists.txt'}:
        try:target.write_text(clean_text(source.read_text(encoding='utf-8-sig')),encoding='utf-8');return
        except UnicodeDecodeError:pass
    shutil.copyfile(source,target)


def write(stage,name,text):
    target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
    if name.endswith(('.cmd','.bat')):target.write_bytes(text.replace('\r\n','\n').replace('\n','\r\n').encode('utf-8'))
    else:target.write_text(text,encoding='utf-8')


def copy_sources(stage,profile):
    for p in ROOT.iterdir():
        if p.is_file() and (p.suffix in {'.py','.md','.txt','.cmd'}):
            if profile=='linux' and p.suffix=='.cmd':continue
            put(p,stage)
    for folder in ('docs','tools','tests','preprocessing','gcvo_native','pointdsc_impl'):
        if profile=='github' and folder=='pointdsc_impl':continue
        for p in walk(ROOT/folder):
            if p.suffix.lower() not in TEXT and p.name!='CMakeLists.txt':continue
            if profile=='linux' and p.suffix in {'.cmd','.bat'}:continue
            put(p,stage)
    # Pinned, permissively licensed inference code; no environments or datasets.
    for folder in ('vendor/bufferx/source','vendor/mac/source'):
        for p in walk(ROOT/folder):
            if p.suffix.lower() in TEXT or p.name in {'LICENSE','CMakeLists.txt'}:put(p,stage)
    sam=ROOT/'vendor/mobilesam/source/MobileSAM-master'
    for p in walk(sam):
        rel=p.relative_to(sam)
        if rel.parts[0]=='mobile_sam' and p.suffix=='.py' or rel.name in ('LICENSE','README.md') and len(rel.parts)==1:put(p,stage)
    for rel in ('vendor/gcvo/runtime.json','vendor/bufferx/runtime.json'):
        put(ROOT/rel,stage)
    buffer_config=json.loads((stage/'vendor/bufferx/runtime.json').read_text())
    buffer_config['python']='vendor/bufferx/env/'+('Scripts/python.exe' if profile!='linux' else 'bin/python')
    write(stage,'vendor/bufferx/runtime.json',json.dumps(buffer_config,indent=2))
    # Preserve all upstream notices as documents, including excluded components.
    licenses={
        'BUFFER-X-MIT.txt':'vendor/bufferx/source/LICENSE',
        'MAC-MIT.txt':'vendor/mac/LICENSE',
        'MobileSAM-Apache-2.0.txt':'vendor/mobilesam/source/MobileSAM-master/LICENSE',
        'Brush-Apache-2.0.txt':'vendor/brush/v0.3.0/bin/LICENSE',
        'GCVO-CC-BY-NC-ND-4.0.txt':'vendor/gcvo/source/LICENSE.txt',
        'NKSR-NVIDIA.txt':'vendor/nksr_windows/source/NKSR/LICENSE.txt',
    }
    for name,rel in licenses.items():put(ROOT/rel,stage,Path('third_party/licenses')/name)
    if profile!='github':
        for folder in ('vendor/gcvo/source','vendor/nksr_windows/source/NKSR'):
            for p in walk(ROOT/folder):
                if p.suffix.lower() in TEXT or p.name in {'LICENSE','CMakeLists.txt','COPYING'}:put(p,stage)
        for folder in ('models/pointdsc',):
            for p in walk(ROOT/folder):put(p,stage)
        for rel in ('vendor/mobilesam/source/MobileSAM-master/weights/mobile_sam.pt',
                    'vendor/bufferx/source/snapshot/threedmatch/Desc/best.pth',
                    'vendor/bufferx/source/snapshot/threedmatch/Pose/best.pth',
                    'vendor/nksr_linux/ks.pth','vendor/nksr_linux/checkpoint.json'):
            put(ROOT/rel,stage)
        for p in (WORKSPACE/'YDLIDAR-ICA-ASC60C/docs').glob('*.pdf'):
            put(p,stage,Path('vendor/camera_docs')/p.name)


def camera_linux(stage):
    pkg=WORKSPACE/'YDLIDAR-ICA-ASC60C/Ubuntu-24.04-Headless/linux_ros.pkg'
    with tarfile.open(fileobj=io.BytesIO(b'\xfd7zXZ'+pkg.read_bytes()),mode='r:xz') as archive:
        for member in archive.getmembers():
            if not member.isfile():continue
            path=Path(member.name)
            if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe camera SDK archive')
            if not member.name.startswith(('linux_ros/libs/','linux_ros/scripts/','linux_ros/configurationfiles/','linux_ros/linux/')):continue
            if path.suffix in {'.o','.a','.log'}:continue
            target=stage/'vendor/camera_linux'/path.relative_to('linux_ros')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())


def windows_native(stage):
    for p in walk(ROOT/'vendor/camera_win64'):put(p,stage)
    for folder in ('vendor/online_geometry','vendor/online_bufferx'):
        for p in (ROOT/folder).iterdir():
            if p.suffix.lower() in {'.dll','.exe'}:put(p,stage)
    for p in (ROOT/'vendor/rtabmap/bin').iterdir():
        if p.is_file() and (p.suffix.lower()=='.dll' and not p.name.lower().startswith('python')
                           or p.name=='rtabmap-export.exe'):put(p,stage)
    for p in walk(ROOT/'vendor/rtabmap/bin/plugins'):
        if p.suffix.lower()=='.dll':put(p,stage)
    for p in walk(ROOT/'vendor/rtabmap/share'):
        if p.suffix in {'.py','.cmake','.txt'}:put(p,stage)
    for p in (ROOT/'vendor/gcvo/bin').glob('*.dll'):put(p,stage)
    for p in (ROOT/'vendor/brush/v0.3.0/bin').iterdir():put(p,stage)


def environment_adapters(stage,profile):
    # The archive carries no interpreter. The target supplies standard Python.
    source=(ROOT/'runtime_paths.py').read_text(encoding='utf-8')
    source=source.replace("ROOT/'vendor/rtabmap/bin/python.exe' if os.name == 'nt' else Path(sys.executable)","Path(sys.executable)")
    write(stage,'runtime_paths.py',source)
    source=(ROOT/'rtab_client.py').read_text(encoding='utf-8').replace('import os,struct,queue,threading,subprocess,json','import os,struct,queue,threading,subprocess,json,sys')
    source=source.replace("PYTHONHOME=str(ROOT/'vendor/rtabmap/bin')","PYTHONHOME=os.environ.get('RECONST_RTAB_PYTHONHOME',sys.base_prefix)")
    write(stage,'rtab_client.py',source)
    # No local NKSR environment path is made the default installer requirement.
    source=(stage/'tools/setup_bufferx.py').read_text(encoding='utf-8')
    source=source.replace("default=ROOT/'vendor/nksr_windows/env/python.exe'","default=Path(sys.executable)")
    old="revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()"
    source=source.replace(old,"revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() if (source/'.git').is_dir() else REVISION")
    write(stage,'tools/setup_bufferx.py',source)
    verify=(ROOT/'tools/verify_installation.py').read_text(encoding='utf-8').replace("'vendor/rtabmap/bin/python.exe'","'vendor/rtabmap/bin/rtabmap_core.dll'")
    write(stage,'tools/verify_installation.py',verify)
    write(stage,'tools/prepare_native_python.py',PREPARE_PYTHON)
    if profile!='linux':write(stage,'install_windows.cmd',INSTALL_WINDOWS)


PREPARE_PYTHON='''"""Bind native RTAB Python DLL imports to the target's installed Python 3.12."""
from pathlib import Path
import shutil,sys
if sys.version_info[:2]!=(3,12):raise RuntimeError('Bundled RTAB native ABI requires Python 3.12 x64')
root=Path(__file__).resolve().parents[1]
if not (root/'vendor/rtabmap/bin/rtabmap_core.dll').exists():
    print('Native RTAB binaries absent; build/provision separately.');sys.exit(0)
for name in ('python312.dll','python3.dll'):
    source=Path(sys.base_prefix)/name
    if not source.is_file():raise FileNotFoundError(source)
    shutil.copy2(source,root/'vendor/rtabmap/bin'/name)
print('Prepared native Python DLLs from target installation; no environment was bundled.')
'''

INSTALL_WINDOWS='''@echo off
setlocal
cd /d "%~dp0"
py -3.12 -c "import struct,tkinter; assert struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 (
  echo Install Python 3.12 x64 with Tcl/Tk and Python Launcher first.
  pause
  exit /b 1
)
if not exist ".venv\\Scripts\\python.exe" py -3.12 -m venv .venv
if errorlevel 1 exit /b 1
".venv\\Scripts\\python.exe" -m pip install -r requirements.txt -r requirements-models.txt -r requirements-geometry.txt
if errorlevel 1 exit /b 1
".venv\\Scripts\\python.exe" tools\\prepare_native_python.py
if errorlevel 1 exit /b 1
if exist "vendor\\camera_win64\\AngstrongCameraSdk.dll" (
  ".venv\\Scripts\\python.exe" tools\\verify_installation.py --windows-sdk
) else (
  ".venv\\Scripts\\python.exe" -c "import main; print('Source UI imports OK. Provision camera/native assets before capture.')"
)
if errorlevel 1 exit /b 1
echo Ready. Use start_main.cmd or start_beta.cmd.
pause
'''


def documents(stage,profile):
    title={'windows':'Windows 迁移包','linux':'Linux 适配迁移包','github':'GitHub 开源源码包'}[profile]
    intro={'windows':'包含现有 Windows x64 相机 SDK 与配置、RTAB/iSAM2 专用运行库、G-CVO CPU DLL、Brush 程序及项目所用权重。Python、虚拟环境、PyTorch、Open3D、CUDA Toolkit 均不随包附带。',
           'linux':'包含算法源码、模型权重和厂商 Linux 相机 SDK（x86_64 / aarch64 / armhf）、udev 规则与示例；不包含 Python 环境。此包供继续适配，尚不是已验证可直接实拍的 Linux 发行版。',
           'github':'提供项目源码、测试、构建脚本、依赖清单和配置模板。第三方二进制、相机 SDK、预训练权重、G-CVO 受限上游源码和许可不明的 PointDSC 推理代码不随开源包发布。独立编写的 G-CVO CPU GN 实现与 Python 接口保留。'}[profile]
    readme=f'''# 基于 RGB-D 的三维场景重建与交互建模

[观看演示视频（哔哩哔哩）]({VIDEO})

{title}。{intro}

流程：采集 → 预处理（可跳过）→ 场景重建 → 场景分割 / 精细建模 → 成果展示。

- 跟踪：默认 RGB-D RTAB-Map；可选 G-CVO CPU，CUDA 接口保留。失跟后支持 RGB-D ORB/PnP 独立恢复及 MAC + ICP 恢复，再接回正常跟踪。
- 几何：Poisson、TSDF、NKSR。Brush 3DGS 默认 8,000 步，右侧直接交互显示，几何与高斯独立保存。
- 分割：MobileSAM。FAL 物体建模为可选云端服务，需自行配置密钥并明确提交。
- Windows 已有运行验证；Linux 的相机绑定、原生进程通信和 WGL 显示仍需目标平台适配。生成测试通过不等于目标电脑实拍验收通过。

## 快速开始

先阅读 [安装说明](INSTALLATION.md) 与 [资源清单](docs/RESOURCES.md)。

Windows 迁移：安装 Python 3.12 x64（含 Tcl/Tk、Launcher），执行 `install_windows.cmd`，然后双击 `start_main.cmd`。Beta 使用 `start_beta.cmd`。

源码环境：

```sh
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt -r requirements-models.txt -r requirements-geometry.txt
python main.py
```

GitHub 源码包需先按资源清单安装 MobileSAM 权重、相机和原生运行库，才能使用对应功能；打开界面不会主动连接相机。Linux 先完成 [平台适配](docs/PORTING.md)，再尝试实拍。

## 配置、数据与测试

`.env.example` 仅为模板；将自己的密钥放到 `.env` 或环境变量。包内不含任何密钥或历史采集。新采集保存到 `results/数据名称`，迁移旧扫描时自行完整复制该数据集目录。

`python tools/run_tests.py` 运行离线回归。部分原生/模型测试需要先安装对应资源，不连接相机的源码检查见 `python tools/check_release.py`。

## 文档与许可

[G-CVO 与恢复](docs/GCVO.md) · [3DGS](docs/BRUSH_3DGS.md) · [工作流](docs/SCENE_WORKFLOW.md) · [相机部署](docs/CAMERA_SETUP.md)

项目自行编写的代码采用 [Apache License 2.0](LICENSE)，第三方组件保留原许可证。该许可不覆盖厂商 SDK、第三方模型或受限算法源码，详见 [第三方说明](THIRD_PARTY_NOTICES.md)。GitHub 包只在本地生成，未自动上传。
'''
    write(stage,'README.md',(ROOT/'README.md').read_text(encoding='utf-8-sig') if profile=='github' else readme)
    write(stage,'LICENSE',(ROOT/'LICENSE').read_text(encoding='utf-8-sig'))
    if (ROOT/'NOTICE').is_file():write(stage,'NOTICE',(ROOT/'NOTICE').read_text(encoding='utf-8-sig'))
    write(stage,'.gitignore',GITIGNORE)
    write(stage,'.env.example','# Optional FAL service; fill locally, never commit real credentials.\nFAL_KEY=\n')
    write(stage,'THIRD_PARTY_NOTICES.md',NOTICES)
    write(stage,'docs/RESOURCES.md',RESOURCES)
    write(stage,'docs/CAMERA_SETUP.md',CAMERA)
    install=f'''# {title}：安装与验收

{intro}

## Windows

1. 解压到可写目录，使用 Python 3.12 x64（专用 RTAB 库链接 Python 3.12 ABI），勾选 Tcl/Tk 与 Launcher。
2. 双击 `install_windows.cmd`。在目标电脑创建 `.venv` 并安装依赖；将目标电脑 Python 的两个运行 DLL 复制到 RTAB 运行库目录，以满足其动态链接。压缩包本身不包含 Python。
3. `start_main.cmd` / `start_beta.cmd` 启动。无需在主环境混入本机原有几何包；几何任务默认使用当前解释器，可用 `RECONST_GEOMETRY_PYTHON` 指定其他解释器。
4. 相机使用 `vendor/camera_win64/configurationfiles/hp60c_v2_00_20230704_configEncrypt.json`。Windows UVC 驱动由系统提供；包内没有发现单独的相机 INF/SYS 安装包，未使用无关的 CP210x 串口驱动替代。详见相机部署文档。
5. 首次实拍前核验 SDK、原生 iSAM2 初始化、RGB-D 同步、暂停恢复、保存/重新导入。未在目标新电脑执行实拍验收。

GitHub 包不含专用二进制、相机 SDK 与权重，先按 RESOURCES.md 补齐；仅安装通用依赖可以检查源码与打开 UI，不能因此声称相机链路已部署完成。

## Linux

```sh
sudo apt install python3-venv python3-tk build-essential cmake libgl1 libusb-1.0-0-dev libudev-dev
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-models.txt -r requirements-geometry.txt
```

不要直接使用 Windows DLL/EXE。Linux 包的 `vendor/camera_linux` 提供厂商 SDK 和 udev 规则；首先选择与系统架构一致的库，查看其 `linux` 示例，再适配 Python 绑定。还需为 RTAB C ABI host 实现 POSIX 进程/管道入口，将显示从 WGL 适配为 GLX/EGL。Qt/OpenCV、GTSAM、RTAB-Map 及编译工具链需在目标系统安装或构建。

## 可选算法与模型

迁移包带 MobileSAM、BUFFER-X、PointDSC、NKSR 的现有推理权重，不带训练数据。模型权重位置与校验见 RESOURCES.md；GitHub 包仅提供获取方式。BUFFER-X 环境可单独创建，`tools/setup_bufferx.py --base-python <已配置解释器>`；NKSR 另见安装脚本，原本环境、WSL 标记和 CUDA Toolkit 已剔除。

Brush Windows 程序仅在 Windows 迁移包；Linux 或 GitHub 按官方发行渠道获取适用平台程序。内嵌场景显示目前仍是 Windows WGL，Linux 需要适配，不能仅替换 Brush 二进制就宣称可用。

## 验证与完整性

`python tools/check_release.py` 验证包内文件 SHA、排除项和演示链接，不连接相机、不调用云服务；`RELEASE_MANIFEST.json` 列出所有发布文件。密钥在目标电脑配置；旧数据由用户独立迁移。
'''
    write(stage,'INSTALLATION.md',install)
    write(stage,'docs/PORTING.md',PORTING)
    write(stage,'tools/check_release.py',CHECK_RELEASE)
    if profile=='linux':
        write(stage,'start_main.sh','#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\nexec "${RECONST_PYTHON:-python3}" main.py "$@"\n')


GITIGNORE='''__pycache__/
*.py[cod]
.venv/
**/env/
**/.git/
.env
.env.*
!.env.example
results/
output/
target/
**/build*/
**/mamba/
*.log
*.db
*.zip
*.mp4
*.pt
*.pth
*.pkl
*.ckpt
*.onnx
*.dll
*.exe
*.pyd
*.obj
vendor/camera_win64/
vendor/camera_linux/
vendor/rtabmap/
vendor/geometry_packages/
vendor/gcvo/source/
vendor/nksr_windows/source/
pointdsc_impl/
!pointdsc_impl/README.md
'''

NOTICES='''# 第三方材料与发布范围

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
'''

RESOURCES='''# 资源位置与补齐方式

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

没有复制本机 Python、site-packages、CUDA Toolkit、WSL/Conda/venv 环境。模型来源和本地文件 SHA 另见 RESOURCE_MANIFEST.json 与第三方许可说明。SuperCluster 自动推理链已移除；NoKSR 作为独立实验保留，不包含其大型权重。
'''

CAMERA='''# 相机部署

## Windows x64

运行使用 vendor/camera_win64/AngstrongCameraSdk.dll 和 configurationfiles/hp60c_v2_00_20230704_configEncrypt.json；camera_win64/CONTENTS.json 保留原资源来源。采用已验证 Viewer v1.2.20 SDK，不以其他 SDK 版本直接替换。

本地资源中未发现单独的该相机 INF/SYS 安装包；Windows 使用设备的 UVC 系统驱动。CP210x 是串口驱动，不是本相机驱动，没有打进相机包。目标机若设备管理器异常，请使用厂商针对该设备提供的驱动，不能将 SDK DLL 当作内核驱动安装。

设置 RECONST_CAMERA_SDK 可指定其他 SDK 目录。RGB/深度均为 640×480，启动后等待1秒再正式采样。实际 USB 访问、固件、标定和首帧图像需目标机实测。

## Linux

迁移包 vendor/camera_linux 包含从现有厂商 linux_ros.pkg 安全展开的 libs/include、libs/lib/{x86_64-linux-gnu,aarch64-linux-gnu,arm-linux-gnueabihf}、scripts/angstrong-camera.rules 和厂商 linux 示例。配置位于 configurationfiles。

先检查目标架构与共享库依赖，再审阅并按目标系统安装 udev 规则、重载规则及重新插拔。没有替用户运行驱动安装脚本，也没有更改本机 USB 权限。主项目 camera_sdk.py 中 Windows ctypes 绑定尚需转换为 Linux SDK API/ABI；不能仅复制 .so 就声称主工作台已适配完成。

SDK 原始资料见 vendor/camera_docs（仅迁移包）。不得将实拍中的私有标定、USB序列号或历史采集作为公共默认配置。
'''

PORTING='''# 平台适配边界

本次交付为 Windows移植.zip、Linux移植.zip、GitHub开源.zip，均不带 Python 环境。当前行为以 README 和 INSTALLATION 为准，历史专题文档中的旧步骤/实验路径只供参考。

Windows 包带已有专用二进制与模型；安装目标 Python 3.12 和依赖，运行 install_windows.cmd 后启动。不能假定目标机的 GPU、驱动、USB、相机固件与本机相同。

Linux 包带可移植 Python/C++ 源码、固定算法资源、Linux 相机 SDK（含多架构库、udev规则和厂商示例），仍需完成：

1. 将 camera_sdk.py/rtabmap_scan.py 的 Windows 绑定适配 Linux SDK 的 ABI 和同步回调，实测 RGB-D 网格与标定。
2. 将 tools/rtab_host.cpp 的 Windows 管道/加载器实现改为 POSIX，构建 GTSAM iSAM2 版 RTAB-Map 与项目 C ABI 桥接，修改 rtab_client.py 的可执行路径。
3. 将 map_cloud_view.py、mesh_view.py、gaussian_renderer.py 的 WGL 改为 GLX/EGL 或相应跨平台窗口系统。
4. G-CVO CPU 用 Eigen/OpenMP/CMake 在目标平台构建；CUDA 需 NVIDIA 驱动/Toolkit并单独验证。ARM64 不可加载 x86_64 动态库。
5. 先离线构造/导入测试，再相机采集、暂停恢复、三维显示、保存/重载和性能验收。未在本次打包中宣称已完成 Linux 实机适配。

GitHub 包为源码发布，不带相机SDK/闭源或许可不明资源、权重、私密配置和历史数据。独立 CPU G-CVO 实现可构建；上游 CUDA 库和 NKSR 等组件的许可分别适用。获取说明见 RESOURCES.md。
'''

CHECK_RELEASE='''"""Check exact package content without camera/network/service access."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
for row in manifest['files']:
    path=root/row['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        raise RuntimeError('Package file mismatch: '+row['path'])
assert 'https://www.bilibili.com/video/BV1Crnt66E6A' in (root/'README.md').read_text(encoding='utf-8')
print('Package content verified:',manifest['profile'],len(manifest['files']),'files')
'''


def audit(stage,profile):
    rows=[];resources=[];parsed=0
    for p in sorted(stage.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(stage).as_posix();parts=rel.split('/')
        if rel in {'RELEASE_MANIFEST.json','RESOURCE_MANIFEST.json'}:continue
        if any(x in {'.git','.venv','venv','env','mamba','site-packages','__pycache__','results','target'} for x in parts):raise ValueError('Excluded environment/data: '+rel)
        if p.name.startswith('.env') and p.name!='.env.example':raise ValueError('Private config: '+rel)
        if p.name.lower() in {'python.exe','pythonw.exe','python3.dll','python312.dll','python312.zip'} or p.suffix.lower() in {'.pyc','.pyd','.log','.obj','.pdb'}:raise ValueError('Environment/build output: '+rel)
        if profile=='github' and (p.suffix.lower() in {'.dll','.exe','.so','.pt','.pth','.pkl','.ckpt','.onnx'} or any(x in rel for x in ('vendor/camera_win64/','vendor/camera_linux/','vendor/gcvo/source/','vendor/nksr_windows/source/'))):raise ValueError('Non-public resource: '+rel)
        if p.suffix=='.py' and parts[0]!='vendor':ast.parse(p.read_text(encoding='utf-8-sig'),filename=rel);parsed+=1
        if p.suffix in TEXT:
            text=p.read_text(encoding='utf-8-sig',errors='ignore')
            if re.search(r'C:(?:\\{1,2}|/)Users(?:\\{1,2}|/)Starry',text,re.I):raise ValueError('Personal path: '+rel)
            if re.search(r'(?:hf_[a-zA-Z0-9]{25,}|sk-[a-zA-Z0-9_-]{30,}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{30,})',text):raise ValueError('Credential-like literal: '+rel)
        digest=hashlib.sha256(p.read_bytes()).hexdigest();row=dict(path=rel,bytes=p.stat().st_size,sha256=digest);rows.append(row)
        if p.suffix.lower() in {'.dll','.exe','.pt','.pth','.pkl','.so','.pdf'}:resources.append(row)
    write(stage,'RESOURCE_MANIFEST.json',json.dumps(dict(resources=resources),ensure_ascii=False,indent=2))
    p=stage/'RESOURCE_MANIFEST.json';rows.append(dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write(stage,'RELEASE_MANIFEST.json',json.dumps(dict(profile=profile,python_environment_included=False,
        demo_video=VIDEO,files=rows),ensure_ascii=False,indent=2))
    return dict(files=len(rows),python_files_parsed=parsed,uncompressed_bytes=sum(r['bytes'] for r in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage-only',action='store_true')
    parser.add_argument('--from-stage',type=Path,help='Re-audit and archive an already validated staging build');a=parser.parse_args()
    outputs=[WORKSPACE/name for name in PROFILES.values()]
    if any(p.exists() for p in outputs):raise FileExistsError('Final archive already exists; refusing accidental overwrite')
    parent=WORKSPACE/'releases';parent.mkdir(exist_ok=True)
    build=a.from_stage.resolve() if a.from_stage else Path(tempfile.mkdtemp(prefix='final_packages_',dir=parent))
    if a.from_stage and (build.parent!=parent.resolve() or not build.name.startswith('final_packages_')):raise ValueError('Invalid owned staging path')
    report=dict(build=str(build),profiles={})
    print('Staging:',build,flush=True)
    for profile,name in PROFILES.items():
        stage=build/profile/'Reconst_3D'
        if not a.from_stage:
            stage.mkdir(parents=True);copy_sources(stage,profile)
            if profile=='windows':windows_native(stage)
            if profile=='linux':camera_linux(stage)
            environment_adapters(stage,profile);documents(stage,profile)
        info=audit(stage,profile);info['stage']=str(stage);report['profiles'][profile]=info
        print(profile,json.dumps(info),flush=True)
    (build/'build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if a.stage_only:return
    for profile,name in PROFILES.items():
        stage=Path(report['profiles'][profile]['stage']);output=WORKSPACE/name
        temporary=build/(name+'.partial')
        with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for p in sorted(stage.rglob('*')):
                if p.is_file():
                    name='Reconst_3D/'+p.relative_to(stage).as_posix()
                    if p.suffix=='.sh':
                        info=zipfile.ZipInfo.from_file(p,name);info.create_system=3;info.external_attr=0o100755<<16
                        archive.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
                    else:archive.write(p,name)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:raise ValueError('Archive CRC failed')
            manifest=json.loads(archive.read('Reconst_3D/RELEASE_MANIFEST.json'))
            for row in manifest['files']:
                if hashlib.sha256(archive.read('Reconst_3D/'+row['path'])).hexdigest()!=row['sha256']:raise ValueError('Archive hash mismatch')
        temporary.replace(output)
        report['profiles'][profile].update(archive=str(output),archive_bytes=output.stat().st_size,
            archive_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),crc_and_sha256_verified=True)
        print('Verified:',output,output.stat().st_size,flush=True)
        (build/'build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
