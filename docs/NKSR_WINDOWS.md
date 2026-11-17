# Windows预编译包优先策略

2026-10-16用户指定优先使用Windows原生预编译包。对照程序默认使用本机Python，不自动切入WSL。取得可验证的Windows wheel后，应在独立Windows环境中安装匹配的Python/PyTorch及运行库，再通过RECONST_NKSR_PYTHON选择该解释器；不替换现有五阶段主环境。

## 已核验内容

- [官方Releases](https://github.com/nv-tlabs/NKSR/releases)为空；GitHub API返回空列表，未提供可下载的Windows发布资产。
- [官方当前README](https://github.com/nv-tlabs/NKSR/tree/public)说明旧预编译包已过期，当前示例使用pip安装本地package目录进行构建。import torch/import nksr展示的是调用接口，不是Windows安装兼容性声明。
- [官方setup.py](https://github.com/nv-tlabs/NKSR/blob/public/package/setup.py)使用CUDAExtension、nvcc及cusparse，并标注仅支持Linux x86-64；未发现可直接使用的官方Windows wheel。该检查只输出提示，不能解释为Windows在原理上永远不可移植。
- [作者的Windows部署回复](https://github.com/nv-tlabs/NKSR/issues/49#issuecomment-1935233445)建议WSL。[另一条作者回复](https://github.com/nv-tlabs/NKSR/issues/41#issuecomment-1725041449)说明当时不能直接在Windows运行。旧回复需与当前源码一起判断，不能单独证明所有未来版本均不支持。
- 实际下载并只读检查了[PyPI的nksr](https://pypi.org/project/nksr/)：版本0.0.0，文件nksr-0.0.0-py3-none-any.whl，1196字节。nksr/__init__.py只有版本号与import os；没有Reconstructor、模型推理实现或本地扩展，因此不能用pip安装成功来证明部署成功。
- 作者旧wheel索引本轮无法访问，不能确认其中还保留哪些平台的包。针对win_amd64与Windows wheel的检索未找到可验证下载项；不把“未找到”表述成“所有第三方Windows包均不存在”。

## 当前结果

尚未取得可验证的Windows原生预编译包。用户随后要求直接按正常流程做Windows原生源码测试，已完成下述真实安装尝试；扩展构建未通过，没有进行真实NKSR几何对照。不会把PyPI占位包、Linux_x86_64 wheel或改名后的Linux动态库当成Windows安装包。

如果提供具体Windows wheel下载地址，可继续核对来源、Python ABI、PyTorch/CUDA版本和是否包含CPU后端，再做独立环境的导入及小规模真实推理验证。本机是AMD Radeon780M，即使取得Windows CUDA wheel也不能据此宣称有CUDA GPU加速；必须分别验证CPU运行路径。

## Windows原生源码实测（2026-10-16）

使用tools/setup_nksr_windows.py，全部通过Windows进程运行，没有通过WSL编译或推理。正式工作台主解释器与原有几何环境不修改。

| 项目 | 实测结果 |
| --- | --- |
| 原有主环境 | Python3.14 / PyTorch2.14 CPU；官方原版pip安装检测提示缺少nvcc/CUDA_HOME |
| 新建独立环境 | vendor/nksr_windows/env，Windows Python3.10 |
| CUDA | 原生nvcc12.8.61安装及版本查询成功 |
| PyTorch | 官方Windows torch2.7.0+cu128安装成功 |
| 其他依赖 | 官方Windows torch-scatter2.1.2+pt27cu128及NumPy/Open3D/PyCG等安装成功 |
| 源码依赖 | Windows Git浅克隆获取官方指定OpenVDB提交、Eigen3.4分支；保留可供原版setup.py检出的Git元数据 |
| NKSR原版构建 | pip install --no-build-isolation package/实际失败：MSVC D8021，无效参数/Wno-unknown-pragmas |
| CUDA独立编译测试 | 当前MSVC19.51超出CUDA12.8头文件接受范围，报C1189 unsupported Microsoft Visual Studio version |
| 模型运行 | 未执行到Reconstructor；未生成NKSR网格 |

这次NKSR的setup.py和算法源码保持原样，未更改Linux平台提示或删除CUDA代码。平台提示本身不会终止安装；实际失败来自编译参数。脚本哈希校验、依赖版本与设备状态存入vendor/nksr_windows/native_test_report.json。

原版构建日志：vendor/nksr_windows/07_native_source_build.log。独立CUDA编译日志：compiler_cuda_probe.log。MSVC诊断：compiler_msvc.log。

下一步应先配置CUDA12.8兼容的VS2022 v143工具集（例如MSVC14.44），再将GCC专用参数替换为MSVC参数；之后还要继续检查链接、扩展导入和CPU真实推理。仅解决这两处已知问题，尚不能保证后续全部通过。本轮没有全局修改Visual Studio或绕过CUDA编译器版本检查。

复现入口：

```powershell
python Reconst_3D/tools/setup_nksr_windows.py --phase probe
python Reconst_3D/tools/setup_nksr_windows.py --phase environment
python Reconst_3D/tools/setup_nksr_windows.py --phase sources
python Reconst_3D/tools/setup_nksr_windows.py --phase install
python Reconst_3D/tools/setup_nksr_windows.py --phase compiler
python Reconst_3D/tools/setup_nksr_windows.py --phase build
```

environment是新建/重建独立环境操作；已有本次环境时无需重复。probe为官方源码安装的dry-run元数据检查；build才是完整原版扩展构建。所有真实扫描数据与setp2/setp3/setp4保持原样。独立环境、源码副本和构建缓存不进入发布包。

后续用户已选择返回 WSL。现有 WSL 环境已编译 NKSR 并通过 CPU 球面推理，详细状态见 [CUDA_TOOLCHAIN.md](CUDA_TOOLCHAIN.md)。上述 Windows 原生失败记录保留，不再作为当前默认部署路线。
