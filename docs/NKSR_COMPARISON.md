# Poisson / NKSR 第一轮几何对照

> 维护状态：旧几何对照窗口已移除；当前 NKSR 重建及其共用推理、测量工具继续保留。 以下历史实验记录不作为当前界面操作说明。

三阶段点击“Poisson / NKSR 对照”，再点击运行；默认使用该页面当前载入的二阶段点云。窗口支持同步视角、线框、原始/统一裁剪结果切换。也可无参数运行 `python Reconst_3D/tools/compare_geometry.py`，自动选择最新完整卡尔曼预处理结果。

输出在数据集下独立的geometry_comparison_时间目录；不替换setp3，不更新正式数据集阶段状态。只在两种方法均成功且源文件哈希不变时，comparison.json中的complete才为true。缺少NKSR时保留Poisson结果，NKSR留空并记录环境错误，不使用其他算法替代。

## 统一条件

- 两种算法读取全部processed_world.npy和surfel_attributes.npz，法线同样单位化，坐标为米。不另做随机下采样。
- Poisson沿用正式流程的depth=9、scale=1.05、4线程；NKSR调用官方预训练Reconstructor，当前正式选用detail_level=0.4、mise_iter=1，不传voxel_size。此前10mm为首轮实验设置。Poisson深度与NKSR体素不是等价容量参数。
- 各保留raw.ply与supported.ply。后者按顶点和面中心距输入均不超过15mm筛选；相同规则不会保证三角形内部每一点都不超过15mm，因此额外统计面上采样的超限比例。
- 均不做平滑、减面、平面优化。正式第三阶段出口仍为既有平面优化Poisson结果，不能把它和原始NKSR直接归因为算法差异。
- 报告全量输入点到网格距离、10万等面积表面采样到输入点距离、5/10/15mm覆盖、面积、边界/非流形边、连通分量、点面数、重建时间。输入本身有噪声；这些是观测一致性，非绝对精度，也不是完整性能占用报告。

## NKSR 独立环境

2026-10-16 已按用户要求返回 WSL 路线。Ubuntu-22.04 独立环境位于 /opt/reconst_3d/nksr/env，CUDA12.8、PyTorch2.7.0+cu128、NKSR1.0.3 已编译并通过真实 CPU 球面重建。无 NVIDIA 显卡时需修补上游 CPU 设备初始化及权重加载，具体备份、校验和与部署步骤见[CUDA工具链](CUDA_TOOLCHAIN.md)。Windows 原生失败记录见[NKSR_WINDOWS.md](NKSR_WINDOWS.md)。

Windows 工作台和 Open3D/Poisson 后台保留原环境；只有 NKSR 模型推理通过 WSL 调用，输出网格后交回 Windows 做统一裁剪及评估。通过扩展导入、真实推理及选用标记三项检查后，默认对照入口使用已验证的 WSL 环境。显式环境变量 RECONST_NKSR_PYTHON 可覆盖自动选择。只迁移源码包不会携带此环境，需要重新部署和验证。

可从 Windows 显式运行：

```powershell
python Reconst_3D/tools/compare_geometry.py --source Reconst_3D/results/real_test/setp2 --nksr-python wsl:Ubuntu-22.04 --device cpu
```

本机无 NVIDIA 显卡，因此测试为 CPU 推理，不能代表 CUDA 加速性能。首次部署需准备官方预训练权重，安装脚本支持经校验的离线缓存。用户于2026-10-19选定NKSR detail_level=0.4，并开始独立测试NoKSR；正式Poisson出口保持不变。Jetson ARM64仍需单独验证，WSL x86-64 编译成功不代表Jetson可直接运行。

### 使用 detail_level 自动选择尺度

对照工具支持与 `--voxel-mm` 互斥的 `--detail-level`，范围0到1。例如：

```powershell
python Reconst_3D/tools/compare_geometry.py --source Reconst_3D/results/real_test/setp2 --detail-level 0.0 --device cpu
python Reconst_3D/tools/compare_geometry.py --source Reconst_3D/results/real_test/setp2 --detail-level 0.8 --device cpu
```

选择此模式时只向模型传入 detail_level，不传 voxel_size；0.0也是有效的显式设置。报告中 nksr_voxel_mm 为null，nksr_detail_level记录所选值，nksr_scale_mode为detail_level。省略两种参数时默认detail_level=0.4；可显式传--voxel-mm继续历史固定尺度实验。工作台正式第三阶段出口不变，mise_iter仍为1。

判断顺序：同视角检查平面波浪、桌沿/薄结构、相邻物体粘连与缺失区，再结合一致性指标与耗时判断。不能用面数更多或输入距离更小直接宣布质量更好。若NKSR有效，再测试是否能减轻现有平滑/平面优化；其效果未确定前不增加NoKSR。

## 本机真实点云首轮结果

完整报告：results/real_test/geometry_comparison_20261016_183037_049909/comparison.json；complete=true，604,018 个输入点，输入校验不变。两者均无后处理，以下为统一15mm支持裁剪版本：

| 指标 | Poisson | NKSR（WSL CPU） |
| --- | ---: | ---: |
| 重建及网格提取时间 | 3.01秒 | 541.34秒 |
| 顶点数 | 230,187 | 1,512,389 |
| 三角面数 | 438,124 | 2,958,734 |
| 输入到网格P50/P95 | 2.64/13.85mm | 2.66/10.56mm |
| 输入10mm覆盖率 | 89.38% | 93.96% |
| 网格表面采样到输入P95 | 9.27mm | 10.37mm |
| 非流形边 | 77 | 919 |

耗时范围包含各自重建和网格提取，NKSR还包含模型构造及缓存权重加载；不含统一导出、距离评估或安装。不是GPU性能对照。运行中观察到 NKSR 进程内存高水位约3.4GiB，仅为阶段性观察，非全流程内存峰值测量。

NKSR输入覆盖有所改善，P50几乎不变，表面到输入P95并未改善；生成更多面且仍有非流形边，不能直接认定为精细高模或直接用于制造。还需在对照窗口检查同一平面、物体边缘与缺失区域。此轮仅验证部署与首轮几何指标，未替换正式Poisson出口，未引入NoKSR。

上游：[源码与安装](https://github.com/nv-tlabs/NKSR)，[使用说明](https://github.com/nv-tlabs/NKSR/blob/public/NKSR-USAGE.md)，[安装扩展约束](https://github.com/nv-tlabs/NKSR/blob/public/package/setup.py)。代码为NVIDIA Source Code License，kitchen-sink预训练权重为CC-BY-SA 4.0，需另核实分发要求。发布包只包含本项目适配代码，不包含NKSR源码、环境或权重。
