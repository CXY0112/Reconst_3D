> 当前NoKSR已从工作台重建选项中移除，本文保留为历史实验与部署记录；历史网格仍可读取。

# NoKSR 实验与复现

NKSR 当前默认固定为 detail_level=0.4、mise_iter=1，不传voxel_size。NoKSR 为独立实验方法，尚未替换正式第三阶段Poisson出口。

## 已完成的本机测试

Windows工作台加WSL Ubuntu-22.04，AMD Radeon780M，无NVIDIA显卡。原生CPU spconv重复参考测试出现不一致，其旧数据已撤回；以下仅使用纯PyTorch稀疏互相关算子的重新验证结果。原网络构造、官方权重严格载入、小范围、完整场景来源及官方10k采样条件均已执行。保留相同604018点作为评估参考，原始数据SHA256不变。

| 方法 | 网络输入 | CPU重建/提取 | 裁剪后三角面 | 输入P95 | 10mm覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Poisson | 604018 | 2.98s | 438119 | 13.85mm | 89.38% |
| NKSR 0.4 | 604018 | 189.31s | 287702 | 18.06mm | 81.12% |
| NoKSR完整来源 | 324466 | 37.34s | 184568 | 21.18mm | 76.20% |
| NoKSR 10k采样条件 | 9830 | 9.91s | 123965 | 26.33mm | 71.28% |

NoKSR按其1cm稀疏编码网格保留唯一索引代表点，完整来源由604018点变为324466点，10k有放回采样后由10000点变为9830点；没有改写原始文件或对几何做平滑。CPU/GPU都使用这一步避免未定义的重复稀疏索引。当前ScanNet Serial权重在本数据上未带来质量收益，10k采样亦未超过完整来源的唯一网格方案。表中距离衡量含噪观测一致性，不是真值精度；不同实际输入点数不能当作公平速度对照。NoKSR从权重载入开始计时，含输入准备、编码、解码、提取，不含网络构造/依赖导入；原NKSR计时包含模型构造，因此时间不是严格同口径微基准。三种方法无额外平滑/减面，统一外部15mm支持裁剪；NoKSR默认2cm表面体素、DMC mise_iter=0，NKSR为自动尺度及mise_iter=1，容量并不等价。

实际报告：results/real_test/noksr_audit_20261019/REPORT.md。两版原网格、裁剪网格、JSON指标、来源信息分别在geometry_comparison_noksr_verified_20261019与geometry_comparison_noksr_10k_verified_20261019。旧原生CPU后端目录明确标记complete=false，不再用于评价。

## 当前正式默认值（2026-10-19更新）

NoKSR已可在工作台第三阶段作为正式构建方法选择。默认解码20mm、提取网格15mm、k_neighbors=8、last_n_layers=4、内部mask_threshold=30mm，完整来源输入后做1cm唯一网格索引。保留模型输出，不追加外部15mm裁剪、平滑或减面。上面的早期表格采用旧15mm内部掩码与20mm提取网格，是历史实验，不代表当前默认输出。

输出mesh.ply/report.json兼容第四阶段RGB和第五阶段展示；compare_default_geometry.py比较三种工作台默认流程。当前主机推理依赖已配置WSL环境；移动到Linux或其他解释器可用RECONST_NKSR_PYTHON/RECONST_NOKSR_PYTHON覆盖；NoKSR设备由RECONST_NOKSR_DEVICE指定。

## CPU适配边界

tools/noksr_infer.py加载原PointTransformerV3、SDF Decoder、mask Decoder，权重strict=True匹配，未训练或改权重。使用官方enable_flash=False分支；保留Serial邻居算法。绕过训练框架、未使用的KNN/GT依赖，误用明确报错。CPU稀疏卷积以纯PyTorch按同一核权重、邻居位置和偏置实现，3x3/5x5核共8个种子的稠密参考校验通过，不再调用原生CPU spconv卷积运算。GPU保留原CUDA卷积。分批查询保持整批序列化深度，缓存同一排序结果；邻居索引逐项一致、SDF/mask输出数值一致均已校验。小范围4096点经唯一网格选点后实际编码2257点。

这不是已验证的CUDA性能结果；还没有和同权重CUDA输出进行端到端一致性对照。GPU使用原CUDA卷积偏置路径，attention=auto在CUDA且安装flash_attn时使用FlashAttention，否则使用普通Attention。迁移后必须先运行smoke再测速度。

上游固定提交73950498ea095ce95e03d4403540833b16eb887e。官方ScanNet_Serial_best.ckpt为555629651字节，下载自README链接的Google Drive ZIP；ZIP CRC32=c0e5ed28，本地SHA256=96b62c3ad77095e5615d29c5f20e0a96927673c409245b6f845650080c16b881。SHA是本地记录，非作者另行发布的摘要。来源记录在vendor/noksr_linux/source.json、checkpoint.json。

## 复现部署

先按CUDA_TOOLCHAIN.md准备已验证的NKSR Linux环境；NoKSR独立venv只读继承其Torch、torch-scatter、NKSR、Open3D等基础包，新增包不写入NKSR环境。默认前缀/opt/reconst_3d/noksr/env，要求Linux x86-64及Python3.10基础环境。当前不代表Windows原生或Jetson ARM64已验证。

在Windows项目根目录下载源码、Linux依赖及单个官方权重：

```powershell
python Reconst_3D/tools/setup_noksr_linux.py --phase sources
python Reconst_3D/tools/setup_noksr_linux.py --phase wheels --accelerator cpu
python Reconst_3D/tools/fetch_noksr_checkpoint.py
```

在WSL有目标目录写权限的账户下，从挂载的项目根目录安装：

```sh
python3 Reconst_3D/tools/setup_noksr_linux.py --phase environment --accelerator cpu
LD_LIBRARY_PATH=/opt/reconst_3d/nksr/env/lib:/opt/reconst_3d/nksr/env/targets/x86_64-linux/lib /opt/reconst_3d/noksr/env/bin/python Reconst_3D/tools/noksr_infer.py --source Reconst_3D/results/real_test/setp2 --packet Reconst_3D/vendor/noksr_linux/cpu_smoke.npz --smoke
```

Windows上运行完整对照，默认完整来源并执行唯一网格选点、CPU；--input-points 10000可测试官方10k有放回采样配方。评估始终针对完整二阶段点云。工具复用同SHA256的Poisson/NKSR0.4结果，缺少时先生成基线：

```powershell
python Reconst_3D/tools/compare_noksr.py --source Reconst_3D/results/real_test/setp2
```

工作台第三阶段点击“几何方法对照”，右侧选noksr，再运行或导入已有对照目录。可切换nksr/noksr与raw/supported，状态栏显示实际网络输入点数。

## 后续NVIDIA电脑

在新电脑上，先验证Torch CUDA设备，再按目标GPU架构重建NKSR扩展；setup_nksr_linux.py接受TORCH_CUDA_ARCH_LIST环境变量。NoKSR wheels/environment两步改用--accelerator cuda（候选包spconv-cu126==2.3.8，与CPU包使用分开的环境），本机未验证这一GPU组合。不要把当前CPU版spconv与GPU版混装。可在Linux环境安装匹配的flash-attn以启用官方加速注意力；未安装时适配器使用普通Attention。

先用noksr_infer.py --smoke --device cuda:0 --attention standard验证同权重输出与CPU版本的一致性，再在Windows运行速度测试（auto可启用已安装的FlashAttention）：

```powershell
python Reconst_3D/tools/compare_noksr.py --source Reconst_3D/results/real_test/setp2 --device cuda:0
```

界面运行可设置RECONST_NOKSR_DEVICE=cuda:0。独立解释器可由RECONST_NOKSR_PYTHON指定；WSL默认Ubuntu-22.04和上述venv路径。未来GPU正式速度测试需统一输入点数、Attention模式、冷/热启动口径与CUDA同步计时，不能直接引用当前CPU时间推断。

发布包只含适配代码、下载工具和说明，不携带第三方源码、模型权重、venv、wheels或测试扫描。

## 参数与连续性诊断

`tools/noksr_infer.py` 支持显式指定 `--k-neighbors`（正偶数，默认8）、
`--last-n-layers`（当前预训练权重只兼容4）、`--mask-threshold-mm`（默认30）、
`--voxel-mm`（默认20）、`--meshing-voxel-mm`（默认15）。毫米参数在内部转换为米；当前默认来自已选实验配置。
`voxel-mm`同时影响解码邻域与DMC网格，不能当作纯提取分辨率。

只用于实验时，`--meshing-voxel-mm`可覆盖DMC提取网格而保留解码尺度，
`--save-mask-diagnostics`会另存`*_before_mask.npz`，含内部裁剪前的顶点、三角面、
每顶点预测距离掩码，可对同一预测场比较多个裁剪阈值。
这一路径含额外诊断开销，不适合直接比较性能；不修改模型源码、权重或输入。
内部裁剪关闭后可能出现无观测支持的表面，不能将更少孔洞直接视为更准确。

[NoKSR官方仓库与权重说明](https://github.com/theialab/noksr) · [ScanNet配置](https://github.com/theialab/noksr/blob/main/config/model/scannet_model.yaml) · [spconv CPU/GPU说明](https://github.com/traveller59/spconv)
