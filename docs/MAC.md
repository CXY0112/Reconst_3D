# MAC CPU 恢复分支（2026-10-28）

选择 G-CVO 后，`vendor/gcvo/runtime.json` 中 `mac.enabled=true` 默认启用。NORMAL 保持原有 G-CVO；进入 RECOVERY 后，继续局部探测，同时启动独立 CPU 恢复进程。普通版和 Beta 共用此链路，无需重新编译 DLL。MAC 使用现有几何 Python/Open3D 运行时，不需要模型权重或 NVIDIA GPU；未来 CUDA G-CVO 仍可使用这个 CPU 恢复器，但本次未实测 GPU。

## 算法与来源

官方源码未修改，位于 `vendor/mac/source`，固定提交 `b0d19635c4c9d69399397495d02422eda94248c2`，MIT 许可证保留在原仓库及 `vendor/mac/LICENSE`。

参考 [官方 basic MAC](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/tree/b0d19635c4c9d69399397495d02422eda94248c2/Python_implement)，本项目增加独立的 NumPy/SciPy CPU 适配，不声称运行的是未经修改的官方二进制或完整 C++ 优化版：

1. 米制点云有界采样至最多 12,000 点，再以 30 mm 体素降采样；Open3D PCA 法向估计半径 2 倍体素、FPFH 半径 5 倍体素。
2. 归一化 FPFH 的双向互近邻匹配；按描述子距离排序，在源/目标各 60 mm 网格内分散取点，最多保留 256 对。空间分散避免重复平面上的集中匹配占满图预算。
3. `FCG=max(0,1-(距离差/sigma)^2)`，清除对角线，阈值 0.99；`SCG=(FCG@FCG)*FCG`。本地 sigma=0.3 m，阈值对应约 30 mm 距离一致性，与当前体素尺度匹配，区别于官方示例 sigma=0.1 m。
4. 以带枢轴的 Bron–Kerbosch 枚举**极大团**，每个节点选择图权重最大的极大团。不是最大团替代，也未改用 RANSAC。
5. 各团 SVD 求解 source→reference 的 SE(3)，以全对应集的截断距离得分选优，并用支持对应点作有界加权 SVD 细化。拒绝共线退化、少于 12 个对应内点或内点比例低于 10% 的候选。没有使用真值来选择细化结果。
6. MAC 候选先通过原有双向重叠/RMSE/SE(3)/运动门槛；再将其作为**当前实时帧** G-CVO 的初值，而不是将较早工作帧的位姿直接当成当前位姿。

## 提交与失败处理

G-CVO 仍须收敛、通过几何与置信度验收、连续确认 3 帧且原生后端接受，才返回 NORMAL。未确认候选不推进可信初值或参考；成功提交后 G-CVO 使用当前解热启动，新关键帧照常刷新参考。世界位姿始终为旧连续里程计锚点乘当前相对 T，不创建新原点。

恢复目标目前是最后可信关键帧，未增加历史关键帧库/全地图检索。原有 0.75 m、60° 运动上限及 35% 双向重叠、35 mm RMSE 上限继续执行；MAC 不会绕过这些限制。它有助于初值失效但仍有重叠的情况，不能恢复完全不重叠的输入。

单进程任务上限 12 秒；团搜索上限 2 秒/10,000 个团，触顶时拒绝该任务而非悄悄截断。失败至少间隔 3 秒再试，种子超过启动后 15 秒失效；参考节点变化、恢复正常或会话结束都会终止旧任务。候选按任务号和参考节点校验，随后必须在当前帧重新配准。后台计算不阻塞预览，但仍会占用 CPU；NORMAL 不启动恢复任务。

参数在 `runtime.json` 的 `mac` 对象中，完整有效配置随扫描保存在 `mac_runtime.json`。`mac.enabled=false` 可关闭后台分支，保留原有局部恢复和 G-CVO。启动器及进程内均有资源界限。

## 界面与日志

参考图仍在 RECOVERY 中保持固定，下方显示“MAC 后台搜索 / MAC 候选确认中 / 本轮未通过”。实时跟踪日志新增 `mac` 状态和 `used_seed`，`tracking.mac_enabled` 反映是否启用；有效种子参与当前帧求解时，`recovery_method=mac_seed_then_gcvo`。

- `mac_worker.log`：独立进程输出。
- `mac_metrics.jsonl`：已完成任务的结果及拒绝原因。
- `mac_input.npz`、`mac_output.json`：最近一次任务输入/输出，后续任务覆盖，避免采集目录无限增长；输入元数据包含源帧时间、参考节点和任务号。
- `gcvo_cpu_metrics.jsonl` / `gcvo_cuda_metrics.jsonl`、`registrations.jsonl`：逐帧当前帧验收、状态和种子使用情况。

## 验证与使用

52 项相关自动测试通过，覆盖团枚举与独立 NetworkX 对照、75% 外点的已知位姿、退化与预算拒绝、超时/过期/旧任务拒绝、只提交当前帧及原有 UI/视频/CPU 配准行为。

真实 Open3D+MAC 子进程对一帧保存点云的 3 组受控变换全部通过，最大平移误差约 10.29 mm、旋转误差约 0.35°；每次含启动和特征提取约 3.2–3.8 秒。这不是跨视角真值精度或现场恢复率。

真实 MAC+G-CVO+原生 RTAB 离线故障注入验证通过：故意将局部初值移出捕获范围，进入 RECOVERY，使用真实 MAC 种子并重新确认后返回 NORMAL。仅初始化及最终确认两次提交，世界坐标连续、可信初值被重新初始化。重复的是一个已保存视角，不代表真实遮挡/环境变化能力。

证据位于 `results/experiments/mac_validation_20261028/final/report.json` 和 `native_pipeline/report.json`。同目录根部与 capped128 是早期参数试验，不能代替 final 的结论。

重启并选择 G-CVO（CPU），正常扫描后遮挡至 RECOVERY，移开遮挡并缓慢展示与参考图有充分重叠的场景，观察 MAC 状态、连续确认及恢复后的地图是否跳变；保存新的数据集供复核。暂未连接相机做这一步验证。
