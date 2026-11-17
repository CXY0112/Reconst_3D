# 第三阶段 Brush 3DGS

在 Reconst_3D 目录执行 `python main.py`，导入完整数据集，进入 **三阶段 · 场景重建**，算法选择 **Brush 3DGS**，点击 **训练 3DGS（8,000 步）**。训练完成后可直接进入成果展示。

Beta 仍按既有流程自动执行 Poisson；手动选择 Brush 请使用默认 main.py 工作台。

## 输入与默认配置

Brush 使用关联一阶段保存的配准 RGB、深度、优化后 RGB 相机位姿和 map_frames 初始几何。可以跳过第二阶段，也可以从第二阶段进入；RGB 与相机始终来自关联的一阶段，第二阶段的平滑/滤波点云不会替代图像观测。单独 PLY 点云没有图像和相机位姿，不能训练。

默认 8,000 步、最长边 320 px、最多 30,000 个高斯、SH=1、种子 42，每 5 帧留 1 帧评估，每 250 步写出评估图。初始点只采自训练视角。相机光学系转 OpenGL 相机基；世界坐标仍为 X前/Y左/Z上，单位米。

至少需要 2 个训练视角和 1 个评估视角。模型沿用保存的相机位姿，本次集成没有加入离线位姿细化，也不会解决已有的动态物体、位姿误差或相机对齐问题。

## 进度、取消与保存

界面根据已实际生成的评估检查点显示训练进度，另显示导出和保存阶段。可以取消本地任务；取消/退出会终止本次启动的训练子进程。失败、取消或检测到输入变化时保留原有第三阶段结果，只有完整模型通过校验才替换。

几何与高斯分别保存，新结果不会跨类型覆盖：

```
results/数据名称/rgbd_scene/setp3/mesh/          # Poisson / TSDF / NKSR 几何结果
results/数据名称/rgbd_scene/setp3/brush_3dgs/   # Brush 高斯结果
  report.json
  gaussians.ply
  previews/                  # 左输入、右渲染的对照图
  dataset/                   # 自包含图像、相机和初始点
  training/eval_8000/         # 最终留出视角渲染
  training/train.log
```

报告声明 `representation=gaussian_splats`，记录模型 SHA-256、输入观测校验值、参数、高斯数量和逐视角 PSNR。它使用第三阶段既有的事务保存及数据集索引，不会作为三角网格加载。数据集导出会包含这些文件，重新导入可恢复模型、预览与指标。移动后的数据集优先按相对引用查找源 RGB-D。

Poisson/TSDF/NKSR 同属几何结果槽，重建时更新该槽；Brush 只更新高斯槽。底层也会拒绝把高斯结果写到已有几何目录，或把几何写到已有高斯目录。旧的 `setp3/base`、`setp4/base` 按报告类型兼容读取，不搬移、不删除，新的两类目录均不占用旧目录。

## 查看与导出

第三、第五阶段右侧自动提供对应的显示类型：**几何结构**或**3DGS 三维效果**。两类都已保存时可以切换查看。第三阶段选择的结果会带到第五阶段。

右侧只显示交互式三维场景，不再包含 Brush 的 Settings、Stats 或“渲染对照”页签。训练仍使用原版 Brush；显示采用项目内的 OpenGL 高斯光栅化视口，在 AMD 780M 上运行。显示真实各向异性高斯、SH 视角颜色与透明度，不是普通点云或静态图。

打开时读取结果目录 `dataset/transforms.json` 中的第一帧相机位置、朝向、焦距和主点；OpenGL 相机基转回 OpenCV 光学系一次，世界坐标不变。原先仅打开 PLY 的查看器没有这些相机信息，固定默认相机导致侧倒、离场景过远或进入高斯内部。现在第三、第五阶段均从真实采集视角打开。

操作：左键环绕、右键转头、中键平移、滚轮前后移动；W/S 前后、A/D 左右、Q/E 上下。按 **R** 或双击恢复当前采集视角，**[ / ]** 切换前后采集视角。视口隐藏时停止刷新，关闭时释放 GPU 资源。界面不显示对照页，原始对照图和最终评估图仍保留在结果目录。

缺少相机文件或不支持的模型会明确报错，不退回任意默认视角。目前支持项目默认的 SH0/SH1 高斯，Windows OpenGL GPU；单独导出的 PLY 不能自动携带相机，移植应导出完整数据集。

**导出当前模型**保存高斯 PLY，可用兼容 3DGS 的查看器打开。普通三角网格工具不能直接按网格处理该 PLY。第四阶段继续对原始场景点云进行物体分割和独立建模，不会把高斯转换为网格或替换场景。

Brush 的 PLY 查看器及 `--with-viewer` 参数见 [官方 v0.3.0 说明](https://github.com/ArthurBrussee/brush/tree/v0.3.0#viewer)。本机使用已验证的 `vendor/brush/v0.3.0/bin/brush_app.exe`，没有改动官方二进制；不需要 NVIDIA CUDA。

## 本次验证（2026-10-29 场景显示修复）

33 项相关测试通过，包括相机基变换、非中心主点、协方差/透明度/SH、深度排序、视角复位、纯场景 UI、两类结果独立保存、导出/移动和取消保护。

`test1029_fix` 当前正式 Brush 结果为 5,000 步、30,000 个高斯、40 个训练视角、11 个评估视角，原训练报告 PSNR 18.99 dB。未重新训练，也未改写模型。

新视口在相同相机与 320×240 分辨率下，与官方训练保存的 11 张渲染图相比，平均 PSNR 51.15 dB、平均每通道绝对误差约 0.5/255。这个数值衡量两个渲染器的一致性，不是重建质量提升。CPU 计算投影与排序，AMD 780M GPU 完成椭圆高斯光栅化及浮点颜色合成；该录制的实际 701px 宽视口单次绘制约 30–45ms。静止时仅重显缓存，不反复计算场景。

模型仍有原始训练的模糊、重影和未观测区域；移到采集轨迹之外或穿入物体仍可能看到拉丝，这需要后续提高采集覆盖与训练质量。显示修复不会凭空恢复未观测的表面。

验证记录：`results/experiments/brush_scene_fix_20261029/REPORT.md`。历史训练、嵌入验证记录分别保留在 `brush_stage3_integration_20261029` 和 `brush_embed_20261029`。

投影与混合规则参考 Brush v0.3.0 的官方 [helpers.wgsl](https://github.com/ArthurBrussee/brush/blob/v0.3.0/crates/brush-render/src/shaders/helpers.wgsl)、[project_visible.wgsl](https://github.com/ArthurBrussee/brush/blob/v0.3.0/crates/brush-render/src/shaders/project_visible.wgsl)、[rasterize.wgsl](https://github.com/ArthurBrussee/brush/blob/v0.3.0/crates/brush-render/src/shaders/rasterize.wgsl)。本地渲染器使用 OpenGL 浮点缓冲由后到前混合，与官方分块由前到后混合存在浮点舍入及提前终止的微小差异。
