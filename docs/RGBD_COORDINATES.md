# RGB-D 坐标契约（2026-10-29）

当前 HP60C SDK 的配准模式中，`pair_cloud` XYZ 和 `depth_raw` 网格均以 RGB 相机为参考。`RGBDCamera.depth_frame` 和 `pointcloud_frame` 明确为 `rgb`。对 test10281 全部 31 帧的数值核验支持这一契约；这不是独立物理标定精度的测量。

## 新采集的处理

- SDK XYZ（毫米）转为米后保留 RGB 坐标，不再重复应用 IR→RGB 外参。
- 缺少同步 SDK XYZ 时，深度图使用所声明相机的内参与畸变模型反投影；当前 HP60C 使用 RGB 内参。
- 真正的 IR 深度或 IR XYZ 显式声明为 `ir`，继续使用 IR 内参并应用一次 IR→RGB 外参。`register_rgbd` 的既有默认 IR 接口保持不变，新增 `points_rgb` 和 `depth_frame`。
- 默认 RGB-D、G-CVO CPU/CUDA 和 BUFFER-X 共用修正后的输入；此次没有修改匹配器、置信度阈值、MAC 算法或原生库。
- 预览使用点云实际坐标系的内参；局部表面包保存其坐标标签。第二阶段融合、第三阶段 TSDF、第四阶段着色按明确坐标声明选择位姿。Brush 导出仍使用 RGB 相机位姿及 RGB 坐标的 `map_frames`，默认 5,000 步不变。

## 保存格式

`frames.jsonl` 新行包含：

```json
{"coordinate_schema":"rgbd_coordinates_v2","raw_depth_frame":"rgb","pointcloud_frame":"rgb","registered_depth_frame":"rgb","map_frame":"rgb"}
```

`pointcloud_frame` 是 `pointcloud_local` 内点和法线的相机光学坐标。`map_frame` 是 `map_frames` 内点的相机坐标，不是全局地图轴。`geometry_poses.json` 仍是世界系（X前、Y左、Z上）从 RGB 相机光学系（X右、Y下、Z前）的变换。

局部 NPZ 的 `metadata.camera_frame` 和 `coordinates` 必须与帧清单一致。未知坐标声明会明确报错。专用 SDK RGB 网格 TSDF 只接受新声明中深度和局部云都为 RGB 的情况。

## 旧数据

原始 test10281 和已有结果没有被重写。未标记的旧文件保留既有读取含义，避免静默改变历史结果；这也意味着历史重复外参及其轨迹影响不会自动消失。不要只修改旧文件标签后当作新数据使用。

离线重新配准工具 `tools/replay_registration.py` 从原始图像重新构建输入，不读取旧位姿作为初值。新记录自动读坐标声明；旧记录必须用 `--depth-frame` 提供已核验的坐标系。稀疏保存帧无法还原原来的连续 10 Hz 流，回放丢帧率和总耗时不能当作实时采集指标。

## 验证与下一次采集

修复报告：`results/experiments/rgbd_coordinates_fix_20261029/REPORT.md`。31 帧点云投影与原始深度的 10 mm 一致率每帧超过 99.99%，RGB 深度回退与原网格完全一致；这是输入一致性指标，不是真实位姿准确率。

重启项目后，用 G-CVO CPU 重新采集约 30–60 秒静态场景，建议命名 `test1029_fix`。关闭动态屏幕内容，缓慢横移并围绕目标小幅移动，保持视野重叠；发生 RECOVERY 时按参考图回到可信视角。先检查新数据的跨帧重投影和轨迹，再用相同的 Brush 5,000 步配置比较。新采集和重建效果尚未验证。
