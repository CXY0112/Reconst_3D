# 二阶段：卡尔曼表面融合

默认且唯一公开处理算法为卡尔曼表面融合，默认表面均匀采样。已移除传统双边平滑及两种深度模型的实验入口。

使用main.py的二阶段选项卡，或直接运行preprocessing/main.py。导入扫描目录或其map_cloud.ply，点击开始预处理，数据集模式完成后自动保存并覆盖本阶段，也可点击保存按钮重试。该算法需要原始pointcloud_raw、frames.jsonl、registrations.jsonl、geometry_poses.json与sdk_calibration.json；只有最终融合点云不能重现观测融合。

固定拍摄位姿，从接受的原始观测重新估计局部表面，以法线、距离和边界门控进行卡尔曼融合。对同帧相关观测限制置信度增长，保留可辨识层面与相反朝向面。最后按表面切向距离均匀化，减少重叠区域密度痕迹，避免跨层和锐边合并。详细门控及限幅见[KALMAN.md](KALMAN.md)。

默认噪声尺度10mm、邻域半径40mm、卡尔曼置信度下限2mm、表面间距5mm，采用保守位移限制。它们不是绝对精度承诺，不用于强制修改拍摄位姿。

处理可取消，取消后保留上一次成功结果；左右显示同步视角。数据集模式保存到results/数据名称/setp2；旧独立导入仍可另存到setp2_时间，包含processed.ply、processed_world.npy、surfel_attributes.npz、被移除点/索引/位移诊断及report.json。不覆盖源数据。保存后可进入第三阶段几何重建。
