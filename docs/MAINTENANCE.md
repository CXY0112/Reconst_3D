# 维护与清理范围

## 本轮已删除

- scene_replacement.py 与 tools/scene_replacement_worker.py：旧物体放回场景/合成替换实现。
- pointmap_window.py：独立像素对齐实验窗口。
- comparison_window.py：旧几何算法对照窗口。
- supercluster_cpu.py、tools/setup_supercluster.py、tools/supercluster_infer.py：SuperCluster 自动分割链。
- scene_segmentation.py 中未调用的 segment_scene() 与 inference_command()。
- brush_reconstruction.py 中未调用的 open_viewer()；当前 Brush 训练和内嵌高斯显示保留。

移除了8个专门测试上述旧功能的用例，其他共享处理、标签兼容、工作流、导出、独立物体建模和几何推理测试继续保留。

## 本轮未拆除的旧链路

PointDSC 配准与旧异步回环、网格 RGB 着色、旧 Workbench 页面/旧 Beta 编排、NoKSR 实验仍保留；拆除这些实现需要另行处理旧入口、继承关系与共享函数。本轮未将其误判为可直接整文件删除。

## 当前共用能力

Workbench、SnapshotSession、ScanWindow 中仍有主流程继承的能力。pointmap_processing.py 提供当前采集所需的反投影和 SDK 网格核验。scene_segmentation.py 继续管理场景与标签。reconstruction_pipeline.py、geometry_methods.py、geometry_comparison.py、tools/compare_geometry.py 继续参与几何重建/NKSR。

Windows/Linux 迁移包保留各自原生库、SDK、模型与许可说明；Python环境、私密配置和历史采集仍不打包。项目名称与 Apache 2.0 许可沿用当前开源目录，第三方材料的原始许可不变。

PointDSC 未接入当前工作台，但仍有历史测试依赖它；开源包不带其授权不明源码和权重。不要将“当前主流程可用”理解为“所有旧实验测试都能在缺少资源时执行”。
