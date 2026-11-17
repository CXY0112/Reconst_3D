# SuperCluster 历史标签兼容

SuperCluster 自动推理链及安装脚本已从本次代码包移除。当前第四阶段使用 MobileSAM 的 RGB 框选提示分割，不再提供自动全场景实例/语义分割。

scene_segmentation.py 的 SegmentedScene、标签读取/保存、类别配色和物体导出继续保留，可读取已有 SuperCluster 标签。这些兼容能力不需要运行 SuperCluster 模型。

过去文档中的自动推理和安装命令不再适用于当前代码。当前使用方法见 [MobileSAM](MOBILESAM.md)、[工作流](SCENE_WORKFLOW.md) 和 [维护范围](MAINTENANCE.md)。
