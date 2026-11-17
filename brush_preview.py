"""Only the interactive Gaussian scene; diagnostics remain saved on disk."""
from scene_reconstruction import result_path
from embedded_brush import EmbeddedBrush

class BrushPreview:
    def __init__(self,parent):
        self.parent=parent;self.folder=None;self.report=None
        self.renderer=EmbeddedBrush(parent)

    def set_result(self,folder,report):
        self.folder=folder;self.report=report
        self.renderer.set_model(result_path(folder,report['model_relative']),report['model_sha256'])

    def clear(self):
        self.folder=self.report=None;self.renderer.clear()

    def shutdown(self):self.renderer.shutdown()
