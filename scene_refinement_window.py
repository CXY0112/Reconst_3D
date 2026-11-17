"""Stage four: prompted scene segmentation and independent object modeling."""
from tkinter import ttk
from segmentation_window import SegmentationWindow
from scene_reconstruction_window import SceneReconstructionWindow


class SceneRefinementWindow:
    def __init__(self,root):
        self.root=root;self.dataset=None;self.closed=False;self.source=None;self.output=None
        self.tabs=ttk.Notebook(root);self.tabs.pack(fill='both',expand=True)
        segment=ttk.Frame(self.tabs);refine=ttk.Frame(self.tabs)
        self.tabs.add(segment,text='物体分割');self.tabs.add(refine,text='物体建模')
        self.segmentation=SegmentationWindow(segment)
        self.refinement=SceneReconstructionWindow(refine,kind='refine')
        self.status=self.segmentation.status
        self.segmentation.on_reconstruct=self.open_refinement
        self.tabs.bind('<<NotebookTabChanged>>',self._changed)
        for view in (self.segmentation,self.refinement):
            view.can_write=lambda v=view:self._can_write(v)
            view.on_stage_saved=self._saved
        self.refinement.on_review=lambda:getattr(self,'on_review',lambda:None)()

    @property
    def working(self):return self.segmentation.working or self.refinement.working

    def _can_write(self,candidate):
        if any(v is not candidate and v.working for v in (self.segmentation,self.refinement)):
            candidate.status.set('请等待当前分割或精细建模任务完成');return False
        return self.can_write(candidate) if getattr(self,'can_write',None) else True

    def set_source(self,source,base=None):
        if self.working:return
        self.source=source
        if base:self.refinement.base_override=base
        for view in (self.segmentation,self.refinement):view.dataset=self.dataset
        from scene_stages import segmentation_for
        saved=segmentation_for(self.dataset) if self.dataset else None
        if saved and (saved/'labels.npz').is_file():
            from scene_segmentation import fingerprint
            from scene_stages import read_report
            if read_report(saved/'report.json').get('source_fingerprint')==fingerprint(source):self.segmentation.load_result(saved,source)
            else:self.segmentation.set_source(source)
        else:self.segmentation.set_source(source)

    def _changed(self,event=None):
        if self.closed or self.working:return
        if self.tabs.select()==str(self.refinement.root):
            if self.segmentation.dirty:
                self.segmentation.status.set('请先保存分割，再进入精细建模');self.tabs.select(self.segmentation.root);return
            if self.segmentation.output:self.open_refinement(self.segmentation.output)
            elif self.source:self.refinement.set_source(self.source)

    def open_refinement(self,saved=None):
        if self.working:return
        self.refinement.dataset=self.dataset
        source=self.segmentation.source or self.source
        if source is None:return
        self.refinement.set_source(source,saved or self.segmentation.output)
        self.tabs.select(self.refinement.root)

    def _saved(self,stage,path):
        self.output=path
        if getattr(self,'on_stage_saved',None):self.on_stage_saved(4,path)
        # A changed segmentation must refresh FAL input guards before the next operation.
        if self.segmentation.output:self.refinement.dataset_refresh=True

    def cancel_job(self):
        for view in (self.segmentation,self.refinement):view.cancel_job()

    def shutdown(self):
        self.closed=True
        for view in (self.segmentation,self.refinement):view.shutdown()
