"""Frontend report schemas must not terminate the Tk polling loop."""
import threading
import unittest
from types import SimpleNamespace
import numpy as np
from scan_window import capture_result_text


def geometry_report(method='G-CVO CPU'):
    return dict(accepted=True,reference=1,node=3,method=method,matching_ms=122.,
        validation=dict(overlap=.98,rmse_m=.0123),
        frontend=dict(inliers=832,inlier_definition='sampled geometry NN distance'))


class CaptureResultDisplayTests(unittest.TestCase):
    def test_gcvo_reads_nested_geometry_without_rgb_or_icp_fields(self):
        for method in ('G-CVO CPU','G-CVO CUDA'):
            text=capture_result_text(3,geometry_report(method))
            for value in (method,'几何支持点 832','98%','12.3mm','122ms'):self.assertIn(value,text)
            self.assertNotIn('RGB内点',text);self.assertNotIn('ICP',text)

    def test_bufferx_rgbd_and_legacy_reports_remain_supported(self):
        report=geometry_report('BUFFER-X');report['frontend'].pop('inlier_definition')
        self.assertIn('配准内点 832',capture_result_text(3,report))
        self.assertIn('RTAB-Map RGB-D F2M',capture_result_text(2,dict(accepted=True,method='RTAB-Map RGB-D F2M',node=2,matching_ms=30)))
        legacy=dict(accepted=True,reference=1,geometry_edges=3,overlap=.75,rmse_mm=8)
        text=capture_result_text(2,legacy)
        self.assertIn('几何约束 3 条',text);self.assertIn('75%',text);self.assertIn('8.0mm',text)

    def test_initialization_skipped_rejection_and_minimal_reports(self):
        self.assertIn('基准位置',capture_result_text(1,dict(accepted=True,reference=None,method='G-CVO CPU')))
        reason='稠密处理繁忙，本节点仅用于原生定位图'
        self.assertEqual(capture_result_text(3,dict(accepted=False,skipped=True,reason=reason)),reason)
        self.assertEqual(capture_result_text(3,dict(accepted=False,reason='G-CVO 无足够重叠')),'G-CVO 无足够重叠')
        self.assertIn('已接收',capture_result_text(2,dict(accepted=True)))

    def test_actual_poll_continues_after_geometry_event_and_schedules_next_tick(self):
        import tkinter as tk
        from scene_workbench import SceneWorkbench
        root=tk.Tk();root.withdraw();app=SceneWorkbench(root)
        view=app.views['acquisition']
        view.session=SimpleNamespace(stop=threading.Event(),paused=threading.Event(),failed=False,video_mode=True)
        view.busy=True
        try:
            for method in ('G-CVO CPU','G-CVO CUDA','BUFFER-X'):
                if view.timer:view.root.after_cancel(view.timer)
                report=geometry_report(method)
                if method.startswith('G-CVO'):
                    report['tracking']=dict(state='RECOVERY',confidence=.8,recovery_good_count=2,recovery_good_required=3)
                view.events.put(('capture_result',(3,report)))
                view.events.put(('backend_times',np.array([122.,3.,832.,1000.,0.])))
                view.poll()
                self.assertIn(method,view.status.get())
                if method.startswith('G-CVO'):
                    self.assertIn('RECOVERY',view.status.get());self.assertIn('2/3',view.status.get())
                self.assertEqual(view.stage_times['odometry'],122.)
                self.assertEqual(view.events.qsize(),0)
                self.assertIn(view.timer,root.tk.call('after','info'))
        finally:
            view.session=None;view.busy=False;app.close()


if __name__=='__main__':unittest.main()
