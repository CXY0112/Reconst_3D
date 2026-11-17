import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gcvo_backend import GCVOBackend
from reconstruct import ROOT
from test_bufferx import Matcher,Native


class ReferencePreviewTests(unittest.TestCase):
    def test_reference_image_owned_by_backend_and_frozen_until_new_trusted_keyframe(self):
        class Hooks(Matcher):
            def __init__(self,folder):super().__init__(folder);self.fail=False
            def estimate(self,*args):
                if self.fail:raise ValueError('not converged')
                return super().estimate(*args)
        cloud=np.random.default_rng(7).uniform([-.2,-.2,.8],[.2,.2,1.2],(1000,3))
        color=np.full((480,640,3),123,np.uint8)
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            b=GCVOBackend(folder,matcher_factory=Hooks,client_factory=Native)
            try:
                b.process(color,None,None,1793152092.661,cloud)
                first=b.reference_preview
                self.assertEqual(first['node'],1);self.assertEqual(first['stamp'],1793152092.661)
                self.assertLessEqual(first['image'].shape[0],135);self.assertLessEqual(first['image'].shape[1],240)
                self.assertFalse(first['image'].flags.writeable);color[:]=0
                self.assertTrue(np.all(first['image']==123));self.assertIsNone(b._frame_rgb)
                b.client.status=1;b.process(color,None,None,1793152093.,cloud)
                self.assertIs(b.reference_preview,first)
                b.matcher.fail=True
                for i in range(3):b.process(color,None,None,1793152094.+i,cloud)
                self.assertEqual(b.report['tracking']['state'],'RECOVERY')
                self.assertIs(b.reference_preview,first)
                b.matcher.fail=False;b.client.status=2
                for i in range(2):
                    b.process(color,None,None,1793152097.+i,cloud);self.assertIs(b.reference_preview,first)
                b.process(color,None,None,1793152099.,cloud)
                self.assertEqual(b.report['tracking']['state'],'NORMAL')
                self.assertEqual(b.reference_preview['revision'],2);self.assertEqual(b.reference_preview['node'],2)
                self.assertTrue(np.all(b.reference_preview['image']==0))
                self.assertNotIn('image',b.report['tracking']['reference_view'])
                # Arrays must never enter persistent JSON reports.
                json.dumps(b.report)
            finally:b.close()

    def test_reference_event_emitted_once_per_keyframe_separately_from_json(self):
        from snapshot_window import SnapshotSession
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            session=SimpleNamespace(folder=Path(folder),events=queue.Queue())
            image=np.zeros((2,2,3),np.uint8)
            backend=SimpleNamespace(reference_preview=dict(revision=1,node=1,stamp=1.,image=image),report=dict(accepted=True,node=1,tracking={}))
            for i in range(2):SnapshotSession._after_process(session,backend,i,2)
            events=list(session.events.queue)
            self.assertEqual([k for k,v in events],['tracking_reference','capture_result','capture_result'])
            self.assertIs(events[0][1]['image'],image)
            backend.reference_preview={**backend.reference_preview,'revision':2,'node':2}
            SnapshotSession._after_process(session,backend,3,2)
            self.assertEqual(len([k for k,v in session.events.queue if k=='tracking_reference']),2)
            self.assertEqual(len((Path(folder)/'registrations.jsonl').read_text().splitlines()),3)

    def test_actual_scene_poll_shows_reference_and_keeps_it_while_live_rgb_changes(self):
        import tkinter as tk
        from scene_workbench import SceneWorkbench
        root=tk.Tk();root.withdraw();app=SceneWorkbench(root)
        view=app.views['acquisition'];panel=view.reference_panel
        view.session=SimpleNamespace(stop=threading.Event(),paused=threading.Event(),failed=False,video_mode=True,live_rgb=None)
        view.busy=True
        try:
            self.assertEqual(panel.frame.winfo_manager(),'')
            view.registration_frontend.set('G-CVO（CPU）');self.assertEqual(panel.frame.winfo_manager(),'pack')
            packet=dict(revision=1,node=90,stamp=1793152092.661,image=np.full((100,200,3),123,np.uint8))
            report=dict(accepted=False,skipped=False,method='G-CVO CPU',reason='test',tracking=dict(state='RECOVERY',recovery_good_count=1,recovery_good_required=3,mac_enabled=True,mac_status='running'))
            if view.timer:view.root.after_cancel(view.timer)
            view.events.put(('tracking_reference',packet));view.events.put(('capture_result',(2,report)))
            view.session.live_rgb=np.zeros((100,200,3),np.uint8);view.poll()
            self.assertIn('#90',panel.caption.get());self.assertIn('09:48:12.661',panel.caption.get())
            self.assertIn('RECOVERY',panel.hint.get());self.assertIn('1/3',panel.hint.get())
            self.assertIn('MAC 后台搜索',panel.hint.get())
            self.assertTrue(np.all(panel.preview.image==123));self.assertTrue(np.all(view.rgb_panel.image==0))
            self.assertIn(view.timer,root.tk.call('after','info'))
            view.registration_frontend.set('RGB-D（现有方案）')
            self.assertEqual(panel.frame.winfo_manager(),'');self.assertIsNone(panel.preview.image)
            view.registration_frontend.set('G-CVO（CUDA）');self.assertEqual(panel.frame.winfo_manager(),'pack')
            panel.set_reference(packet)
            with patch('snapshot_window.VideoWindow.start',lambda window:setattr(window,'session',SimpleNamespace())):
                view.start()
            self.assertIsNone(panel.key);self.assertIsNone(panel.preview.image)
        finally:view.session=None;view.busy=False;app.close()


if __name__=='__main__':unittest.main()
