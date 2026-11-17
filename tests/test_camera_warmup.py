import ctypes as ct
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from rtabmap_scan import RGBDCamera


class WarmupTests(unittest.TestCase):
    def camera(self):
        camera=RGBDCamera.__new__(RGBDCamera)
        camera.condition=threading.Condition();camera.quit=threading.Event()
        camera.rgb=camera.depth=camera.last_pair=camera.frame=None
        camera.native_cloud=camera.pair_cloud=None;camera.capture_native_cloud=False
        camera._warmup_until=None;camera.sequence=camera.consumed=0
        camera.error=None;camera.calibration=object()
        return camera

    def frame(self,camera,index,now):
        rgb=(ct.c_uint8*3)(index,index,index);depth=(ct.c_uint16*1)(1000)
        def meta(buffer):
            return NS(data=ct.addressof(buffer),size=ct.sizeof(buffer),width=1,height=1,frame_id=index,timestamp=index*50)
        data=NS(contents=NS(rgb=meta(rgb),depth=meta(depth)))
        with patch('rtabmap_scan.time.monotonic',return_value=now):camera._on_frame(None,data,None)
        self.assertIsNone(camera.error)

    def test_startup_frames_discarded_and_first_delivered_frame_is_fresh(self):
        camera=self.camera()
        # Slow device discovery does not consume the one-second warmup.
        self.frame(camera,1,20.);self.frame(camera,2,20.999)
        self.assertIsNone(camera.frame);self.assertEqual(camera.sequence,0)
        self.frame(camera,3,21.)
        pair,calibration=camera.read()
        self.assertEqual(pair[0][2],3);self.assertEqual(int(pair[0][0][0,0,0]),3)
        self.frame(camera,4,21.05)
        self.assertEqual(camera.read()[0][0][2],4)

    def test_reopening_camera_restarts_warmup(self):
        camera=self.camera();self.frame(camera,1,0.);self.frame(camera,2,1.1);camera.read()
        with patch('rtabmap_scan.DepthCamera._open') as opened:camera._open(object())
        opened.assert_called_once();self.assertIsNone(camera._warmup_until)
        self.frame(camera,3,10.);self.assertEqual(camera.sequence,1)
        self.frame(camera,4,10.9);self.assertEqual(camera.sequence,1)
        self.frame(camera,5,11.);self.assertEqual(camera.read()[0][0][2],5)


if __name__=='__main__':unittest.main()
