import unittest,tempfile,time,sys
from pathlib import Path
from async_loop import AsyncLoop

class AsyncLoopTests(unittest.TestCase):
    def test_bounded_queue_replaces_pending_and_accepts_new_job_after_idle(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            folder=Path(temp);script=folder/'fake.py'
            script.write_text("import sys,json,time\nfor line in sys.stdin:\n job=json.loads(line);time.sleep(.08);print('@@LOOP '+json.dumps(dict(source=job['source'],links=[])),flush=True)\n")
            worker=AsyncLoop(folder,[sys.executable,'-u',str(script)])
            def job(i):
                path=folder/f'{i}.npz';path.write_bytes(b'packet');return dict(packet=str(path),source=i)
            def wait():
                deadline=time.monotonic()+10
                while time.monotonic()<deadline:
                    value=worker.poll()
                    if value is not None:return value
                    time.sleep(.01)
                self.fail('No background response')
            try:
                worker.submit(job(1));worker.submit(job(2));worker.submit(job(3))
                self.assertEqual(worker.stats(),dict(running=1,pending=1,replaced=1,cancelled=0))
                self.assertFalse((folder/'2.npz').exists())
                self.assertEqual(wait()['source'],1);self.assertEqual(wait()['source'],3)
                worker.submit(job(4));self.assertEqual(wait()['source'],4)
                worker.submit(job(5));worker.submit(job(6))
            finally:worker.close()
            self.assertIsNotNone(worker.process.poll());self.assertFalse(worker.reader.is_alive())
            self.assertFalse(list(folder.glob('*.npz')))

    def test_cancel_stops_owned_refiner_before_dataset_can_be_saved(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            folder=Path(temp);script=folder/'tree.py';marker=folder/'late_write.txt'
            child="import time;from pathlib import Path;time.sleep(2);Path("+repr(str(marker))+").write_text('late')"
            script.write_text("import subprocess,sys,json,time\nfor line in sys.stdin:\n p=subprocess.Popen([sys.executable,'-c',"+repr(child)+"]);print('@@REFINER '+json.dumps(dict(pid=p.pid)),flush=True);time.sleep(20)\n")
            worker=AsyncLoop(folder,[sys.executable,'-u',str(script)])
            packet=folder/'packet.npz';packet.write_bytes(b'packet')
            try:
                worker.submit(dict(packet=str(packet),source=1))
                deadline=time.monotonic()+5
                while worker.refiner_pid is None and time.monotonic()<deadline:time.sleep(.01)
                self.assertIsNotNone(worker.refiner_pid)
            finally:worker.close()
            time.sleep(2.1)
            self.assertFalse(marker.exists());self.assertFalse(packet.exists())
