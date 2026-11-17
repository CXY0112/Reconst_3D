"""One running loop job, one replaceable pending job, one result. No graph writes."""
import json,os,queue,subprocess,sys,threading
from pathlib import Path

ROOT=Path(__file__).resolve().parent

class AsyncLoop:
    def __init__(self,folder,command=None):
        self.folder=Path(folder);self.pending=None;self.running=None;self.closed=False
        self.results=queue.Queue(maxsize=1);self.replaced=0;self.refiner_pid=None;self.cancelled=0
        self.log=(self.folder/'async_loop_worker.log').open('w',encoding='utf-8')
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
        self.process=subprocess.Popen(command or [sys.executable,'-u',str(ROOT/'loop_worker.py')],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',env=env,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                if line.startswith('@@REFINER '):self.refiner_pid=int(json.loads(line[10:])['pid'])
                elif line.startswith('@@LOOP '):self.results.put(json.loads(line[7:]))
        except (OSError,ValueError):pass
        if not self.closed:
            try:self.results.put_nowait(dict(error='后台回环进程退出'))
            except queue.Full:pass

    @staticmethod
    def discard(job):
        if job:Path(job['packet']).unlink(missing_ok=True)

    def _dispatch(self,job):
        self.running=job
        try:self.process.stdin.write(json.dumps(job)+'\n');self.process.stdin.flush()
        except (BrokenPipeError,OSError) as exc:
            self.results.put_nowait(dict(error=str(exc)))

    def submit(self,job):
        if self.closed:self.discard(job);return
        if self.running is None:self._dispatch(job)
        else:
            if self.pending:self.replaced+=1;self.discard(self.pending)
            self.pending=job

    def poll(self):
        try:result=self.results.get_nowait()
        except queue.Empty:return None
        self.discard(self.running);self.running=None
        if self.pending:
            job=self.pending;self.pending=None
            self._dispatch(job)
        return result

    def stats(self):return dict(running=int(self.running is not None),pending=int(self.pending is not None),replaced=self.replaced,cancelled=self.cancelled)

    def close(self):
        if self.closed:return
        self.closed=True
        self.cancelled+=int(self.running is not None)+int(self.pending is not None)
        if self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps(dict(command='close'))+'\n');self.process.stdin.flush()
                self.process.wait(timeout=.5)
            except (OSError,subprocess.TimeoutExpired):
                if self.refiner_pid is not None:
                    import signal
                    try:os.kill(self.refiner_pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.process.stdin.close();self.reader.join(timeout=2);self.process.stdout.close();self.log.close()
        self.discard(self.running);self.discard(self.pending);self.running=self.pending=None
