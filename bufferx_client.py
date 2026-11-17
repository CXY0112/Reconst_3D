"""Bounded, persistent BUFFER-X worker with explicit runtime configuration."""
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import numpy as np

ROOT=Path(__file__).resolve().parent


def runtime_settings():
    path=Path(os.environ.get('RECONST_BUFFERX_CONFIG',ROOT/'vendor/bufferx/runtime.json'))
    if not path.is_file():raise RuntimeError('BUFFER-X 环境未安装，请参阅 docs/BUFFER_X.md')
    info=json.loads(path.read_text(encoding='utf-8'))
    for key in ('python','source','weights'):
        value=Path(info[key]);info[key]=str(value if value.is_absolute() else ROOT/value)
    if not Path(info['python']).is_file():raise RuntimeError('BUFFER-X Python 不存在：'+info['python'])
    for stage in ('Desc','Pose'):
        if not (Path(info['weights'])/stage/'best.pth').is_file():raise RuntimeError('BUFFER-X 官方预训练权重不完整')
    return info


class BufferXClient:
    def __init__(self,folder):
        self.folder=Path(folder);self.closed=False;self.process=None;self.responses=queue.Queue(maxsize=1)
        self.settings=runtime_settings();self.timeout=float(self.settings.get('timeout_seconds',600))
        self.log=(self.folder/'bufferx_worker.log').open('w',encoding='utf-8')
        config=self.folder/'bufferx_runtime.json';config.write_text(json.dumps(self.settings,indent=2),encoding='utf-8')
        env=os.environ.copy();env.pop('PYTHONHOME',None);env.pop('PYTHONPATH',None)
        cache=self.folder/'bufferx_cache';cache.mkdir(exist_ok=True)
        env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONIOENCODING='utf-8',MPLCONFIGDIR=str(cache.resolve()))
        try:
            self.process=subprocess.Popen([self.settings['python'],'-u',str(ROOT/'bufferx_worker.py'),str(config)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',env=env,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start();ready=self._receive()
            self.settings['effective_parameters']=ready.get('parameters',{})
            config.write_text(json.dumps(self.settings,indent=2),encoding='utf-8')
        except Exception:self.close();raise

    def _read(self):
        try:
            for line in self.process.stdout:
                if line.startswith('@@BUFFERX '):self.responses.put(json.loads(line[len('@@BUFFERX '):]))
                else:self.log.write(line);self.log.flush()
            self.responses.put(dict(error='BUFFER-X 进程退出，请检查 bufferx_worker.log'))
        except Exception as exc:
            try:self.responses.put_nowait(dict(error=str(exc)))
            except queue.Full:pass

    def _receive(self):
        try:value=self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.close();raise RuntimeError('BUFFER-X 超时，工作进程已终止；请检查 bufferx_worker.log')
        if 'error' in value:
            if value.get('rejected'):raise ValueError(value['error'])
            raise RuntimeError(value['error'])
        return value

    def estimate(self,source,target):
        packet=self.folder/'bufferx_pair.npz';np.savez(packet,source=source,target=target)
        try:
            self.process.stdin.write(json.dumps(dict(command='estimate',packet=str(packet.resolve())))+'\n');self.process.stdin.flush()
            return self._receive()
        finally:packet.unlink(missing_ok=True)

    def close(self):
        if self.closed:return
        self.closed=True
        if self.process is not None:
            if self.process.poll() is None:
                try:self.process.stdin.write('{"command":"close"}\n');self.process.stdin.flush();self.process.wait(timeout=2)
                except (OSError,subprocess.TimeoutExpired):self.process.kill();self.process.wait()
            if hasattr(self,'reader'):self.reader.join(timeout=2)
            self.process.stdin.close();self.process.stdout.close()
        self.log.close()
