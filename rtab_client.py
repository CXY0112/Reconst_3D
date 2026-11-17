"""Binary IPC to a standalone native RTAB host (never loads RTAB into UI Python)."""
import os,struct,queue,threading,subprocess,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

class RtabClient:
    def __init__(self,folder,timeout=30,optimizer='default',bridge_dir='online_geometry'):
        if optimizer not in ('default','isam2'):raise ValueError('未知轨迹优化方案')
        self.optimizer=optimizer
        self.folder=Path(folder);self.timeout=timeout;self.closed=False;self.closing=False;self.responses=queue.Queue(maxsize=1)
        self.log=(self.folder/'rtab_native.log').open('wb')
        env=os.environ.copy();env.update(PYTHONHOME=os.environ.get('RECONST_RTAB_PYTHONHOME',sys.base_prefix),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
        try:
            native=ROOT/'vendor'/bridge_dir
            command=[str(native/'rtab_host.exe'),str(ROOT/'vendor/rtabmap/bin'),str(native),str(self.folder/'map.db')]
            if optimizer!='default':command.append(optimizer)
            self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()
            response=self._receive()
            self.optimizer_info=json.loads(response['reason']) if response['reason'] else {'requested':'default','verified':False}
            if optimizer=='isam2' and not (self.optimizer_info.get('requested')=='isam2' and
                    self.optimizer_info.get('verified') is True and self.optimizer_info.get('incremental') is True and self.optimizer_info.get('strategy')==2):
                raise RuntimeError('原生运行库未确认启用 iSAM2，请更新原生桥接文件')
            (self.folder/'optimizer.json').write_text(json.dumps(self.optimizer_info,ensure_ascii=False,indent=2),encoding='utf-8')
        except Exception:
            self.close(force=True);raise

    def _exact(self,size):
        data=bytearray()
        while len(data)<size:
            block=self.process.stdout.read(size-len(data))
            if not block:raise EOFError('原生RTAB-Map进程退出，请检查rtab_native.log')
            data.extend(block)
        return bytes(data)

    def _read(self):
        try:
            while True:
                status,node=struct.unpack('<ii',self._exact(8));timing=struct.unpack('<5d',self._exact(40));count=struct.unpack('<i',self._exact(4))[0]
                if count<0 or count>1000000:raise ValueError('Invalid native pose count')
                poses=np.frombuffer(self._exact(count*104),dtype='<f8').reshape(count,13).copy()
                length=struct.unpack('<i',self._exact(4))[0]
                if length<0 or length>1048576:raise ValueError('Invalid native error length')
                reason=self._exact(length).decode('utf-8',errors='replace')
                self.responses.put(dict(status=status,node=node,poses=poses,timings=np.array(timing),reason=reason))
        except Exception as exc:
            if not self.closing:
                try:self.responses.put_nowait(exc)
                except queue.Full:pass

    def _receive(self):
        try:response=self.responses.get(timeout=self.timeout)
        except queue.Empty:raise RuntimeError('RTAB-Map处理超时，请检查rtab_native.log')
        if isinstance(response,Exception):raise RuntimeError(str(response))
        if response['status']<0:raise RuntimeError(response['reason'])
        return response

    def frame(self,color,depth,k,stamp,pose=None,variance=.0004):
        if depth.ndim!=2 or color.shape!=(*depth.shape,3) or k.shape!=(3,3):raise ValueError('Invalid RGB-D input')
        if not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.isfinite(stamp):raise ValueError('Invalid calibration or timestamp')
        if pose is not None:
            if not self.optimizer_info.get('external_pose'):raise RuntimeError('原生桥接未支持外部位姿，请重新构建 rtab_host/online_bridge')
            pose=np.asarray(pose,float)
            if (pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1])
                    or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-4)
                    or np.linalg.det(pose[:3,:3])<.99 or not np.isfinite(variance) or variance<=0):raise ValueError('Invalid external pose/variance')
        h,w=depth.shape
        self.process.stdin.write(struct.pack('<iii5d',4 if pose is not None else 1,w,h,k[0,0],k[1,1],k[0,2],k[1,2],stamp))
        self.process.stdin.write(np.ascontiguousarray(color,np.uint8).tobytes());self.process.stdin.write(np.ascontiguousarray(depth,dtype='<u2').tobytes())
        if pose is not None:self.process.stdin.write(np.ascontiguousarray(pose[:3],dtype='<f8').tobytes()+struct.pack('<d',variance))
        self.process.stdin.flush()
        return self._receive()

    def close(self,force=False):
        if self.closed:return
        self.closing=True;response=None
        process=getattr(self,'process',None)
        if process is not None:
            try:
                if process.poll() is None and not force:
                    process.stdin.write(struct.pack('<i',3));process.stdin.flush();response=self._receive()
                    try:process.wait(timeout=2)
                    except subprocess.TimeoutExpired:pass
            except (OSError,RuntimeError):
                if not force:raise
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=2)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                try:process.stdin.close()
                except OSError:pass
                if hasattr(self,'reader'):self.reader.join(timeout=2)
                process.stdout.close();self.log.close();self.closed=True
        self.log.close();self.closed=True
        return response
