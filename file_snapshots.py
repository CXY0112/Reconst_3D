"""Atomic JSON snapshots with Windows delete-sharing readers and bounded commits."""
from pathlib import Path
from functools import lru_cache
import ctypes as ct
import errno,json,os,tempfile,threading,time,weakref

_locks=weakref.WeakValueDictionary()
_locks_guard=threading.Lock()


def _path_lock(path):
    key=os.path.normcase(str(Path(path).resolve()))
    with _locks_guard:
        lock=_locks.get(key)
        if lock is None:lock=threading.RLock();_locks[key]=lock
        return lock


@lru_cache(maxsize=1)
def _windows_files():
    kernel=ct.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[ct.c_wchar_p,ct.c_uint32,ct.c_uint32,ct.c_void_p,ct.c_uint32,ct.c_uint32,ct.c_void_p]
    kernel.CreateFileW.restype=ct.c_void_p
    kernel.CloseHandle.argtypes=[ct.c_void_p];kernel.CloseHandle.restype=ct.c_int
    kernel.ReplaceFileW.argtypes=[ct.c_wchar_p,ct.c_wchar_p,ct.c_wchar_p,ct.c_uint32,ct.c_void_p,ct.c_void_p]
    kernel.ReplaceFileW.restype=ct.c_int
    return kernel


def open_shared_binary(path):
    """Readers keep an immutable old file handle while the writer replaces its name."""
    if os.name!='nt':return Path(path).open('rb')
    import msvcrt
    kernel=_windows_files()
    # GENERIC_READ, FILE_SHARE_READ|WRITE|DELETE, OPEN_EXISTING.
    deadline=time.monotonic()+.25
    while True:
        handle=kernel.CreateFileW(str(Path(path).resolve()),0x80000000,0x1|0x2|0x4,None,3,0x80,None)
        if handle!=ct.c_void_p(-1).value:break
        code=ct.get_last_error();remaining=deadline-time.monotonic()
        if code not in (2,5,32,33) or remaining<=0:
            exc=ct.WinError(code);exc.filename=str(path);raise exc
        time.sleep(min(.005,remaining))
    try:fd=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle);raise
    try:return os.fdopen(fd,'rb')
    except BaseException:
        os.close(fd);raise


def read_shared_bytes(path):
    # Protect the short ReplaceFileW name transition between internal readers
    # and writers; JSON decoding and hashing run after the handle is closed.
    with _path_lock(path):
        with open_shared_binary(path) as stream:return stream.read()


def read_shared_json(path):return json.loads(read_shared_bytes(path))


def _temporary_conflict(exc):
    if os.name=='nt':return getattr(exc,'winerror',None) in (5,32,33,1175)
    return exc.errno in (errno.EACCES,errno.EBUSY,errno.EAGAIN)


def _replace_snapshot(source,target):
    # os.replace uses MoveFileExW, which can reject an open destination even
    # when readers share deletion. ReplaceFileW supports this reader pattern.
    if os.name=='nt' and target.exists():
        kernel=_windows_files()
        if kernel.ReplaceFileW(str(target.resolve()),str(source.resolve()),None,0,None,None):return
        code=ct.get_last_error()
        if code!=2:
            exc=ct.WinError(code);exc.filename=str(source);exc.filename2=str(target);raise exc
    source.replace(target)


class AtomicJsonWriter:
    """One latest checkpoint, no truncation of the committed reader-visible file.

    Live callers defer transient sharing failures without blocking tracking.
    Owners MUST flush at finalization; a persistent error is raised, not hidden.
    Only temporary files created by this writer instance may be removed.
    """
    def __init__(self,path):
        self.path=Path(path);self.pending=None;self.stale=[];self.deferred_count=0
        self.lock=threading.RLock()
        self.io_lock=_path_lock(path)

    def _cleanup(self):
        remaining=[]
        for path in self.stale:
            try:path.unlink(missing_ok=True)
            except OSError:remaining.append(path)
        self.stale=remaining

    def update(self,value):
        payload=json.dumps(value,ensure_ascii=False).encode('utf-8')
        with self.lock:
            fd,name=tempfile.mkstemp(prefix='.'+self.path.name+'.',suffix='.pending',dir=self.path.parent)
            temporary=Path(name)
            try:
                with os.fdopen(fd,'wb') as stream:stream.write(payload)
            except BaseException:
                temporary.unlink(missing_ok=True);raise
            if self.pending is not None:self.stale.append(self.pending)
            self.pending=temporary;self._cleanup()
            return self.flush(timeout=0,strict=False)

    def flush(self,timeout=2.,strict=True):
        with self.lock:
            deadline=time.monotonic()+timeout
            while self.pending is not None:
                try:
                    with self.io_lock:_replace_snapshot(self.pending,self.path)
                except OSError as exc:
                    if not _temporary_conflict(exc):raise
                    remaining=deadline-time.monotonic()
                    if remaining<=0:
                        self.deferred_count+=1
                        if strict:raise
                        return False
                    time.sleep(min(.02,remaining))
                else:self.pending=None
            self._cleanup();return True
