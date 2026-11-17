"""Windows x64 HP60C RGB reader using the supplied vendor SDK."""
import ctypes as ct
import os
from pathlib import Path
import queue
import re
import threading
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
# This device has been verified with the Viewer v1.2.20 SDK. The separate
# EaiCameraSdk v1.2.27 copy fails StartStream (-91) on the same device.
SDK = Path(os.environ.get('RECONST_CAMERA_SDK', str(ROOT / 'vendor/camera_win64')))
DEFAULT_DLL_DIR = SDK
DEFAULT_CONFIG = SDK / 'configurationfiles/hp60c_v2_00_20230704_configEncrypt.json'


class Frame(ct.Structure):
    _fields_ = [('type',ct.c_int),('width',ct.c_uint),('height',ct.c_uint),
                ('data',ct.c_void_p),('size',ct.c_uint),('buffer_size',ct.c_uint),
                ('frame_id',ct.c_uint),('timestamp',ct.c_ulonglong)]


class Data(ct.Structure):
    # The SDK appends a C++ vector after these fields; it is never accessed here.
    _fields_ = [(name,Frame) for name in ('depth','rgb','ir','pointcloud','yuyv','peak','mjpeg')]


class WindowsUSB(ct.Structure):
    _fields_ = [('vid',ct.c_int),('pid',ct.c_int),('device_id',ct.c_int),
                ('symbol_link',ct.c_char*2048),('location_path',ct.c_char*2048),
                ('serial_port',ct.c_char*2048)]


class AttrUnion(ct.Union):
    _fields_ = [('win',WindowsUSB)]  # Largest member of vendor union.


class Attr(ct.Structure):
    _fields_ = [('type',ct.c_int),('attr',AttrUnion)]


ListenerFn = ct.CFUNCTYPE(None,ct.POINTER(Attr),ct.c_void_p)
StreamFn = ct.CFUNCTYPE(None,ct.c_void_p,ct.POINTER(Data),ct.c_void_p)


class Listener(ct.Structure):
    _fields_ = [('attached',ListenerFn),('detached',ListenerFn),('private',ct.c_void_p)]


class Stream(ct.Structure):
    _fields_ = [('callback',StreamFn),('private',ct.c_void_p)]


class StreamParam(ct.Structure):
    _fields_ = [('width',ct.c_int),('height',ct.c_int),('fps',ct.c_int)]


class HP60CRGB:
    def __init__(self, dll_dir=DEFAULT_DLL_DIR, config=DEFAULT_CONFIG):
        self.dll_dir,self.config = Path(dll_dir).resolve(),Path(config).resolve()
        self.condition = threading.Condition()
        self.events = queue.Queue()
        self.quit = threading.Event()
        self.frame,self.sequence,self.consumed,self.error = None,0,0,None
        self.handle = ct.c_void_p()
        self.opened,self.streaming = False,False
        self.dll_cookie = None

    def _bind(self):
        path = self.dll_dir/'AngstrongCameraSdk.dll'
        if os.name != 'nt' or ct.sizeof(ct.c_void_p) != 8:
            raise RuntimeError('HP60C SDK requires Windows x64 Python.')
        if not path.is_file() or not self.config.is_file():
            raise FileNotFoundError(f'HP60C SDK/config missing: {path}, {self.config}')
        self.dll_cookie = os.add_dll_directory(str(self.dll_dir))
        self.lib = ct.CDLL(str(path))
        symbols = [s.decode('ascii') for s in re.findall(rb'[\x20-\x7e]{8,}',path.read_bytes())]
        definitions = {
            'Init':[], 'Deinit':[], 'StopListener':[],
            'StartListener':[ct.POINTER(Listener),ct.c_int,ct.c_bool],
            'CreateCamHandle':[ct.POINTER(ct.c_void_p),ct.POINTER(Attr)],
            'DestoryCamHandle':[ct.c_void_p], 'OpenCamera':[ct.c_void_p,ct.c_char_p],
            'CloseCamera':[ct.c_void_p], 'StartStream':[ct.c_void_p,ct.c_int],
            'StopStream':[ct.c_void_p,ct.c_int],
            'SetStreamParam':[ct.c_void_p,ct.c_int,ct.POINTER(StreamParam)],
            'RegisterStreamCallback':[ct.c_void_p,ct.POINTER(Stream)]}
        self.api = {}
        for name,args in definitions.items():
            matching = [s for s in symbols if s.startswith('?AS_SDK_'+name+'@@')]
            if len(matching) != 1:
                raise RuntimeError(f'Cannot resolve SDK export AS_SDK_{name}')
            fn = getattr(self.lib,matching[0]);fn.argtypes=args;fn.restype=ct.c_int
            self.api[name] = fn

    def _call(self,name,*args):
        result = self.api[name](*args)
        if result != 0:
            raise RuntimeError(f'AS_SDK_{name}: error {result}')

    def _on_attach(self,attr,_):
        if attr and attr.contents.type == 1:
            self.events.put(('attach',ct.string_at(attr,ct.sizeof(Attr))))

    def _on_detach(self,attr,_):
        if attr and attr.contents.type == 1:
            self.events.put(('detach',None))

    def _on_frame(self,handle,data,_):
        if self.quit.is_set() or not data:
            return
        try:
            f = data.contents.rgb
            if not f.data or f.width == 0 or f.height == 0 or f.size != f.width*f.height*3:
                return
            # Despite the SDK field name "rgb", this HP60C/Viewer SDK outputs
            # BGR bytes (also documented in the project's Camera.cpp).
            # Do not swap R/B here: imshow and OpenCV TrackerNano expect BGR.
            bgr = np.frombuffer(ct.string_at(f.data,f.size),np.uint8).reshape(f.height,f.width,3).copy()
            with self.condition:
                self.frame = bgr
                self.sequence += 1
                self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc);self.condition.notify_all()

    def _release(self):
        if self.handle.value:
            if self.streaming:
                self.api['StopStream'](self.handle,0)
            if self.opened:
                self.api['CloseCamera'](self.handle)
            self.api['DestoryCamHandle'](self.handle)
        self.handle = ct.c_void_p()
        self.opened=self.streaming=False

    def _open(self,attr):
        self._call('CreateCamHandle',ct.byref(self.handle),ct.byref(attr))
        self._call('OpenCamera',self.handle,str(self.config).encode('mbcs'));self.opened=True
        # Match the vendor Demo::onCameraOpen: both streams must have equal FPS.
        # AS_MEDIA_TYPE_DEPTH=0, AS_MEDIA_TYPE_RGB=1 in as_camera_sdk_def.h.
        depth_param=StreamParam(640,480,20)
        rgb_param=StreamParam(640,480,20)
        for media,param in ((0,depth_param),(1,rgb_param)):
            result=self.api['SetStreamParam'](self.handle,media,ct.byref(param))
            if result != 0:
                print(f'HP60C stream {media} rejected 640x480@20: {result}; attempting SDK default start.')
        self._call('RegisterStreamCallback',self.handle,ct.byref(self._stream))
        self._call('StartStream',self.handle,0);self.streaming=True

    def _run(self):
        initialized=listening=False
        try:
            self._bind()
            self._attach_ref=ListenerFn(self._on_attach);self._detach_ref=ListenerFn(self._on_detach)
            self._stream_ref=StreamFn(self._on_frame)
            self._listener=Listener(self._attach_ref,self._detach_ref,None)
            self._stream=Stream(self._stream_ref,None)
            self._call('Init');initialized=True
            self._call('StartListener',ct.byref(self._listener),0,True);listening=True
            while not self.quit.is_set():
                try:
                    event,payload=self.events.get(timeout=.1)
                except queue.Empty:
                    continue
                if event == 'detach':
                    self._release()
                    with self.condition:
                        self.frame=None
                    continue
                if self.handle.value:
                    continue
                attr=Attr.from_buffer_copy(payload)
                self._open(attr)
        except Exception as exc:
            with self.condition:
                self.error=str(exc);self.condition.notify_all()
        finally:
            if initialized:
                self._release()
                if listening:
                    self.api['StopListener']()
                self.api['Deinit']()
            if self.dll_cookie:
                self.dll_cookie.close()

    def __enter__(self):
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.thread.start()
        return self

    def read(self,timeout=15):
        deadline=time.monotonic()+timeout
        with self.condition:
            while self.frame is None or self.sequence == self.consumed:
                if self.error:
                    raise RuntimeError(self.error)
                left=deadline-time.monotonic()
                if left <= 0:
                    raise TimeoutError('No new HP60C RGB frame. Connect camera and close other camera viewers.')
                self.condition.wait(left)
            self.consumed=self.sequence
            return self.frame.copy()

    def __exit__(self,*args):
        self.quit.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('HP60C SDK shutdown did not finish.')
