"""Scene-only anisotropic Gaussian rasterization on the existing Windows GL GPU.

CPU projects/sorts all Gaussian ellipses; GPU evaluates their footprint and
composites in a float framebuffer (no premature RGB clipping/8-bit rounding).
Training remains the unmodified official Brush executable.
"""
import ctypes as ct
import numpy as np
from map_cloud_view import _WGLRenderer

VERTEX='''#version 120
attribute vec2 vertex;
attribute vec2 offset;
attribute vec3 conic;
attribute vec4 color;
varying vec2 d;
varying vec3 c;
varying vec4 rgba;
void main(){gl_Position=vec4(vertex,0.,1.);d=offset;c=conic;rgba=color;}
'''
FRAGMENT='''#version 120
varying vec2 d;
varying vec3 c;
varying vec4 rgba;
void main(){
 float power=.5*(c.x*d.x*d.x+c.z*d.y*d.y)+c.y*d.x*d.y;
 float a=min(.999,rgba.a*exp(-power));
 if(power<0. || a<1./255.) discard;
 gl_FragColor=vec4(rgba.rgb,a);
}
'''


class GaussianRenderer(_WGLRenderer):
    def __init__(self,widget):
        self.program=0;self.fbo=ct.c_uint();self.texture=ct.c_uint();self.size=None
        super().__init__(widget)
        try:
            self.create_shader=self._required('glCreateShader',ct.c_uint,ct.c_uint)
            self.shader_source=self._required('glShaderSource',None,ct.c_uint,ct.c_int,ct.POINTER(ct.c_char_p),ct.c_void_p)
            self.compile_shader=self._required('glCompileShader',None,ct.c_uint)
            self.get_shader=self._required('glGetShaderiv',None,ct.c_uint,ct.c_uint,ct.POINTER(ct.c_int))
            self.get_log=self._required('glGetShaderInfoLog',None,ct.c_uint,ct.c_int,ct.c_void_p,ct.c_char_p)
            self.delete_shader=self._required('glDeleteShader',None,ct.c_uint)
            create=self._required('glCreateProgram',ct.c_uint)
            attach=self._required('glAttachShader',None,ct.c_uint,ct.c_uint)
            link=self._required('glLinkProgram',None,ct.c_uint)
            get_program=self._required('glGetProgramiv',None,ct.c_uint,ct.c_uint,ct.POINTER(ct.c_int))
            self.use=self._required('glUseProgram',None,ct.c_uint)
            self.delete_program=self._required('glDeleteProgram',None,ct.c_uint)
            self.attribute=self._required('glGetAttribLocation',ct.c_int,ct.c_uint,ct.c_char_p)
            self.enable_attribute=self._required('glEnableVertexAttribArray',None,ct.c_uint)
            self.disable_attribute=self._required('glDisableVertexAttribArray',None,ct.c_uint)
            self.pointer=self._required('glVertexAttribPointer',None,ct.c_uint,ct.c_int,ct.c_uint,ct.c_ubyte,ct.c_int,ct.c_void_p)
            self.gen_fbo=self._required('glGenFramebuffers',None,ct.c_int,ct.POINTER(ct.c_uint))
            self.bind_fbo=self._required('glBindFramebuffer',None,ct.c_uint,ct.c_uint)
            self.attach_texture=self._required('glFramebufferTexture2D',None,ct.c_uint,ct.c_uint,ct.c_uint,ct.c_uint,ct.c_int)
            self.check_fbo=self._required('glCheckFramebufferStatus',ct.c_uint,ct.c_uint)
            self.delete_fbo=self._required('glDeleteFramebuffers',None,ct.c_int,ct.POINTER(ct.c_uint))
            self.blit=self._required('glBlitFramebuffer',None,*([ct.c_int]*8),ct.c_uint,ct.c_uint)
            signatures={'glBlendFunc':([ct.c_uint]*2,None),'glGenTextures':([ct.c_int,ct.POINTER(ct.c_uint)],None),
                'glBindTexture':([ct.c_uint]*2,None),'glDeleteTextures':([ct.c_int,ct.POINTER(ct.c_uint)],None),
                'glTexParameteri':([ct.c_uint,ct.c_uint,ct.c_int],None),
                'glTexImage2D':([ct.c_uint,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_uint,ct.c_uint,ct.c_void_p],None),
                'glReadPixels':([ct.c_int]*4+[ct.c_uint,ct.c_uint,ct.c_void_p],None),
                'glPixelStorei':([ct.c_uint,ct.c_int],None)}
            for name,(args,result) in signatures.items():
                fn=getattr(self.gl,name);fn.argtypes=args;fn.restype=result
            self.program=create()
            for kind,source in ((0x8B31,VERTEX),(0x8B30,FRAGMENT)):
                shader=self.create_shader(kind)
                try:
                    text=ct.c_char_p(source.encode());self.shader_source(shader,1,ct.byref(text),None);self.compile_shader(shader)
                    ok=ct.c_int();self.get_shader(shader,0x8B81,ct.byref(ok))
                    if not ok.value:
                        log=ct.create_string_buffer(4096);self.get_log(shader,len(log),None,log)
                        raise RuntimeError(log.value.decode(errors='replace'))
                    attach(self.program,shader)
                finally:self.delete_shader(shader)
            link(self.program);ok=ct.c_int();get_program(self.program,0x8B82,ct.byref(ok))
            if not ok.value:raise RuntimeError('Gaussian shader link failed')
            self.layout=[(self.attribute(self.program,n),size,offset) for n,size,offset in
                         ((b'vertex',2,0),(b'offset',2,8),(b'conic',3,16),(b'color',4,28))]
            self.gen_fbo(1,ct.byref(self.fbo));self.gl.glGenTextures(1,ct.byref(self.texture))
            self.gl.glDisableClientState(0x8074)
        except Exception:
            self.close();raise

    def _required(self,name,result,*args):
        fn=self._extension(name,result,*args)
        if fn is None:raise RuntimeError('GPU 不支持 '+name)
        return fn

    def draw_scene(self,scene,pose,intrinsics,width,height,readback=False,present=True):
        gl=self.gl;gl.wglMakeCurrent(self.dc,self.context)
        if self.size!=(width,height):
            gl.glBindTexture(0x0DE1,self.texture.value)
            gl.glTexParameteri(0x0DE1,0x2801,0x2600);gl.glTexParameteri(0x0DE1,0x2800,0x2600)
            gl.glTexImage2D(0x0DE1,0,0x8814,width,height,0,0x1908,0x1406,None) # RGBA32F
            self.bind_fbo(0x8D40,self.fbo.value)
            self.attach_texture(0x8D40,0x8CE0,0x0DE1,self.texture.value,0)
            if self.check_fbo(0x8D40)!=0x8CD5:raise RuntimeError('无法建立浮点高斯绘图缓冲')
            self.size=(width,height)
        self.bind_fbo(0x8D40,self.fbo.value)
        gl.glViewport(0,0,width,height);gl.glDisable(0x0B71);gl.glDisable(0x0C11);gl.glDisable(0x8DB9)
        gl.glClearColor(0,0,0,0);gl.glClear(0x4000)
        means,conic,opacity,color,extent=scene.project(pose,intrinsics,width,height)
        # Clip quad rectangles to the viewport, retaining exact pixel deltas.
        lo=np.maximum(means-extent,0);hi=np.minimum(means+extent,[width,height])
        corners=np.array([[0,0],[1,0],[1,1],[0,0],[1,1],[0,1]],np.float32)
        pixels=lo[:,None,:]+corners[None,:,:]*(hi-lo)[:,None,:]
        packed=np.empty((len(means),6,11),np.float32)
        packed[:,:,:2]=pixels/[width,height]*[2,-2]+[-1,1]
        packed[:,:,2:4]=pixels-means[:,None,:]
        packed[:,:,4:7]=conic[:,None,:]
        packed[:,:,7:10]=color[:,None,:];packed[:,:,10]=opacity[:,None]
        self.use(self.program);gl.glEnable(0x0BE2);gl.glBlendFunc(0x0302,0x0303)
        self.bind(0x8892,self.buffer.value);self.upload(0x8892,packed.nbytes,packed.ctypes.data,0x88E0)
        for loc,size,offset in self.layout:
            self.enable_attribute(loc);self.pointer(loc,size,0x1406,0,44,ct.c_void_p(offset))
        gl.glDrawArrays(4,0,len(means)*6)
        for loc,_,_ in self.layout:self.disable_attribute(loc)
        self.bind(0x8892,0);self.use(0);gl.glDisable(0x0BE2)
        image=None
        if readback:
            pixels=np.empty((height,width,3),np.float32);gl.glPixelStorei(0x0D05,1)
            gl.glReadPixels(0,0,width,height,0x1907,0x1406,pixels.ctypes.data)
            image=np.clip(pixels[::-1]*255,0,255).astype(np.uint8)
        if present:
            self.present()
        self.bind_fbo(0x8D40,0);gl.glFinish()
        return image

    def present(self):
        """Re-present cached float image after Tk repaints/resizes its child HWND."""
        if not self.size:return
        self.gl.wglMakeCurrent(self.dc,self.context)
        self.bind_fbo(0x8CA8,self.fbo.value);self.bind_fbo(0x8CA9,0)
        self.blit(0,0,*self.size,0,0,self.widget.winfo_width(),self.widget.winfo_height(),0x4000,0x2601)
        self.bind_fbo(0x8D40,0);self.gl.glFlush();self.gdi.SwapBuffers(self.dc)

    def close(self):
        if getattr(self,'context',None):
            self.gl.wglMakeCurrent(self.dc,self.context)
            if self.program and hasattr(self,'delete_program'):self.delete_program(self.program);self.program=0
            if self.fbo.value and hasattr(self,'delete_fbo'):self.delete_fbo(1,ct.byref(self.fbo));self.fbo.value=0
            if self.texture.value:self.gl.glDeleteTextures(1,ct.byref(self.texture));self.texture.value=0
        super().close()
