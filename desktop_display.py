"""Initialize physical-pixel rendering before creating Tk windows."""
import os


def enable_dpi_awareness():
    if os.name!='nt':return
    import ctypes as ct
    user=ct.WinDLL('user32',use_last_error=True)
    try:
        user.SetProcessDpiAwarenessContext.argtypes=[ct.c_void_p]
        if user.SetProcessDpiAwarenessContext(ct.c_void_p(-4)):return
        # A host/manifest may already have selected its DPI policy. Respect it.
        if ct.get_last_error()==5:return
    except AttributeError:pass
    try:ct.windll.shcore.SetProcessDpiAwareness(1)
    except (OSError,AttributeError):user.SetProcessDPIAware()


def configure_window(root,beta=False):
    """Size the initial window for its display, without forcing a low Tk scaling."""
    scale=float(root.tk.call('tk','scaling'))/(96/72)
    width=min(round(1450*scale),root.winfo_screenwidth()-64)
    height=min(round(950*scale),root.winfo_screenheight()-100)
    root.geometry(f'{max(800,width)}x{max(600,height)}')
    if beta:root.minsize(min(width,round(1180*scale)),min(height,round(760*scale)))
