"""Bounded fal HTTPS transport. Never log credentials or inline input data."""
import base64,json,os,struct,urllib.error,urllib.parse,urllib.request
from pathlib import Path

if os.name=='nt':
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:pass


def env_file_key(path):
    path=Path(path)
    if not path.is_file():return ''
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        if line.startswith('export '):line=line[7:].lstrip()
        name,separator,value=line.partition('=')
        if not separator or name.strip()!='FAL_KEY':continue
        value=value.strip()
        if len(value)>=2 and value[0] in ('\"',"'") and value[-1]==value[0]:value=value[1:-1]
        return value.strip()
    return ''


def credentials(env_file=None):
    key=env_file_key(env_file) if env_file is not None else ''
    if not key:key=os.environ.get('FAL_KEY','').strip()
    if not key and os.name=='nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as registry:key=str(winreg.QueryValueEx(registry,'FAL_KEY')[0]).strip()
        except OSError:pass
    if not key:raise RuntimeError('未配置FAL_KEY，请选择.env配置文件或设置本机环境变量；尚未提交请求')
    return key


def data_uri(path,mime):return 'data:'+mime+';base64,'+base64.b64encode(Path(path).read_bytes()).decode('ascii')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def safe_error(value,key):
    if isinstance(value,dict):return {k:safe_error(v,key) for k,v in value.items() if k.lower() not in ('input','authorization','headers','image_url','pointmap_url','mask_urls')}
    if isinstance(value,list):return [safe_error(v,key) for v in value[:30]]
    if isinstance(value,str):
        if value.startswith('data:'):return '[omitted data URI]'
        return value.replace(key,'[REDACTED]')[:3000] if key else value[:3000]
    return value


def http_json(url,key,method='GET',payload=None):
    parsed=urllib.parse.urlparse(url)
    trusted=parsed.hostname=='queue.fal.run' or (parsed.hostname=='rest.fal.ai' and parsed.path=='/storage/auth/token')
    if parsed.scheme!='https' or not trusted or parsed.username:raise ValueError('拒绝向未知主机发送API密钥')
    body=None if payload is None else json.dumps(payload,allow_nan=False).encode('utf-8')
    request=urllib.request.Request(url,data=body,method=method,headers={
        'Authorization':'Key '+key,'Content-Type':'application/json','X-Fal-No-Retry':'1'})
    try:
        with urllib.request.build_opener(NoRedirect()).open(request,timeout=60) as response:return json.loads(response.read(16*2**20))
    except urllib.error.HTTPError as exc:
        try:detail=safe_error(json.loads(exc.read(2**20)),key)
        except (ValueError,UnicodeError):detail={'message':'Non-JSON error response'}
        raise RuntimeError(f'FAL HTTP {exc.code}；未自动重试。'+json.dumps(detail,ensure_ascii=False)) from None
    except Exception as exc:raise RuntimeError(safe_error(str(exc),key)) from None


def validate_glb(path):
    with Path(path).open('rb') as stream:header=stream.read(12)
    if len(header)!=12 or header[:4]!=b'glTF' or struct.unpack('<I',header[4:8])[0]!=2 or struct.unpack('<I',header[8:12])[0]!=Path(path).stat().st_size:
        raise ValueError('返回文件不是完整GLB 2.0网格容器')


def download_glb(url,destination):
    parsed=urllib.parse.urlparse(url);host=parsed.hostname or ''
    if parsed.scheme!='https' or parsed.username or not (host in ('fal.media','fal.ai','fal.run') or host.endswith(('.fal.media','.fal.ai','.fal.run'))):
        raise ValueError('模型下载地址不是已知FAL媒体主机')
    destination=Path(destination);temporary=destination.with_suffix('.download');maximum=256*2**20
    try:
        with urllib.request.urlopen(url,timeout=60) as response,temporary.open('xb') as stream:
            total=0
            for block in iter(lambda:response.read(2**20),b''):
                total+=len(block)
                if total>maximum:raise ValueError('模型超过256 MiB下载上限')
                stream.write(block)
        validate_glb(temporary);os.replace(temporary,destination)
    finally:temporary.unlink(missing_ok=True)
