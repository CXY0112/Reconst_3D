"""Fetch a single official checkpoint from the public ZIP using validated HTTP ranges."""
from pathlib import Path
import io,json,urllib.request,zipfile,hashlib,shutil,time

ROOT=Path(__file__).resolve().parents[1]/'vendor/noksr_linux'
ROOT.mkdir(parents=True,exist_ok=True)
URL='https://drive.usercontent.google.com/download?id=14Xrbbox87tLjwE4vmQkEM2u5-vrWW008&export=download&confirm=t'
class Remote(io.RawIOBase):
    def __init__(self):
        with urllib.request.urlopen(urllib.request.Request(URL,method='HEAD'),timeout=60) as r: self.size=int(r.headers['Content-Length'])
        self.position=0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0: raise ValueError('Invalid position')
        return self.position
    def read(self,n=-1):
        n=min(n if n>=0 else self.size-self.position,self.size-self.position)
        if n<=0: return b''
        if n>64*1024*1024: raise ValueError('Unexpected large range')
        start=self.position; end=start+n-1
        request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
        cache=ROOT/'range_cache'; cache.mkdir(exist_ok=True); block=cache/f'{start}_{end}.bin'
        if block.is_file() and block.stat().st_size==n: content=block.read_bytes()
        else:
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(request,timeout=120) as r:
                        if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                            raise RuntimeError('Server did not honor the requested range')
                        content=r.read(n+1)
                    if len(content)!=n: raise RuntimeError('Incomplete range')
                    block.write_bytes(content); break
                except Exception:
                    if attempt==4: raise
                    print('Retrying range',start,'attempt',attempt+1,flush=True); time.sleep(2)
        self.position+=len(content); return content

with zipfile.ZipFile(Remote()) as bundle:
    members=[dict(name=i.filename,bytes=i.file_size,compressed=i.compress_size) for i in bundle.infolist()]
    (ROOT/'checkpoint_members.json').write_text(json.dumps(members,indent=2),encoding='utf-8')
    candidates=[i for i in bundle.infolist() if Path(i.filename).name=='ScanNet_Serial_best.ckpt']
    if len(candidates)!=1: raise RuntimeError('Expected one ScanNet Serial checkpoint')
    entry=candidates[0]; print('Fetching',entry.filename,entry.file_size,'bytes',flush=True)
    destination=ROOT/'ScanNet_Serial_best.ckpt'; temp=destination.with_suffix('.partial')
    with bundle.open(entry) as source,temp.open('wb') as output:
        while chunk:=source.read(4*1024*1024): output.write(chunk)
    if temp.stat().st_size!=entry.file_size: raise ValueError('Checkpoint size mismatch')
    temp.replace(destination)
    sha=hashlib.file_digest(destination.open('rb'),'sha256').hexdigest()
    (ROOT/'checkpoint.json').write_text(json.dumps(dict(source='official NoKSR README Google Drive checkpoints.zip',
        file_id='14Xrbbox87tLjwE4vmQkEM2u5-vrWW008',member=entry.filename,bytes=entry.file_size,
        zip_crc32=f'{entry.CRC:08x}',sha256=sha,verification='ZIP CRC32 and size; local SHA256, not an author-provided digest'),indent=2))
    print('Checkpoint downloaded and ZIP CRC checked:',sha,flush=True)
