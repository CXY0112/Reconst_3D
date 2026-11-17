"""Prepare pinned NoKSR inference sources/wheels on Windows and a Linux runtime."""
from pathlib import Path
import argparse,hashlib,json,platform,shutil,subprocess,sys,tarfile,urllib.request
ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'vendor/noksr_linux'
COMMIT='73950498ea095ce95e03d4403540833b16eb887e'
ARCHIVE_SHA='895b26d6e98d1db4c6966cb58ee32848a377fdfa327786312bdc6f79cfdc534d'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('sources','wheels','environment'),required=True)
    p.add_argument('--accelerator',choices=('cpu','cuda'),default='cpu')
    args=p.parse_args(); CONTROL.mkdir(parents=True,exist_ok=True)
    if args.phase=='sources':
        url='https://codeload.github.com/theialab/noksr/tar.gz/'+COMMIT
        archive=CONTROL/'noksr.tar.gz'
        if not archive.exists():
            with urllib.request.urlopen(url,timeout=120) as response,archive.open('wb') as output: shutil.copyfileobj(response,output)
        sha=hashlib.sha256(archive.read_bytes()).hexdigest()
        if sha!=ARCHIVE_SHA: raise RuntimeError('Pinned source archive checksum mismatch')
        folder=CONTROL/'source'; folder.mkdir(exist_ok=True); links=[]
        with tarfile.open(archive) as bundle:
            for member in bundle:
                relative=Path(*Path(member.name).parts[1:]); target=(folder/relative).resolve()
                if not target.is_relative_to(folder.resolve()): raise ValueError('Unsafe archive path')
                if member.issym() or member.islnk(): links.append(dict(path=str(relative),target=member.linkname)); continue
                if '__pycache__' in relative.parts: continue
                if member.isdir(): target.mkdir(parents=True,exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with bundle.extractfile(member) as source,target.open('wb') as output: shutil.copyfileobj(source,output)
        (CONTROL/'source.json').write_text(json.dumps(dict(commit=COMMIT,url=url,sha256=sha,bytes=archive.stat().st_size,skipped_links=links),indent=2),encoding='utf-8')
        print('Pinned NoKSR sources ready'); return
    spconv='spconv==2.3.8' if args.accelerator=='cpu' else 'spconv-cu126==2.3.8'
    wheels=CONTROL/'wheels'; wheels.mkdir(exist_ok=True)
    if args.phase=='wheels':
        common=[sys.executable,'-m','pip','download','--platform','manylinux2014_x86_64','--platform','manylinux_2_28_x86_64',
                '--python-version','310','--implementation','cp','--abi','cp310','--only-binary=:all:','--dest',str(wheels)]
        subprocess.run(common+[spconv],check=True)
        subprocess.run(common+['--no-deps','addict==2.4.0','timm==1.0.15'],check=True)
        return
    if platform.system()!='Linux' or platform.machine()!='x86_64': raise RuntimeError('Run environment phase inside Linux x86-64 / WSL')
    base=Path('/opt/reconst_3d/nksr/env/bin/python'); prefix=Path('/opt/reconst_3d/noksr/env')
    if not base.is_file(): raise RuntimeError('Prepare the verified NKSR Python/CUDA environment first')
    marker=prefix.parent/'environment.json'
    if marker.exists() and json.loads(marker.read_text())['accelerator']!=args.accelerator:
        raise RuntimeError('Do not mix CPU/GPU spconv in an existing environment; use a fresh machine/environment')
    if not (prefix/'bin/python').exists(): subprocess.run([str(base),'-m','venv','--system-site-packages',str(prefix)],check=True)
    python=str(prefix/'bin/python')
    subprocess.run([python,'-m','pip','install','--no-index','--find-links',str(wheels),spconv],check=True)
    subprocess.run([python,'-m','pip','install','--no-index','--no-deps','--find-links',str(wheels),'addict==2.4.0','timm==1.0.15'],check=True)
    marker.write_text(json.dumps(dict(accelerator=args.accelerator,base=str(base),prefix=str(prefix),gpu_inference_verified=False),indent=2))
    print('NoKSR inference environment ready; actual reconstruction validation is still required')

if __name__=='__main__': main()
