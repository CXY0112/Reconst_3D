"""Fetch pinned official MobileSAM sources/weights; never changes Python packages."""
from pathlib import Path
import argparse,hashlib,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
from mobilesam_worker import COMMIT,WEIGHT_SHA256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args();vendor=ROOT/'vendor/mobilesam';vendor.mkdir(parents=True,exist_ok=True)
    source=vendor/'source'/('MobileSAM-'+COMMIT)
    if not (source/'mobile_sam/__init__.py').is_file():
        archive=vendor/'source.zip'
        urllib.request.urlretrieve('https://codeload.github.com/ChaoningZhang/MobileSAM/zip/'+COMMIT,archive)
        with zipfile.ZipFile(archive) as stream:
            if stream.comment.decode()!=COMMIT:raise RuntimeError('下载的源码提交不匹配')
            target=(vendor/'source').resolve()
            for member in stream.infolist():
                path=(target/member.filename).resolve()
                if target!=path and target not in path.parents:raise ValueError('不安全归档路径')
            stream.extractall(target)
    checkpoint=source/'weights/mobile_sam.pt'
    if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=WEIGHT_SHA256:
        raise RuntimeError('官方权重校验失败，请重新下载源码归档')
    print('Official MobileSAM source/weights ready:',source)
    print('Install matching torch + torchvision, timm, pillow, numpy in the selected model environment.')


if __name__=='__main__':main()
