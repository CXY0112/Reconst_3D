"""Check exact package content without camera/network/service access."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
for row in manifest['files']:
    path=root/row['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        raise RuntimeError('Package file mismatch: '+row['path'])
assert 'https://www.bilibili.com/video/BV1Crnt66E6A' in (root/'README.md').read_text(encoding='utf-8')
print('Package content verified:',manifest['profile'],len(manifest['files']),'files')
