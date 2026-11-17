"""Bind native RTAB Python DLL imports to the target's installed Python 3.12."""
from pathlib import Path
import shutil,sys
if sys.version_info[:2]!=(3,12):raise RuntimeError('Bundled RTAB native ABI requires Python 3.12 x64')
root=Path(__file__).resolve().parents[1]
if not (root/'vendor/rtabmap/bin/rtabmap_core.dll').exists():
    print('Native RTAB binaries absent; build/provision separately.');sys.exit(0)
for name in ('python312.dll','python3.dll'):
    source=Path(sys.base_prefix)/name
    if not source.is_file():raise FileNotFoundError(source)
    shutil.copy2(source,root/'vendor/rtabmap/bin'/name)
print('Prepared native Python DLLs from target installation; no environment was bundled.')
