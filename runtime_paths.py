"""Select native Python dependencies without mixing Windows and Linux ABIs."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def geometry_python():
    configured = os.environ.get('RECONST_GEOMETRY_PYTHON')
    return Path(configured) if configured else Path(sys.executable)


def activate_geometry_packages():
    if os.name == 'nt' and sys.version_info[:2] == (3, 12):
        sys.path.insert(0, str(ROOT/'vendor/geometry_packages'))
