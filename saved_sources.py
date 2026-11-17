"""Resolve saved provenance after moving a dataset between computers."""
from pathlib import Path, PureWindowsPath
import json


def reference(value, owner):
    if not value:
        return None
    owner = Path(owner)
    value = str(value)
    name = PureWindowsPath(value).name if '\\' in value else Path(value).name
    candidates = [owner/value, owner.parent/name, Path(__file__).parent/'results'/name, Path(value)]
    return next((p.resolve() for p in candidates if p.exists()), None)


def metadata(folder):
    try:
        return json.loads((Path(folder)/'report.json').read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return {}


def review_sources(folder):
    folder = Path(folder)
    report = metadata(folder)
    if not report.get('complete') or report.get('pipeline') != 'rgb_vertex_color_baseline':
        raise ValueError('请选择完整的第四阶段着色结果目录')
    geometry = reference(report.get('input'), folder)
    preprocessed = reference(metadata(geometry).get('input'), geometry) if geometry else None
    session = reference(report.get('source_session'), folder)
    if session is None and preprocessed:
        session = reference(metadata(preprocessed).get('source_session'), preprocessed)
    frames = []
    if session:
        for directory in ('rgb_raw', 'rgb'):
            frames = sorted(p for p in (session/directory).glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'))
            if frames:
                break
    raw = next((session/name for name in ('full_geometry_world.npy', 'map_cloud.ply') if (session/name).is_file()), None) if session else None
    return dict(report=report, geometry=geometry, preprocessed=preprocessed, session=session, frames=frames, raw=raw)
