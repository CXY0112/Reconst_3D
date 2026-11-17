"""Stage-labelled output folders, with discovery of historical result names."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent/'results'


def stage_result_path(stage, parent=None, tag=None, now=None):
    if stage not in (1, 2, 3, 4): raise ValueError('Stage must be 1, 2, 3 or 4')
    if tag is not None and (not tag or not all(c.isalnum() or c == '_' for c in tag)):
        raise ValueError('Invalid result tag')
    stamp = (now or datetime.now(timezone(timedelta(hours=8)))).strftime('%Y%m%d_%H%M%S_%f')
    return Path(parent or RESULTS)/f'setp{stage}_{stamp}{"_"+tag if tag else ""}'


def result_reports(kind, parent=None):
    parent = Path(parent or RESULTS)
    patterns = {
        'final': ('geometry_final_*/report.json', 'setp3_*/report.json', '*/setp3/report.json'),
        'comparison': ('geometry_comparison_*/metrics.json', 'setp3_*_comparison/metrics.json'),
        'refined': ('geometry_refined_*/metrics.json', 'setp3_*_refined/metrics.json'),
        'planar': ('geometry_planar_*/metrics.json', 'setp3_*_planar/metrics.json'),
    }
    found = []
    for pattern in patterns[kind]:
        for path in parent.glob(pattern):
            try: data = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError): continue
            if not data.get('complete'): continue
            if kind == 'final':
                from geometry_methods import is_geometry_report
                if not is_geometry_report(data):continue
            found.append((path, data))
    return sorted(found, key=lambda item: item[0].stat().st_mtime)


def latest_preprocessed(parent=None):
    candidates = []
    root=Path(parent or RESULTS)
    for folder in [*root.iterdir(),*root.glob('*/setp2')]:
        report = folder/'report.json'
        if folder.is_dir() and report.is_file():
            try: data = json.loads(report.read_text(encoding='utf-8'))
            except (OSError, ValueError): continue
            if data.get('complete'):
                if data.get('backend')=='kalman' and (folder/'processed_world.npy').is_file() and (folder/'surfel_attributes.npz').is_file():candidates.append(report)
                elif data.get('backend')=='skip' and data.get('skipped'):
                    from geometry_inputs import source_info
                    try:source_info(folder)
                    except (ValueError,FileNotFoundError):continue
                    candidates.append(report)
    if not candidates: raise FileNotFoundError('尚无完整二阶段结果，请先预处理或选择跳过')
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def scan_clouds(parent=None):
    parent = Path(parent or RESULTS)
    return list(parent.glob('scan_*/map_cloud.ply'))+list(parent.glob('setp1_*/map_cloud.ply'))+list(parent.glob('*/setp1/map_cloud.ply'))
