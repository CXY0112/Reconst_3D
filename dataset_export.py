"""Export only the named dataset manifest and saved pipeline stages."""
from pathlib import Path
import os
import tempfile
import zipfile
from dataset_store import inspect_dataset


def export_dataset(source,destination,overwrite=False):
    source=Path(source).resolve();destination=Path(destination).resolve()
    inspect_dataset(source)
    if destination==source or source in destination.parents:raise ValueError('请将导出文件保存到当前数据集目录之外')
    if destination.exists() and not overwrite:raise FileExistsError(destination)
    files=[source/'dataset.json'] if (source/'dataset.json').is_file() else []
    folders=[source/f'setp{stage}' for stage in range(1,5)]
    folders += [source/'rgbd_scene'/f'setp{stage}' for stage in (2,3,4)]
    for folder in folders:
        if folder.is_symlink():raise ValueError('阶段目录不能是符号链接')
        if folder.is_dir():files.extend(path for path in folder.rglob('*') if path.is_file())
    if not files:raise ValueError('当前数据集没有可导出的文件')
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(source):raise ValueError('导出来源包含数据集外部链接')
    fingerprints={path:(path.stat().st_size,path.stat().st_mtime_ns) for path in files}
    handle,temporary=tempfile.mkstemp(prefix='.dataset_export_',suffix='.zip',dir=destination.parent)
    os.close(handle);temporary=Path(temporary)
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for path in files:archive.write(path,source.name+'/'+path.relative_to(source).as_posix())
        if any((p.stat().st_size,p.stat().st_mtime_ns)!=value for p,value in fingerprints.items()):raise RuntimeError('导出期间数据发生变化，请停止处理后重试')
        os.replace(temporary,destination)
    finally:temporary.unlink(missing_ok=True)
    return destination
