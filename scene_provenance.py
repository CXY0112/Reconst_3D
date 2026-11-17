"""Dependency-free scene hashes, shared by host and isolated geometry runtime."""
from pathlib import Path
import hashlib


def point_source_fingerprint(folder):
    folder=Path(folder);digest=hashlib.sha256()
    if (folder/'processed_world.npy').is_file():names=['processed_world.npy','surfel_attributes.npz']
    else:
        names=['map_cloud.ply','geometry_poses.json','sdk_calibration.json','frames.jsonl','registrations.jsonl']
        for directory in ('rgb_raw','depth_raw'):
            names += [p.relative_to(folder).as_posix() for p in sorted((folder/directory).glob('*')) if p.is_file()]
    for name in names:
        digest.update(name.encode())
        with (folder/name).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
