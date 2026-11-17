"""Disk-resident reversible voxel contributions and mmap frame clouds."""
from collections.abc import MutableMapping
from pathlib import Path
import sqlite3
import os
import numpy as np
from voxel_map import voxel_groups


class CloudStore(MutableMapping):
    def __init__(self,folder):self.folder=Path(folder);self.folder.mkdir(exist_ok=True)
    def __getitem__(self,key):
        path=self.folder/f'{int(key)}.npy'
        if not path.is_file():raise KeyError(key)
        return np.load(path,mmap_mode='r',allow_pickle=False)
    def __setitem__(self,key,value):
        path=self.folder/f'{int(key)}.npy';temp=path.with_suffix('.pending')
        try:
            with temp.open('wb') as stream:np.save(stream,value,allow_pickle=False)
            os.replace(temp,path)
        finally:temp.unlink(missing_ok=True)
    def __delitem__(self,key):(self.folder/f'{int(key)}.npy').unlink()
    def __iter__(self):return (int(p.stem) for p in self.folder.glob('*.npy'))
    def __len__(self):return sum(1 for _ in self.folder.glob('*.npy'))


class DiskVoxelMap:
    def __init__(self,path,size=.003,cache_mib=16,check_same_thread=True):
        if not np.isfinite(size) or size<=0:raise ValueError('Invalid voxel size')
        self.size=size;self.path=Path(path);self.poses={};self.cache_mib=cache_mib
        self.db=sqlite3.connect(str(path),check_same_thread=check_same_thread)
        self.db.execute(f'PRAGMA cache_size={-int(cache_mib*1024)}')
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.executescript('CREATE TABLE IF NOT EXISTS config(size REAL); CREATE TABLE IF NOT EXISTS cells(x INTEGER,y INTEGER,z INTEGER,sx REAL,sy REAL,sz REAL,n INTEGER,PRIMARY KEY(x,y,z)) WITHOUT ROWID; CREATE TABLE IF NOT EXISTS parts(node INTEGER,x INTEGER,y INTEGER,z INTEGER,sx REAL,sy REAL,sz REAL,n INTEGER,PRIMARY KEY(node,x,y,z)) WITHOUT ROWID; CREATE TABLE IF NOT EXISTS poses(node INTEGER PRIMARY KEY,pose BLOB);')
        existing=self.db.execute('SELECT size FROM config').fetchone()
        if existing and not np.isclose(existing[0],size):self.db.close();raise ValueError('Voxel size differs from archived map')
        if not existing:self.db.execute('INSERT INTO config VALUES(?)',(size,));self.db.commit()
        for node,data in self.db.execute('SELECT node,pose FROM poses'):self.poses[node]=np.frombuffer(data,dtype=np.float64).reshape(3,4).copy()

    def _remove(self,node):
        # Restrict both operations to this node's indexed contributions. Never
        # scan every map cell once per corrected historical node.
        self.db.execute('UPDATE cells AS c SET sx=c.sx-p.sx,sy=c.sy-p.sy,sz=c.sz-p.sz,n=c.n-p.n FROM parts AS p WHERE p.node=? AND c.x=p.x AND c.y=p.y AND c.z=p.z',(node,))
        self.db.execute('DELETE FROM cells WHERE (x,y,z) IN (SELECT x,y,z FROM parts WHERE node=?) AND n<=0',(node,))
        self.db.execute('DELETE FROM parts WHERE node=?',(node,));self.db.execute('DELETE FROM poses WHERE node=?',(node,))

    def remove(self,node):
        with self.db:self._remove(node)
        self.poses.pop(node,None)

    def update(self,node,points,pose):
        pose=np.asarray(pose,dtype=np.float64)
        if pose.shape!=(3,4) or not np.isfinite(pose).all():raise ValueError('Invalid pose')
        if node in self.poses and np.allclose(self.poses[node],pose,rtol=0,atol=1e-6):return False
        world=np.asarray(points)@pose[:,:3].T+pose[:,3]
        keys,sums,counts=voxel_groups(world,self.size)
        def rows():
            return ((int(k[0]),int(k[1]),int(k[2]),float(s[0]),float(s[1]),float(s[2]),int(n)) for k,s,n in zip(keys,sums,counts))
        with self.db:
            if node in self.poses:self._remove(node)
            self.db.executemany('INSERT INTO parts VALUES(?,?,?,?,?,?,?,?)',((int(node),*r) for r in rows()))
            self.db.execute('INSERT INTO cells SELECT x,y,z,sx,sy,sz,n FROM parts WHERE node=? ON CONFLICT(x,y,z) DO UPDATE SET sx=sx+excluded.sx,sy=sy+excluded.sy,sz=sz+excluded.sz,n=n+excluded.n',(node,))
            self.db.execute('INSERT INTO poses VALUES(?,?)',(int(node),pose.tobytes()))
        self.poses[node]=pose.copy();return True

    def cloud(self):
        # No persistent full-map Python cache. The caller owns this display/export snapshot.
        count=self.db.execute('SELECT count(*) FROM cells').fetchone()[0]
        result=np.empty((count,3));offset=0
        cursor=self.db.execute('SELECT sx/n,sy/n,sz/n FROM cells ORDER BY x,y,z')
        while True:
            rows=cursor.fetchmany(4096)
            if not rows:break
            result[offset:offset+len(rows)]=rows;offset+=len(rows)
        return result

    def stats(self):
        return dict(storage='sqlite',nodes=len(self.poses),cells=self.db.execute('SELECT count(*) FROM cells').fetchone()[0],
                    contribution_rows=self.db.execute('SELECT count(*) FROM parts').fetchone()[0],sqlite_cache_target_bytes=self.cache_mib*1024**2,
                    database_bytes=self.path.stat().st_size)

    def close(self):self.db.close()
