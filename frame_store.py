"""Disk-backed LRU for registration frames; archived frames remain retrievable."""
from collections import OrderedDict
import numpy as np


class FrameStore:
    def __init__(self,folder,loader,budget=256*1024**2,max_frames=64):
        self.folder=folder;folder.mkdir(exist_ok=True)
        self.loader=loader;self.budget=budget;self.max_frames=max_frames
        self.cache=OrderedDict();self.sizes={};self.count=0
        self.descriptors=[];self.loads=0;self.evictions=0

    def __len__(self):return self.count

    def append(self,frame):
        index=self.count
        frame.save(self.folder/f'{index}.npz')
        descriptor=np.r_[frame.features.mean(axis=0),frame.features.std(axis=0)]
        self.descriptors.append(descriptor/(np.linalg.norm(descriptor)+1e-12))
        self.count+=1;self._keep(index,frame)

    def _keep(self,index,frame):
        size=frame.nbytes()
        self.cache[index]=frame;self.cache.move_to_end(index);self.sizes[index]=size
        while self.cache and (sum(self.sizes.values())>self.budget or len(self.cache)>self.max_frames):
            old,_=self.cache.popitem(last=False);self.sizes.pop(old);self.evictions+=1

    def __getitem__(self,index):
        if index<0:index+=self.count
        if index<0 or index>=self.count:raise IndexError(index)
        if index in self.cache:
            frame=self.cache[index];self.cache.move_to_end(index);return frame
        frame=self.loader(self.folder/f'{index}.npz');self.loads+=1;self._keep(index,frame);return frame

    def match_data(self,index):
        if index in self.cache:return self.cache[index].xyz,self.cache[index].features
        with np.load(self.folder/f'{index}.npz') as data:return data['xyz'].copy(),data['features'].copy()

    def candidates(self,frame,recent=4,old_limit=8):
        start=max(0,self.count-recent)
        local=list(range(start,self.count))
        if not start:return local
        descriptor=np.r_[frame.features.mean(axis=0),frame.features.std(axis=0)]
        descriptor/=np.linalg.norm(descriptor)+1e-12
        scores=np.asarray(self.descriptors[:start])@descriptor
        old=np.argsort(-scores,kind='stable')[:old_limit].tolist()
        return local+old

    def stats(self):
        return dict(resident_frames=len(self.cache),resident_array_bytes=sum(self.sizes.values()),
                    archived_frames=self.count,disk_loads=self.loads,evictions=self.evictions,
                    budget_bytes=self.budget)
