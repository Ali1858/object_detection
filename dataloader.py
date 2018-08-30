from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler,RandomSampler,BatchSampler
import os
import numpy as np
import collections

from fastai.dataset import open_image

class FilesDataset(Dataset):
    def __init__(self, fnames,y, transform, path):
        self.path,self.fnames,self.transform,self.y = path,fnames,transform,y
        self.n = self.get_n()
        self.c = self.get_c()
        
    def get_sz(self): return self.transform.sz
    def get_x(self, i): return open_image(os.path.join(self.path, self.fnames[i]))
    def get_y(self, i): return self.y[i]
    def get_n(self): return len(self.fnames)
    def get_c(self): return self.y.shape[1] if len(self.y.shape)>1 else 0

    def get1item(self, idx):
        x,y = self.get_x(idx),self.get_y(idx)
        return self.get(self.transform, x, y)

    def __getitem__(self, idx):
        if isinstance(idx,slice):
            xs,ys = zip(*[self.get1item(i) for i in range(*idx.indices(self.n))])
            return np.stack(xs),ys
        return self.get1item(idx)

    def __len__(self): return self.n

    def get(self, tfm, x, y):
        return (x,y) if tfm is None else tfm(x,y)
    
    
    
class ConcatDataset(Dataset):
    
    def __init__(self,ds,y2):
        
        self.ds = ds
        self.y2 = y2
        
    def __getitem__(self, idx):
        x,y = self.ds[idx]
        return (x,(y,self.y2[idx]))
        
    def __len__(self): return len(self.ds)
    
    
def T(a,half=False):
    
    if not torch.is_tensor(a):
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8,np.int16,np.int32,np.int64):
            return torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32,np.float64):
            return torch.FloatTensor(a)
    return a


## may be combine this function and collate function in future
def get_Tensor(batch,pin):
    
    if isinstance(batch,(np.ndarray,np.generic)): return T(batch).contiguous()
    elif isinstance(batch,(str,bytes)):return batch
    elif isinstance(batch, collections.Mapping):
        return {k: get_Tensor(sample,pin) for key,sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [get_Tensor(samples,pin) for samples in batch]
    return TypeError(("batch must contain numbers,dict or list for converting into tensor"))         
        
class CustomDataLoader(object):
    
    def __init__(self,dataset,batch_size,shuffle=True,pin_memory=False):
        
        self.dataset, self.bs = dataset, batch_size 
        self.shuffle,self.pin_memory = shuffle, pin_memory

        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        self.batch_sampler = BatchSampler(sampler,self.bs,drop_last=False)
        
    def __len__(self):return len(self.batch_sampler)
    
    def add_pad(self,b):
        
        if len(b[0].shape) not in (1,2):return np.stack(b)
        length = [len(o) for o in b]
        ml = max(length)
        if min(length) ==  ml : return np.stack(b)
        
        res = np.zeros((len(b),ml),dtype=b[0].dtype)
        
        for i,o in enumerate(b):
            
            res[i,-len(o):] = o
        return res
        
        
    
    def collate(self,batch):
     
        b = batch[0]

        if isinstance(b,(np.ndarray,np.generic)):return self.add_pad(batch)
        elif isinstance(b,(int,float)):return np.array(batch)
        elif isinstance(b,(str,bytes)):return batch
        elif isinstance(b, collections.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in b}
        elif isinstance(b, collections.Sequence):
            transposed = zip(*batch)
            return [self.collate(samples) for samples in transposed]
        return TypeError(("batch must contain numbers,dict or list"))

    def get_batch(self,indices):
        
        return self.collate([self.dataset[i] for i in indices])
    
    def __iter__(self):
        for batch in map(self.get_batch,iter(self.batch_sampler)):
            yield get_Tensor(batch,self.pin_memory)
        
        
        
    

def get_data_loader(input_data,label,tfms,PATH,bs):
    
    train_dataloader = DataLoader(FilesDataset(input_data['train'],label['train'],tfms['train'],PATH), 
                                  batch_size=bs, shuffle=True,pin_memory=False)
    
    valid_dataloader = DataLoader(FilesDataset(input_data['valid'],label['valid'],tfms['valid'],PATH), 
                                 batch_size=bs, pin_memory=False)
    
    
    aug_dataloader = DataLoader(FilesDataset(input_data['valid'],label['valid'],tfms['train'],PATH), 
                                 batch_size=bs, pin_memory=False)
    
    return train_dataloader, valid_dataloader, aug_dataloader




def get_concat_data_loader(input_data,label,label2,tfms,PATH,bs):
    
    trn_ds = FilesDataset(input_data['train'],label['train'],tfms['train'],PATH)
    trn_final_ds = ConcatDataset(trn_ds,label2['train'])
    train_dataloader = CustomDataLoader(trn_final_ds,batch_size=bs)
    
    val_ds = FilesDataset(input_data['valid'],label['valid'],tfms['valid'],PATH)
    val_final_ds = ConcatDataset(val_ds,label2['valid'])
    valid_dataloader = CustomDataLoader(val_final_ds, batch_size=bs,shuffle=False)
    
    
    aug_ds = FilesDataset(input_data['valid'],label['valid'],tfms['valid'],PATH)
    aug_final_ds = ConcatDataset(val_ds,label2['valid'])
    aug_dataloader = CustomDataLoader(val_final_ds, batch_size=bs,shuffle=False)
    
    
    return train_dataloader, valid_dataloader,aug_dataloader