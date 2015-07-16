import h5py
import numpy

class H5Loader(object):
    def __init__(self, path, key, channels):
        
    
        self.channels  = channels
        self.h5File = h5py.File(path,'r')
        self.h5Ds = self.h5File[key]

    def loadAllChannels(self, slicing):
        shape = self.h5Ds.shape
        ndim = len(self.h5Ds.shape) 

        if ndim == 3:
            return self.h5Ds[tuple(slicing)].astype('float32')[:,:,:,None]
        elif ndim == 4:
            s = s
            s = tuple(slicing + [slice(0,shape[3])] )
            return self.h5Ds[s].astype('float32')

    def load(self,slicing):

        ch = self.channels
        nCh = len(ch)
        ndim = len(self.h5Ds.shape) 
        if nCh == 1:
            if ndim == 3:
                if ch[0] != 0:
                    raise RuntimeError("channel of 3D array must be [0]")
                a = self.h5Ds[tuple(slicing)].astype('float32')[:,:,:,None]
            elif ndim == 4:
                a = self.h5Ds[tuple(slicing+[ch[0]])].astype('float32')
            else:
                raise RuntimeError("wrong dimensions in dataset")
        else:
            raise RuntimeError("not yet implemented")
        
        return a


    def close(self):
        self.h5File.close()
