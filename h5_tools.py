import h5py
import numpy

class H5Loader(object):
    def __init__(self, path, key, channels):
        
    
        self.channels  = channels
        self.h5File = h5py.File(path,'r')
        self.h5Ds = self.h5File[key]



    def load(self,slicing):

        
        ndim = len(self.h5Ds.shape) 
        if self.channels == [0]:
            if ndim == 3:
                a = self.h5Ds[tuple(slicing)].astype('float32')[:,:,:,None]
            elif ndim == 4:
                a = self.h5Ds[tuple(slicing+[0])].astype('float32')[:,:,:,None]
            else:
                raise RuntimeError("wrong dimensions in dataset")
        else:
            raise RuntimeError("not yet implemented")
        
        return a


    def close(self):
        self.h5File.close()
