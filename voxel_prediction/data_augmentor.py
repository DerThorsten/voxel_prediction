from __future__ import print_function
import numpy
import vigra

def normalVol(shape, center,scale):
    size = numpy.prod(shape)
    a = numpy.random.normal(center,scale,size).reshape(shape)
    a = vigra.taggedView(a, 'xyz')
    return a



def augmentGaussian(data, lAdd, gAdd, gMult,clip):
    """
        lAdd : sigma of local additive gaussian noise
        gAdd : sigma of global additive gaussian noise
        gMult : sigma of global multiplicative guasian noise
    """
    data = vigra.taggedView(data, 'xyz')
    shape = data.shape

    # local and global additive and multiplicative
    # gaussian noise
    toAdd =  normalVol(shape,0.0,lAdd)+numpy.random.normal(0.0,gAdd)
    augmentedData = data.copy()
    augmentedData += toAdd
    augmentedData *= numpy.abs(numpy.random.normal(1.0,gMult))
    augmentedData = numpy.clip(augmentedData,clip[0],clip[1])

    return augmentedData


def augmentRaw(data, lAdd=8.0, gAdd=10.0, gMult=0.4, clip=(0, 255)):
    """
        lAdd : sigma of local additive gaussian noise
        gAdd : sigma of global additive gaussian noise
        gMult : sigma of global multiplicative guasian noise
    """
    



    # apply gaussian augmentation
    gaussianAugmentedData = augmentGaussian(data=data, lAdd=lAdd,
                                            gAdd=gAdd, gMult=gMult,
                                            clip=clip)

    augmentedData = gaussianAugmentedData

    return augmentedData





class DataAugmentor(object):
    
    def __init__(self,nChannels, n, sigmaScaling, clipScaling, channelSettings):
        self.n = n
        self.sigmaScaling = sigmaScaling
        self.clipScaling = clipScaling
        self.nChannels = nChannels

        self.channelKwargs = [None]*nChannels

        for cs in channelSettings:
            channels = cs.pop('channels')
            for channel in channels:
                self.channelKwargs[channel] = cs

        for ckwarg in self.channelKwargs:
            print("ckwarg",ckwarg)

    def nAugmentations(self):
        return self.n

    def margin(self, fCompMargin):
        minScale = float(self.clipScaling[0]) 
        rMinScale = 1.0/minScale
        return tuple([int(float(m)*rMinScale+0.5) for m in fCompMargin])






    def __call__(self, inputData, labels):
        if inputData.ndim == 3:
            inputData = inputData[:, :, :, None]

        # get the new shape
        shape = inputData.shape[0:3]
        fac = numpy.random.normal(1,self.sigmaScaling)
        #print("fac",fac)
        fac = numpy.clip(fac,self.clipScaling[0], self.clipScaling[1])
        print("fac",fac)
        newShape = [int(float(s)*fac +0.5) for s in shape]


        # resize data
        inputData = vigra.taggedView(inputData,'xyzc')
        sInputData = vigra.sampling.resize(inputData, newShape, order=3)

        # resize labels
        flabels = labels.astype('float32').squeeze()
        flabels = vigra.taggedView(flabels,'xyz')
        sLabels = vigra.sampling.resize(flabels, newShape, order=0).astype('uint32')
            

        assert numpy.isnan(numpy.sum(sLabels)) == False
        assert numpy.isnan(numpy.sum(sInputData)) == False

        augmentedInputData = numpy.zeros(sInputData.shape, dtype='float32')
        for c in range(self.nChannels):
            cs = self.channelKwargs[c]
            inputDataC = sInputData[:, :, :, c] 
            if cs is None:
                augmentedInputData[:, :, :, c] = inputDataC
            else:
                augmentedInputData[:, :, :, c] = augmentRaw(inputDataC,**cs)

        assert numpy.isnan(numpy.sum(augmentedInputData)) == False


        return augmentedInputData.astype('float32'),sLabels
