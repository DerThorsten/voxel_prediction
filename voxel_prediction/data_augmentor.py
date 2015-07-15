import numpy
import vigra

def normalVol(shape, center,scale):
    size = numpy.prod(shape)
    a = numpy.random.normal(center,scale,size).reshape(shape)
    a = vigra.taggedView(a, 'xyz')
    return a



def augmentGaussian(data, lAdd, gAdd, gMult):
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
    augmentedData = numpy.clip(augmentedData,0,255)

    return augmentedData


def augmentRaw(data, lAdd=8.0, gAdd=10.0, gMult=0.4):
    """
        lAdd : sigma of local additive gaussian noise
        gAdd : sigma of global additive gaussian noise
        gMult : sigma of global multiplicative guasian noise
    """
    



    # apply gaussian augmentation
    gaussianAugmentedData = augmentGaussian(data=data, lAdd=lAdd,
                                            gAdd=gAdd, gMult=gMult)

    augmentedData = gaussianAugmentedData

    return augmentedData
