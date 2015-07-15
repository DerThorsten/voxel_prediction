from __future__ import print_function
import numpy


def augmentGaussianPertubation(featureArray, labelsArray,
                               varMult=0.05, n=15):
    """
        augment a flat training set 
        X,Y, default machine setup.
        Add a Gaussian noise to each
        sample. The variance for each
        feature is estiamted from the training
        set.
    """
    fshape = featureArray.shape
    fvar = numpy.var(featureArray,axis=0)
    assert fvar.size == featureArray.shape[1]
    fvar = numpy.sqrt(fvar)*varMult

    f = [featureArray]
    for i in range(n):
        featureArray2 = featureArray.copy()
        r = numpy.random.normal(0,1.0, featureArray.size).reshape(fshape)
        r *=fvar[None,:]
        featureArray2 += r
        f.append(featureArray2)
    featureArray = numpy.concatenate(f,axis=0)
    labelsArray = numpy.concatenate([labelsArray]*len(f),axis=0)



    print("New Features ",featureArray.shape)


    return featureArray, labelsArray
