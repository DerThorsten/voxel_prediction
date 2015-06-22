from __future__ import print_function
from colorama import Fore, Back, Style
import h5py
import vigra
import skneuro
import skneuro.utilities as skut
import concurrent.futures
import sys
import pickle
from block_yielder import *
from h5_tools import *
import os
import thread
import threading
from multiprocessing import cpu_count
from sklearn.ensemble import RandomForestClassifier


from progressbar import *               # just a simple progress bar




def redStr(s):
    rs = Fore.RED+s+Fore.RESET + Back.RESET + Style.RESET_ALL
    return rs

def greenStr(s):
    rs = Fore.GREEN+s+Fore.RESET + Back.RESET + Style.RESET_ALL
    return rs

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def getTrainingData(projectFile, name):

    dataPath,dataKey = projectFile["input_data"][name]["data"]
    gtPath,gtKey = projectFile["input_data"][name]["gt"]

    return (dataPath,dataKey ), (gtPath,gtKey)





class IlastikFeatureComputor(object):
    def __init__(self,**kwargs):
        
        #self.inputDataSets = []
        #for inputInfo in inputs:
        #    print(inputInfo)
        #    # get the h5 path
        #    path,key,channels = voxelPredict.getH5PathAndChannels(inputInfo, datasetName)

        #    h5File = h5py.File(path, 'r')
        #    h5DSet = h5File[key]

        #    self.inputDataSets.append((h5File, h5DSet, channels))
        pass
    def close(self):
        #for h5File, h5DSet, channels in self.inputDataSets:
        #    h5File.close()
        pass

    def loadInput(self,block):
        inputArrays = []
        slicing = block.slicing
        for h5File, h5DSet, channels in self.inputDataSets:
            if channels == [0]:
                s = slicing #+ channels
            else:
                s = slicing + channels
            array = h5DSet[tuple(s)].astype('float32')
            if array.ndim == 3:
                array = array[:,:,:,None]
            inputArrays.append(array)

        return  numpy.concatenate(inputArrays,axis=3)

    def margin(self):
        return 5

    def computeFeatures(self, inputArray, blockWithMargin):
        featurerList = []
        for sigma in [1,2]:#,3,4]:
            inputArray = vigra.taggedView(inputArray,'xyzc')
            smoothed = vigra.filters.gaussianSmoothing(inputArray,sigma).squeeze()
            smoothedL = smoothed[blockWithMargin.localInnerBlock.slicing][:,:,:,None]
            featurerList.append(smoothedL)


        for sigma in [1,2]:#,3,4]:
            inputArray = vigra.taggedView(inputArray,'xyzc')
            hessian = vigra.filters.hessianOfGaussianEigenvalues(inputArray,sigma).squeeze()
            hessianL = hessian[blockWithMargin.localInnerBlock.slicing+[slice(0,3)] ][:,:,:,:]
            featurerList.append(hessianL)

        featuresArray = numpy.concatenate(featurerList,axis=3)
        return featuresArray

nameToFeatureComp = dict(ilastik_features=IlastikFeatureComputor)


def rmLastAxisSingleton(shape):
    if shape[-1] == 1:
        return shape[0:-1]
    else:
        return shape


def appendSingleton(slices):
    return slices +[ slice(0,1)]


class VoxelPredict(object):

    def __init__(self, projectFile):
        self.projectFile = projectFile
        #for layer in self.architecture:
        #    layerName = layer['name']
        self.blockShape = tuple(self.projectFile['settings']['block_shape'])

    def getH5Path(self, dataName):
        """
            inputName: something as 'input_data/raw'
                to indicate the channel named 'raw'
                fromm the 'input_data' layer
                and 'initLayer/mito' would 
                take the prediction for the class
                mito from the 'initLayer' predictions.

            dataName: name of the dataset/image.
                Something like 'denk_block_1'
        """
        dataInfo = self.projectFile['input_data'][dataName]['data']
        return dataInfo['path'],dataInfo['key']


    def getGt(self, name):
        return self.projectFile['input_data'][name]['gt']

    def getAndOpenGt(self, name):
        gtInfo = self.getGt(name)
        gtFile = h5py.File(gtInfo['path'],'r')
        gtDataset = gtFile[gtInfo['key']]
        return gtFile, gtDataset

    def hasLabelsNaive(self, gt, labels):
        for l in labels :
            if (gt == l).any():
                return True
        return False

    def mapLabelsNaive(self,targetLabelLists,gt):
        actualLabelArray = numpy.zeros(gt.shape,dtype='uint32')
        for actualLabelIndex, labelList in enumerate(targetLabelLists):
            for l in labelList:
                actualLabelArray[numpy.where(gt==l)] = actualLabelIndex+1
        return actualLabelArray


    def findBlocksWithLabels(self,blocking,gtDataset, flatLabels):

        # check and remember which block has labels 
        # (check for the specific needed labels)
        blocksWithLabels = []

        widgets = ['FindBlocksWithLabels: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
           ' ', ETA()] #see docs for other options

        pbar = ProgressBar(widgets=widgets, maxval=blocking.nBlocks)
        pbar.start()
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            lock = threading.Lock()
            doneBlocks = [0]
            def appendBlocksWithLabels(block,doneBlocks):
                # get slicing of the block
                #lock.acquire(True) 
                #print(block)
                #lock.release()
                s = appendSingleton(block.slicing)
                gt = gtDataset[tuple(s)]
                hasLabels = self.hasLabelsNaive(gt,flatLabels)
                if hasLabels:     
                    lock.acquire(True)               
                    blocksWithLabels.append(block)
                    lock.release()


                lock.acquire(True)
                doneBlocks[0] += 1
                pbar.update(doneBlocks[0])
                lock.release()


            for block in blocking.yieldBlocks():
                executor.submit(appendBlocksWithLabels,block=block,doneBlocks=doneBlocks)
                #appendBlocksWithLabels(block=block)
        pbar.finish()
        return blocksWithLabels



    def getFeatureComps(self, path, key):



        features = self.projectFile['prediction']['features']
        fComps = [None]*len(features)
        dataLoaders = [None]*len(features)
        for fCompIndex, featureGroup in enumerate(features):
            fCls = nameToFeatureComp[featureGroup['name']]
            fComp = fCls(**featureGroup['kwargs'])
            fComps[fCompIndex] = fComp
            input_channels = featureGroup['input_channels']
            dataLoaders[fCompIndex] = H5Loader(path=path,key=key,channels=input_channels)

        return fComps, dataLoaders

    def doTraining(self):

        # training data names for  
        # initial layer
        layer = self.projectFile['prediction']


        labels = []
        features = []

        # get targets for this layer
        targetNames = [ t[0] for t in layer['targets']]
        targetLabelLists = [ t[1] for t in layer['targets']]


        flatLabels = numpy.concatenate(targetLabelLists)
        print(flatLabels)

        # iterate over images
        trainingData = layer['training_data']
        for dsName in trainingData:

            print("dataset",dsName)
            # open the gt dataset
            gtFile,gtDataset = self.getAndOpenGt(dsName)
            print(gtDataset.shape)
            blocking = Blocking(shape=rmLastAxisSingleton(gtDataset.shape), 
                                blockShape=self.blockShape)

            # prepare feature comp. for this image
            path,key = self.getH5Path(dsName)
            fComps,dataLoaders = self.getFeatureComps(path, key)


            # check and remember which block has labels 
            # (check for the specific needed labels)
            blocksWithLabels = self.findBlocksWithLabels(blocking, gtDataset, flatLabels)

            print("#blocksWithLabels",len(blocksWithLabels))



            widgets = ['ComputeFeatures: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
            ' ', ETA()] #see docs for other options

            pbar = ProgressBar(widgets=widgets, maxval=blocking.nBlocks)
            pbar.start()

            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                lock = threading.Lock()
                doneBlocks=[0]
                def fThread(block_,lock_,labels_,doneBlocks):

                    # get slicing of the block
                    s = appendSingleton(block_.slicing)
                    gt = gtDataset[tuple(s)].squeeze()

                    actualGt = self.mapLabelsNaive(targetLabelLists,gt)
                    gtVoxels = numpy.where(actualGt!=0)

                    

                    gtVoxelsLabels = actualGt[gtVoxels]
                    #print(gtVoxelsLabels)

                    # compute the features
                    blockFeatures = []
                    for fCompIndex, featureGroup in enumerate(layer['features']):
                        fName = featureGroup['name']
                        fComp = fComps[fCompIndex]
                        dataLoader = dataLoaders[fCompIndex]
                        neededMargin = fComp.margin()
                        blockWithMargin = block_.blockWithMargin(neededMargin)
                        
                        dataArray = dataLoader.load(blockWithMargin.outerBlock.slicing)
                        featureArray = fComp.computeFeatures(dataArray,blockWithMargin)
                        blockFeatures.append(featureArray)


                    blockFeatureArray = numpy.concatenate(blockFeatures, axis=3)
                    gtVoxelFeatures = blockFeatureArray[gtVoxels[0],gtVoxels[1],gtVoxels[2],:]

                    lock_.acquire(True)
                    labels_.append(gtVoxelsLabels)
                    features.append(gtVoxelFeatures)
                    doneBlocks[0] += 1 
                    pbar.update(doneBlocks[0])

                    lock_.release()

                for block in blocksWithLabels:
                    executor.submit(fThread,block_=block,lock_=lock,labels_=labels,doneBlocks=doneBlocks)
                    #fThread(block_=block,lock_=lock,labels_=labels,doneBlocks=doneBlocks)

            pbar.finish()

            # close all fcomps
            for dataLoader in dataLoaders:
                dataLoader.close()

            # close the gt dataset
            gtFile.close()

        labelsArray =  numpy.concatenate(labels)
        labelsArray = numpy.require(labelsArray, dtype='uint32')[:,None]
        featuresArray = numpy.concatenate(features,axis=0)



        if False:
            print(labelsArray.shape, featuresArray.shape)

            RF = RandomForestClassifier
            rf = RF(n_estimators=20, verbose=0,
                    n_jobs=cpu_count(),oob_score=True)
            print("start fitting")
            rf.fit(featuresArray,labelsArray-1)
            print("end fitting")
            print("OOB",rf.oob_score_)

            self.saveRf(layer, rf)
        if True:
            print("learn rf")
            rf = vigra.learning.RandomForest(treeCount=20)
            oob = rf.learnRF(featuresArray, labelsArray)
            print("OOB",oob)
            self.saveRf(layer, rf)
    def saveRf(self,layer, rf):


        outputFolder = self.projectFile['output']['output_folder']
        ensure_dir(outputFolder)

        rf.writeHDF5(outputFolder+"vigra_rf.h5")
        #pickledRF = pickle.dumps(rf)
        #open(outputFolder+"rf3.dump",'w').write(pickledRF) 

        
    def loadRf(self):

        outputFolder = self.projectFile['output']['output_folder']
        #pickledRFStr = open(outputFolder+"rf3.dump",'r').read() 
        #rf = pickle.loads(pickledRFStr)
        rf = vigra.learning.RandomForest(outputFolder+"vigra_rf.h5")
        return rf


    def predict(self, dataPath, dataKey, outPath):
        

        layer = self.projectFile['prediction']
        blockShape = self.blockShape



        # get number of targets
        nTargets = len(layer['targets'])

        # inspect the shape and get blocking
        f = h5py.File(dataPath, 'r')
        spatialShape = f[dataKey].shape[0:3]
        f.close()


        # create outfile
        outfileShape = tuple(spatialShape) + (nTargets,)
        h5OutFile = h5py.File(outPath,'w')
        chucks = blockShape + (nTargets,)
        outDset = h5OutFile.create_dataset("data", outfileShape, chunks=chucks)


        #blocking
        blocking = Blocking(shape=spatialShape,blockShape=self.blockShape)

        # feature comps
        fComps,dataLoaders = self.getFeatureComps(dataPath, dataKey)

        # load classifier
        rf = self.loadRf()


        widgets = ['Predict: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
           ' ', ETA()] #see docs for other options

        pbar = ProgressBar(widgets=widgets, maxval=blocking.nBlocks)
        pbar.start()

        

        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:

            lock = threading.Lock()
            doneBlocks = [0]
            

            def threadFunc(block_,doneBlocks_):
                global doneBlocks
                totalFeatures = []

                for fCompIndex, featureGroup in enumerate(layer['features']):

                    fComp = fComps[fCompIndex]
                    dataLoader = dataLoaders[fCompIndex]
                    neededMargin = fComp.margin()
                    blockWithMargin = block_.blockWithMargin(neededMargin)

                    dataArray = dataLoader.load(blockWithMargin.outerBlock.slicing)
                    featureArray = fComp.computeFeatures(dataArray,blockWithMargin)
                    totalFeatures.append(featureArray)

                allFeatures = numpy.concatenate(totalFeatures,axis=3)
                nFeat = allFeatures.shape[3]
                allFeaturesFlat = allFeatures.reshape([-1,nFeat])

                probs = rf.predictProbabilities(allFeaturesFlat.view(numpy.ndarray))
                probs = probs.reshape(allFeatures.shape[0:3]+(nTargets,))

                #print("probs",probs.shape)
                slicing = block_.slicing + [slice(0,nTargets)]
                outDset[tuple(slicing)] = probs[:,:,:,:]


                lock.acquire(True)
                doneBlocks_[0] += 1
                pbar.update(doneBlocks_[0])
                lock.release()

            for block in blocking.yieldBlocks(): 
                executor.submit(threadFunc, block_=block,doneBlocks_=doneBlocks)
                #threadFunc(block_=block,doneBlocks_=doneBlocks)

        pbar.finish()

        for dataLoader in dataLoaders:
                dataLoader.close()