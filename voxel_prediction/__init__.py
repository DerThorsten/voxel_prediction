from __future__ import print_function
from colorama import Fore, Back, Style
import h5py
import vigra
import skneuro
import skneuro.utilities as skut
import skneuro.learning as skl
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
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


def reraise(future):
    ex = future.exception()
    if ex :
        raise ex

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


def getPbar(maxval,name=""):
    cname = redStr(name)
    widgets = [' %s: '%cname, Percentage(), ' ', Bar(marker='*',left='[',right=']'),
           ' ',ETA()] #see docs for other options

    pbar = ProgressBar(widgets=widgets, maxval=maxval)
    #pbar.start()
    return pbar






nameToFeatureComp = dict(ilastik_features=skl.IlastikFeatureOperator,
                         slic_features=skl.SlicFeatureOp)


def rmLastAxisSingleton(shape):
    if shape[-1] == 1:
        return shape[0:-1]
    else:
        return shape


def appendSingleton(slices):
    return slices +[ slice(0,1)]


def saveChunks(sShape, sChunks, ):
    pass


class VoxelPredict(object):

    def __init__(self, projectFile):
        self.projectFile = projectFile
        #for layer in self.architecture:
        #    layerName = layer['name']
        self.blockShape = tuple(self.projectFile['settings']['block_shape'])

    def getH5Path(self, dataName):
        dataInfo = self.projectFile['input_data'][dataName]['data']
        return dataInfo['path'],dataInfo['key']


    def getGt(self, name):
        return self.projectFile['input_data'][name]['gt']

    def getAndOpenGt(self, name):
        gtInfo = self.getGt(name)
        gtFile = h5py.File(gtInfo['path'],'r')
        gtDataset = gtFile[gtInfo['key']]
        return gtFile, gtDataset


    def blockRoiToGlobal(self,roiBeginLocal, roiEndLocal, block):
        roiBegin = [None]*3
        roiEnd = [None]*3
        for d in range(3):
            roiBegin[d] = roiBeginLocal[d] + block.begin[d]
            roiEnd[d] = roiEndLocal[d] + block.begin[d]
        return roiBegin, roiEnd

    def findBlocksWithLabels(self,blocking,gtDataset, flatLabels):

        blocksWithLabels = []
        pbar = getPbar(blocking.nBlocks,name="Find Block With Labels").start()

        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            lock = threading.Lock()
            doneBlocks = [0]
            def appendBlocksWithLabels(block,doneBlocks):

                s = appendSingleton(block.slicing)
                gt = gtDataset[tuple(s)]
               
                nLabelsInBlock,roiBeginLocal, roiEndLocal = skl.countLabelsAndFindRoi(gt.squeeze(), flatLabels)
                
                if nLabelsInBlock > 0 :
                    roiBegin, roiEnd = self.blockRoiToGlobal(roiBeginLocal,roiEndLocal,block)
                
                lock.acquire(True)
                if nLabelsInBlock > 0 :
                    newBlock = Block(roiBegin, roiEnd,block.blocking)       
                    blocksWithLabels.append((newBlock, nLabelsInBlock))

                doneBlocks[0] += 1
                pbar.update(doneBlocks[0])
                lock.release()


            for block in blocking.yieldBlocks():
                futureRes = executor.submit(appendBlocksWithLabels,block=block,doneBlocks=doneBlocks)
                futureRes.add_done_callback(reraise)
        pbar.finish()
        return blocksWithLabels



    def getFeatureComps(self, path, key):

        features = self.projectFile['prediction']['features']
        fComps = [None]*len(features)
        dataLoaders = [None]*len(features)
        for fCompIndex, featureGroup in enumerate(features):
            fCls = nameToFeatureComp[featureGroup['name']]
            extraKwargs = featureGroup['kwargs']
            fComp = fCls(**featureGroup['kwargs'])
            #fComp = fCls()
            fComps[fCompIndex] = fComp
            input_channels = featureGroup['input_channels']
            dataLoaders[fCompIndex] = H5Loader(path=path,key=key,channels=input_channels)

        return fComps, dataLoaders



    def doTraining(self):

        layer = self.projectFile['prediction']

        ######################################################
        # THIS IS THE ACTUAL TRAINING SET WHICH WE WILL FILL
        ######################################################
        labels = []
        features = []

        ######################################################
        #   Labels (map from input labels to training labels)
        ######################################################
        # get targets 
        targetNames = [ t[0] for t in layer['targets']]
        targetLabelLists = [ t[1] for t in layer['targets']]
        # all needed labels to predict
        flatLabels = numpy.concatenate(targetLabelLists).astype('uint32')
        # compute remapping from old labels (starting at 1)
        # to new labels (starting at 0)
        self.rLabels = numpy.zeros(flatLabels.max()+1,dtype='uint32')
        for newLabelIndex, oldLabels in enumerate(targetLabelLists):
            for ol in oldLabels:
                self.rLabels[ol] = newLabelIndex + 1

        
        ######################################################
        # Iterate over images
        ######################################################
        trainingData = layer['training_data']
        for dsName in trainingData:

            ######################################################
            # open the gt dataset and get the blocking for the
            # dataset
            ######################################################
            gtFile,gtDataset = self.getAndOpenGt(dsName)
            print(redStr("dataset"),greenStr(dsName),gtDataset.shape)
            blocking = Blocking(shape=rmLastAxisSingleton(gtDataset.shape), 
                                blockShape=self.blockShape)

            ######################################################
            # prepare feature comp. for this image
            ######################################################
            path,key = self.getH5Path(dsName)
            fComps,dataLoaders = self.getFeatureComps(path, key)

            ######################################################
            # check and remember which block has labels 
            ######################################################
            blocksWithLabels = self.findBlocksWithLabels(blocking, gtDataset, flatLabels)            
            print(redStr("#blocksWithLabels"),len(blocksWithLabels))


            ######################################################
            # parallel blockwise feature computation
            ######################################################
            pbar = getPbar(len(blocksWithLabels),"ComputeFeatures").start()
            with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                lock = threading.Lock()
                doneBlocks=[0]
                def fThread(block_,nLabelsInBlock,lock_,labels_,doneBlocks):

                    # get slicing of the block
                    s = appendSingleton(block_.slicing)

                    gt = gtDataset[tuple(s)]#.squeeze()
                    if gt.ndim == 4:
                        gt = gt.reshape(gt.shape[0:3])

                    #print("GT uniQUE",numpy.unique(gt),gt.min(),gt.max())
                    gtVoxelsLabels, whereGt =  skl.getLabelsAndLocation(gt,self.rLabels, nLabelsInBlock)
                    #print(gtVoxelsLabels)
                    gtVoxels = (whereGt[:,0],whereGt[:,1],whereGt[:,2])

                    # compute the features
                    blockFeatures = []
                    for fCompIndex, featureGroup in enumerate(layer['features']):
                        fName = featureGroup['name']
                        fComp = fComps[fCompIndex]
                        dataLoader = dataLoaders[fCompIndex]
                        blockWithMargin = block_.blockWithMargin(fComp.margin())
                        dataArray = dataLoader.load(blockWithMargin.outerBlock.slicing)                      
                        if dataArray.ndim == 4 and dataArray.shape[3] == 1:
                            dataArray = dataArray.reshape(dataArray.shape[0:3])

                        # heavy load (C++)
                        newFeatures = fComp.trainFeatures(
                            array=dataArray,
                            roiBegin=blockWithMargin.localInnerBlock.begin,
                            roiEnd=blockWithMargin.localInnerBlock.end,
                            whereGt=whereGt
                        )

                        blockFeatures.append(newFeatures)

                    blockFeatureArray = numpy.concatenate(blockFeatures, axis=0)

                    lock_.acquire(True)
                    labels_.append(gtVoxelsLabels)
                    features.append(blockFeatureArray)
                    doneBlocks[0] = doneBlocks[0] + 1
                    pbar.update(doneBlocks[0])
                    lock_.release()

                for block,nLabelsInBlock in blocksWithLabels:
                    futureRes = executor.submit(fThread,block_=block,nLabelsInBlock=nLabelsInBlock,lock_=lock,labels_=labels,doneBlocks=doneBlocks)
                    futureRes.add_done_callback(reraise)
                    #fThread(block_=block,nLabelsInBlock=nLabelsInBlock,lock_=lock,labels_=labels,doneBlocks=doneBlocks)

            pbar.finish()

            # close all fcomps
            for dataLoader in dataLoaders:
                dataLoader.close()

            # close the gt dataset
            gtFile.close()

        labelsArray =  numpy.concatenate(labels)
        labelsArray = numpy.require(labelsArray, dtype='uint32')[:,None]
        featuresArray = numpy.concatenate(features,axis=1).T

        print("training",featuresArray.shape)
        print("labelsArray",labelsArray.shape)
        print("labelsArray Min max",labelsArray.min(),labelsArray.max())
        if False:
            print(labelsArray.shape, featuresArray.shape)

            RF = RandomForestClassifier
            rf = RF(n_estimators=256, verbose=0,
                    n_jobs=cpu_count(),oob_score=True)
            print("start fitting")
            rf.fit(featuresArray,labelsArray-1)
            print("end fitting")
            print("OOB",rf.oob_score_)

            self.saveRf(layer, rf)
        if True:
            print("learn rf")
            rf = vigra.learning.RandomForest(treeCount=256)
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



    def predictROI(self, dataPath, dataKey, roiBegin, roiEnd , outPath):
        outputFolder = self.projectFile['output']['output_folder']
        blockShape = self.blockShape
        fAll = h5py.File(dataPath,'r')
        dAll = fAll[dataKey]

        print ("data shape",dAll.shape)
        print(type(roiBegin),type(roiEnd),type(roiBegin[0]))

        if len(dAll.shape) == 3:

            fSub = h5py.File(outputFolder+"tmp.h5",'w')
            outShape = [roiEnd[d]-roiBegin[d] for d in range(3)]
            print("outShape ",outShape,"blockshape",blockShape)
            print(tuple([min(outShape[d]-1,blockShape[d]) for d in range(3)]))


            chunks = tuple([min(outShape[d],blockShape[d]) for d in range(3)])
            dSub = fSub.create_dataset("data", outShape, chunks=(30,30,30))

            dSub[:,:,:] = dAll[roiBegin[0]:roiEnd[0],
                               roiBegin[1]:roiEnd[1],
                               roiBegin[2]:roiEnd[2]]
            
        fSub.close()                   
        fAll.close()

        self.predict(dataPath=outputFolder+"tmp.h5",
                     dataKey="data", outPath=outPath)


    def predict(self, dataPath, dataKey, outPath, downsample = 1):
        

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
        chunks = list(blockShape + (nTargets,))
        for d in  range(3):
            chunks[d] = min(spatialShape[d],chunks[d])
        outDset = h5OutFile.create_dataset("data", outfileShape, chunks=tuple(chunks))


        #blocking
        blocking = Blocking(shape=spatialShape,blockShape=self.blockShape)

        # feature comps
        fComps,dataLoaders = self.getFeatureComps(dataPath, dataKey)

        # load classifier
        rf = self.loadRf()


        pbar = getPbar(blocking.nBlocks,'Predict').start()
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:

            lock = threading.Lock()
            doneBlocks = [0]
            

            def threadFunc(block_,doneBlocks_):
                totalFeatures = []
                for fCompIndex, featureGroup in enumerate(layer['features']):

                    fComp = fComps[fCompIndex]
                    dataLoader = dataLoaders[fCompIndex]
                    neededMargin = fComp.margin()
                    blockWithMargin = block_.blockWithMargin(neededMargin)

                    dataArray = dataLoader.load(blockWithMargin.outerBlock.slicing)

                    if dataArray.ndim == 4 and dataArray.shape[3] ==1:
                        dataArray = dataArray.reshape(dataArray.shape[0:3])

                    featureArray = fComp.testFeatures(
                        array=dataArray,
                        roiBegin=blockWithMargin.localInnerBlock.begin,
                        roiEnd=blockWithMargin.localInnerBlock.end
                    )
                    totalFeatures.append(featureArray)

                allFeatures = numpy.concatenate(totalFeatures,axis=0)
                nFeat = allFeatures.shape[0]
                va = vigra.taggedView(allFeatures, "cxyz")
                sshape = allFeatures.shape[1:4]
                subshape = [None]*3
                #print("nFeat",nFeat,"initshape",allFeatures.shape)

                canDoDs = True
                if downsample>1:
                    for d in range(3):
                        if sshape[d]<6:
                            canDoDs = False
                            break


                if downsample>1 and canDoDs:
                    for d in range(3):
                        if sshape[d]<15:
                            subshape[d] = sshape[d]
                        else:
                            subshape[d] = int((float(sshape[d])/downsample)+0.5)
                    #print("subshape",subshape)

                    allFeaturesV = numpy.rollaxis(allFeatures,0,4)
                    #print("allFeaturesV",allFeaturesV.shape)
                    va = vigra.taggedView(allFeaturesV, "xyzc")
                    allFeatures = vigra.sampling.resize(va, subshape,order=0)
                    allFeatures = numpy.rollaxis(allFeatures,3)

                    #print("after reshape shape",allFeatures.shape)


                

                allFeaturesFlat = allFeatures.reshape([nFeat,-1]).T

                #print("allFeaturesFlat",allFeaturesFlat.shape)

                probs = rf.predictProbabilities(allFeaturesFlat.view(numpy.ndarray))
                #print("probs",probs.shape)

                probs = probs.reshape(allFeatures.shape[1:4]+(nTargets,))
                if downsample>1 and canDoDs:
                    probs = vigra.taggedView(probs,'xyzc')
                    probs = vigra.sampling.resize(probs,sshape,order=3)
    


                #print("probs",probs.shape)
                slicing = block_.slicing + [slice(0,nTargets)]
                outDset[tuple(slicing)] = probs[:,:,:,:]


                lock.acquire(True)
                doneBlocks_[0] += 1
                pbar.update(doneBlocks_[0])
                lock.release()

            for block in blocking.yieldBlocks(): 
                futureRes = executor.submit(threadFunc, block_=block,doneBlocks_=doneBlocks)
                futureRes.add_done_callback(reraise)
                #threadFunc(block_=block,doneBlocks_=doneBlocks)

        pbar.finish()

        for dataLoader in dataLoaders:
                dataLoader.close()
