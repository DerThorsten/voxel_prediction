from __future__ import print_function
import h5py
import vigra
import numpy
import skneuro
import skneuro.learning as skl
from concurrent.futures import ThreadPoolExecutor
import threading
from multiprocessing import cpu_count
import operator

# others from this package
from block_yielder import *
from data_augmentor import *
from tools import *
from h5_tools import *
from pertubator import *





nameToFeatureComp = dict(ilastik_features=skl.IlastikFeatureOperator,
                         slic_features=skl.SlicFeatureOp)



class VoxelPredict(object):

    def __init__(self, projectFile):
        self.projectFile = projectFile
        self.nChannels  = projectFile['input_data']['nChannels']
        self.blockShape = tuple(self.projectFile['settings']['block_shape'])

    def getH5Path(self, dataName):
        dataInfo = self.projectFile['input_data'][dataName]['data']
        return dataInfo['path'],dataInfo['key']

    def getAndOpenGt(self, name):
        gtInfo = self.projectFile['input_data'][name]['gt']
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
            input_channels = featureGroup['input_channels']
            fComp.inputChannels = input_channels
            #fComp = fCls()
            fComps[fCompIndex] = fComp
           
            dataLoaders[fCompIndex] = H5Loader(path=path,key=key,channels=input_channels)

        return fComps, dataLoaders


    def maxMargin(self, fComps):
        """
            returns the maximal margin of
            all the feature computors
        """
        margin = (0,0,0)
        for f in fComps:
            m = f.margin()
            margin = map(max, zip(m, margin))
        return margin


    def createDataAugmentor(self):
        kwargs = self.projectFile['prediction']['augmentation']
        self.dataAugmentor = DataAugmentor(nChannels=self.nChannels,**kwargs)

    def doTraining(self):

        self.createDataAugmentor()

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

            maxM = self.maxMargin(fComps)
            maxM = self.dataAugmentor.margin(maxM)
            print("the total margin",maxM)

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

                    #gt = gtDataset[tuple(s)]#.squeeze()
                    #if gt.ndim == 4:
                    #    gt = gt.reshape(gt.shape[0:3])

                    #gt = skneuro.loadFromDataset(gtDataset, block_.slicing)

                    #print("get the total data array")
                    blockWithTotalMargin = block_.blockWithMargin(maxM)
                    dataWithAllChannels_ = dataLoaders[0].loadAllChannels(blockWithTotalMargin.outerBlock.slicing)
                    s = appendSingleton(blockWithTotalMargin.outerBlock.slicing)

                    gtWithMargin = gtDataset[tuple(s)]

                    maskedGtWithMargin = skl.maskData(gtWithMargin.squeeze(),
                                    blockWithTotalMargin.localInnerBlock.begin,
                                    blockWithTotalMargin.localInnerBlock.end)

                    assert maskedGtWithMargin.shape[0:3] == dataWithAllChannels_.shape[0:3]

                    for i in range(self.dataAugmentor.nAugmentations() +1):
                        
                        if i==0:
                            dataWithAllChannels = dataWithAllChannels_
                            labels = maskedGtWithMargin
                        else:
                            dataWithAllChannels,labels = self.dataAugmentor(dataWithAllChannels_, 
                                                                            labels=maskedGtWithMargin)



                        #grayData = [(dataWithAllChannels, "raw")]
                        #segData  = [(labels,"ll")]
                        #skneuro.addHocViewer(grayData, segData)
                            


                        # WHERE is gt
                        nLabelsInBlock,roiBeginLocal, roiEndLocal = skl.countLabelsAndFindRoi(labels.squeeze(), flatLabels)
                        if nLabelsInBlock == 0 :
                            #print("ZERO")
                            continue
                        gtVoxelsLabels, whereGt =  skl.getLabelsAndLocation(labels.squeeze(),self.rLabels, nLabelsInBlock)
                        b = numpy.array(roiBeginLocal,dtype='uint32')
                        whereGt -= b[None,:]
                        #print("where gt",whereGt.shape,nLabelsInBlock)

                        #print(gtVoxelsLabels)
                        gtVoxels = (whereGt[:,0],whereGt[:,1],whereGt[:,2])


                        # compute the features
                        blockFeatures = []
                        for fCompIndex, featureGroup in enumerate(layer['features']):
                            fName = featureGroup['name']
                            fComp = fComps[fCompIndex]
                            dataLoader = dataLoaders[fCompIndex]
                            blockWithMargin = block_.blockWithMargin(fComp.margin())
                            #dataArray = dataLoader.load(blockWithMargin.outerBlock.slicing)   
                            dataArray = dataWithAllChannels[:,:,:,fComp.inputChannels[0]].copy()
                            dataArray = vigra.taggedView(dataArray,'xyz')


                            # assert numpy.isinf(numpy.sum(dataArray)) == False
                            #assert numpy.isnan(numpy.sum(dataArray)) == False

                            if dataArray.ndim == 4 and dataArray.shape[3] == 1:
                                dataArray = dataArray.reshape(dataArray.shape[0:3])

                            newFeatures = fComp.trainFeatures(
                                array=dataArray,
                                roiBegin=roiBeginLocal,
                                roiEnd=roiEndLocal,
                                whereGt=whereGt
                            )
                            #print("...DONE")
                            newFeatures = numpy.nan_to_num(newFeatures)
                            blockFeatures.append(newFeatures)

                        blockFeatureArray = numpy.concatenate(blockFeatures, axis=0)
                        #print("totalFShape",blockFeatureArray.shape)
                        lock_.acquire(True)
                        features.append(blockFeatureArray)
                        labels_.append(gtVoxelsLabels)
                        lock_.release()

                    lock_.acquire(True)
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
        featureArray = numpy.concatenate(features,axis=1).T

        self.learnRf(X=featureArray, Y=labelsArray)




    def prepareTrainingSet(self,featureArray, labelsArray,varMult, n):
        print("Raw Features ",featureArray.shape)
        #print("labelsArray",labelsArray.shape)
        outputFolder = self.projectFile['output']['output_folder']
        ensure_dir(outputFolder)
        vigra.impex.writeHDF5(featureArray,outputFolder+"features_X.h5","data")
        vigra.impex.writeHDF5(featureArray,outputFolder+"labels.h5","data")

        return augmentGaussianPertubation(featureArray,labelsArray,
                                          varMult=varMult,n=n)
        
    

    def learnRf(self,X,Y):
        
        print("learn rf (#examples %d #features %d)"%X.shape)


        cOpt = self.projectFile['prediction']['classifier']

        varMult = cOpt['varMult']
        nPertubations = cOpt['nPertubations']
        treeCount = cOpt['treeCount']
        mTry = cOpt['mTry']
        if mTry == 'sqrt':
            mTry = int(numpy.sqrt(float(X.shape[1]))+0.5)
        minSplitNodeSize = cOpt['minSplitNodeSize']
        sampleClassesIndividually = cOpt['sampleClassesIndividually']
        minSplitNodeSize = cOpt['minSplitNodeSize']

        if nPertubations > 0 :
            X, Y = self.prepareTrainingSet(X,Y, varMult=varMult, n=nPertubations)

        Rf = vigra.learning.RandomForest
        rf = Rf(treeCount=treeCount, mtry=mTry,
                min_split_node_size=minSplitNodeSize,
                sample_classes_individually=sampleClassesIndividually)

        oob = rf.learnRF(X, Y)
        print("OOB",oob)

        outputFolder = self.projectFile['output']['output_folder']
        ensure_dir(outputFolder)
        rf.writeHDF5(outputFolder+"vigra_rf.h5")

        
    def loadRf(self):
        outputFolder = self.projectFile['output']['output_folder']
        rf = vigra.learning.RandomForest(outputFolder+"vigra_rf.h5")
        return rf

    def predictROI(self, dataPath, dataKey, roiBegin, roiEnd , outPath):
        outputFolder = self.projectFile['output']['output_folder']
        blockShape = self.blockShape
        fAll = h5py.File(dataPath,'r')
        dAll = fAll[dataKey]

        print ("data shape",dAll.shape)
        print(type(roiBegin),type(roiEnd),type(roiBegin[0]))

        inputDataShape = dAll.shape
        spatialShape = inputDataShape[0:3]
        hasAxis = len(inputDataShape) == 4
        spatialRoiShape = map(operator.sub,roiEnd,roiBegin)

        subShape = list(spatialRoiShape)
        chunks = [30,30,30]
        chunks = map(min, chunks, subShape)
        if(hasAxis):
            subShape += [inputDataShape[3]]
            chunks += [1]

        fSub = h5py.File(outputFolder+"tmp.h5",'w')   
        dSub = fSub.create_dataset("data", subShape, chunks=tuple(chunks))


        if len(dAll.shape) == 3:
            dSub[:,:,:] = dAll[roiBegin[0]:roiEnd[0],
                               roiBegin[1]:roiEnd[1],
                               roiBegin[2]:roiEnd[2]]
        elif len(dAll.shape) == 4:
            dSub[:,:,:,:] = dAll[roiBegin[0]:roiEnd[0],
                                 roiBegin[1]:roiEnd[1],
                                 roiBegin[2]:roiEnd[2],:]
        else:
            raise RuntimeError("wrong dimension")

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

               

                allFeaturesFlat = allFeatures.reshape([nFeat,-1]).T

                #print("allFeaturesFlat",allFeaturesFlat.shape)

                probs = rf.predictProbabilities(allFeaturesFlat.view(numpy.ndarray))
                #print("probs",probs.shape)

                probs = probs.reshape(allFeatures.shape[1:4]+(nTargets,))

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
