
# coding: utf-8

# In[ ]:

from __future__ import print_function
import numpy as np
from subprocess import check_output
import sys, argparse, os

from skimage.transform import resize
from sklearn.preprocessing import normalize

import timeit


# In[ ]:

def getLocalCC(ft,labels,classInds,numFeatureMaps):
    """
    ft of shape [rows*cols,numFeatureMaps]
    labels of shape [rows*cols,]
    
    Returns array [numClasses,numFeatureMaps]
    """
    
    cc = np.zeros([len(classInds),numFeatureMaps])
    for i in range(len(classInds)):
        inds = np.argwhere(labels==classInds[i])[:,0]
        if inds.size:
            ft_c = ft[inds,...]
            ftMean = np.mean(ft_c,axis=0)
            cc[i,:] = ftMean
    return cc

def getEncoding(cc,inData,membership):

    diffMat = abs(cc[:,None,:] - inData[None,:,:])
    
    lostVec = np.sum(diffMat*membership[:,:,None],axis=1)
    
    return normalize(lostVec)

def getTemporalMeanofClusterCenters(ccData,win=10):
    ccDataMod = ccData.copy()
    for i in range(ccData.shape[0]):
         ll = max(0,i-win/2)
         ul = min(ccData.shape[0],i+win/2)
        
         ccDataMod[i] = np.mean(ccData[ll:ul].reshape([ul-ll,-1]),axis=0).reshape(ccData[0].shape)
    return ccDataMod



# In[ ]:

def compute_Lost_and_Kp(dataPath,cc_is_precomputed=False):
    denseDescPath = dataPath + "denseDescs/"
    consideredClasses = [0,2,8] # road, building, vegetation
    nameFormat = '{0:07d}'

    # Initialize some variables
    lostArr = []
    numImages = len(check_output(["ls",denseDescPath]).split("\n"))-1
    print("Found ",numImages, " files to read.")
    ccAll = []
    kpLocInds = []
    
    # load visual semantic data
    semLabelProbs = np.load(dataPath+"semLabelProbs.npz")['arr_0']
    denseSemLabels = np.load(dataPath+"denseSemLabels.npz")['arr_0']
    numSemanticClasses = denseSemLabels.shape[1]

    # this file comprises cluster centers per image
    ccFileName = dataPath+"lost_clusterCenters.npz"
    
    # if cluster centers are pre-computed
    if cc_is_precomputed:
        ccAll = np.load(ccFileName)['arr_0']
        ccAll = getTemporalMeanofClusterCenters(ccAll,win=15)

    # compute a descriptor per image
    for i in range(numImages):
        resFt = np.load(denseDescPath+nameFormat.format(i)+'.npz')['arr_0']
        
        conv5Reso = resFt.shape[1:]
        numFeatureMapsConv5 = resFt.shape[0]
        resFt = np.reshape(resFt,[numFeatureMapsConv5,-1]).transpose([1,0])

        # semantic labels
        resLabel = denseSemLabels[i].flatten()

        if cc_is_precomputed:
            ccData = ccAll[i]
        else:
            ccData = getLocalCC(resFt,resLabel,consideredClasses,numFeatureMapsConv5)
            ccAll.append(ccData)
            
        lost = getEncoding(ccData,resFt,semLabelProbs[i][consideredClasses,:])
        
        lostArr.append(lost.flatten())
        
        # Extract the keypoints from the conv feature maps      
        kpLocIdxFlat = np.argmax(resFt,axis=0)
        kpLocInds.append(kpLocIdxFlat)
        if i%100==0 and i!=0:
            print("Files Read - ",i)

    outFileID = "lost"
    if cc_is_precomputed:
        outFileID = "lost-DbCC"
    np.savez(dataPath+outFileID,np.array(lostArr))
    np.savez(ccFileName,np.array(ccAll))
    np.savez(dataPath+"kpLocInds",kpLocInds)
    print("\nProcessed ", numImages, " files.")



# In[ ]:

def argParser():
    parser = argparse.ArgumentParser(description="Computes LoST descriptor and extract keypoints (kp) using the     reformatted dense desciptors and semantic scores' tensor.")

    parser.add_argument("--dataPath", "-p",help="Path where refineNet's output is stored",type=str)
    parser.add_argument("--ccPrecomputeFlag", "-c",help="Flag to be set 1 if cluster centers are to computed     using a sequence of images, especially for the reference database. Deafult is set to 1.",type=int, default=1)

    if len(sys.argv) < 2:
        print(parser.format_help())
        sys.exit(1)    

    args=parser.parse_args()
    
    return args


# In[ ]:

def main():
    
    args = argParser()
    dataPath = args.dataPath + "/vpr/"
    ccPrecomputeFlag = args.ccPrecomputeFlag 
    
    print("\nComputing LoST descriptor ...")
    compute_Lost_and_Kp(dataPath,0)
    # TO DO: Remove redundant computation of re-running the compute function, when ccPrecomputeFlag is set to 1
    if ccPrecomputeFlag:
        print("\nRe-running to consider temporal clusters ...")
        compute_Lost_and_Kp(dataPath,1)
    print("\nDone.")
    
if __name__== "__main__":
    main()    

