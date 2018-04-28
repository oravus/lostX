
# coding: utf-8

# In[ ]:

from __future__ import print_function
import numpy as np
from subprocess import check_output
import sys, argparse, os

import h5py
from skimage.transform import resize
from sklearn.preprocessing import normalize

import timeit


# In[ ]:

def convert_denseDesc_txt2npz(loadPath, outPath, numFeatureMaps = 2048):
    loadPath += "denseDesc/"
    nameFormat = '{0:07d}'

    numImages = len(check_output(["ls",loadPath]).split("\n"))-1
    print("Num Files - ",numImages)

    for i in range(numImages):
        descData = np.loadtxt(loadPath+str(i)+'.txt',delimiter=',')
        descDataReshape = np.array(np.split(descData,numFeatureMaps,1))
        np.savez(outPath+nameFormat.format(i),descDataReshape)
        if i%10==0 and i!=0:
            print("Files Read - ",i)
    print("\nProcessed ", numImages, " files.")
        
def process_semScores(scoresData,numClasses,shape):
    shape = np.insert(shape,2,numClasses)

    scores_rsz = resize(scoresData.transpose([2,1,0]),shape,order=0,mode='constant',preserve_range=True)
    
    denseLabels = np.argmax(scores_rsz,axis=2)
    
    scores_flat = np.reshape(scores_rsz.transpose([2,0,1]),[numClasses,-1])
    scores_prob = normalize(scores_flat,'l1',axis=0)
    
    return scores_prob, denseLabels
        
def convert_semScores_mat2npz(scoreLoadPath, outPath):
    fMapReso = np.loadtxt(scoreLoadPath+"denseDescShapes.txt",delimiter=',')
    
    scoreLoadPath += "predict_result_full/" 
    nameFormat = '{0:07d}'
    
    fileNames = check_output(["ls",scoreLoadPath]).split("\n")
    
    semLabelProb_s, denseSemLabel_s = [],[]
    for i in range(len(fileNames)-1):
        
        fobj = h5py.File(scoreLoadPath+fileNames[i],'r')
        semScoreTensor = np.array(fobj['data_obj']['score_map'])
        
        numSemClasses = semScoreTensor.shape[0]
        
        semLabelProb, denseSemLabel = process_semScores(semScoreTensor,numSemClasses,fMapReso[i,:2])
        
        semLabelProb_s.append(semLabelProb)
        denseSemLabel_s.append(denseSemLabel)
        if i%100==0 and i!=0:
            print("Files Read - ",i)
          
    np.savez(outPath+"semLabelProbs",semLabelProb_s)
    np.savez(outPath+"denseSemLabels",denseSemLabel_s)


# In[ ]:

def argParser():
    parser = argparse.ArgumentParser(description="Reformats the dense conv descriptors and semantic scores tensor     loaded from RefineNet's output. Stores the output in a subfolder /vpr/")

    parser.add_argument("--loadPath", "-p",help="Path where refineNet's output is stored",type=str)

    if len(sys.argv) < 2:
        print(parser.format_help())
        sys.exit(1)    

    args=parser.parse_args()

    return args

def main():
    
    args = argParser()
    
    loadPath = args.loadPath
    outPath = loadPath + "/vpr/"

    print("\nReformatting Dense Descriptor Data ...\nThis may take a long while ...")
    denseDescPath = outPath + "denseDescs/"
    if not os.path.exists(denseDescPath):
        os.makedirs(denseDescPath)
    convert_denseDesc_txt2npz(loadPath,denseDescPath)


    print("\nReformatting Semantic Scores Data ...")
    convert_semScores_mat2npz(loadPath,outPath)

if __name__== "__main__":
    main()    

