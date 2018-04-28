
# coding: utf-8

# In[ ]:

from __future__ import print_function
import numpy as np
import argparse, sys, os
from subprocess import check_output

from scipy.spatial.distance import cdist,cosine
from sklearn.preprocessing import normalize


# In[ ]:

def unravel_flat_kp(ft,shape,flip=False):
    ftR = ft.astype(int)
    kps = np.array(np.unravel_index(ftR,shape)).transpose()
    if flip:
        kps[:,1] = shape[1]-1-kps[:,1]
    return kps

def filter_kp_semantically(kpInds1,kpInds2,semLabels1,semLabels2):
    matchInds = np.argwhere(semLabels1[kpInds1] == semLabels2[kpInds2])[:,0]
    kpInds1_, kpInds2_ = kpInds1[matchInds], kpInds2[matchInds]
    return kpInds1_, kpInds2_,matchInds
    
def readDenseDesc(denseDesPath,idx,nameFormat='{0:07d}'):
    desc = np.load(denseDesPath+nameFormat.format(idx)+".npz")['arr_0']        
    desc_ = np.reshape(desc,[desc.shape[0],-1]).transpose([1,0])
    return desc_

def readDenseDesc_loop(denseDesPath,nameFormat='{0:07d}'):
    numImages = len(check_output(["ls",denseDesPath]).split("\n"))-1
    print("\n Loading Files - ",numImages)
    denseDescs = []
    for i in range(numImages):
        desc = readDenseDesc(denseDesPath,i)
        denseDescs.append(desc)
    print("\n Loaded.")
    return np.array(denseDescs)

def get_cosine_corresponding_vectors(d1,d2):
    return 1 - np.sum(d1*d2,axis=1)/(np.linalg.norm(d1,axis=1)*np.linalg.norm(d2,axis=1))

def getDenseDescriptorDistances(kp1,kp2,desc1,desc2):
    dists = get_cosine_corresponding_vectors(desc1[kp1],desc2[kp2])
    return np.array(dists)

def getSpatialCheckMat(semLabels1,semLabels2,res5_ArgMax_1,res5_ArgMax_2,topN=10,baseDiff=None,flipFlag=False,
                       preLoadDenseDescRef=True,denseDesPath1=None,denseDesPath2=None,nameFormat = '{0:07d}'):

    if preLoadDenseDescRef and denseDesPath2 is not None:
        print("\n Loading Dense Descriptor data...\n This may take a while...")
        descsRef = readDenseDesc_loop(denseDesPath1)
        
    labShape1 = semLabels1.shape
    labShape2 = semLabels2.shape
    
    matchesTopN = np.argsort(baseDiff,axis=0)[:topN,:]

    unravelIdxMap_f = unravel_flat_kp(np.arange(labShape1[1] * labShape1[2]),[labShape1[1],labShape1[2]])
    unravelIdxMap_r = unravel_flat_kp(np.arange(labShape2[1] * labShape2[2]),[labShape2[1],labShape2[2]],flipFlag)
    
    matches_db, matchConf_db = [], []
    for j in range(matchesTopN.shape[1]):
        match_q = []
        if denseDesPath1 is not None:
            desc2 = readDenseDesc(denseDesPath2,j)
            
        semLabel2 = semLabels2[j]
        if flipFlag:
            semLabel2 = np.fliplr(semLabels2[j])
        
        inds2Test = matchesTopN[:,j].copy()
        
        for i in inds2Test:
            if denseDesPath1 is not None:
                if preLoadDenseDescRef:
                    desc1 = descsRef[i]
                else:
                    desc1 = readDenseDesc(denseDesPath1,i)
            
#             kpIndsFil_1,kpIndsFil_2 = res5_ArgMax_1[i],res5_ArgMax_2[j]
            kpIndsFil_1,kpIndsFil_2,matchInds = filter_kp_semantically(res5_ArgMax_1[i],res5_ArgMax_2[j],
                                              semLabels1[i].flatten(),semLabel2.flatten()) 

            kpFil_1 = unravelIdxMap_f[kpIndsFil_1][:,1]
            kpFil_2 = unravelIdxMap_r[kpIndsFil_2][:,1]

            if denseDesPath1 is not None:
                denseDescDists = getDenseDescriptorDistances(kpIndsFil_1,kpIndsFil_2,desc1,desc2)
                distVals = normalize(denseDescDists.reshape(1,-1),'l1',axis=1)
                distVals = np.max(distVals) - distVals
                mch_loc = cosine(kpFil_1*distVals,kpFil_2*distVals)/np.linalg.norm(distVals)
            else:
                mch_loc = cosine(kpFil_1,kpFil_2)
            
            mch_loc /= len(kpFil_1)
            
            match_q.append(mch_loc*baseDiff[i,j])

        matches_db.append(inds2Test[np.argmin(match_q)])
        matchConf_db.append(np.min(match_q))   
        if j%10==0 and j!=0:
            print("Query Files Processed - ",j) 
    
    print("\n Processed ", matchesTopN.shape[1], " files.")
    return np.vstack([matches_db,matchConf_db]).transpose()

def normalize_wid(ftIn):
    return normalize(normalize(ftIn),axis=0)

def compute_distanceMatrix(ft1,ft2):
    distMat = cdist(ft1,ft2,"cosine")
    return distMat


# In[ ]:

def load_input_data(dataPath,ref=False):
    dataPath += "/vpr/"
    print("\tReading Pixel-wise semantic labels ...")
    denseSemLabels = np.load(dataPath + "denseSemLabels.npz")['arr_0']
    
    denseDescPath = dataPath + "denseDescs/"

    print("\tReading Whole-Image Descriptors - LoST ...")
    if os.path.exists(dataPath+"lost-DbCC.npz") and ref:
        print("\t\t from lost-DbCC.npz")
        lost_wid = np.load(dataPath+"lost-DbCC.npz")['arr_0']
    else:
        if ref:
            print("\t\t\t\nINFO: lost-DbCC.npz was not found for reference database, using lost.npz.",
            "\nlost-DbCC can be generated by setting '-c' option as 1 when running LoST.py\n")
        print("\t\t from lost.npz")
        lost_wid = np.load(dataPath+"lost.npz")['arr_0']        

    print("\tReading Keypoint Location data ...")
    kpLocInds = np.load(dataPath+"kpLocInds.npz")['arr_0']

    return lost_wid, denseDescPath, denseSemLabels,kpLocInds

def parseArguments():
    parser = argparse.ArgumentParser(description='Loads the whole-image-descriptors (wid),     dense conv descriptors, and semantic labels, then computes a distance matrix,     and then finally performs spatial-layout-verification on top-N candidates to retrieve the final match.')

    parser.add_argument("--data_path_1", "-p1",help="Path where dataset 1 (reference) files are stored",type=str)
    parser.add_argument("--data_path_2", "-p2",help="Path where dataset 2 (query) files are stored",type=str)
    parser.add_argument("--out_path", "-po",help="Path where output will be stored. \nDefault is './bin/'",
                        type=str,default="./bin/")
    parser.add_argument("--topN", "-n",help="Number of top candidates to be considered for spatial match. \nDefault     is set to 10.",type=int,default=10)
    parser.add_argument("--flipFlag", "-f",help="Flag to be set 1 for front-rear matching. \nDefault is set to 0",
                        type=int,default=0)
    parser.add_argument("--preLoadDenseDescRef", "-d",help="Flag to be set 0 to save memory at cost of increased    computation time. \nDefault is set to 1",type=int,default=1)
    
    if len(sys.argv) < 2:
        print(parser.format_help())
        sys.exit(1)  

    args=parser.parse_args()

    if args.data_path_1 is None or args.data_path_2 is None:
        print("Error: Path to two datasets is needed, see usage below:\n")
        print(parser.format_help())
        sys.exit(1)    
        
    return args


# In[ ]:

def main():
    
    args = parseArguments()
    
    print("\nReading first dataset ...")
    wid_1, denseDescPath_1, denseSemLabels_1, kpLocInds_1 = load_input_data(args.data_path_1,ref=True)
    print("\n\nReading second dataset ...")
    wid_2, denseDescPath_2, denseSemLabels_2, kpLocInds_2 = load_input_data(args.data_path_2)

    wid_1, wid_2 = normalize_wid(wid_1), normalize_wid(wid_2)

    print("\nComputing Distance Matrix ...")
    baseDiff = compute_distanceMatrix(wid_1,wid_2)
    print("\nShape of Distance Matrix:",baseDiff.shape)
    np.savetxt(args.out_path+"baseDiffMat.txt",baseDiff,fmt='%f')

    print("\nVerifying spatial layout for top N candidates ...")
    matchResult_db = getSpatialCheckMat(denseSemLabels_1,denseSemLabels_2,
                                        kpLocInds_1,kpLocInds_2,
                                        args.topN,baseDiff,flipFlag=args.flipFlag,
                                        preLoadDenseDescRef=args.preLoadDenseDescRef,
                                        denseDesPath1=denseDescPath_1,denseDesPath2=denseDescPath_2
                                       )
    np.savetxt(args.out_path+"matchInds.txt",matchResult_db,fmt=['%d','%f'])
    print("\nDone.")
    print("\nOutput saved to ", args.out_path+"matchInds.txt ",
          "as a 2-column matrix: column 1 contains matched indices from reference data and",
          " column 2 has the matched distance value. Number of rows are same as query data files.")    

    

if __name__== "__main__":
    main()

