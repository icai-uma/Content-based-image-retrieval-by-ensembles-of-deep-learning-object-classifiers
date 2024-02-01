# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 06:11:45 2020

@author: Safa
"""


import os
import statistics
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from paths import setPaths

def getQueryPrecAtK(k,p,thresh, queryIdx):
    imagesProTrain = np.load(vecsTrainPath, mmap_mode=None, allow_pickle=True,
                             fix_imports=True, encoding='ASCII')
    imagesProTest = np.load(vecsTestPath, mmap_mode=None, allow_pickle=True,
                            fix_imports=True, encoding='ASCII')

    queryPrec = 0
    Prec = 0
    queryRec = 0
    Rec = 0
    if queryIdx is None:
      testIdx = 0
    else:
      testIdx = queryIdx

    while (testIdx < len(imagesProTest)):
        queryTP = 0
        CurrentTrainIdx = 0
        queryPro = imagesProTest[testIdx]
        queryId,queryClass= getqueryClass(testIdx,'test')
        if k is not None:
          similarImagesIndex = searchTopK(imagesProTrain,queryPro,k,p)
        else:
          similarImagesIndex = searchThresh(imagesProTrain,queryPro,thresh,p)
        for subdir, dirs, files in sorted(os.walk(splittedDatasetDir+'train/')):
            for imageName in sorted(files):
                classNameTrain = subdir.rsplit('/', 1)[1]
                if  (CurrentTrainIdx in similarImagesIndex) and (classNameTrain == queryClass):
                    queryTP +=1

                CurrentTrainIdx += 1

        if (len(similarImagesIndex)!= 0):
          queryPrec = queryTP / len(similarImagesIndex)
          queryRec = queryTP / relImgs
        else:
          queryPrec = 0
          queryRec = 0

        Prec = queryPrec + Prec
        Rec = queryRec + Rec
        if queryIdx is not None:
          break

        testIdx += 1

    if queryIdx is None:
      Prec = Prec/ len(imagesProTest)
      Rec = Rec/ len(imagesProTest)

    return round(Prec,4),round(Rec,4)


def mAP(p):
  imagesProTrain = np.load(vecsTrainPath, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
  imagesProTest = np.load(vecsTestPath, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')

  thresh= None
  queryIdx = 0
  sumAP = 0
  while (queryIdx < len(imagesProTest)):
    rank = 1
    ap = 0
    lastRec = 0
    rec = 0

    while (rank <= relImgs ) :
      [prec,rec] = getQueryPrecAtK(rank,p,thresh,queryIdx)
      ap = (rec-lastRec) * prec + ap
      rank += 1
      lastRec = rec
    sumAP = ap + sumAP
    queryIdx += 1
  meanAP = sumAP/len(imagesProTest)
  return round(meanAP,4)

def pnorm_dist(x,y,p):
    if (p == 1): #Manhattan
        distance = np.linalg.norm(x-y, ord=1)
    elif (p==1.5):
        distance = np.linalg.norm(x-y, ord = 1.5)
    elif (p==2): #euclidean
        distance = np.linalg.norm(x-y, ord=2)
    else: #Tchebychev
        distance = np.linalg.norm(x-y, ord=float('inf'))
    return distance

def searchTopK(imagesProTrain,queryPro,k,p):
    distArray,distArraySorted = ([] for m in range(2))
    j = 0
    while (j< len(imagesProTrain)):
        distArray.append(pnorm_dist(queryPro,imagesProTrain[j],p))
        j+=1
    distArraySorted = sorted(distArray)
    similarImagesIndex = np.argsort(distArray)
    similarImagesIndex = similarImagesIndex[:k]
    return similarImagesIndex

def searchThresh(imagesProTrain,queryPro,thresh,p):
    distArray = []
    j = 0
    while (j< len(imagesProTrain)):
        distArray.append(pnorm_dist(queryPro,imagesProTrain[j],p))
        j+=1
    distArray = np.array(distArray)
    silimarImagesIdx = np.nonzero(distArray < thresh)
    silimarImagesIdx = (silimarImagesIdx[0])
    return silimarImagesIdx


def getqueryClass(testIdx,dataset):
    global className
    CurrentTestIdx = 0
    classId = -2
    if dataset== 'test':
      path = splittedDatasetDir +'test/'
    else:
      path = splittedDatasetDir +'train/'
    for subdir, dirs, files in sorted(os.walk(path)):

            classId += 1
            for imageName in files:
                    if (CurrentTestIdx == testIdx):
                        className = subdir.rsplit('/', 1)[1]
                        return classId,className

                    CurrentTestIdx += 1


def getmAPP(datasetDir, method):
  filemAP = parentDir + datasetDir+'/figures files/'+datasetDir+'_'+ method+ '_mAP_p.txt'
  filemAP = open(filemAP, 'w')
  pValues = [1, 1.5, 2, float('inf')]
  ensDir = 'ensemble'
  mAPPpVals = []
  if datasetDir =='imagenet':
    ensSize = 18
  elif datasetDir == 'caltech20' or datasetDir == 'caltech50':
    ensSize = 20
  setPaths(ensDir=ensDir, ensSize=ensSize, datasetDir=datasetDir, method=method, cnnId = None, kq=None)

  for p in pValues:
    mAPPp = mAP(p)
    filemAP.write('%s:%s \n' % (p, mAPPp))
    mAPPpVals.append(mAPPp)

  filemAP.close()
  mAPPpVals = np.array(mAPPpVals)
  idx = np.argmax(mAPPpVals)
  return pValues[idx]


def getmAPensSize(datasetDir, method,p):

  ensDirs = ['ensemble', 'ensembleBest', 'ensembleWorst']
  setPaths(ensDir=None, ensSize=None, datasetDir=datasetDir, method=None, cnnId=None, kq=None)
  fileBstmAP = parentDir + datasetDir+'/figures files/'+datasetDir+'_'+ method+ '_BestmAP.txt'
  fileBstmAP = open(fileBstmAP, 'w')


  if datasetDir == 'caltech20' or datasetDir =='caltech50':
    ensDirs = ensDirs [:-1]
  LastMapEns = 0
  for ensDir in ensDirs:
    filemAP = parentDir + datasetDir+'/figures files/'+datasetDir+'_'+ method+ '_mAP_ensSize_'+ensDir+'.txt'
    filemAP = open(filemAP, 'w')

    # map for ensemble size = 1
    ensSize = 1
    if ensDir == 'ensemble':
      setPaths(ensDir=None, ensSize=None, datasetDir=datasetDir, method='base', cnnId = 1, kq=None)
      mapEns = mAP(p)
      filemAP.write('%s:%s \n' % (ensSize , mapEns))
    elif ensDir == 'ensembleBest':
      setPaths(ensDir=None, ensSize=None, datasetDir=datasetDir, method='baseBest', cnnId =1, kq=None)
      mapEns = mAP(p)
      filemAP.write('%s:%s \n' % (ensSize , mapEns))
    elif ensDir == 'ensembleWorst':
      setPaths(ensDir=None, ensSize=None, datasetDir=datasetDir, method='baseWorst', cnnId = 1, kq=None)
      mapEns = mAP(p)
      filemAP.write('%s:%s \n' % (ensSize , mapEns))
    # map for ensemble size > 1
    lista = os.listdir(vecsTrainDir+ensDir+'/'+method)
    filesCount = len(lista)
    for ensSize in range(2,filesCount+2):
      setPaths(ensDir=ensDir, ensSize=ensSize, datasetDir=datasetDir, method=method, cnnId = None, kq=None)
      mapEns = mAP(p)
      filemAP.write('%s:%s \n' % (ensSize , mapEns))
      if mapEns> LastMapEns:
        bstEnsSize = ensSize
        datasetEnsDir = ensDir
        LastMapEns = mapEns
  fileBstmAP.write('%s' % LastMapEns)
  filemAP.close()
  fileBstmAP.close()
  return bstEnsSize, datasetEnsDir



def getPrecRecK(datasetDir, method,ensSizeBest,ensSizeDir,p):
  filePrec = parentDir + datasetDir+'/figures files/'+datasetDir+'_'+ method+ '_precision_k.txt'
  fileRec = parentDir + datasetDir+'/figures files/'+datasetDir+'_'+ method+ '_recall_k.txt'
  filePrec = open(filePrec, 'w')
  fileRec = open(fileRec, 'w')
  kValues = [1, 10, 50, 100]
  setPaths(ensDir=ensSizeDir, ensSize=ensSizeBest, datasetDir=datasetDir,
           method=method, cnnId = None, kq=None)
  for k in kValues:
    prec,rec = getQueryPrecAtK(k=k,p=1,thresh=None,queryIdx=None)
    filePrec.write('%s:%s \n' % (k , prec))
    fileRec.write('%s:%s \n' % (k , rec))
  filePrec.close()
  fileRec.close()

def getPrecRecThresh(datasetDir, method, ensSizeBest,ensSizeDir,p):
  filePrec = parentDir + datasetDir+'/figures files/'+datasetDir+'_'+ method+ '_precision_thresh.txt'
  fileRec = parentDir + datasetDir+'/figures files/'+datasetDir+'_'+ method+ '_recall_thresh.txt'
  filePrec = open(filePrec, 'w')
  fileRec = open(fileRec, 'w')

  threshValues = [x/10 for x in range(1, 21)]
  setPaths(ensDir=ensSizeDir, ensSize=ensSizeBest, datasetDir=datasetDir, method=method,cnnId=None, kq=None)
  for thresh in threshValues:
    prec,rec = getQueryPrecAtK(k=None,p=1,thresh=thresh,queryIdx=None)
    print('precision = ',prec, 'recall = ',rec)
    filePrec.write('%s:%s \n' % (thresh , prec))
    fileRec.write('%s:%s \n' % (thresh , rec))
  filePrec.close()
  fileRec.close()


def getQEk(datasetDir, method, ensSizeBest,ensSizeDir,p):
  filekQE = parentDir + datasetDir+'/figures files/'+datasetDir+'_'+ method+ '_QE.txt'
  filekQE = open(filekQE, 'w')
  kQEValues = [1,2,5,10,15,20]
  for kQE in kQEValues:
      setPaths(ensDir=ensSizeDir, ensSize=ensSizeBest, datasetDir=datasetDir, method=method, cnnId=None, kq=kQE)
      mAPK = mAP(p)
      filekQE.write('%s:%s \n' % (kQE , mAPK))
  filekQE.close()


def Q_Expansion(dataset, method,ensSizeBest,ensSizeDir,p,k):
  setPaths(ensDir=ensSizeDir, ensSize=ensSizeBest, datasetDir=dataset, method=method,cnnId=None, kq=None)

  testPros = np.load(vecsTestPath, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
  trainPros = np.load(vecsTrainPath, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')

  ImgsCount = len(testPros)
  i = 0
  imagesMean = []
  imagesMedian = []
  loadedClassifiers = []

  j = 1
  while (j<=ensSizeBest):
    loadedClassifiers.append(np.load(vecsTrainDir + 'base/classesProbabilitiesOfmodel_'+str(j)+'.npy'))
    j+=1

  while (i < ImgsCount):
    imageMean = []
    selectedVectorCount = 0
    vectorsPro = [] #Contains all vectors to generate the new query vector
    allVectorsPro = []
    vecTest = []
    vectrain = []
    vectorsPronp = []
    Id, _ = getqueryClass(i, 'test')
    smlrImgsClassesId = []
    #step1: search k nearest neighbors and return images with their classes
    smlrOutProfilesIdx = searchTopK(trainPros,testPros[i], k, p)
    s = 0
    while(s<len(smlrOutProfilesIdx)):
      trainclassId, _ = getqueryClass(smlrOutProfilesIdx[s], 'train')
      smlrImgsClassesId.append(trainclassId)
      s += 1

    j = 0
    while (j<ensSizeBest):
      n = 0
      currentClassifierProTrain = loadedClassifiers[j]
      crctSamples = 0
      vecTest = testPros[i]
      #step2: For all k nearest neighbors, extract probability vector for current classifier
      while (n<len(smlrImgsClassesId)):
        currentTrainId = smlrOutProfilesIdx[n]
        vecTrain = currentClassifierProTrain[currentTrainId]
        belongingClassId = smlrImgsClassesId[n]
        corrctClassAcc = vecTrain[belongingClassId]
        allVectorsPro.append(vecTrain)
        #step3: If neighbor is correctly classified by current classifier, add the probability vector to the list:
        if (corrctClassAcc == np.amax(vecTrain)):
            vectorsPro.append(vecTrain)
            selectedVectorCount += 1

        n+= 1
      j += 1
    # step4: add query vector
    vectorsPro.append(vecTest)
    allVectorsPro.append(vecTest)
    if (selectedVectorCount >= 1): #if at least one neighbor vector is added in the list of vectors
      vectorsPronp = np.array(vectorsPro)
    else:
      vectorsPronp = np.array(allVectorsPro) #add all neighbors vectors
    #step5: mean/median of neighbors and query to generate new query:

    i += 1
    if method=='mean':
      imagesMean.append(vectorsPronp.mean(axis=0))
      np.save(vecsTestMeanQEDir + 'meanQEk'+str(k), np.array(imagesMean), allow_pickle=True, fix_imports=True)
    else:
      medianVec = np.median(vectorsPronp,axis=0)
      medianvec = 1/LA.norm(medianVec, 1) *  medianVec
      imagesMedian.append(medianvec )
      np.save(vecsTestMedianQEDir + 'medianQEk'+str(k), np.array(imagesMedian), allow_pickle=True, fix_imports=True)


def getAllMeasures():
  methods = ['mean','median']
  datasets = [ 'caltech20','caltech50','imagenet']
  kVals = [1,2,5,10,15,20]
  for dataset in datasets:
    getPrecRecK(dataset, 'base', None,None,1)
    getPrecRecThresh(dataset, 'base', None,None,1)
    for method in methods:
      p = getmAPP(dataset, method)
      ensSizeBest, ensSizeDir = getmAPensSize(dataset, method,p)
      getPrecRecK(dataset, method, ensSizeBest,ensSizeDir,p)
      getPrecRecThresh(dataset, method,ensSizeBest,ensSizeDir,p)
  for dataset in datasets[:-1]:
    for method in methods:
      for k in kVals:
        Q_Expansion(dataset, method,ensSizeBest,ensSizeDir,p,k)
      getQEk(dataset, method, ensSizeBest,ensSizeDir,p)

def PlotAndSave(datasetDir,method,vals,exprmnt,ensDir):
  y1Vals = []
  y2Vals = []
  xVals = []
  e = ''

  for val in vals:
    if exprmnt == 0:
      fileName = parentDir + datasetDir+'/figures files/'+datasetDir+'_'+ method+'_'+val+'_thresh'
      xl = 'Threshold'
      yl='Precision and recall'
      yl1='precision'
      yl2='recall'
      intrvl = 0.2
    elif exprmnt == 1:
      fileName = parentDir + datasetDir + '/figures files/'+datasetDir+'_'+ val+ '_mAP_ensSize_'+ensDir
      if ensDir == 'ensemble':
        e = ' (M)'
      elif ensDir == 'ensembleBest':
        e = ' (B)'
      elif ensDir == 'ensembleWorst':
        e = ' (W)'
      xl = 'Ensemble size'+e
      yl='mAP'
      yl1='Mean'
      yl2='Median'
      intrvl = 2
    elif exprmnt == 2:
      fileName = parentDir + datasetDir + '/figures files/'+datasetDir+'_'
      +val+ '_QE'
      xl = 'k'
      yl='mAP'
      yl1='Mean'
      yl2='Median'
      intrvl = 1

    txtFileName = fileName + '.txt'
    f = open(txtFileName , 'r')
    lines = [line.rstrip('\n') for line in f]
    if val == vals[0]:
      i=0
      while (i<len(lines)):
        val = lines[i].split(':')[1]
        y1Vals.append(float(val))
        val = lines[i].split(':')[0]
        xVals.append(float(val))
        i+=1
    elif val ==vals[1]:
      i=0
      while (i<len(lines)):
        val = lines[i].split(':')[1]
        y2Vals.append(float(val))
        i+=1

  outFile = fileName + '.pdf'

  xTicks = np.arange(min(xVals), max(xVals), intrvl)
  plotFigure(x=xVals,xt=xTicks,y1=y1Vals,y2=y2Vals,xl=xl,yl=yl,yl1=yl1,yl2=yl2,
             outFile=outFile)


def plotFigure(x,xt,y1,y2,xl,yl,yl1,yl2,outFile):
  plt.plot(x, y1,'ro-',label=yl1)
  plt.plot(x, y2,'bo-',label=yl2)
  plt.xlabel(xl)
  plt.ylabel(yl)
  plt.grid(True)
  plt.legend()
  plt.xticks(xt)
  plt.xlim(min(xt),max(xt)+min(xt))
  plt.savefig(outFile, bbox_inches='tight')
  plt.show()


def plotAll():

    datasets = ['caltech20','caltech50','imagenet']
    methods = ['mean', 'median','base']

    for datasetDir in datasets:
      setPaths(ensDir=None, ensSize=None, datasetDir=datasetDir, method=None,
               cnnId=None, kq=None)
      #precision and recall:
      for method in methods:
        vals = ['precision','recall']
        exprmnt = 0
        PlotAndSave(datasetDir,method,vals,exprmnt,None)
      #ensemble curves:
      if datasetDir =='caltech20' or datasetDir == 'caltech50':
        ensDirs = ['ensemble', 'ensembleBest']
      else:
        ensDirs = ['ensemble', 'ensembleBest', 'ensembleWorst']
      for ensDir in ensDirs:
        vals = ['mean','median']
        exprmnt = 1
        PlotAndSave(datasetDir,None,vals,exprmnt,ensDir)
    for datasetDir in datasets[:-1]:
      exprmnt = 2
      PlotAndSave(datasetDir,None,vals,exprmnt,ensDir)
