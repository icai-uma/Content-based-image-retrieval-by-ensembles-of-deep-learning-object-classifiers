# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 06:12:04 2020

@author: Safa
"""

def setPaths(ensDir, ensSize, datasetDir, method, cnnId, kq):

  global relImgs,maxEnsSize, parentDir,splittedDatasetDir, vecsTrainPath,vecsTestPath, vecsTrainBaseDir,vecsTestBaseDir, vecsTestQEPath, vecsTestDir,vecsTrainDir, vecsTestMeanQEDir, vecsTestMedianQEDir,compiledModelsDir,trainedModelsDir,classesNb,vecsTrainBaseDir,vecsTestBaseDir

  parentDir = ''
  if datasetDir=='imagenet':
    relImgs = 5
    maxEnsSize = 18
  elif datasetDir=='caltech20' or datasetDir=='caltech50' :
    relImgs = 40
    maxEnsSize = 20

  compiledModelsDir = parentDir + datasetDir + '/compiled models/'
  trainedModelsDir = parentDir + datasetDir + '/trained models/'
  vecsDir = parentDir +  datasetDir + '/vectors probabilities/'
  vecsTrainDir = vecsDir +'train/'
  vecsTestDir = vecsDir + 'test/'

  splittedDatasetDir = parentDir + datasetDir + '/splitted dataset/'
  classesNb = len([name for name in os.listdir( parentDir + datasetDir
                                               +'/splitted dataset/train/')])

  #probability vectors dirs and paths:

  if method =='mean':
    vecsTrainMeanDir = vecsTrainDir + ensDir + '/mean/'
    vecsTestMeanDir = vecsTestDir + ensDir + '/mean/'
    vecsTestMeanQEDir = vecsTestDir + 'ensemble/mean/query expansion/'
    vecsTestPath = vecsTestMeanDir + 'mean-'+str(ensSize)+'CNN.npy'
    vecsTrainPath = vecsTrainMeanDir + 'mean-'+str(ensSize)+'CNN.npy'
    vecsTestQEPath  = vecsTestMeanQEDir + 'meanQEkssaa'+str(kq)+'.npy'
  elif method=='median':
    vecsTrainMedianDir = vecsTrainDir + ensDir + '/median/'
    vecsTestMedianDir = vecsTestDir + ensDir + '/median/'
    vecsTestMedianQEDir = vecsTestDir + 'ensemble/median/query expansion/'
    vecsTestPath = vecsTestMedianDir+ 'median-' + str(ensSize) + 'CNN.npy'
    vecsTrainPath   = vecsTrainMedianDir + 'median-' + str(ensSize) + 'CNN.npy'
    vecsTestQEPath   = vecsTestMedianQEDir + 'medianQEk' + str(kq) + '.npy'
  elif method=='base':
    vecsTrainBaseDir = vecsTrainDir + method + '/'
    vecsTestBaseDir = vecsTestDir + method + '/'
    if cnnId is not None:
      vecsTrainPath =  vecsTrainBaseDir + 'classesProbabilitiesOfmodel_'
      +str(cnnId)+'.npy'
      vecsTestPath =   vecsTestBaseDir+ 'classesProbabilitiesOfmodel_'
      +str(cnnId)+'.npy'
    else:
      vecsTrainPath =  vecsTrainBaseDir + 'classesProbabilitiesOfmodel_base.npy'
      vecsTestPath =   vecsTestBaseDir+ 'classesProbabilitiesOfmodel_base.npy'
