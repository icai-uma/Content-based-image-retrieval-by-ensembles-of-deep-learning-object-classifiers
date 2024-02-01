# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 06:01:08 2020

@author: Safa
"""

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input,VGG16
from keras.preprocessing import image
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout
from keras import optimizers
from numpy import linalg as LA

from paths import setPaths

def compileModels():
  vgg16_model = VGG16(weights='imagenet', include_top=False,
                      input_shape=(224,224,3))
  model = Sequential()
  model.add(vgg16_model)
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(classesNb, activation='softmax'))
  for subdir, dirs, files in os.walk(trainedModelsDir):
    for modelName in files:
      model.load_weights(trainedModelsDir + modelName)
      model.compile(loss='categorical_crossentropy',
                    optimizer= optimizers.sgd(lr = 1e-4, momentum = 0.9),
                    metrics = ['accuracy'])
      model.save(compiledModelsDir + modelName)

def predict(data,modelName):
    global cpArray

    if 'train' in data:
         folder = vecsTrainBaseDir
    else:
         folder = vecsTestBaseDir

    i = 1
    while i<= maxEnsSize:
        classesProbabilities =[]
        if modelName == 'trained':
          model = load_model(compiledModelsDir +'model'+str(i)+'.h5')
          fileName = folder + 'classesProbabilitiesOfmodel_'+str(i)
        else:
          model = load_model(compiledModelsDir +'base.h5')
          fileName = folder + 'classesProbabilitiesOfmodel_base'
          n = 1

        for subdir,dirs,files in sorted(os.walk(splittedDatasetDir +data+'/')):

            for imageName in sorted(files):
                imagePath = subdir + os.sep + imageName
                folderName = subdir.rsplit('/', 1)[1]
                img = image.load_img(imagePath, target_size=(224,224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                y=model.predict(img)[0]
                classesProbabilities.append(y)


        cpArray = np.array(classesProbabilities)

        #remove model & save data:
        model = None
        np.save(fileName, cpArray, allow_pickle=True, fix_imports=True)
        i += 1

def predictEns(path,method):
    global maxEnsSize
    if 'train' in path:
        folder =  vecsTrainDir
    else:
        folder = vecsTestDir

    fileCount = len(np.load(folder+'base/classesProbabilitiesOfmodel_1.npy',
                 mmap_mode=None, allow_pickle=True, fix_imports=True,
                 encoding='ASCII'))
    ensSize = 2
    if 'Best' in method:
      ens = 'ensembleBest'
      if 'caltech' in dataset:
        maxEnsSize = 10
    elif 'Worst' in method:
      ens = 'ensembleWorst'
    else:
      ens = 'ensemble'

    while (ensSize <= maxEnsSize):

      imagesMedian, imagesMean = ([] for i in range(2))
      i = 0
      while i< fileCount:
          imagecnnProbabilities = []
          k = 1
          while k<= ensSize:
              fileName = folder + method+'/classesProbabilitiesOfmodel_'
              + str(k) + '.npy'
              #display(fileName)
              cp = np.load(fileName, mmap_mode=None, allow_pickle=True,
                           fix_imports=True, encoding='ASCII')
              imagecnnProbabilities.append(cp[i])
              k += 1
          i += 1
          imagecnnProbabilitiesnp = np.array(imagecnnProbabilities)

        #mean:
          imagesMean.append(imagecnnProbabilitiesnp.mean(axis=0))
        #median:
          medianVec = np.median(imagecnnProbabilitiesnp,axis=0)
          medianvec = 1/LA.norm(medianVec, 1) *  medianVec
          imagesMedian.append(medianvec )
      np.save(folder + ens+ '/mean/mean-'+str(ensSize)+'CNN',
              np.array(imagesMean), allow_pickle=True, fix_imports=True)
      np.save(folder + ens + '/median/median-'+str(ensSize)+'CNN',
              np.array(imagesMedian), allow_pickle=True, fix_imports=True)
      ensSize += 1


datasets = ['caltech20', 'caltech50']
methods = ['base','baseBest']

for dataset in datasets:
  setPaths(ensDir='ensemble', ensSize=None, datasetDir=dataset, method='base',
           cnnId=None, kq=None)
  compileModels()
  #calculate predictions for testing  images for bagged and base models:
  predict('test','trained')
  predict('test','base')
  #calculate predictions for the base model:
  predict('train','base')
  predict('train','trained')
  for method in methods:
    predictEns('train',method)
    predictEns('test',method)
