# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 06:11:24 2020

@author: Safa
"""

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
from numpy import linalg as LA
import os
import numpy as np
import glob
import pathlib
import keras

from paths import setPaths
from caltech_prepare_models import predictEns

def selectModel(i):
        if i==1:

          from keras.applications.vgg16 import preprocess_input
          model =  keras.applications.vgg16.VGG16(include_top=True,
                                                  weights='imagenet',
                                                  input_tensor=None,
                         input_shape=None, pooling=None, classes=1000)
          target_size=(224, 224)

        elif i==2:
          from keras.applications.vgg19 import preprocess_input
          model =  keras.applications.vgg19.VGG19(include_top=True,
                                                  weights='imagenet',
                                                  input_tensor=None,
                         input_shape=None, pooling=None, classes=1000)
          target_size=(224, 224)
        elif i==3:
          from keras.applications.resnet50 import preprocess_input
          model =  keras.applications.resnet.ResNet50(include_top=True,
                                                      weights='imagenet',
                            input_tensor=None, input_shape=None, pooling=None,
                            classes=1000)
          target_size=(224, 224)
        elif i ==4:
          from keras.applications.resnet import preprocess_input
          model =  keras.applications.resnet.ResNet152(include_top=True,
                                                       weights='imagenet',
                             input_tensor=None, input_shape=None, pooling=None,
                             classes=1000)
          target_size=(224, 224)
        elif i==5:
          from keras.applications.resnet import preprocess_input
          model = keras.applications.resnet.ResNet101(include_top=True,
                                                      weights='imagenet',
                            input_tensor=None, input_shape=None, pooling=None,
                            classes=1000)
          target_size = (224, 224)
        elif i==6:
          from keras.applications.resnet import preprocess_input
          model = keras.applications.resnet_v2.ResNet50V2(include_top=True,
                                                          weights='imagenet',
                             input_tensor=None, input_shape=None, pooling=None,
                             classes=1000)
          target_size = (224, 224)
        elif i ==7:
          from  keras.applications.resnet_v2 import preprocess_input
          model = keras.applications.resnet_v2.ResNet152V2(include_top=True,
                                                           weights='imagenet',
                              input_tensor=None, input_shape=None,pooling=None,
                              classes=1000)
          target_size = (224, 224)
        elif i ==8:
          from  keras.applications.resnet_v2 import preprocess_input
          model = keras.applications.resnet_v2.ResNet101V2(include_top=True,
                                                           weights='imagenet',
                              input_tensor=None, input_shape=None,
                              pooling=None,
                              classes=1000)
          target_size = (224, 224)
        elif i ==9:
          from keras.applications.inception_v3 import preprocess_input
          model = keras.applications.inception_v3.InceptionV3(include_top=True,
                                                              weights='imagenet',
                              input_tensor=None, input_shape=None,
                              pooling=None,
                              classes=1000)
          target_size=(299,299)
        elif i ==10:
          from keras.applications.xception import preprocess_input
          model = keras.applications.xception.Xception(include_top=True,
                                                       weights='imagenet',
                           input_tensor=None, input_shape=None, pooling=None,
                           classes=1000)
          target_size = (299, 299)
        elif i==11:
          from keras.applications.inception_resnet_v2 import preprocess_input
          model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True,
                                    weights='imagenet',
                                    input_tensor=None, input_shape=None,
                                    pooling=None, classes=1000)
          target_size = (299, 299)
        elif i==12:
          from keras.applications.mobilenet import preprocess_input
          model = keras.applications.mobilenet.MobileNet(input_shape=None,
                                                         alpha=1.0,
                                                         depth_multiplier=1,
                            dropout=1e-3, include_top=True, weights='imagenet',
                            input_tensor=None, pooling=None, classes=1000)
          target_size = (224, 224)
        elif i ==13:
          from keras.applications.mobilenet_v2 import preprocess_input
          model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=None,
                                                             alpha=1.0,
                                        include_top=True, weights='imagenet',
                                input_tensor=None, pooling=None, classes=1000)
          target_size = (224, 224)
        elif i ==14:
          from keras.applications.densenet import preprocess_input
          model = keras.applications.densenet.DenseNet121(include_top=True,
                                                          weights='imagenet',
                              input_tensor=None, input_shape=None,
                              pooling=None, classes=1000)
          target_size = (224, 224)
        elif i==15:
          from keras.applications.densenet import preprocess_input
          model = keras.applications.densenet.DenseNet169(include_top=True,
                                                          weights='imagenet',
                              input_tensor=None, input_shape=None,
                              pooling=None,
                              classes=1000)
          target_size = (224, 224)
        elif i == 16:
          from keras.applications.densenet import preprocess_input
          model = keras.applications.densenet.DenseNet201(include_top=True,
                                                          weights='imagenet',
                              input_tensor=None, input_shape=None,
                              pooling=None, classes=1000)
          target_size = (224, 224)
        elif i ==17:
          from keras.applications.nasnet import preprocess_input
          model = keras.applications.nasnet.NASNetMobile(input_shape=None,
                                                         include_top=True,
                               weights='imagenet', input_tensor=None,
                               pooling=None, classes=1000)
          target_size = (224, 224)
        elif i ==18:
          from keras.applications.nasnet import preprocess_input
          model = keras.applications.nasnet.NASNetLarge(input_shape=None,
                                                        include_top=True,
                              weights='imagenet', input_tensor=None,
                              pooling=None, classes=1000)
          target_size = (331,331)

        return model, target_size, preprocess_input

def predict(data):
    global cpArray

    if 'train' in data:
         folder = vecsTrainBaseDir
    else:
         folder = vecsTestBaseDir

    i = 1
    n = maxEnsSize
    while i<= n:
        classesProbabilities =[]
        fileName = folder + 'classesProbabilitiesOfmodel_'+str(i)
        sub = 1
        model,target_size,preprocess_input = selectModel(i)

        for subdir,dirs,files in sorted(os.walk(splittedDatasetDir +data+'/')):
            subdir = splittedDatasetDir +data+'/'+str(sub)
            for file in pathlib.Path(subdir).glob('*.JPEG'):
                imagePath = str(file)
                img = image.load_img(imagePath, target_size=target_size)
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                y=model.predict(img)[0]
                classesProbabilities.append(y)
            sub = sub +1


        cpArray = np.array(classesProbabilities)

        #remove model & save data:
        model = None
        np.save(fileName, cpArray, allow_pickle=True, fix_imports=True)
        i += 1

#calculate predictions for trained models:
dataset='imagenet'
setPaths(ensDir='ensemble', ensSize=None, datasetDir=dataset, method='base',
         cnnId=None, kq=None)
predict('train')
predict('test')

methods = ['base','baseBest','baseWorst']
for method in methods:
    predictEns('train',method)
    predictEns('test',method)
