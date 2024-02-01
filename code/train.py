# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 06:12:55 2020

@author: Safa
"""

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import callbacks as kc
from keras.applications.vgg16 import VGG16
import os

parentDir = ''
datasetDir = parentDir + 'caltech20/'
splittedDatasetDir = datasetDir + 'splitted dataset/'
validationPath = splittedDatasetDir + 'val/'
trainPath = splittedDatasetDir + 'train/'
weightsDir =  datasetDir + 'checkpoints/'
classesNb = len([name for name in os.listdir(datasetDir+'/images')])
epochs = 200 #define the number of epochs

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential()
model.add(vgg_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classesNb, activation='softmax'))

for layer in model.layers[:-4]:
    layer.trainable = False

    #model.load_weights(weightsDir + 'bag'+ str(bagNum)+'/weights.20-0.27-3.72-0.13-3.94.h5')

    def dataTrain(bagNum, trainPath, weightsPath):

      model.compile(loss='categorical_crossentropy',
                    optimizer= optimizers.sgd(lr = 1e-4, momentum = 0.9),
                    metrics = ['accuracy'])

      train_data_gen = ImageDataGenerator(rescale=1./255,rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        horizontal_flip = True)

      validation_data_gen = ImageDataGenerator(rescale=1./255)

      train_gen= train_data_gen.flow_from_directory(trainPath,
                                                  target_size=(224, 224),
                                                  batch_size = 64,
                                                  class_mode='categorical' )

      validation_gen= validation_data_gen.flow_from_directory(validationPath, target_size=(224, 224), batch_size = 64, class_mode='categorical' )

      checkpoint= kc.ModelCheckpoint(weightsPath, monitor='val_acc',
                                   verbose=2, save_best_only=True,
                                   save_weights_only=True, mode='max', period=10)

      callbacks_list = [checkpoint]

      history = model.fit_generator(train_gen,
                                  steps_per_epoch=train_gen.samples/train_gen.batch_size,
                                  epochs=epochs, verbose=2,
                                  validation_data=validation_gen,
                                  validation_steps=validation_gen.samples/validation_gen.batch_size,
                                  shuffle= False,
                                  callbacks=callbacks_list )

#train for base:
bagNum = None
trainPath = splittedDatasetDir+ 'train'
weightsPath = weightsDir+ 'base/weights.{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}-{acc:.2f}-{loss:.2f}.h5'
dataTrain(bagNum, trainPath, weightsPath)
#train for 20 bags:
bagNum = 1
while(bagNum <=20):
      weightsPath = weightsDir+ 'bag' + str(bagNum)+'/weights.{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}-{acc:.2f}-{loss:.2f}.h5'
      trainPath = splittedDatasetDir+ 'bag' + str(bagNum)+'/train'
      dataTrain(bagNum, trainPath, weightsPath)
      bagNum+=1
