# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 06:10:59 2020

@author: Safa
"""

import split_folders
import shutil
import os

parentDir = ''
datasetDir = parentDir + 'imagenet/'
datasetCatDir  = datasetDir +'imagenet categories/'
datasetimgsDir = datasetDir + 'val/'
splitDatasetDir = datasetDir + 'splitted dataset/'

for i in range(1, 1001):
    os.mkdir(datasetCatDir + str(i))

i = 0
valSize = 5
testSize  = 1
for subdir, dirs, files in sorted(os.walk(datasetimgsDir)):
    for imageName in sorted(files):
        gt = open(datasetDir + 'groundtruth.txt', 'r')
        line = gt.readlines()[i]
        line = line.rstrip('\n')
        dest = datasetCatDir + line
        shutil.copy2(datasetimgsDir+imageName, dest + '/' + imageName)
        i += 1

split_folders.fixed(datasetCatDir, output=splitDatasetDir, fixed=(valSize,
                                                                  testSize))
shutil.rmtree(splitDatasetDir+'train')
os.rename(splitDatasetDir+'val', splitDatasetDir+'train')
