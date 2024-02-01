import split_folders
from random import randint,sample
import os
import shutil

parentDir = ''
datasetDir = parentDir + 'caltech20/' #or caltech50
datasetimgsDir =  datasetDir + ' images/'
trainSize = 40
valSize = 15
testSize = 25
bagsNb = 20
splitDatasetDir = datasetDir + 'splitted dataset/'

#1 - split folder into validation/test/train:
split_folders.fixed(datasetimgsDir, output=splitDatasetDir,
                    fixed=(valSize, testSize))

#2- in train set, keep only 40 samples in each category:
for subdir, dirs, files in os.walk(splitDatasetDir+'/train/'):
    categoryName = subdir.rsplit('/', 1)[1]
    if categoryName != "":
        i = trainSize
        while (i<len(files)):
            os.remove(splitDatasetDir+'/train/'+ categoryName + '/'+  files[i])
            i+= 1

#3- create bags:
i=1
while(i<=bagsNb):
    seed = randint(0, 5000)
    split_folders.ratio(splitDatasetDir+'/train',
                        output=splitDatasetDir+'/bag'+str(i), seed=seed,
                        ratio=(0.6, 0.4)) #create a bag
    shutil.rmtree(splitDatasetDir+"/bag"+str(i)+"/val") #delete remaining 40% of data
    i += 1
    
