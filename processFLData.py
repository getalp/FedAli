#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Uncomment if running on googlecolab 
# !pip install hickle
# from google.colab import drive
# drive.mount('/content/drive/')
# %cd drive/MyDrive/PerCom2021-FL-master/


# In[ ]:


import hickle as hkl 
import numpy as np
import os
import warnings
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


# In[ ]:


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
randomSeed = 0
np.random.seed(randomSeed)


# In[ ]:


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[ ]:


mainDir = './Datasets'


# In[ ]:


datasetList = ['HHAR','RealWorld'] 


# In[ ]:


dirName =  mainDir + '/FL_Clients'
os.makedirs(dirName, exist_ok=True)


# In[ ]:


fineTuneDir = 'trainData'
testDir = 'testData'
datasetDir = 'datasets'
# os.makedirs(dirName+'/'+datasetDir, exist_ok=True)
os.makedirs(dirName+'/'+fineTuneDir, exist_ok=True)
os.makedirs(dirName+'/'+testDir, exist_ok=True)


# In[ ]:


HHAR_Activities = ['Sitting', 'Standing', 'Walking', 'Upstair', 'Downstair', 'Biking']
RW_Activities = ['Downstair','Upstair','Running','Sitting','Standing','Walking','Lying','Jumping']
AlignedLabels = ['Sitting', 'Standing', 'Walking', 'Upstair', 'Downstair', 'Biking','Running','Lying','Jumping']


# In[ ]:





# In[ ]:


RWMapping = [4,3,6,0,1,2,7,8]


# In[ ]:


for datasetIndex,dataSetName in enumerate(datasetList):
    datasetLabel = hkl.load(mainDir + '/processedDatasets/'+dataSetName+'/clientsLabel.hkl')
    datasetTrain = hkl.load(mainDir + '/processedDatasets/'+dataSetName+'/clientsData.hkl')
    
    trainingData = []
    testingData = []
    trainingLabel = []
    testingLabel = []
    
    alignedTrainingLabel = []
    alignedTestingLabel = []
    
    for datasetData, datasetLabels in zip(datasetTrain,datasetLabel):        
        skf = StratifiedKFold(n_splits=10,shuffle = False)
        skf.get_n_splits(datasetData, datasetLabels)
        testIndex = []
        
        for train_index, test_index in skf.split(datasetData, datasetLabels):
            testIndex.append(test_index)

        trainIndex = np.hstack((testIndex[:7]))
        testIndex = np.hstack((testIndex[7:]))

        X_train = tf.gather(datasetData,trainIndex).numpy()
        X_test = tf.gather(datasetData,testIndex).numpy()
        
        y_train = tf.gather(datasetLabels,trainIndex).numpy()
        y_test = tf.gather(datasetLabels,testIndex).numpy()

        if(dataSetName == 'RealWorld'):

            y_train_onehot = tf.one_hot(y_train,len(RW_Activities))
            y_test_onehot = tf.one_hot(y_test,len(RW_Activities))

            alignedLabel = np.asarray([RWMapping[labelIndex] for labelIndex in datasetLabels])
            
            y_train_aligned = tf.gather(alignedLabel,trainIndex).numpy()
            y_test_aligned = tf.gather(alignedLabel,testIndex).numpy()
            
            y_train_aligned_onehot = tf.one_hot(y_train_aligned,len(AlignedLabels))
            y_test_aligned_onehot = tf.one_hot(y_test_aligned,len(AlignedLabels))
            

        else:
            y_train_onehot = tf.one_hot(y_train,len(HHAR_Activities))
            y_test_onehot = tf.one_hot(y_test,len(HHAR_Activities))

            y_train_aligned_onehot = tf.one_hot(y_train,len(AlignedLabels))
            y_test_aligned_onehot = tf.one_hot(y_test,len(AlignedLabels))
            
        trainingData.append(X_train)
        testingData.append(X_test)
        
        trainingLabel.append(y_train_onehot)
        testingLabel.append(y_test_onehot)


        alignedTrainingLabel.append(y_train_aligned_onehot)
        alignedTestingLabel.append(y_test_aligned_onehot)

    trainingData = np.asarray(trainingData, dtype=object)
    trainingLabel = np.asarray(trainingLabel, dtype=object)
    alignedTrainingLabel = np.asarray(alignedTrainingLabel, dtype=object)

    testingData = np.asarray(testingData, dtype=object)
    testingLabel = np.asarray(testingLabel, dtype=object)
    alignedTestingLabel = np.asarray(alignedTestingLabel, dtype=object)

    
    hkl.dump(trainingData,dirName+'/'+fineTuneDir+ '/'+dataSetName+'_data.hkl')
    hkl.dump(trainingLabel,dirName+'/'+fineTuneDir+ '/'+dataSetName+'_label.hkl')
    hkl.dump(alignedTrainingLabel,dirName+'/'+fineTuneDir+ '/'+dataSetName+'_aligned_label.hkl')

    
    hkl.dump(testingData,dirName+'/'+testDir+ '/'+dataSetName+'_data.hkl' )
    hkl.dump(testingLabel,dirName+'/'+testDir+ '/'+dataSetName+'_label.hkl' )
    hkl.dump(alignedTestingLabel,dirName+'/'+testDir+ '/'+dataSetName+'_aligned_label.hkl')




# In[ ]:





# In[ ]:


dirLabels =  mainDir + '/FL_Clients/labelNames'
os.makedirs(dirLabels, exist_ok=True)
hkl.dump(HHAR_Activities,dirLabels+'/HHAR.hkl')
hkl.dump(RW_Activities,dirLabels+'/RealWorld.hkl')
hkl.dump(AlignedLabels,dirLabels+'/Combined.hkl')

