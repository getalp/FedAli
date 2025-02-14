#!/usr/bin/env python
# coding: utf-8


import numpy as np


from sklearn.model_selection import KFold,StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
import os
import hickle as hkl 

from sklearn import preprocessing
import scipy.signal
import scipy.stats
import tensorflow as tf
import seaborn as sns
import logging
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']

class dataHolder:
    clientDataTrain = []
    clientLabelTrain = []
    clientDataTest = []
    clientLabelTest = []
    centralTrainData = []
    centralTrainLabel = []
    centralTestData = []
    centralTestLabel = []
    clientOrientationTrain = []
    clientOrientationTest = []
    orientationsNames = None
    activityLabels = []
    clientCount = None

def generatePrototypeCounts(baseProtoypeCount,blocks):
    result = [baseProtoypeCount]
    for _ in range(blocks - 1): 
        baseProtoypeCount //= 2  
        result.append(baseProtoypeCount)
    return result


class LinearLearningRateScheduler(LearningRateSchedule):
    def __init__(self, initial_lr, end_lr, num_epochs):
        super(LinearLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_epochs = num_epochs

    def __call__(self, epoch):
        if(epoch <= self.num_epochs):
            current_lr = (1 - epoch / self.num_epochs) * self.initial_lr + (epoch / self.num_epochs) * self.end_lr
        else:
            current_lr = self.end_lr
        return current_lr
    
def returnClientByDataset(dataSetName):
    if(dataSetName=='UCI' or dataSetName ==  'UCI_ORIGINAL'):
        return 5
    elif(dataSetName == "REALWORLD_CLIENT"):
        return 15
    elif(dataSetName == "SHL_128_PreviewLowPass"):
        return 3
    elif(dataSetName == "SHL_128_Body_PreviewLowPass"):
        return 12
    elif(dataSetName == "Motion_Sense" or dataSetName == "Motion_Sense_Sensors"):
        return 24 
    elif(dataSetName == "SHL_128_Time_PreviewLowPass"):
        return 9
    elif(dataSetName == "HHAL_DEVICE"):
        return 51
    else:
        raise ValueError('Unknown dataset')
    
def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None)
    return dataframe.values


def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    loaded = np.dstack(loaded)
    return loaded


def create_segments_and_labels_Mobiact(df, time_steps, step, label_name = "LabelsEncoded", n_features= 6):
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        acc_x = df['acc_x'].values[i: i + time_steps]
        acc_y = df['acc_y'].values[i: i + time_steps]
        acc_z = df['acc_z'].values[i: i + time_steps]

        gyro_x = df['gyro_x'].values[i: i + time_steps]
        gyro_y = df['gyro_y'].values[i: i + time_steps]
        gyro_z = df['gyro_z'].values[i: i + time_steps]

    

        # Retrieve the most often used label in this segment
        label = scipy.stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, n_features)
    labels = np.asarray(labels)

    return reshaped_segments, labels


def load_dataset(group,mainDir,prefix=''):
    filepath = mainDir + 'datasetStandardized/'+prefix + '/' + group + '/'
    filenames = list()
    filenames += ['AccX'+prefix+'.csv', 'AccY' +
                prefix+'.csv', 'AccZ'+prefix+'.csv']
    filenames += ['GyroX'+prefix+'.csv', 'GyroY' +
                prefix+'.csv', 'GyroZ'+prefix+'.csv']
    X = load_group(filenames, filepath)
    y = load_file(mainDir + 'datasetStandardized/'+prefix +
                '/' + group + '/Label'+prefix+'.csv')
    return X, y

def projectTSNE(fileName,filepath,ACTIVITY_LABEL,labels_argmax,tsne_projections,unique_labels,globalPrototypesIndex = None ):
    plt.figure(figsize=(16,16))
#     plt.title('HART Embeddings T-SNE')
    graph = sns.scatterplot(
        x=tsne_projections[:,0], y=tsne_projections[:,1],
        hue=labels_argmax,
        palette=sns.color_palette(n_colors = len(unique_labels)),
        s=50,
        alpha=1.0,
        rasterized=True
    )
    legend = graph.legend_
    for j, label in enumerate(unique_labels):
        legend.get_texts()[j].set_text(ACTIVITY_LABEL[int(label)]) 
        

    plt.tick_params(
    axis='both',         
    which='both',     
    bottom=False,     
    top=False,         
    labelleft=False,        
    labelbottom=False)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    if(globalPrototypesIndex != None):
        plt.scatter(tsne_projections[globalPrototypesIndex,0],tsne_projections[globalPrototypesIndex,1], s=400,linewidth=3, facecolors='none', edgecolor='black')
    plt.savefig(filepath+fileName+".png", bbox_inches="tight")
    plt.show()
    plt.clf()
def projectTSNEWithPosition(dataSetName,fileName,filepath,ACTIVITY_LABEL,labels_argmax,orientationsNames,clientOrientationTest,tsne_projections,unique_labels):
    classData = [ACTIVITY_LABEL[i] for i in labels_argmax]
    orientationData = [orientationsNames[i] for i in np.hstack((clientOrientationTest))]
    if(dataSetName == 'REALWORLD_CLIENT'):
        orientationName = 'Position'
    else:
        orientationName = 'Device'
    pandaData = {'col1': tsne_projections[:,0], 'col2': tsne_projections[:,1],'Classes':classData, orientationName :orientationData}
    pandaDataFrame = pd.DataFrame(data=pandaData)

    plt.figure(figsize=(16,16))
#     plt.title('HART Embeddings T-SNE')
    sns.scatterplot(data=pandaDataFrame, x="col1", y="col2", hue="Classes", style=orientationName,
                    palette=sns.color_palette(n_colors = len(unique_labels)),
                    s=50, alpha=1.0,rasterized=True,)
    plt.tick_params(
    axis='both',          
    which='both',     
    bottom=False,     
    top=False,         
    labelleft=False,       
    labelbottom=False)

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.savefig(filepath+fileName+".png", bbox_inches="tight")
    plt.show()
    plt.clf()
    
def loadDataset(dataSetName, clientCount, dataConfig, randomSeed, mainDir, StratifiedSplit = True):
# loading datasets
    clientDataTrain = []
    clientLabelTrain = []
    clientDataTest = []
    clientLabelTest = []
    centralTrainData = []
    centralTrainLabel = []
    centralTestData = []
    centralTestLabel = []
    clientOrientationTrain = []
    clientOrientationTest = []
    orientationsNames = None
    orientationsNames = None
    
    
    if(dataSetName == "UCI"):

        trainData, trainLabel = load_dataset('train',mainDir, dataSetName)
        evalData, evalLabel = load_dataset('eval',mainDir, dataSetName)
        allData = np.float32(np.vstack((trainData, evalData)))
        allLabel = np.vstack((trainLabel, evalLabel))

        # split data into 80 - 20 
        skf = StratifiedKFold(n_splits=5,shuffle = True,random_state = randomSeed)
        skf.get_n_splits(allData, allLabel)
        partitionedData = list()
        partitionedLabel = list()
        for train_index, test_index in skf.split(allData, allLabel):
            partitionedData.append(allData[test_index])
            partitionedLabel.append(allLabel[test_index])

        centralTrainData = np.vstack((partitionedData[:4]))
        centralTrainLabel = np.vstack((partitionedLabel[:4]))
        centralTestData = partitionedData[4]
        centralTestLabel = partitionedLabel[4]

        trainData = list()
        trainLabel = list()
        testData = list()
        testLabel = list()

        if(dataConfig == "BALANCED"):
            skf = StratifiedKFold(n_splits=clientCount,shuffle = True , random_state = randomSeed)
            skf.get_n_splits(centralTrainData, centralTrainLabel)
            for train_index, test_index in skf.split(centralTrainData, centralTrainLabel):
                trainData.append(centralTrainData[test_index])
                trainLabel.append(centralTrainLabel[test_index].ravel())
        else:
        # unbalanced
            kf = KFold(n_splits=clientCount, shuffle=True,random_state = randomSeed)
            kf.get_n_splits(centralTrainData)
            for train_index, test_index in kf.split(centralTrainData):
                trainData.append(centralTrainData[test_index])
                trainLabel.append(centralTrainLabel[test_index].ravel())

        #splittestSetInto5
        skf.get_n_splits(centralTestData, centralTestLabel)
        for train_index, test_index in skf.split(centralTestData, centralTestLabel):
            testData.append(centralTestData[test_index])
            testLabel.append(centralTestLabel[test_index].ravel())

        clientDataTrain = trainData
        clientLabelTrain = trainLabel
        clientDataTest = testData
        clientLabelTest = testLabel
        
        centralTrainData = (np.vstack((clientDataTrain)))
        centralTrainLabel = (np.hstack((clientLabelTrain)))

        centralTestData = (np.vstack((clientDataTest)))
        centralTestLabel = (np.hstack((clientLabelTest)))
        
        
    elif(dataSetName == "UCI_ORIGINAL"):
        centralTrainData, centralTrainLabel = load_dataset('train',mainDir, 'UCI')
        centralTestData, centralTestLabel = load_dataset('eval',mainDir, 'UCI')
        centralTrainLabel = np.squeeze(centralTrainLabel)
        centralTestLabel = np.squeeze(centralTestLabel)
    elif(dataSetName == "REALWORLD_CLIENT"):
        clientData = []
        clientLabel = []
        orientations = hkl.load(mainDir + 'datasetStandardized/REALWORLD_CLIENT/clientsOrientationRW.hkl')
        orientationsNames = ['chest','forearm','head','shin','thigh','upperarm','waist']
        
        dataSetName = 'REALWORLD_CLIENT'
        for i in range(0,clientCount):
            accX = hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/'+str(i)+'/AccX'+dataSetName+'.hkl')
            accY = hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/'+str(i)+'/AccY'+dataSetName+'.hkl')
            accZ = hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/'+str(i)+'/AccZ'+dataSetName+'.hkl')
            gyroX = hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/'+str(i)+'/GyroX'+dataSetName+'.hkl')
            gyroY = hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/'+str(i)+'/GyroY'+dataSetName+'.hkl')
            gyroZ = hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/'+str(i)+'/GyroZ'+dataSetName+'.hkl')
            label = hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/'+str(i)+'/Label'+dataSetName+'.hkl')
            clientData.append(np.dstack((accX,accY,accZ,gyroX,gyroY,gyroZ)))
            clientLabel.append(label)
        
        if(dataConfig == "BALANCED"):
            for i in range (0,clientCount):
                skf = StratifiedKFold(n_splits=5, shuffle=True,random_state = randomSeed)
                skf.get_n_splits(clientData[i], clientLabel[i])
                partitionedData = list()
                partitionedLabel = list()    
                dataIndex = []

                for train_index, test_index in skf.split(clientData[i], clientLabel[i]):
                    partitionedData.append(clientData[i][test_index])
                    partitionedLabel.append(clientLabel[i][test_index])
                    dataIndex.append(test_index)
                    
                clientDataTrain.append((np.vstack((partitionedData[:4]))))
                clientLabelTrain.append((np.hstack((partitionedLabel[:4]))))
                clientDataTest.append((partitionedData[4]))
                clientLabelTest.append((partitionedLabel[4]))
                clientOrientationTrain.append(np.hstack((dataIndex[:4])))
                clientOrientationTest.append(dataIndex[4]) 
        else:
            for i in range (0,clientCount):
                kf = KFold(n_splits=5, shuffle=True,random_state = randomSeed)
                kf.get_n_splits(clientData[i])
                partitionedData = list()
                partitionedLabel = list()    
                for train_index, test_index in kf.split(clientData[i]):
                    partitionedData.append(clientData[i][test_index])
                    partitionedLabel.append(clientLabel[i][test_index])
                clientDataTrain.append((np.vstack((partitionedData[:4]))))
                clientLabelTrain.append((np.hstack((partitionedLabel[:4]))))
                clientDataTest.append((partitionedData[4]))
                clientLabelTest.append((partitionedLabel[4]))
                
        for i in range(0,clientCount):
            clientOrientationTest[i] = orientations[i][clientOrientationTest[i]]
            clientOrientationTrain[i] = orientations[i][clientOrientationTrain[i]]

        centralTrainData = (np.vstack((clientDataTrain)))
        centralTrainLabel = (np.hstack((clientLabelTrain)))

        centralTestData = (np.vstack((clientDataTest)))
        centralTestLabel = (np.hstack((clientLabelTest)))
    else:
        clientData = []
        clientLabel = []

        for i in range(0,clientCount):
            clientData.append(hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/UserData'+str(i)+'.hkl'))
            clientLabel.append(hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/UserLabel'+str(i)+'.hkl'))
            
            
        if(StratifiedSplit and dataSetName == "SHL_128_Time_PreviewLowPass"):
            timePerUser = 3 
            tempData = {new_list: [] for new_list in range(clientCount)}
            tempLabel = {new_list: [] for new_list in range(clientCount)}
            for i in range(0,clientCount):
                startCount = int(i/timePerUser)*timePerUser
                skf = StratifiedKFold(n_splits=timePerUser, shuffle=True, random_state = randomSeed)
                skf.get_n_splits(clientData[i], clientLabel[i])
                for train_index, test_index in skf.split(clientData[i], clientLabel[i]):
                    tempData[startCount].append(clientData[i][test_index])
                    tempLabel[startCount].append(clientLabel[i][test_index])
                    startCount += 1 
            for i in range(0,clientCount):
                clientData[i] =  np.vstack((tempData[i]))
                clientLabel[i] = np.hstack((tempLabel[i]))
            del tempData,tempLabel
        if(dataSetName == "HHAL_DEVICE"):
            orientations = hkl.load(mainDir + 'datasetStandardized/HHAL_DEVICE/deviceIndex.hkl')
            orientationsNames = ['nexus4', 'lgwatch','s3', 's3mini','gear','samsungold']

        for i in range (0,clientCount):
            skf = StratifiedKFold(n_splits=5, shuffle=True,random_state = randomSeed)
            skf.get_n_splits(clientData[i], clientLabel[i])
            partitionedData = list()
            partitionedLabel = list()    
            dataIndex = []

            for train_index, test_index in skf.split(clientData[i], clientLabel[i]):
                partitionedData.append(clientData[i][test_index])
                partitionedLabel.append(clientLabel[i][test_index])
                dataIndex.append(test_index)

            clientDataTrain.append((np.vstack((partitionedData[:4]))))
            clientLabelTrain.append((np.hstack((partitionedLabel[:4]))))
            clientDataTest.append((partitionedData[4]))
            clientLabelTest.append((partitionedLabel[4]))
            clientOrientationTrain.append(np.hstack((dataIndex[:4])))
            clientOrientationTest.append(dataIndex[4]) 
            
        if(dataSetName == "HHAL_DEVICE"):        
            for i in range(0,clientCount):
                clientOrientationTest[i] = orientations[i][clientOrientationTest[i]]
                clientOrientationTrain[i] = orientations[i][clientOrientationTrain[i]]

            
            
        centralTrainData = (np.vstack((clientDataTrain)))
        centralTrainLabel = (np.hstack((clientLabelTrain)))

        centralTestData = (np.vstack((clientDataTest)))
        centralTestLabel = (np.hstack((clientLabelTest)))


    dataReturn = dataHolder
    dataReturn.clientDataTrain = clientDataTrain
    dataReturn.clientLabelTrain = clientLabelTrain
    dataReturn.clientDataTest = clientDataTest
    dataReturn.clientLabelTest = clientLabelTest
    dataReturn.centralTrainData = centralTrainData
    dataReturn.centralTrainLabel = centralTrainLabel
    dataReturn.centralTestData = centralTestData
    dataReturn.centralTestLabel = centralTestLabel
    dataReturn.clientOrientationTrain = clientOrientationTrain
    dataReturn.clientOrientationTest = clientOrientationTest
    dataReturn.orientationsNames = orientationsNames
    return dataReturn


def load_checkpoint(filepath,nbOfBlocks = 6):
    if(os.path.exists(filepath+'checkpoint.hkl')):
        checkpointProp = hkl.load(filepath+'checkpoint.hkl')
        logging.warning("Checkpoint Found")
        logging.warning("Communication Round:"+str(checkpointProp["CommunicationRound"]))
    else:
        checkpointProp = {}
        checkpointProp["CommunicationRound"] = 0
        checkpointProp['bestServerVal'] = 0
        checkpointProp['pretrained_f1_score'] = 0
        checkpointProp['roundTrainingTime'] = []

        # Initialization of metrics during training
        checkpointProp['adaptiveLoss'] = []
        checkpointProp['adaptiveLossStd'] = []

        # client models test againts own test-set
        checkpointProp['trainLossHistory'] = []
        checkpointProp['trainAccHistory'] = []
        checkpointProp['testLossHistory'] = []
        checkpointProp['testAccHistory'] = []

        checkpointProp['stdTrainLossHistory'] = []
        checkpointProp['stdTrainAccHistory'] = []
        checkpointProp['stdTestLossHistory'] = []
        checkpointProp['stdTestAccHistory'] = []

        # client models test againts all test-set

        checkpointProp['clientTrainLossHistory'] = []
        checkpointProp['clientTrainAccHistory'] = []
        checkpointProp['clientTestLossHistory'] = []
        checkpointProp['clientTestAccHistory'] = []

        checkpointProp['clientStdTrainLossHistory'] = []
        checkpointProp['clientStdTrainAccHistory'] = []
        checkpointProp['clientStdTestLossHistory'] = []
        checkpointProp['clientStdTestAccHistory'] = []
        
        checkpointProp['globalTestLossHistory'] = []
        checkpointProp['globalTestAccHistory'] = []


        checkpointProp['globalTestAlignZeroLossHistory']  = []
        checkpointProp['globalTestAlignZeroAccHistory'] = []
        # server test againts all test-set

#         checkpointProp['serverTrainLossHistory'] = []
#         checkpointProp['serverTrainAccHistory'] = []
        checkpointProp['meanHistoryDist'] = []
        checkpointProp['stdHistoryDist'] = []

        checkpointProp['meanRoundLayerHistory'] = []
        checkpointProp['stdRoundLayerHistory'] = []

        checkpointProp['meanRoundGeneralLayerHistory'] = []
        checkpointProp['stdRoundGeneralLayerHistory'] = []

        checkpointProp['bestModelRound'] = 0
        checkpointProp['currentAccuracy'] = 0.0
        checkpointProp['currentGeneralizationAccuracy'] = 0.0
        checkpointProp['serverCurrentAccuracy'] = 0.0
        checkpointProp['serverbestModelRound'] = 0
        checkpointProp['bestServerModelWeights'] = None
        checkpointProp['modelEmbeddings'] = None
        checkpointProp['best_local_weights'] = []
        checkpointProp['autoEncoderHistory'] = [] 
        checkpointProp['totalEmission'] = 0.0

        checkpointProp['prototypeStabilityEpoch'] = {i: [] for i in range(nbOfBlocks)}
        checkpointProp['previousPrototype'] = {i: [] for i in range(nbOfBlocks)}

        hkl.dump(checkpointProp,filepath+'checkpoint.hkl')
    return checkpointProp
def load_data(dataSetName, randomSeed, mainDir, clientCount = 0, oneHot = True,  dataConfig = 'BALANCED',StratifiedSplit = True):
    if(clientCount == 0):
        if(dataSetName=='UCI'):
            clientCount = 5
        elif(dataSetName == "REALWORLD_CLIENT"):
            clientCount = 15
        elif(dataSetName == "SHL_128_PreviewLowPass"):
            clientCount = 3
        elif(dataSetName == "SHL_128_Body_PreviewLowPass"):
            clientCount = 12
        elif(dataSetName == "SHL_128_Time_PreviewLowPass"):
            clientCount = 9
        elif(dataSetName == "Motion_Sense"):
            clientCount = 24   
        elif(dataSetName == "HHAL_DEVICE"):
            clientCount = 51
        else:
            raise ValueError('Unknown Dataset')

    if(dataSetName == 'UCI' or dataSetName ==  'UCI_ORIGINAL'):
        ACTIVITY_LABEL = ['Walking', 'Upstair','Downstair', 'Sitting', 'Standing', 'Lying']
    elif(dataSetName == "REALWORLD_CLIENT"):
        ACTIVITY_LABEL = ['Downstairs','Upstairs', 'Jumping','Lying', 'Running', 'Sitting', 'Standing', 'Walking']
    elif(dataSetName == "Motion_Sense"):
        ACTIVITY_LABEL = ['Downstairs', 'Upstairs', 'Sitting', 'Standing', 'Walking', 'Jogging']
    elif(dataSetName == "HHAL_DEVICE"):
        ACTIVITY_LABEL = ['Sitting', 'Standing', 'Walking', 'Upstair', 'Downstairs', 'Biking']
    elif(dataSetName == "HHAL_DEVICE"):
        ACTIVITY_LABEL = ['Downstairs', 'Upstairs', 'Sitting', 'Standing', 'Walking', 'Jogging']
    else:
        ACTIVITY_LABEL = ['Standing','Walking','Runing','Biking','Car','Bus','Train','Subway']
    activityCount = len(ACTIVITY_LABEL)
    clientDataTrain = []
    clientLabelTrain = []
    clientDataTest = []
    clientLabelTest = []
    centralTrainData = []
    centralTrainLabel = []
    centralTestData = []
    centralTestLabel = []
    clientOrientationTrain = []
    clientOrientationTest = []
    orientationsNames = None
    orientationsNames = None
    
    
    if(dataSetName == "UCI"):

        trainData, trainLabel = load_dataset('train',mainDir, dataSetName)
        evalData, evalLabel = load_dataset('eval',mainDir, dataSetName)
        allData = np.float32(np.vstack((trainData, evalData)))
        allLabel = np.vstack((trainLabel, evalLabel))

        # split data into 80 - 20 
        skf = StratifiedKFold(n_splits=5,shuffle = True,random_state = randomSeed)
        skf.get_n_splits(allData, allLabel)
        partitionedData = list()
        partitionedLabel = list()
        for train_index, test_index in skf.split(allData, allLabel):
            partitionedData.append(allData[test_index])
            partitionedLabel.append(allLabel[test_index])

        centralTrainData = np.vstack((partitionedData[:4]))
        centralTrainLabel = np.vstack((partitionedLabel[:4]))
        centralTestData = partitionedData[4]
        centralTestLabel = partitionedLabel[4]

        trainData = list()
        trainLabel = list()
        testData = list()
        testLabel = list()

        if(dataConfig == "BALANCED"):
            skf = StratifiedKFold(n_splits=clientCount,shuffle = True , random_state = randomSeed)
            skf.get_n_splits(centralTrainData, centralTrainLabel)
            for train_index, test_index in skf.split(centralTrainData, centralTrainLabel):
                trainData.append(centralTrainData[test_index])
                trainLabel.append(centralTrainLabel[test_index].ravel())
        else:
        # unbalanced
            kf = KFold(n_splits=clientCount, shuffle=True,random_state = randomSeed)
            kf.get_n_splits(centralTrainData)
            for train_index, test_index in kf.split(centralTrainData):
                trainData.append(centralTrainData[test_index])
                trainLabel.append(centralTrainLabel[test_index].ravel())

        #splittestSetInto5
        skf.get_n_splits(centralTestData, centralTestLabel)
        for train_index, test_index in skf.split(centralTestData, centralTestLabel):
            testData.append(centralTestData[test_index])
            testLabel.append(centralTestLabel[test_index].ravel())

        clientDataTrain = trainData
        clientLabelTrain = trainLabel
        clientDataTest = testData
        clientLabelTest = testLabel
        
        centralTrainData = (np.vstack((clientDataTrain)))
        centralTrainLabel = (np.hstack((clientLabelTrain)))

        centralTestData = (np.vstack((clientDataTest)))
        centralTestLabel = (np.hstack((clientLabelTest)))
        
        
    elif(dataSetName == "UCI_ORIGINAL"):
        centralTrainData, centralTrainLabel = load_dataset('train',mainDir, 'UCI')
        centralTestData, centralTestLabel = load_dataset('eval',mainDir, 'UCI')
        centralTrainLabel = np.squeeze(centralTrainLabel)
        centralTestLabel = np.squeeze(centralTestLabel)
    elif(dataSetName == "REALWORLD_CLIENT"):
        clientData = []
        clientLabel = []
        orientations = hkl.load(mainDir + 'datasetStandardized/REALWORLD_CLIENT/clientsOrientationRW.hkl')
        orientationsNames = ['chest','forearm','head','shin','thigh','upperarm','waist']
        
        dataSetName = 'REALWORLD_CLIENT'
        for i in range(0,clientCount):
            accX = hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/'+str(i)+'/AccX'+dataSetName+'.hkl')
            accY = hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/'+str(i)+'/AccY'+dataSetName+'.hkl')
            accZ = hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/'+str(i)+'/AccZ'+dataSetName+'.hkl')
            gyroX = hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/'+str(i)+'/GyroX'+dataSetName+'.hkl')
            gyroY = hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/'+str(i)+'/GyroY'+dataSetName+'.hkl')
            gyroZ = hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/'+str(i)+'/GyroZ'+dataSetName+'.hkl')
            label = hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/'+str(i)+'/Label'+dataSetName+'.hkl')
            clientData.append(np.dstack((accX,accY,accZ,gyroX,gyroY,gyroZ)))
            clientLabel.append(label)
        
        if(dataConfig == "BALANCED"):
            for i in range (0,clientCount):
                skf = StratifiedKFold(n_splits=5, shuffle=True,random_state = randomSeed)
                skf.get_n_splits(clientData[i], clientLabel[i])
                partitionedData = list()
                partitionedLabel = list()    
                dataIndex = []

                for train_index, test_index in skf.split(clientData[i], clientLabel[i]):
                    partitionedData.append(clientData[i][test_index])
                    partitionedLabel.append(clientLabel[i][test_index])
                    dataIndex.append(test_index)
                    
                clientDataTrain.append((np.vstack((partitionedData[:4]))))
                clientLabelTrain.append((np.hstack((partitionedLabel[:4]))))
                clientDataTest.append((partitionedData[4]))
                clientLabelTest.append((partitionedLabel[4]))
                clientOrientationTrain.append(np.hstack((dataIndex[:4])))
                clientOrientationTest.append(dataIndex[4]) 
        else:
            for i in range (0,clientCount):
                kf = KFold(n_splits=5, shuffle=True,random_state = randomSeed)
                kf.get_n_splits(clientData[i])
                partitionedData = list()
                partitionedLabel = list()    
                for train_index, test_index in kf.split(clientData[i]):
                    partitionedData.append(clientData[i][test_index])
                    partitionedLabel.append(clientLabel[i][test_index])
                clientDataTrain.append((np.vstack((partitionedData[:4]))))
                clientLabelTrain.append((np.hstack((partitionedLabel[:4]))))
                clientDataTest.append((partitionedData[4]))
                clientLabelTest.append((partitionedLabel[4]))
                
        for i in range(0,clientCount):
            clientOrientationTest[i] = orientations[i][clientOrientationTest[i]]
            clientOrientationTrain[i] = orientations[i][clientOrientationTrain[i]]

        centralTrainData = (np.vstack((clientDataTrain)))
        centralTrainLabel = (np.hstack((clientLabelTrain)))

        centralTestData = (np.vstack((clientDataTest)))
        centralTestLabel = (np.hstack((clientLabelTest)))
    else:
        clientData = []
        clientLabel = []

        for i in range(0,clientCount):
            clientData.append(hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/UserData'+str(i)+'.hkl'))
            clientLabel.append(hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/UserLabel'+str(i)+'.hkl'))
            
            
        if(StratifiedSplit and dataSetName == "SHL_128_Time_PreviewLowPass"):
            timePerUser = 3 
            tempData = {new_list: [] for new_list in range(clientCount)}
            tempLabel = {new_list: [] for new_list in range(clientCount)}
            for i in range(0,clientCount):
                startCount = int(i/timePerUser)*timePerUser
                skf = StratifiedKFold(n_splits=timePerUser, shuffle=True, random_state = randomSeed)
                skf.get_n_splits(clientData[i], clientLabel[i])
                for train_index, test_index in skf.split(clientData[i], clientLabel[i]):
                    tempData[startCount].append(clientData[i][test_index])
                    tempLabel[startCount].append(clientLabel[i][test_index])
                    startCount += 1 
            for i in range(0,clientCount):
                clientData[i] =  np.vstack((tempData[i]))
                clientLabel[i] = np.hstack((tempLabel[i]))
            del tempData,tempLabel
        if(dataSetName == "HHAL_DEVICE"):
            orientations = hkl.load(mainDir + 'datasetStandardized/HHAL_DEVICE/deviceIndex.hkl')
            orientationsNames = ['nexus4', 'lgwatch','s3', 's3mini','gear','samsungold']

        for i in range (0,clientCount):
            skf = StratifiedKFold(n_splits=5, shuffle=True,random_state = randomSeed)
            skf.get_n_splits(clientData[i], clientLabel[i])
            partitionedData = list()
            partitionedLabel = list()    
            dataIndex = []

            for train_index, test_index in skf.split(clientData[i], clientLabel[i]):
                partitionedData.append(clientData[i][test_index])
                partitionedLabel.append(clientLabel[i][test_index])
                dataIndex.append(test_index)

            clientDataTrain.append((np.vstack((partitionedData[:4]))))
            clientLabelTrain.append((np.hstack((partitionedLabel[:4]))))
            clientDataTest.append((partitionedData[4]))
            clientLabelTest.append((partitionedLabel[4]))
            clientOrientationTrain.append(np.hstack((dataIndex[:4])))
            clientOrientationTest.append(dataIndex[4]) 
            
        if(dataSetName == "HHAL_DEVICE"):        
            for i in range(0,clientCount):
                clientOrientationTest[i] = orientations[i][clientOrientationTest[i]]
                clientOrientationTrain[i] = orientations[i][clientOrientationTrain[i]]

            
            
        centralTrainData = (np.vstack((clientDataTrain)))
        centralTrainLabel = (np.hstack((clientLabelTrain)))

        centralTestData = (np.vstack((clientDataTest)))
        centralTestLabel = (np.hstack((clientLabelTest)))




    dataReturn = dataHolder
    dataReturn.clientDataTrain = clientDataTrain
    dataReturn.clientLabelTrain = clientLabelTrain
    dataReturn.clientDataTest = clientDataTest
    dataReturn.clientLabelTest = clientLabelTest
    dataReturn.centralTrainData = centralTrainData
    dataReturn.centralTrainLabel = centralTrainLabel
    dataReturn.centralTestData = centralTestData
    dataReturn.centralTestLabel = centralTestLabel
    dataReturn.clientOrientationTrain = clientOrientationTrain
    dataReturn.clientOrientationTest = clientOrientationTest
    dataReturn.orientationsNames = orientationsNames
    dataReturn.clientCount = clientCount
    dataReturn.activityLabels = ACTIVITY_LABEL

    return dataReturn

def plot_learningCurve(history, epochs, filepath):
    # Plot training & validation accuracy values
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'],markevery=[np.argmax(history.history['val_accuracy'])], ls="", marker="o",color="orange")
    plt.plot(epoch_range, history.history['accuracy'],markevery=[np.argmax(history.history['accuracy'])], ls="", marker="o",color="blue")

    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.savefig(filepath+"LearningAccuracy.png", bbox_inches="tight")
    plt.show()
    plt.clf()
    # Plot training & validation loss values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.plot(epoch_range, history.history['loss'],markevery=[np.argmin(history.history['loss'])], ls="", marker="o",color="blue")
    plt.plot(epoch_range, history.history['val_loss'],markevery=[np.argmin(history.history['val_loss'])], ls="", marker="o",color="orange")
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig(filepath+"ModelLoss.png", bbox_inches="tight")
    plt.show()
    plt.clf()
def prepareContrastiveData(dataInput,dataLabel,folds = 2):
    maxValue = 0 
    maxIndex = 0

    classIndexInData = {}
    uniqueLabelsCont = np.unique(dataLabel)
    for i in uniqueLabelsCont:
        classIndexInData[i] = np.asarray(np.where(dataLabel==i)).ravel()
    
    for key, value in classIndexInData.items():
        if(len(value) > maxValue):
            maxValue = len(value)
            maxIndex = key
    targetLength = maxValue
    outputArray = []
    for f in range(folds):
        foldArray = []
        for i in range(len(classIndexInData)):
            if(i!=maxIndex):
                old_list = classIndexInData[i]
                new_list =  list(old_list)
                np.random.shuffle(new_list)
                deficit = maxValue - len(old_list)
                for x in range(deficit):
                    new_list.append(np.random.choice(old_list))
            else:
                new_list = list(classIndexInData[i])
                np.random.shuffle(new_list)
            foldArray.append(new_list)
        outputArray.append(np.asarray(foldArray).T)
        
#     dataInput[np.vstack((outputArray))]
    return  dataInput[np.vstack((outputArray))]

def roundNumber(toRoundNb):
    return round(toRoundNb, 4) * 100
def converTensor(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

def extract_intermediate_model_from_head_model(base_model, intermediate_layer=221):
    input_shape = base_model.layers[intermediate_layer].get_input_shape_at(0)
    layer_input = tf.keras.layers.Input(shape=input_shape)
    x = layer_input
    for layer in base_model.layers[intermediate_layer:]:
        x = layer(x)
    new_model = tf.keras.Model(layer_input, x)
    return new_model

def extract_intermediate_model_from_base_model(base_model, intermediate_layer=7):
    model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[intermediate_layer].output, name=base_model.name + "_layer_" + str(intermediate_layer))
    return model

# model = tf.keras.Model(inputs=base_model.inputs, outputs=[base_model.layers[intermediate_layer-1].output, base_model.output])

def multi_output_model(base_model, intermediate_layer=7):
    model = tf.keras.Model(inputs=base_model.inputs, outputs=[base_model.layers[intermediate_layer-1].output, base_model.output])
    return model