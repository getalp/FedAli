#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pandas as pd
import zipfile
from sklearn.preprocessing import StandardScaler
import hickle as hkl 
import requests 
import urllib.request
from scipy import signal
import tensorflow as tf
import resampy
np.random.seed(0)


# In[2]:


# fomating data to adjust for dataset sensor errors
def formatData(data,dim):
    remainders = data.shape[0]%dim
    max_index = data.shape[0] - remainders
    data = data[:max_index,:]
    new = np.reshape(data, (-1, 128,3))
    return new

# segment data into windows
def segmentData(accData,time_step,step):
#     print(accData.shape)
    segmentAccData = list()
    for i in range(0, accData.shape[0] - time_step,step):
        segmentAccData.append(accData[i:i+time_step,:])
    return np.asarray(segmentAccData)

# load a single file as a numpy array
def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=0,usecols=[2,3,4])
    return dataframe.values
 
# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, filepath='',trainOrEval=0):
    loaded = list()
    for name in filenames:
        data = load_file(filepath + name)
        data = np.asarray(data)
#         print(data.shape)
#         data = segmentData(data,128,64)
        data = np.asarray(data)
        loaded.append(data)
    return loaded

# check if file exist
def isReadableFile(file_path, file_name,flag):
    full_path = file_path + "/" + file_name
    try:
        if not os.path.exists(file_path):
            return False
        elif not os.path.isfile(full_path):
            return False
        elif not os.access(full_path, os.R_OK):
            return False
        else:
            return True
    except IOError as ex:
        print ("I/O error({0}): {1}".format(ex.errno, ex.strerror))
    except Error as ex:
        print ("Error({0}): {1}".format(ex.errno, ex.strerror))
    return False


# stairs down 0 
# stairs Up   1
# jumping     2
# lying       3
# standing    4 
# sitting     5
# running/jogging 6
# Walking     7

# load a dataset group
def load_dataset(group, datasetName='',activity='',orientation='',trainOrEval=0,client = 0):
    filepath = 'dataset/'+datasetName +'/'+ group + '/'
    filenames = list()
    if(isReadableFile('dataset/'+datasetName +'/'+ group, str(client)+'/'+group+'_'+activity+'_'+orientation+'.csv',0)):
        filenames += [str(client)+'/'+group+'_'+activity+'_'+orientation+'.csv']
    if(isReadableFile('dataset/'+datasetName +'/'+ group, str(client)+'/'+group+'_'+activity+'_2_'+orientation+'.csv',1)):
        filenames += [str(client)+'/'+group+'_'+activity+'_2_'+orientation+'.csv']
    if(isReadableFile('dataset/'+datasetName +'/'+ group, str(client)+'/'+group+'_'+activity+'_3_'+orientation+'.csv',1)):
        filenames += [str(client)+'/'+group+'_'+activity+'_3_'+orientation+'.csv']
    X = load_group(filenames, filepath,trainOrEval)
    return X

# download function for datasets
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


# In[3]:


# definign activities and orientations of REALWORLD dataset
# activities = ['climbingdown','climbingup','jumping','lying','running','sitting','standing','walking'] 
activities = ['climbingdown','climbingup','running','sitting','standing','walking','lying','jumping']
# activities = ['standing','sitting','walking','climbingup','climbingdown'] 

orientations = ['chest','forearm','head','shin','thigh','upperarm','waist']
# orientations = ['waist']


# In[4]:


orientationKeyMap = dict() 
for index,value in enumerate(orientations):
    orientationKeyMap[value] = index


# In[5]:


# download and unzipping dataset
os.makedirs('dataset',exist_ok=True)
print("downloading...")            
data_directory = os.path.abspath("dataset/realworld2016_dataset.zip")
if not os.path.exists(data_directory):
    download_url("http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip",data_directory)
    print("download done")
else:
    print("dataset already downloaded")
    
data_directory2 = os.path.abspath("dataset/realworld2016_dataset")
if not os.path.exists(data_directory2): 
    print("extracting data")
    with zipfile.ZipFile(data_directory, 'r') as zip_ref:
        zip_ref.extractall(os.path.abspath(data_directory2))
    print("data extracted in " + data_directory2)
else:
    print("Data already extracted in " + data_directory2)


# In[6]:


# # unzipping REALWORLD dataset
for id in range(1,16):
    id = str(id)
    for activity in activities:
        for sensor in ["acc","gyr"]:
            dirName = sensor
            if(sensor == "gyr"):
                dirName = "Gyroscope"
            with zipfile.ZipFile('dataset/realworld2016_dataset/proband'+id+'/data/'+sensor+'_'+str(activity)+'_csv.zip', 'r') as zip_ref:
                os.makedirs('dataset/REALWORLD/'+dirName+'/'+id, exist_ok=True)
                zip_ref.extractall('dataset/REALWORLD/'+dirName+'/'+id)

            for i in range (1,4):
                if os.path.exists('dataset/REALWORLD/'+dirName+'/'+id+'/'+sensor+'_'+str(activity)+'_'+str(i)+'_csv.zip'):
                    with zipfile.ZipFile('dataset/REALWORLD/'+dirName+'/'+id+'/'+sensor+'_'+str(activity)+'_'+str(i)+'_csv.zip', 'r') as zip_ref:
                        zip_ref.extractall('dataset/REALWORLD/'+dirName+'/'+id)
                    os.remove('dataset/REALWORLD/'+dirName+'/'+id+'/'+sensor+'_'+str(activity)+'_'+str(i)+'_csv.zip') 
            if os.path.exists('dataset/REALWORLD/'+dirName+'/'+id+'/'+sensor+'_'+str(activity)+'_csv.zip'):
                with zipfile.ZipFile('dataset/REALWORLD/'+dirName+'/'+id+'/'+sensor+'_'+str(activity)+'_csv.zip', 'r') as zip_ref:
                    zip_ref.extractall('dataset/REALWORLD/'+dirName+'/'+id)
                os.remove('dataset/REALWORLD/'+dirName+'/'+id+'/'+sensor+'_'+str(activity)+'_csv.zip') 


# In[7]:


def downSampleLowPass(toDownSampleData,factor):
    accX = signal.decimate(toDownSampleData[:,0],factor)
    accY = signal.decimate(toDownSampleData[:,1],factor)
    accZ = signal.decimate(toDownSampleData[:,2],factor)
    return np.dstack((accX,accY,accZ)).squeeze()


# In[9]:


# Reading and processing all data
xAccListClient = list()
xGyrListClient = list()
yListClient = list()

xAccListWatchClient = list()
xGyrListWatchClient = list()
yListWatchClient = list()
nbAnomalies = 0 
for k in range(1,16):
    for orientation in orientations:
        xAccList = list()
        xGyrList = list()
        yList = list()
        for activity in activities:
            tempAcc = load_dataset('acc','REALWORLD',activity,orientation,0,k)
            tempGyro = load_dataset('Gyroscope','REALWORLD',activity,orientation,0,k)
            orientationLength = 0 
            for i in range(0, len(tempAcc)):  
                accDataLength = len(tempAcc[i])
                gyroDataLength = len(tempGyro[i])
                difference = accDataLength - gyroDataLength
                differenceAbs = abs(difference)


                differenceAcc = accDataLength - gyroDataLength
                differenceGyro = gyroDataLength - accDataLength
                if(differenceGyro > 100):
                    print("Client Number "+str(k) +" Activity : "+str(activity) + " Orientation :"+str(orientation))
                    print("Disalignment of:" +str(differenceAbs) + " found")
                    print("Acc data: "+str(accDataLength))
                    print("Gyro data: "+str(gyroDataLength))
                    tempGyro[i] = resampy.resample(tempGyro[i], gyroDataLength, accDataLength,axis = 0)
                elif(differenceAcc > 100):
                    print("Client Number "+str(k) +" Activity : "+str(activity) + " Orientation :"+str(orientation))
                    print("Disalignment of:" +str(differenceAbs) + " found")
                    print("Acc data: "+str(accDataLength))
                    print("Gyro data: "+str(gyroDataLength))
                    tempGyro[i] = resampy.resample(tempAcc[i], accDataLength, gyroDataLength,axis = 0)
                tempAcc[i] = segmentData(tempAcc[i],128,64)
                tempGyro[i] = segmentData(tempGyro[i],128,64)
    
                accDataLength = len(tempAcc[i])
                gyroDataLength = len(tempGyro[i])
    
                difference = accDataLength - gyroDataLength
                differenceAbs = abs(difference)
                if(differenceAbs <= 10):
                    toAddShape = 0
                    if(difference > 0):
                        maxIndex = accDataLength-differenceAbs
                        xAccList.append(tempAcc[i][:maxIndex,:])
                        xGyrList.append(tempGyro[i])
                        toAddShape = tempGyro[i].shape[0]
                        yList.append(np.full((toAddShape), activities.index(activity)))   
                    else:
                        maxIndex = gyroDataLength-differenceAbs
                        xAccList.append(tempAcc[i])
                        xGyrList.append(tempGyro[i][:maxIndex,:])
                        toAddShape = tempAcc[i].shape[0]
                        yList.append(np.full((toAddShape), activities.index(activity)))
                else:
                    print("Difference: "+str(differenceAbs))
                    print("Big misalignment, skipping")
        if(orientation == 'forearm'):
            xAccListWatchClient.append(np.vstack((xAccList)))
            xGyrListWatchClient.append(np.vstack((xGyrList)))
            yListWatchClient.append(np.hstack((yList)))
        else:
            xAccListClient.append(np.vstack((xAccList)))
            xGyrListClient.append(np.vstack((xGyrList)))
            yListClient.append(np.hstack((yList)))
            



# In[10]:


def normalizedData(accData, gyroData):
    allAcc = np.vstack((accData))
    allGyro = np.vstack((gyroData))
    
    # Calculating features
    meanAcc = np.mean(allAcc)
    stdAcc = np.std(allAcc)
    meanGyro = np.mean(allGyro)
    stdGyro = np.std(allGyro)


    numpyAccData = np.asarray(accData, dtype=object)
    numpyGyroData = np.asarray(gyroData, dtype=object)
    normalizedAcc = (numpyAccData - meanAcc)/stdAcc
    normalizedGyro = (numpyGyroData - meanGyro)/stdGyro

    subjectData = np.asarray([np.dstack((normAcc,normGyro))for normAcc,normGyro in zip(normalizedAcc,normalizedGyro)],dtype=object)
    return subjectData


# In[11]:


subjectData = normalizedData(xAccListClient,xGyrListClient)


# In[12]:


subjectWatchData = normalizedData(xAccListWatchClient,xGyrListWatchClient)


# In[13]:


yListClient = np.asarray(yListClient, dtype=object)
yListWatchClient = np.asarray(yListWatchClient, dtype=object)


# In[14]:


allLabels = np.hstack((yListClient,yListWatchClient))
allData =  np.hstack((subjectData,subjectWatchData))


# In[15]:


dataName = 'RealWorld'
os.makedirs('processedDatasets/'+dataName, exist_ok=True)
hkl.dump(allData,'processedDatasets/'+dataName+ '/clientsData.hkl' )
hkl.dump(allLabels,'processedDatasets/'+dataName+ '/clientsLabel.hkl' )


# In[ ]:




