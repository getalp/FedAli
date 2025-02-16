if __name__ == "__main__":
    #!/usr/bin/env python
    # coding: utf-8

    # In[ ]:


    # if __name__ == "__main__":


    # In[ ]:


    from tensorflow.keras.optimizers import SGD
    import tensorflow as tf 
    import os
    import numpy as np
    import random
    import logging
    import time
    import concurrent.futures
    import hickle as hkl 
    import copy
    import matplotlib.pyplot as plt
    import csv
    import pandas as pd
    import seaborn as sns
    import gc
    import sklearn.manifold
    from tensorflow.python.keras import backend as K
    from sklearn.metrics import f1_score
    from sklearn.utils import class_weight
    from utils import LinearLearningRateScheduler,projectTSNE,load_data,load_checkpoint,get_available_gpus,get_available_cpus,extract_intermediate_model_from_base_model,converTensor,prepareContrastiveData,multi_output_model
    import argparse
    import __main__ as main
    from multiprocessing import get_context,Value,Array,Manager,set_start_method
    from ctypes import c_char_p  
    from distutils.util import strtobool
    from os import listdir


    # In[ ]:


    import model 
    import fed_util
    from sklearn.cluster import KMeans
    import mae_model
    import alp_model


    # In[ ]:


    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


    # In[ ]:


    algorithm = "FEDAVG"
    # FEDALI, FEDAVG, FEDPROTO
    # MOON,FEDAVG,FEDPROX, FEDPAC

    dataSetName = 'HHAR'

    input_shape = (128,6)

    # Show training verbose: 0,1
    showTrainVerbose = 0

    # input window size 
    segment_size = 128

    # input channel count
    num_input_channels = 6

    GeneralizationTest = True

    clientLearningRate =  1e-4

    adaptiveLearningRate =  clientLearningRate * 0.5

    batch_size = 64

    localEpoch = 5

    randomSeed = 1

    communicationRound = 200

    parallelInstancesCPU = 4

    parallelInstancesGPU = 4

    mu = 1.0

    projection_dim = 192 

    architecture = "HART"

    prototypeNum = 256

    clusterMethod = "kmean"

    loadPretrain = True

    initial_lr = 0.999
    end_lr = 0.999
    influenceFactor = 0.2
    usePersonalPrototype = False

    singleUpdate = False

    useGLU = True


    # In[ ]:


    prototypeLayers = [2048,1024,512,256,128,64]


    # In[ ]:


    nbOfBlocks = 6 


    # In[ ]:


    if(algorithm == "FEDPROX"):
        mu = 0.2
    else:
        mu = 1.0
    # MOON,FEDAVG,FEDALI,FEDPROX


    # In[ ]:


    def is_interactive():
        return not hasattr(main, '__file__')
    def add_fit_args(parser):
        parser.add_argument('--dataset', type=str, default=dataSetName, 
            help='Dataset')  
        parser.add_argument('--algorithm', type=str, default=algorithm, 
            help='Algorithm')
        parser.add_argument('--mu', type=float, default=mu, 
            help='Mu')  
        parser.add_argument('--parallelInstancesGPU', type=int, default=parallelInstancesGPU, 
            help='Number of tasks per GPU')  
        parser.add_argument('--localEpoch', type=int, default=localEpoch, 
            help='Number of tasks per GPU')  
        parser.add_argument('--clientLearningRate', type=float, default=clientLearningRate, 
            help='Number of tasks per GPU')  
        parser.add_argument('--influenceFactor', type=float, default=influenceFactor, 
            help='Number of tasks per GPU')  
        parser.add_argument('--initial_lr', type=float, default=initial_lr, 
            help='Number of tasks per GPU')  
        parser.add_argument('--clusterMethod', type=str, default=clusterMethod, 
            help='clusterMethod')  
        parser.add_argument('--loadPretrain', type=lambda x: bool(strtobool(x)), default=loadPretrain,
            help='loadPretrain')  
        parser.add_argument('--communicationRound', type=int, default=communicationRound, 
            help='Number of communicationRound')  
        parser.add_argument('--prototypeNum', type=int, default=prototypeNum, 
            help='Number of prototypeNum')  
        parser.add_argument('--usePersonalPrototype', type=lambda x: bool(strtobool(x)), default=usePersonalPrototype,
            help='usePersonalPrototype')  
        parser.add_argument('--useGLU', type=lambda x: bool(strtobool(x)), default=useGLU,
            help='usePersonalPrototype')  
        parser.add_argument('--singleUpdate', type=lambda x: bool(strtobool(x)), default=singleUpdate,
            help='usePersonalPrototype') 
        args = parser.parse_args()
        return args

    # clusterMethod

    if not is_interactive():
        args = add_fit_args(argparse.ArgumentParser(description='Federated Learning Experiments'))
        dataSetName = args.dataset
        algorithm = args.algorithm
        localEpoch = args.localEpoch
        clientLearningRate = args.clientLearningRate
        mu = args.mu
        clusterMethod = args.clusterMethod
        parallelInstancesGPU = args.parallelInstancesGPU
        loadPretrain = args.loadPretrain
        influenceFactor = args.influenceFactor
        communicationRound = args.communicationRound
        prototypeNum = args.prototypeNum
        usePersonalPrototype = args.usePersonalPrototype
        useGLU = args.useGLU
        initial_lr = args.initial_lr
        singleUpdate = args.singleUpdate


    # In[ ]:


    prototype_linear_scheduler = LinearLearningRateScheduler(initial_lr,end_lr, 100)


    # In[ ]:


    prototypeNum = prototypeLayers[0]


    # In[ ]:


    # classifierEpoch
    architectureType = str(algorithm)+'_'+str(architecture)
    if(communicationRound < 20):
        architectureType =  "Tests/"+str(architectureType)

    if(loadPretrain):
        architectureType = architectureType +"_pretrain"


    architectureType = architectureType +'_clientLearningRate'+str(clientLearningRate)    

    if(algorithm == 'FEDALI'):
        architectureType = architectureType +'_influenceFactor_'+str(influenceFactor)
        architectureType = architectureType +'_initialLr_'+str(initial_lr)

        if(usePersonalPrototype):
            architectureType = architectureType +'_personalPrototype'
        if(singleUpdate):
            architectureType = architectureType +'_single'
        architectureType = architectureType +"_v4"

    if(algorithm == 'MOON' or algorithm == 'FEDPROX'):
        architectureType = architectureType +"_mu_"+str(mu)


    mainDir = ''
    folderName = 'results'
    filepath = mainDir + folderName+'/'+architectureType+'/'+dataSetName+'/'
    os.makedirs(filepath, exist_ok=True)

    # # # # uncomment for testing
    # if(os.path.exists(filepath+'checkpoint.hkl')):
    #     os.remove(filepath+'checkpoint.hkl')
    # # # # keep on for testing

    bestModelPath = filepath + 'bestModels/'
    os.makedirs(bestModelPath, exist_ok=True)

    trainModelPath = filepath + 'trainModels/'
    os.makedirs(trainModelPath, exist_ok=True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


    np.random.seed(randomSeed)
    tf.keras.utils.set_random_seed(randomSeed)
    tf.random.set_seed(randomSeed)
    random.seed(randomSeed)
    checkpointProp = load_checkpoint(filepath)


    # In[ ]:


    dataDirectory = './Datasets/FL_Clients/'
    trainDataDirectory = dataDirectory +'trainData/'
    testDataDirectory = dataDirectory +'testData/'


    # In[ ]:


    # load the data
    if(dataSetName =='Combined'):
        datasetNames = ['HHAR','RealWorld']
        
        clientDataTrain = []
        clientLabelTrain = []
        clientDataTest =  []
        clientLabelTest = []
        datasetClientCounts = []
        
        for datasetName in datasetNames:
            with open(trainDataDirectory + datasetName + '_data.hkl', 'rb') as f:
                clientDataTrain.append(hkl.load(f))
            with open(trainDataDirectory + datasetName + '_aligned_label.hkl', 'rb') as f:
                clientLabelTrain.append(hkl.load(f))
            with open(testDataDirectory + datasetName + '_data.hkl', 'rb') as f:
                clientDataTest.append(hkl.load(f))
            with open(testDataDirectory + datasetName + '_aligned_label.hkl', 'rb') as f:
                clientLabelTest.append(hkl.load(f))
            datasetClientCounts.append(len(clientLabelTest[-1]))
        clientDataTrain = np.hstack((clientDataTrain))
        clientLabelTrain = np.hstack((clientLabelTrain))
        clientDataTest = np.hstack((clientDataTest))
        clientLabelTest = np.hstack((clientLabelTest))
        with open(dataDirectory + 'labelNames/Combined.hkl', 'rb') as f:
            ACTIVITY_LABEL = hkl.load(f)
        activityCount = len(ACTIVITY_LABEL)
    else:
        with open(trainDataDirectory + str(dataSetName) + '_data.hkl', 'rb') as f:
            clientDataTrain = hkl.load(f)
        with open(trainDataDirectory + str(dataSetName) + '_label.hkl', 'rb') as f:
            clientLabelTrain = hkl.load(f)
        with open(testDataDirectory + str(dataSetName) + '_data.hkl', 'rb') as f:
            clientDataTest = hkl.load(f)
        with open(testDataDirectory + str(dataSetName) + '_label.hkl', 'rb') as f:
            clientLabelTest = hkl.load(f)
        with open(dataDirectory + 'labelNames/' + str(dataSetName) + '.hkl', 'rb') as f:
            ACTIVITY_LABEL = hkl.load(f)

        activityCount = len(ACTIVITY_LABEL)


    # In[ ]:


    clientCount = clientDataTest.shape[0]
    # to test/develop, you can set clients count manually to a lower number eg clientCount = 2 


    # In[ ]:


    for index,clientData in enumerate(clientDataTrain):
        clientDataTrain[index] = clientData.astype('float32')
    for index,clientData in enumerate(clientDataTest):
        clientDataTest[index] = clientData.astype('float32')


    # In[ ]:


    centralTrainLabel = np.vstack((clientLabelTrain))
    centralTestData = np.vstack((clientDataTest))
    centralTestLabel = np.vstack((clientLabelTest))


    # In[ ]:


    availableGPUPOOl = get_available_gpus()

    resourcePool = availableGPUPOOl 
    if(len(resourcePool) == 0):
        resourcePool = availableCPUPOOl * parallelInstancesCPU
    else:
        resourcePool = resourcePool * parallelInstancesGPU
        GPUPool = [availableGPUPOOl[client%len(availableGPUPOOl)] for client in range(clientCount)]
        GPUPoolIndex = [client%len(availableGPUPOOl) for client in range(clientCount)]

    # limits the amount of workers from more than neccesary
    if(len(resourcePool) > clientCount):
        resourcePool = resourcePool[:clientCount]
    modelPool = np.arange(len(resourcePool)).tolist()


    # In[ ]:


    # client models test againts own test-set
    trainLossHistory = checkpointProp['trainLossHistory'] 
    trainAccHistory = checkpointProp['trainAccHistory'] 
    testLossHistory = checkpointProp['testLossHistory'] 
    testAccHistory = checkpointProp['testAccHistory']

    stdTrainLossHistory = checkpointProp['stdTrainLossHistory']
    stdTrainAccHistory = checkpointProp['stdTrainAccHistory']
    stdTestLossHistory = checkpointProp['stdTestLossHistory']
    stdTestAccHistory = checkpointProp['stdTestAccHistory']

    clientTestLossHistory = checkpointProp['clientTestLossHistory']
    clientTestAccHistory = checkpointProp['clientTestAccHistory']

    clientStdTestLossHistory = checkpointProp['clientStdTestLossHistory']
    clientStdTestAccHistory = checkpointProp['clientStdTestAccHistory']
    roundTrainingTime = checkpointProp['roundTrainingTime']

    # server test againts all test-set

    globalTestLossHistory = checkpointProp['globalTestLossHistory'] 
    globalTestAccHistory = checkpointProp['globalTestAccHistory']


    globalTestAlignZeroLossHistory = checkpointProp['globalTestAlignZeroLossHistory'] 
    globalTestAlignZeroAccHistory = checkpointProp['globalTestAlignZeroAccHistory']

    meanHistoryDist = checkpointProp['meanHistoryDist']
    stdHistoryDist = checkpointProp['stdHistoryDist']

    meanRoundLayerHistory = checkpointProp['meanRoundLayerHistory']
    stdRoundLayerHistory = checkpointProp['stdRoundLayerHistory'] 

    meanRoundGeneralLayerHistory = checkpointProp['meanRoundGeneralLayerHistory']
    stdRoundGeneralLayerHistory = checkpointProp['stdRoundGeneralLayerHistory']

    bestModelRound = checkpointProp['bestModelRound']
    currentAccuracy = checkpointProp['currentAccuracy']
    serverCurrentAccuracy = checkpointProp['serverCurrentAccuracy']
    serverbestModelRound = checkpointProp['serverbestModelRound']
    bestServerModelWeights = checkpointProp['bestServerModelWeights']
    best_local_weights = checkpointProp['best_local_weights']
    totalEmission = checkpointProp['totalEmission'] 
    currentGeneralizationAccuracy = checkpointProp['currentGeneralizationAccuracy']


    adaptiveLoss = checkpointProp['adaptiveLoss'] 
    adaptiveLossStd = checkpointProp['adaptiveLossStd'] 

    prototypeStabilityEpoch =  checkpointProp['prototypeStabilityEpoch'] 
    previousPrototype =  checkpointProp['previousPrototype']


    # In[ ]:


    # initialization for asynchronous client training, client selection
    roundEnd = []
    trainPool = range(clientCount)
    startRound = checkpointProp["CommunicationRound"]


    # In[ ]:


    trainingInit = os.path.exists(filepath+'serverWeights.h5') 

    if(algorithm == "FEDALI"):
        pretraindir = './pretrained_models/MAE_ALP_FE.h5'
        serverModel = model.createAndLoadHART_ALP(prototypeLayers,
                                                  activityCount,
                                                  loadPretrain = loadPretrain, 
                                                  pretrain_dir = pretraindir,
                                                  useGLU = useGLU,
                                                  influenceFactor = influenceFactor,
                                                  singleUpdate = singleUpdate)

    else:
        pretraindir = './pretrained_models/MAE_FE.h5'
        serverModel = model.createAndLoadHART(activityCount,
                                              loadPretrain = loadPretrain, 
                                              pretrain_dir = pretraindir)

    if(trainingInit):
        print("Weights Found, Loading Server Model Weights")
        # serverModel.load_weights(filepath+'serverWeights.h5')
        serverModel.load_weights(filepath+'serverWeights.h5')
    else:
        serverModel.save_weights(filepath+'serverWeights.h5')

    # # optmizer is random, because server model will not be used for training, just evaluating
    serverModel.compile(optimizer=tf.keras.optimizers.SGD(0.005),loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])   


    # In[ ]:


    def checkPointProgress():
        # Initialization of metrics during training
        checkpointProp["SimCLRFinished"] = True

        checkpointProp["CommunicationRound"] = roundNum + 1

        checkpointProp['trainLossHistory'] =  trainLossHistory
        checkpointProp['trainAccHistory'] =  trainAccHistory
        checkpointProp['testLossHistory'] =  testLossHistory
        checkpointProp['testAccHistory'] = testAccHistory
        checkpointProp['roundTrainingTime'] = roundTrainingTime 

        checkpointProp['adaptiveLoss'] = adaptiveLoss
        checkpointProp['adaptiveLossStd'] = adaptiveLossStd

        checkpointProp['stdTestLossHistory'] = stdTestLossHistory
        checkpointProp['stdTestAccHistory'] = stdTestAccHistory

        # client models test againts all test-set

        checkpointProp['clientTestLossHistory'] = clientTestLossHistory
        checkpointProp['clientTestAccHistory'] = clientTestAccHistory

        checkpointProp['clientStdTestLossHistory'] = clientStdTestLossHistory
        checkpointProp['clientStdTestAccHistory'] = clientStdTestAccHistory

        # server test againts all test-set

        checkpointProp['globalTestLossHistory'] =  globalTestLossHistory
        checkpointProp['globalTestAccHistory'] = globalTestAccHistory
        
        checkpointProp['meanHistoryDist'] = meanHistoryDist
        checkpointProp['stdHistoryDist'] = stdHistoryDist

        checkpointProp['meanRoundLayerHistory'] = meanRoundLayerHistory
        checkpointProp['stdRoundLayerHistory'] =  stdRoundLayerHistory

        checkpointProp['meanRoundGeneralLayerHistory'] = meanRoundGeneralLayerHistory
        checkpointProp['stdRoundGeneralLayerHistory'] = stdRoundGeneralLayerHistory

        checkpointProp['bestModelRound'] = bestModelRound
        checkpointProp['currentAccuracy'] = currentAccuracy
        checkpointProp['currentGeneralizationAccuracy'] = currentGeneralizationAccuracy

        checkpointProp['serverCurrentAccuracy'] = serverCurrentAccuracy
        checkpointProp['serverbestModelRound'] = serverbestModelRound
        checkpointProp['bestServerModelWeights'] = bestServerModelWeights
        checkpointProp['best_local_weights'] = best_local_weights
        checkpointProp['totalEmission'] = totalEmission

        checkpointProp['prototypeStabilityEpoch'] = prototypeStabilityEpoch
        checkpointProp['previousPrototype'] = previousPrototype

        checkpointProp['globalTestAlignZeroLossHistory'] = globalTestAlignZeroLossHistory
        checkpointProp['globalTestAlignZeroAccHistory'] = globalTestAlignZeroAccHistory
        hkl.dump(checkpointProp,filepath+'checkpoint.hkl')



    # In[ ]:


    allTrainDataSize = np.float32(len(centralTrainLabel))
    local_coeffs = {}
    for i in range(0,clientCount):
        local_coeffs[i] = np.float32(len(clientLabelTrain[i])) / allTrainDataSize
    del allTrainDataSize


    # In[ ]:


    def limit_memory():
        """ Release unused memory resources. Force garbage collection """
        K.clear_session()
        tf.compat.v1.keras.backend.get_session().close()
        tf.compat.v1.reset_default_graph()
        K.set_session(tf.compat.v1.Session())
        gc.collect()


    # In[ ]:


    def enderOutputIndexSearch(model, layerSearchName = "pooling"):
        representationLayer = 0
        for i, layer in enumerate(serverModel.layers):
            layer_name = layer.name
            if layerSearchName in layer.name:
                representationLayer = i
                break
        if(representationLayer == 0):
            raise Exception("Unrecognized architecture, Please manually set the 'embedLayerIndex' variable to the layer index of the encoder's output")
        return representationLayer


    # In[ ]:


    embedLayerIndex = enderOutputIndexSearch(serverModel,layerSearchName = 'pooling')
    layerCount = len(serverModel.get_weights())


    # In[ ]:


    manager = Manager()
    context =  get_context('spawn')

    if(loadPretrain):
        clientsEmbedLayer = embedLayerIndex - 1
        # -1 because the input layer doesn't show in model.layer with the pre-trained model declared at the client, but shows on the server
    else:
        clientsEmbedLayer = embedLayerIndex


    embedLayerIndexTemp = Value('i', clientsEmbedLayer, lock=False)
    segment_sizeTemp = Value('i', segment_size, lock=False)
    num_input_channelsTemp = Value('i', num_input_channels, lock=False)
    activityCountTemp = Value('i', activityCount, lock=False)
    showTrainVerboseTemp = Value('i', showTrainVerbose, lock=False)
    clientLearningRateTemp = Value('d', clientLearningRate, lock=False)
    batch_sizeTemp = Value('i', batch_size, lock=False)
    localEpochTemp = Value('i', localEpoch, lock=False)
    centralTestDataTemp = Array('d',centralTestData.flatten(),lock=False)
    centralTestLabelTemp = Array('d',tf.reshape(centralTestLabel, (-1)),lock=False)
    generalizationTestTemp = Value('i',True,lock=False)
    GPULockIndexTemp = Array('i',GPUPoolIndex,lock=False)
    filepathTemp = manager.Value(c_char_p, filepath,lock=False)
    muTemp = Value('d', mu, lock=False)
    client_ids = context.Value('i', -1)


    shared_vars = (embedLayerIndexTemp,
                   segment_sizeTemp,
                   num_input_channelsTemp,
                   activityCountTemp,
                   showTrainVerboseTemp,
                   clientLearningRateTemp,
                   batch_sizeTemp,
                   localEpochTemp,
                   centralTestDataTemp,
                   centralTestLabelTemp,
                   generalizationTestTemp,
                   filepathTemp,
                   muTemp)


    # In[ ]:





    # In[ ]:


    if(algorithm == 'FEDALI'):
        layerIndexTracker = 0
        layerAdaptIndex = []
        alignLayerIndex = []
        layerSearchName = "alignment"
        if(loadPretrain):
            maeLayerIndex = None
            blockSearchName = "mae_encoder"
            maeEncoderLayerIndex = 0
            for i, layer in enumerate(serverModel.layers):
                layer_name = layer.name
                num_params = len(layer.get_weights())
                if blockSearchName in layer.name:
                    maeLayerIndex = i
                    maeEncoderLayerIndex = layerIndexTracker
                    break
                layerIndexTracker+= num_params
            

            # this is very hard coded for HART with the adaptive layer.
            layerAdaptIndex = []
            layerSearchName = "alignment"
            
            adaptLayerLocation = []
            for i, layer in enumerate(serverModel.layers[3].layers):
                layer_name = layer.name
                num_params = len(layer.get_weights())
                if layerSearchName in layer.name:
                    adaptLayerLocation.append(i)
                    print("Index :"+str(i))
                    alignLayerIndex.append(i)
                    if(useGLU):
                        print(layerIndexTracker + 2)
                        layerAdaptIndex.append(layerIndexTracker + 2)
                    else:
                        print(layerIndexTracker)
                        layerAdaptIndex.append(layerIndexTracker)
                layerIndexTracker+= num_params
        else:
            
            adaptLayerLocation = []
            for i, layer in enumerate(serverModel.layers):
                layer_name = layer.name
                num_params = len(layer.get_weights())
                if layerSearchName in layer.name:
                    adaptLayerLocation.append(i)
                    print("Index :"+str(i))
                    alignLayerIndex.append(i)
                    if(useGLU):
                        print(layerIndexTracker + 2)
                        layerAdaptIndex.append(layerIndexTracker + 2)
                    else:
                        print(layerIndexTracker)
                        layerAdaptIndex.append(layerIndexTracker)
                layerIndexTracker+= num_params
        
        
        adaptLayerLocation = np.repeat(np.expand_dims(adaptLayerLocation, axis=0), clientCount, axis=0)
        prototypeLayersDistribute = np.tile(prototypeLayers,(clientCount, 1))
        
        globalPrototypeIndex = [localIndex+1 for localIndex in layerAdaptIndex]


        # can delete this later
        localPrototypeDir = filepath + 'clientsLocalPrototypes.hkl'
        clientsLocalPrototypes = []
        if(os.path.exists(localPrototypeDir)):
            clientsLocalPrototypes = hkl.load(localPrototypeDir)
            print("Local prototypes found, loading them.")
        
        else:
            for clientIndex in range(clientCount):
                clientLocalPrototype = [serverModel.get_weights()[localPrototypeIndex] for localPrototypeIndex in layerAdaptIndex]
                clientsLocalPrototypes.append(np.asarray(clientLocalPrototype,dtype=object))
            print("Local prototypes not found, generating new ones")
            hkl.dump(clientsLocalPrototypes,localPrototypeDir)

    elif(algorithm == 'MOON'):
        prevModelPath = filepath + 'prevModels/'
        os.makedirs(prevModelPath, exist_ok=True)

        # initalize previous model for first com round
        if(not os.path.exists(prevModelPath+'clientModel0.hkl')):
            clientPrevModelDir = []
            for clientIdx in range(clientCount):
                prevClientPath = prevModelPath+'clientModel'+str(clientIdx)+'.h5'
                clientPrevModelDir.append(prevClientPath)
                serverModel.save_weights(prevClientPath)
                # hkl.dump(local_weights[clientIdx],prevClientPath )

        


    # In[ ]:


    loadPretrains = np.tile(loadPretrain,clientCount)


    # In[ ]:


    if(algorithm == "FEDPROTO"):
        os.makedirs(filepath + 'clientModels/', exist_ok=True)
        if(os.path.exists(filepath+'globalPrototypes.hkl')):
            globalPrototype = hkl.load(filepath+'globalPrototypes.hkl')
        else:
            globalPrototype = tf.Variable(tf.random.normal((activityCount,projection_dim)),trainable= False)
    elif(algorithm == "FEDPAC"):
        os.makedirs(filepath + 'clientModels/', exist_ok=True)
        if(os.path.exists(filepath+'globalPrototypes.hkl')):
            globalPrototype = hkl.load(filepath+'globalPrototypes.hkl')
        else:
            globalPrototype = None
    elif(algorithm == "FEDPER"):
        os.makedirs(filepath + 'clientModels/', exist_ok=True)


    # In[ ]:


    # Federated learning training
    start_time = time.time()
    for roundNum in range(startRound,communicationRound):
        limit_memory()
        roundStartTime = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(resourcePool),
                                                mp_context=context,
                                                initializer=fed_util.set_global, 
                                                initargs=(shared_vars,GPULockIndexTemp,client_ids)
                                               ) as executor:
            
            if(algorithm == 'FEDALI'):
                prototypeDecay = np.tile(prototype_linear_scheduler(roundNum),(clientCount))
                # we do use the global prototype in the 1st communication round
                if(roundNum == 0):
                    influenceFactors = np.tile(0.0,clientCount)
                else:    
                    influenceFactors = np.tile(influenceFactor,clientCount)
                usePersonalPrototypes = np.tile(usePersonalPrototype,clientCount)

                useGLUs = np.tile(useGLU,clientCount)
                singleUpdates = np.tile(singleUpdate,clientCount)
                comRoundResults = [x for x in executor.map(fed_util.fedAli_global_Trainer, 
                                   trainPool,
                                   clientDataTrain, 
                                   clientLabelTrain,
                                   clientDataTest,
                                   clientLabelTest,
                                   prototypeLayersDistribute,
                                   clientsLocalPrototypes,
                                   adaptLayerLocation,
                                   loadPretrains,
                                   prototypeDecay,
                                   influenceFactors,
                                   usePersonalPrototypes,
                                   useGLUs,
                                   singleUpdates
                                  )]
            elif(algorithm == 'FEDPROX'):
                comRoundResults = [x for x in executor.map(fed_util.fedProx_Trainer, 
                                       trainPool,
                                       clientDataTrain, 
                                       clientLabelTrain,
                                       clientDataTest,
                                       clientLabelTest,
                                       loadPretrains
                                      )]
            elif(algorithm == 'MOON'):
                comRoundResults = [x for x in executor.map(fed_util.Moon_Trainer, 
                                       trainPool,
                                       clientDataTrain, 
                                       clientLabelTrain,
                                       clientDataTest,
                                       clientLabelTest,
                                       clientPrevModelDir,
                                       loadPretrains
                                      )]
            elif(algorithm == 'FEDPROTO'):
                globalPrototypes = np.tile(globalPrototype,(clientCount,1,1))
                comRoundResults = [x for x in executor.map(fed_util.fedProto_Trainer, 
                                       trainPool,
                                       clientDataTrain, 
                                       clientLabelTrain,
                                       clientDataTest,
                                       clientLabelTest,
                                       globalPrototypes,
                                       loadPretrains
                                      )]
            elif(algorithm == 'FEDPER'):
                comRoundResults = [x for x in executor.map(fed_util.fedPer_Trainer, 
                                       trainPool,
                                       clientDataTrain, 
                                       clientLabelTrain,
                                       clientDataTest,
                                       clientLabelTest,
                                       loadPretrains
                                      )]
            elif(algorithm == 'FEDAVG'):
                comRoundResults = [x for x in executor.map(fed_util.fedAvg_Trainer, 
                                       trainPool,
                                       clientDataTrain, 
                                       clientLabelTrain,
                                       clientDataTest,
                                       clientLabelTest,
                                       loadPretrains,
                                      )]
            elif(algorithm == 'FEDPAC'):
                if(globalPrototype is None):
                    globalPrototypes = np.tile(globalPrototype,(clientCount))
                else:
                    globalPrototype = tf.convert_to_tensor(globalPrototype, dtype=tf.float32)
                    globalPrototypes = np.tile(globalPrototype,(clientCount,1,1))
                comRoundResults = [x for x in executor.map(fed_util.fedPac_Trainer, 
                                       trainPool,
                                       clientDataTrain, 
                                       clientLabelTrain,
                                       clientDataTest,
                                       clientLabelTest,
                                       globalPrototypes,
                                       loadPretrains
                                      )]
            else:
                raise Exception("Unrecognized strategy")

        comRoundResults = np.asarray(comRoundResults, dtype=object)
        local_weights = comRoundResults[:,1]
        aggregationVars =  comRoundResults[:,2]
        trainAcc = comRoundResults[:,3]
        trainLoss = comRoundResults[:,4]
        testAcc = comRoundResults[:,5]
        testLoss = comRoundResults[:,6]
        clientTestAcc = comRoundResults[:,7]
        clientTestLoss = comRoundResults[:,8]
        fitTime = comRoundResults[:,9]
        clientPrototypes = comRoundResults[:,10]

        
        if(usePersonalPrototype):
            clientsLocalPrototypes = []
            for clientIndex in range(clientCount):
                clientLocalPrototype = [local_weights[clientIndex][localPrototypeIndex] for localPrototypeIndex in layerAdaptIndex]
                clientsLocalPrototypes.append(np.asarray(clientLocalPrototype,dtype=object))
            hkl.dump(clientsLocalPrototypes,localPrototypeDir)
        trainAccHistory.append(np.mean(trainAcc))
        stdTrainAccHistory.append(np.std(trainAcc))
        trainLossHistory.append(np.mean(trainLoss))
        stdTrainLossHistory.append(np.std(trainLoss))
        meanTestAcc = np.mean(testAcc)
        testAccHistory.append(meanTestAcc)
        stdTestAccHistory.append(np.std(testAcc))
        meanTestLoss = np.mean(testLoss)
        testLossHistory.append(meanTestLoss)
        stdTestLossHistory.append(np.std(testLoss))
        
        if(meanTestAcc > currentAccuracy):
            logging.warning("Better Personalization Accuracy Observed")
            logging.warning("Previous Score: "+str(currentAccuracy)+" from round "+str(bestModelRound)+", Now:"+str(meanTestAcc)+" from round "+str(roundNum))
            best_local_weights = []
            for clientID in trainPool:
                hkl.dump(local_weights[clientID],bestModelPath + "bestModel"+str(clientID)+".hkl" )
            currentAccuracy = meanTestAcc
            bestModelRound = roundNum 



        meanGerneralizationAcc = np.mean(clientTestAcc)


        clientTestLossHistory.append(np.mean(clientTestLoss))
        clientTestAccHistory.append(meanGerneralizationAcc)

        clientStdTestLossHistory.append(np.std(clientTestLoss))
        clientStdTestAccHistory.append(np.std(clientTestAcc))
        startAggregationTime = time.time()

        #FedAvg weightedAveraging 


        newWeight = []
        globalPrototype = []
        # localPrototypes = []
        if(algorithm == 'FEDALI'):
            for index,adaptIndex in enumerate(layerAdaptIndex):
                concatClientPrototypes = tf.concat([local_weights[clientIndex][adaptIndex] for clientIndex in range(clientCount)],axis = 0 )
                averagedPrototypes = np.sum([local_weights[clientIndex][adaptIndex] * local_coeffs[clientIndex] for clientIndex in range(clientCount)],axis = 0 )
                clustering = KMeans(n_clusters=prototypeLayers[index], init=averagedPrototypes, n_init = 1).fit(concatClientPrototypes)
                labels = clustering.labels_
                unique_cluster = np.unique(labels)
                n_clusters = len(unique_cluster)
                labels_tensor = tf.constant(labels, dtype=tf.int32)
                print("Number of cluster: " +str(n_clusters))                
                cluster_means = []
                for label in unique_cluster:
                    cluster_indices = tf.where(tf.equal(labels_tensor, label))  # Find indices of data points in the cluster
                    cluster_data = tf.gather(concatClientPrototypes, cluster_indices)  # Extract data points in the cluster
                    cluster_mean = tf.reduce_mean(cluster_data, axis=0)  # Compute the mean of the cluster
                    cluster_means.append(cluster_mean)
                cluster_means_tensor = tf.squeeze(tf.stack(cluster_means)) 
                globalPrototype.append(cluster_means_tensor)

        if(algorithm == 'FEDPAC'):
            hList, vList, clientSizeList = zip(*[(h, v, size) for h, v, size in aggregationVars])
            headCalculationStart = time.time()                
            clientHeadCoefs = fed_util.get_head_agg_weight_optimized(hList,vList,len(trainPool),activityCount)
            # clientHeadCoefs = fed_util.get_head_agg_weight(hList,vList,len(trainPool),activityCount)
            headCalculation = (time.time() - headCalculationStart)/60 

            logging.warning("Head calculation time: "+str(headCalculation))

            serverBaseWeights = []
            for modelBackBoneIdx in range(layerCount - 4):
                serverBaseWeights.append(np.sum([local_weights[idx][modelBackBoneIdx] * local_coeffs[clientIndex] for idx,clientIndex in enumerate(trainPool)],axis = 0))
            
            for clientIndex,clientID in enumerate(trainPool):
                clientHeadWeights = []
                for modelHeadIdx in range(layerCount - 4,layerCount):
                    clientNewHeadWeight = np.zeros(local_weights[clientIndex][modelHeadIdx].shape)
                    for innerClientID, clientHeadCoef in enumerate(clientHeadCoefs[clientIndex]):
                        clientNewHeadWeight += local_weights[innerClientID][modelHeadIdx] * clientHeadCoef
                    clientHeadWeights.append(clientNewHeadWeight)
                clientModelWeights = serverBaseWeights + clientHeadWeights
                hkl.dump(clientModelWeights,filepath+'clientModels/localModel'+str(clientID)+'.hkl', mode='w')


            globalPrototype = fed_util.fedPacProtoAgggregation(clientSizeList,clientPrototypes)
            hkl.dump(globalPrototype,filepath+'globalPrototypes.hkl' )

        else:
            for index,i in enumerate(trainPool):
                for j in range(0,len(local_weights[i])):
                    local_weights[i][j] = local_weights[i][j] * local_coeffs[i]
        
        
            blockIndex = 0
            for layerIndex in range(layerCount):
                averagedLayerWeight = np.sum([local_weights[clientIndex][layerIndex] for clientIndex in range(len(local_weights))],axis = 0)
                newWeight.append(averagedLayerWeight)
        
                if(algorithm == 'FEDALI'):
                    if(layerIndex in layerAdaptIndex):
                        if(roundNum != 0):
                            prototypeStabilityEpoch[blockIndex].append(tf.reduce_mean(tf.math.abs(globalPrototype[blockIndex] - previousPrototype[blockIndex])))
                        previousPrototype[blockIndex] = globalPrototype[blockIndex]
                        blockIndex += 1

                
        if(algorithm == 'FEDALI'):
            for idx, globalIndex in enumerate(globalPrototypeIndex):
                newWeight[globalIndex] = globalPrototype[idx]
            for idx, localIndex in enumerate(layerAdaptIndex):
                newWeight[localIndex] = globalPrototype[idx]

        if(algorithm != 'FEDPAC'):
            serverModel.set_weights((newWeight))
            del averagedLayerWeight
            if(algorithm != 'FEDPER' and algorithm != 'FEDPROTO' and algorithm != 'FEDPAC' ):
                logging.warning("Evaluating Server Model")
                globalTestMetrics = serverModel.evaluate(centralTestData, centralTestLabel,verbose = showTrainVerbose)
                globalTestLossHistory.append(globalTestMetrics[0])
                globalTestAccHistory.append(globalTestMetrics[1])
                if(globalTestMetrics[1]>serverCurrentAccuracy):
                    logging.warning("Better Global Accuracy Observed")
                    logging.warning("Previous Score: "+str(serverCurrentAccuracy)+" from round "+str(serverbestModelRound)+", Now:"+str(globalTestMetrics[1])+" from round "+str(roundNum))
                    serverCurrentAccuracy = globalTestMetrics[1]
                    serverbestModelRound = roundNum
                    serverModel.save_weights(filepath+'bestServerWeights.h5')
                    bestServerModelWeights = copy.deepcopy(serverModel.get_weights())
            serverModel.save_weights(filepath+'serverWeights.h5')

        
        roundEndTime = time.time() - roundStartTime
        roundTrainingTime.append(roundEndTime / 60)



        
        if(algorithm == 'FEDPROTO'):

            clientActivitySampleCounts = [[len(clientPrototypes[clientIdx][activityIdx]) for activityIdx in range(activityCount)] for clientIdx in trainPool]
            totalActivitySampleCounts = tf.math.reduce_sum(clientActivitySampleCounts,axis = 0)
            
            clientLocalPrototypes = []
            for clientIdx in range(clientCount):
                activityMean = []
                for activityIdx in range(activityCount): 
                    clientActivityCoef = tf.cast(clientActivitySampleCounts[clientIdx][activityIdx] / totalActivitySampleCounts[activityIdx],dtype = tf.float32) 
                    if(len(clientPrototypes[clientIdx][activityIdx]) > 0 ):
                        activityMean.append(tf.math.reduce_mean(clientPrototypes[clientIdx][activityIdx], axis = 0) * clientActivityCoef)
                    else:
                        activityMean.append(tf.zeros(projection_dim))
                clientLocalPrototypes.append(activityMean)
            globalPrototype = tf.reduce_sum(clientLocalPrototypes,axis = 0)
            hkl.dump(globalPrototype,filepath+'globalPrototypes.hkl' )

        aggregationTime = (time.time() - startAggregationTime)/60 

        del newWeight

        # for clientID in trainPool:
        #     hkl.dump(local_weights[clientID],trainModelPath + "trainModel"+str(clientID)+".hkl" )    
        checkPointProgress()
        logging.warning("Client fit time on communcation round " +str(roundNum)+ " is :"+str(np.mean(fitTime)) +" minutes")
        logging.warning("Aggregation time on communcation round " +str(roundNum)+ " is :"+str(aggregationTime / 60) +" minutes")
        logging.warning("Total Training time on communcation round " +str(roundNum)+ " is :"+str(roundEndTime / 60) +" minutes")
        logging.warning("Fit time " +str(np.max(fitTime)))
        logging.warning("Personalization Accuracy " +str(meanTestAcc) +" Loss: " +str(meanTestLoss))
        logging.warning("Generalization Accuracy " +str(clientTestAccHistory[-1]) + " Loss: " +str(clientTestLossHistory[-1]))
        if(algorithm != 'FEDPER' and algorithm != 'FEDPROTO' and algorithm != 'FEDPAC'):
            logging.warning("Global Accuracy " +str(globalTestAccHistory[-1]) +" Loss: " +str(globalTestLossHistory[-1]))


    # In[ ]:


    # convert datatypes to a np formats
    # std of all clients
    stdTrainLossHistory = np.asarray(stdTrainLossHistory[:communicationRound])
    stdTrainAccHistory = np.asarray(stdTrainAccHistory[:communicationRound])
    stdTestLossHistory = np.asarray(stdTestLossHistory[:communicationRound])
    stdTestAccHistory = np.asarray(stdTestAccHistory[:communicationRound])

    clientStdTestLossHistory = np.asarray(clientStdTestLossHistory[:communicationRound])
    clientStdTestAccHistory = np.asarray(clientStdTestAccHistory[:communicationRound])

    trainLossHistory = np.asarray(trainLossHistory[:communicationRound])
    trainAccHistory = np.asarray(trainAccHistory[:communicationRound])
    testLossHistory = np.asarray(testLossHistory[:communicationRound])
    testAccHistory = np.asarray(testAccHistory[:communicationRound])

    clientTestLossHistory = np.asarray(clientTestLossHistory[:communicationRound])
    clientTestAccHistory = np.asarray(clientTestAccHistory[:communicationRound])


    if(algorithm != 'FEDPER' and algorithm != 'FEDPROTO' ):
        globalTestLossHistory = np.asarray(globalTestLossHistory[:communicationRound])
        globalTestAccHistory = np.asarray(globalTestAccHistory[:communicationRound])


    # In[ ]:


    # Saving the training statistics and results
    os.makedirs(filepath+'trainingStats', exist_ok=True)

    hkl.dump(trainLossHistory,filepath + "trainingStats/trainLossHistory.hkl" )
    hkl.dump(trainAccHistory,filepath + "trainingStats/trainAccHistory.hkl" )
    hkl.dump(stdTrainLossHistory,filepath + "trainingStats/stdTrainLossHistory.hkl" )
    hkl.dump(stdTrainAccHistory,filepath + "trainingStats/stdTrainAccHistory.hkl" )

    hkl.dump(testLossHistory,filepath + "trainingStats/testLossHistory.hkl" )
    hkl.dump(testAccHistory,filepath + "trainingStats/testAccHistory.hkl" )
    hkl.dump(stdTestLossHistory,filepath + "trainingStats/stdTestLossHistory.hkl" )
    hkl.dump(stdTestAccHistory,filepath + "trainingStats/stdTestAccHistory.hkl" )
        
    # hkl.dump(adaptiveLoss,filepath + "trainingStats/adaptiveLoss.hkl" )
    # hkl.dump(adaptiveLossStd,filepath + "trainingStats/adaptiveLossStd.hkl" )
        
    if(GeneralizationTest == True):
        hkl.dump(clientStdTestLossHistory,filepath + "trainingStats/clientStdTestLossHistory.hkl" )
        hkl.dump(clientStdTestAccHistory,filepath + "trainingStats/clientStdTestAccHistory.hkl" )

        hkl.dump(clientTestLossHistory,filepath + "trainingStats/clientTestLossHistory.hkl" )
        hkl.dump(clientTestAccHistory,filepath + "trainingStats/clientTestAccHistory.hkl" )

    if(algorithm != 'FEDPER' and algorithm != 'FEDPROTO' ):
        hkl.dump(globalTestLossHistory,filepath + "trainingStats/globalTestLossHistory.hkl" )
        hkl.dump(globalTestAccHistory,filepath + "trainingStats/globalTestAccHistory.hkl" )


    # In[ ]:


    # generate line chart function
    def saveGraph(title = "",accuracyOrLoss = "Accuracy",asyTest = False,legendLoc = 'lower right'):
        plt.title(title)
        plt.ylabel(accuracyOrLoss)
        plt.xlabel('Communication Round')
        plt.legend(loc=legendLoc)
        plt.savefig(filepath+title.replace(" ", "")+'.png', bbox_inches="tight", format="png")
        plt.clf()


    # In[ ]:


    # Plotting results
    epoch_range = range(1, communicationRound+1)

    if(algorithm != 'FEDPER' and algorithm != 'FEDPROTO' and algorithm != 'FEDPAC' ):
        plt.plot(epoch_range, globalTestAccHistory, label= 'Global Test',color="orange")
        plt.plot(epoch_range, globalTestAccHistory,markevery=[np.argmax(globalTestAccHistory)], ls="", marker="o",color="orange") 


    plt.errorbar(epoch_range, trainAccHistory, yerr=stdTrainAccHistory, label='Personalization Train',alpha=0.6, color= "green")
    plt.errorbar(epoch_range, testAccHistory, yerr=stdTestAccHistory, label='Personalization Test',alpha=0.6, color='red')

    plt.plot(epoch_range, trainAccHistory,markevery=[np.argmax(trainAccHistory)], ls="", marker="o",color="green")
    plt.plot(epoch_range, testAccHistory,markevery=[np.argmax(testAccHistory)], ls="", marker="o",color="red")  

    if(GeneralizationTest == True):
        plt.errorbar(epoch_range, clientTestAccHistory, yerr=clientStdTestAccHistory, label='Generalization Test',alpha=0.6, color="brown")
        plt.plot(epoch_range, clientTestAccHistory,markevery=[np.argmax(clientTestAccHistory)], ls="", marker="o",color="brown")  

    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Communication Round')
    plt.legend(loc='lower right')
    plt.savefig(filepath+'LearningAccuracy.png', bbox_inches="tight", format="png")
    plt.clf()

    if(algorithm != 'FEDPER' and algorithm != 'FEDPROTO' and algorithm != 'FEDPAC'):
        plt.plot(epoch_range, globalTestLossHistory, label= 'Global Test',color="orange")
        plt.plot(epoch_range, globalTestLossHistory,markevery=[np.argmin(globalTestLossHistory)], ls="", marker="o",color="orange") 
        
    plt.errorbar(epoch_range, trainLossHistory, yerr=stdTrainLossHistory, label='Personalization Train',alpha=0.6, color='green')
    plt.errorbar(epoch_range, testLossHistory, yerr=stdTestLossHistory, label='Personalization Test',alpha=0.6, color='red')
    plt.plot(epoch_range, trainLossHistory,markevery=[np.argmin(trainLossHistory)], ls="", marker="o",color="green")
    plt.plot(epoch_range, testLossHistory,markevery=[np.argmin(testLossHistory)], ls="", marker="o",color="red")  

    if(GeneralizationTest == True):
        plt.errorbar(epoch_range, clientTestLossHistory, yerr=clientStdTestLossHistory, label='Generalization Test',alpha=0.6,color="brown")
        plt.plot(epoch_range, clientTestLossHistory,markevery=[np.argmin(clientTestLossHistory)], ls="", marker="o",color="brown")  

    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Communication Round')
    plt.legend(loc= 'upper right')
    plt.savefig(filepath+'LearningLoss.png', bbox_inches="tight", format="png")
    plt.clf()


    # In[ ]:


    # Rounding number function 
    def roundNumber(toRoundNb):
        return round(np.mean(toRoundNb), 4)


    # In[ ]:


    #Generating personalized accuracy
    indiWeightedTest = []
    indiMicroTest = []
    indiMacroTest = []

    genWeightedTest = []
    genMicroTest = []
    genMacroTest = []

    genBestMacroTest = []

    os.makedirs(filepath+'models/' , exist_ok=True)

    for i in trainPool:
        print("Loading Client "+str(i))
        clientWeightsLoad = hkl.load(bestModelPath + "bestModel"+str(i)+".hkl")
        serverModel.set_weights(clientWeightsLoad)
        y_pred = np.argmax(serverModel.predict(clientDataTest[i],verbose = showTrainVerbose), axis=-1)
        y_test = np.argmax(clientLabelTest[i], axis=-1)

        indiWeightedTest.append(f1_score(y_test, y_pred,average='weighted' ))
        indiMicroTest.append(f1_score(y_test, y_pred,average='micro' ))
        indiMacroTest.append(f1_score(y_test, y_pred,average='macro' ))


        y_pred = np.argmax(serverModel.predict(centralTestData,verbose = showTrainVerbose), axis=-1)
        y_test = np.argmax(centralTestLabel, axis=-1)    

        genWeightedTest.append(f1_score(y_test, y_pred,average='weighted'))
        genMicroTest.append(f1_score(y_test, y_pred,average='micro'))
        genMacroTest.append(f1_score(y_test, y_pred,average='macro'))

        del y_pred
        del y_test
        del clientWeightsLoad
        gc.collect()
        tf.keras.backend.clear_session()


    modelStatistics = {
        "Personalization Accuracy:" : '',
        "\nBestModelRound:": bestModelRound,
        "\nweighted f1:" : roundNumber(np.mean(indiWeightedTest)) * 100,
        "\nmicro f1:": roundNumber(np.mean(indiMicroTest)) * 100,
        "\nmacro f1:": roundNumber(np.mean(indiMacroTest)) * 100,
        "\nmacro std f1:": roundNumber(np.std(indiMacroTest)) * 100,
        "\nround mean time:": np.mean(roundTrainingTime),
        "\nround std time:": np.std(roundTrainingTime),
    }    
    with open(filepath +'PersonalizationACC.csv','w') as f:
        w = csv.writer(f)
        w.writerows(modelStatistics.items())


    modelStatistics = {
    "Generalization Accuracy:" : '',
    "\Generalization Best Model Round:": bestModelRound,
    "\nGeneralization weighted f1:" : roundNumber(np.mean(genWeightedTest)) * 100,
    "\nGeneralization micro f1:": roundNumber(np.mean(genMicroTest)) * 100,
    "\nGeneralization macro f1:": roundNumber(np.mean(genMacroTest)) * 100,
    "\nGeneralization macro std f1:": roundNumber(np.std(genMacroTest)) * 100,
    "\nGeneralization macro std f1:": roundNumber(np.std(genMacroTest)) * 100,
    }    
    with open(filepath +'GeneralizationACC.csv','w') as f:
        w = csv.writer(f)
        w.writerows(modelStatistics.items())

    hkl.dump(indiMacroTest,filepath + 'indiMacroTest.hkl') 
    hkl.dump(genMacroTest,filepath + 'genMacroTest.hkl') 


    # In[ ]:


    def extract_intermediate_model_from_base_model(base_model, intermediate_layer=4):
        model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[intermediate_layer].output, name=base_model.name + "_layer_" + str(intermediate_layer))
        return model


    # In[ ]:


    perplexity = 30


    # In[ ]:


    clientEmbeddingFilePath = filepath + "clientEmbeddings/"
    os.makedirs(clientEmbeddingFilePath, exist_ok=True)


    # In[ ]:


    if(algorithm != 'FEDPER' and algorithm != 'FEDPROTO' and algorithm != 'FEDPAC' ):
        serverModel.set_weights(bestServerModelWeights)
        y_pred = np.argmax(serverModel.predict(centralTestData,verbose = showTrainVerbose), axis=-1)
        y_test = np.argmax(centralTestLabel, axis=-1)    
        
        weightVal_f1 = f1_score(y_test, y_pred,average='weighted' )
        microVal_f1 = f1_score(y_test, y_pred,average='micro')
        macroVal_f1 = f1_score(y_test, y_pred,average='macro')
        
        modelStatistics = {
        "Global Accuracy" : '',
        "\nServer Best Model Round": serverbestModelRound,
        "\nGlobal Accuracy:" : roundNumber(serverCurrentAccuracy) * 100,
        "\nGlobal weighted f1:" : roundNumber(weightVal_f1) * 100,
        "\nGlobal micro f1:": roundNumber(microVal_f1) * 100,
        "\nGlobal macro f1:": roundNumber(macroVal_f1) * 100,
        }    
        with open(filepath +'GlobalACC.csv','w') as f:
            w = csv.writer(f)
            w.writerows(modelStatistics.items())
        hkl.dump(macroVal_f1,filepath + 'macroVal_f1.hkl') 


    # In[ ]:


    if(algorithm == "FEDALP"):
        memoryStabilityPath = filepath+"protoTypeImages/"
        os.makedirs(memoryStabilityPath, exist_ok=True)
        prototypeStabilityCR = [prototypeStabilityEpoch[key] for key in prototypeStabilityEpoch]
        for index, layerMemoryStability in enumerate(prototypeStabilityCR):
            epoch_range = range(1, communicationRound)
            plt.plot(epoch_range, layerMemoryStability)
            plt.title('Prototype Displacements For Block '+str(index))
            plt.ylabel('L1 Distance')
            plt.xlabel('Epoch')
            plt.savefig(memoryStabilityPath+"B_"+str(index+1)+"_memoryDisplacement.png", bbox_inches="tight")
            plt.show()
            plt.clf()
        epoch_range = range(1, communicationRound)
        meanMemory = np.mean(prototypeStabilityCR,axis = 0)
        stdMemory = np.std(prototypeStabilityCR,axis = 0)
        plt.errorbar(epoch_range, meanMemory, yerr=stdMemory)
        plt.title('Mean Prototype Displacements')
        plt.ylabel('L1 Distance')
        plt.xlabel('Epoch')
        plt.savefig(memoryStabilityPath+"meanMemoryDisplacement.png", bbox_inches="tight")
        # plt.show()
        plt.clf()


    # In[ ]:


    clientOneIndex = 0
    clientTwoIndex = 1

    if(dataSetName == 'Combined'):
        clientOneIndex = 0
        clientTwoIndex = datasetClientCounts[0]


    labels_argmax = np.argmax(np.vstack((clientLabelTest[clientOneIndex],clientLabelTest[clientTwoIndex])), axis=-1)
    unique_labels = np.unique(labels_argmax)
    clientIndex = np.hstack((np.full(len(clientLabelTest[clientOneIndex]),0),np.full(len(clientLabelTest[clientTwoIndex]),1)))

    serverModel.set_weights(hkl.load(bestModelPath + "bestModel"+str(clientOneIndex)+".hkl"))
    embed1 = extract_intermediate_model_from_base_model(serverModel,embedLayerIndex)(clientDataTest[clientOneIndex])
    serverModel.set_weights(hkl.load(bestModelPath + "bestModel"+str(clientTwoIndex)+".hkl"))
    embed2 = extract_intermediate_model_from_base_model(serverModel,embedLayerIndex)(clientDataTest[clientTwoIndex])


    # In[ ]:


    tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=showTrainVerbose, random_state=randomSeed)
    tsne_projections = tsne_model.fit_transform(np.vstack((embed1,embed2)))
    pandaData = {'col1': tsne_projections[:,0], 'col2': tsne_projections[:,1],'Classes':labels_argmax,'Client':clientIndex}
    pandaDataFrame = pd.DataFrame(data=pandaData)

    plt.figure(figsize=(14,14))
    plt.title('Embeddings Between 2 Clients')
    graph = sns.scatterplot(data=pandaDataFrame, x="col1", y="col2", hue="Classes", style="Client",
                    palette=sns.color_palette(n_colors = len(unique_labels)),
                    s=100, alpha=1.0,rasterized=True,)
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
    legend = graph.legend_
    for i in range(len(ACTIVITY_LABEL)):
        legend.get_texts()[i+1].set_text(ACTIVITY_LABEL[i]) 
    plt.savefig(filepath+'Overlap_Embeddings.png', dpi=200,bbox_inches='tight')
    plt.show()
    plt.clf()
    hkl.dump(tsne_projections,filepath + "overlappingRepresentations.hkl" )


    # In[ ]:


    print("Training Done!")


    # In[ ]:





    # In[ ]:




