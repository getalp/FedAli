from unittest.result import failfast
import tensorflow as tf
import time
import numpy as np
from model import HART
from utils import extract_intermediate_model_from_base_model,extract_intermediate_model_from_head_model,multi_output_model
import functools
import model
import os
import logging
import hickle as hkl
import copy
import gc

logging.getLogger('tensorflow').setLevel(logging.ERROR)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


randomSeed = 1
LARGE_NUM = 1e9
EPISILON = 1e-07
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def set_global(shared_vars,GPULIST,devID):
    global deviceIndex 
    deviceIndex = devID
    with deviceIndex.get_lock():
        globals()['gpu_id'] = deviceIndex.value
        if(deviceIndex.value >= len(GPULIST)-1):
            deviceIndex.value = -1
        deviceIndex.value += 1
    print(f" client_id: {deviceIndex.value} training on GPU id:{GPULIST[deviceIndex.value]}",flush= True)
    set_specific_gpu(GPULIST[deviceIndex.value])    

    global embedLayerIndex
    global batch_fold
    global segment_size
    global num_input_channels
    global activityCount
    global showTrainVerbose
    global clientLearningRate
    global adaptiveLearningRate
    global adaptiveEpoch
    global batch_size
    global localEpoch
    global centralTestData
    global centralTestLabel
    global GeneralizationTest
    global filepath
    global mu
    global architecture
#     global GPUPoolIndex
    embedLayerIndex = shared_vars[0].value
    batch_fold = shared_vars[1].value
    segment_size = shared_vars[2].value
    num_input_channels = shared_vars[3].value
    activityCount = shared_vars[4].value
    showTrainVerbose = shared_vars[5].value
    clientLearningRate = shared_vars[6].value
    adaptiveLearningRate = shared_vars[7].value
    batch_size = shared_vars[8].value
    localEpoch = shared_vars[9].value
    centralTestData =  tf.reshape(shared_vars[10], (-1,segment_size,num_input_channels))
    centralTestLabel =  tf.reshape(shared_vars[11], (-1,activityCount))
    GeneralizationTest = shared_vars[12].value
    filepath = shared_vars[13].value
    mu =  shared_vars[14].value
    architecture = shared_vars[15].value
    # print(f" Architecture: {architecture} ",flush= True)
    tf.random.set_seed(1)

def set_specific_gpu(ID):  

    gpus_all_physical_list = tf.config.list_physical_devices(device_type='GPU')    
    tf.config.set_visible_devices(gpus_all_physical_list[ID], 'GPU')
    # tf.config.experimental.set_memory_growth(gpus_all_physical_list[ID], True)
    # tf.config.set_logical_device_configuration(
    #         gpus_all_physical_list[ID],
    #         [tf.config.LogicalDeviceConfiguration(memory_limit=4092)])
    

    
    # gpus_all_physical_list = tf.config.list_physical_devices(device_type='GPU')    
    # # tf.config.set_visible_devices(gpus_all_physical_list[ID], 'GPU')
    # # tf.config.set_memory_growth(gpus_all_physical_list[ID], True)

    # tf.config.experimental.set_visible_devices(gpus_all_physical_list[ID], 'GPU')

    # # Set memory growth for the selected GPU
    # for gpu in specific_gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)






def fedAvg_Trainer(clientNumber,clientDataTrain,clientLabelTrain,clientDataTest,clientLabelTest,loadPretrain):

    if(loadPretrain):
        localModel = model.HART_MAE((segment_size,num_input_channels),activityCount)
    else:
        localModel = model.HART((segment_size,num_input_channels),activityCount)
    localModel.load_weights(filepath+'serverWeights.h5')
    startTime = time.time()
    localModel.compile(optimizer=tf.keras.optimizers.Adam(clientLearningRate),loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])      
    trainHistory = localModel.fit(clientDataTrain, clientLabelTrain,batch_size = batch_size, epochs = localEpoch,verbose=showTrainVerbose)
    fitTime = (time.time() - startTime)/60 

    return [clientNumber,
            localModel.get_weights(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            fitTime,
            None]

def fedProx_Trainer(clientNumber,clientDataTrain,clientLabelTrain,clientDataTest,clientLabelTest):
    localModel = model.HART((segment_size,num_input_channels),activityCount)
    localModel.load_weights(filepath+'serverWeights.h5')
    serverModel = model.HART((segment_size,num_input_channels),activityCount)
    serverModel.load_weights(filepath+'serverWeights.h5')


    startTime = time.time()
    foptimizer = tf.keras.optimizers.Adam(clientLearningRate)
    localModel.compile(optimizer=foptimizer,loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])      
    
    fedProxFit(localModel,serverModel,clientDataTrain,clientLabelTrain,mu,foptimizer,batch_size,localEpoch)

    fitTime = (time.time() - startTime)/60 
    
    
    return [clientNumber,
            localModel.get_weights(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            fitTime,
            None]


def fedALP_prototypeOnly(clientNumber,clientDataTrain,clientLabelTrain,clientDataTest,
                          clientLabelTest,prototypeCount,globalPrototype,adaptLayerLocation,
                          pretrainModel,prototypeDecay,influenceFactor,usePersonalPrototype,useGLU,singleUpdate):

    clientModelPath = filepath+'clientModels/localModel'+str(clientNumber)+'.h5'
    
    localModel = model.HART_ALP(input_shape =  (segment_size,num_input_channels),
                          activityCount = activityCount,
                          prototypeCount = prototypeCount, 
                          prototypeDecay = prototypeDecay, 
                          influenceFactor = influenceFactor, 
                          useGLU = useGLU,
                          singleUpdate = singleUpdate)

    if(os.path.exists(clientModelPath)):
        # take the server model for the first time
        localModel.load_weights(clientModelPath)
    else:
        # use old client model, since model is not transmitted
        localModel.load_weights(filepath+'serverWeights.h5')

    for blockIndex,layerIndex in enumerate(adaptLayerLocation):
        localModel.layers[layerIndex].localPrototypes.assign(globalPrototype[blockIndex])
        localModel.layers[layerIndex].globalPrototypes.assign(globalPrototype[blockIndex])
    startTime = time.time()
    localModel.compile(optimizer=tf.keras.optimizers.Adam(clientLearningRate),loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])      
    trainHistory = localModel.fit(clientDataTrain, clientLabelTrain,batch_size = batch_size, epochs = localEpoch,verbose=showTrainVerbose)
    fitTime = (time.time() - startTime)/60 

    localModel.save_weights(filepath+'clientModels/localModel'+str(clientNumber)+'.h5')

    # we return local Model, but in reality we can use just return prototype. Returning whole model because the script was build to already extract localPrototype from model weights directly
    return [clientNumber,
            localModel.get_weights(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            fitTime,
            None]


def fedALP_global_Trainer(clientNumber,clientDataTrain,clientLabelTrain,clientDataTest,
                          clientLabelTest,prototypeCount,adaptLayerLocation,
                          pretrainModel,prototypeDecay,influenceFactor,usePersonalPrototype,useGLU,singleUpdate):


    if(pretrainModel):
        localModel = model.HART_ALP_GLOBAL(input_shape = (segment_size,num_input_channels),
                                           activityCount = activityCount,
                                           prototypeCount = prototypeCount, 
                                           prototypeDecay = prototypeDecay,
                                           influenceFactor = influenceFactor,
                                           useGLU = useGLU,
                                           singleUpdate = singleUpdate)
    else:
        localModel = model.HART_ALP(input_shape =  (segment_size,num_input_channels),
                              activityCount = activityCount,
                              prototypeCount = prototypeCount, 
                              prototypeDecay = prototypeDecay, 
                              influenceFactor = influenceFactor, 
                              useGLU = useGLU,
                              singleUpdate = singleUpdate)

    localModel.load_weights(filepath+'serverWeights.h5')
    if(usePersonalPrototype):
        for blockIndex,layerIndex in enumerate(adaptLayerLocation):
            localModel.layers[layerIndex].localPrototypes.assign(localPrototype[blockIndex])
    startTime = time.time()
    localModel.compile(optimizer=tf.keras.optimizers.Adam(clientLearningRate),loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])      
    trainHistory = localModel.fit(clientDataTrain, clientLabelTrain,batch_size = batch_size, epochs = localEpoch,verbose=showTrainVerbose)
    fitTime = (time.time() - startTime)/60 

    return [clientNumber,
            localModel.get_weights(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            fitTime,
            None]


def Moon_Trainer(clientNumber,clientDataTrain,clientLabelTrain,clientDataTest,clientLabelTest,clientPrevModelDir):

    localModel = model.HART((segment_size,num_input_channels),activityCount)
    foptimizer = tf.keras.optimizers.Adam(clientLearningRate)
    floss = tf.keras.losses.CategoricalCrossentropy()
    localModel.compile(optimizer=foptimizer,loss=floss, metrics=['acc'])    
    localModel.load_weights(clientPrevModelDir)    
    modelFE = extract_intermediate_model_from_base_model(localModel,embedLayerIndex)
    # prevModelEmbeeddings = modelFE(clientDataTrain)
    # localModel.load_weights(filepath+'serverWeights.h5')
    # serverModelEmbeddings = modelFE(clientDataTrain)

    prevModelEmbeeddings = modelFE.predict(clientDataTrain, batch_size = batch_size,verbose=0)
    localModel.load_weights(filepath+'serverWeights.h5')
    serverModelEmbeddings = modelFE.predict(clientDataTrain, batch_size = batch_size,verbose=0)
    # gc.collect()
    
    startTime = time.time()
    localMOONModel = tf.keras.Model(inputs=localModel.inputs, outputs=[localModel.layers[embedLayerIndex].output, localModel.output])


    
    _ = MoonFiT(localMOONModel,
               prevModelEmbeeddings,
               serverModelEmbeddings,
               clientDataTrain,
               clientLabelTrain,
               foptimizer,
               mu,
               batch_size,
               localEpoch,
               embedLayerIndex,
               showTrainVerbose)
    fitTime = (time.time() - startTime)/60 
    
    localModel.save_weights(clientPrevModelDir)

    return [clientNumber,
            localModel.get_weights(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            fitTime,
            None]

def MoonFiT(localMOONModel, prevModelEmbeeddings, serverModelEmbeddings, clientDataTrain,clientLabelTrain, optimizer, mu=1.0,batch_size = 32, epochs=5, embedLayerIndex = 221,verbose=0):
    epoch_wise_loss = []
    epoch_wise_acc = []
    for epoch in range(epochs):
        indices = tf.range(start=0, limit = clientDataTrain.shape[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        shuffled_train = tf.gather(clientDataTrain, shuffled_indices, axis=0)
        shuffled_label = tf.gather(clientLabelTrain, shuffled_indices, axis=0)
        for i in range(0, shuffled_train.shape[0] ,batch_size):   
            batched_train = shuffled_train[i:i+batch_size]
            batched_label = shuffled_label[i:i+batch_size]
            Moon_NT_Xent_gradients(localMOONModel,
                                   embedLayerIndex,
                                   batched_train,
                                   batched_label,
                                   prevModelEmbeeddings[i:i+batch_size],
                                   serverModelEmbeddings[i:i+batch_size],
                                   optimizer,
                                   mu=mu)
    return None

def Moon_NT_Xent_gradients(model, embedLayerIndex, trainData,trainLabel, prevModelEmbeds, serverEmbeds,optimizer, temperature=1.0, mu=1.0,):
    with tf.GradientTape() as tape:
        # client_features = extract_intermediate_model_from_base_model(model,embedLayerIndex)(trainData,training=True)
        client_features,outputs = model(trainData,training=True)
        contrastive_loss = moon_contrastive(client_features,serverEmbeds,prevModelEmbeds, temperature = temperature, mu = mu)
        cce_loss = tf.keras.losses.CategoricalCrossentropy()(trainLabel, outputs)
        moon_loss = cce_loss + contrastive_loss
    grads = tape.gradient(moon_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return moon_loss

@tf.function
def cosine_similarity(x1, x2):
    # Normalize the vectors to unit length
    x1_normalized = tf.nn.l2_normalize(x1, axis=1)
    x2_normalized = tf.nn.l2_normalize(x2, axis=1)
    
    # Compute the cosine similarity
    return tf.reduce_sum(tf.multiply(x1_normalized, x2_normalized), axis=1)
@tf.function
def moon_contrastive(z, zglob, zprev, temperature=1.0, mu=1.0):
    # Compute similarities
    sim_z_zglob = cosine_similarity(z, zglob)
    sim_z_zprev = cosine_similarity(z, zprev)

    # Exponentiate the similarities divided by tau
    exp_sim_z_zglob = tf.exp(sim_z_zglob / temperature)
    exp_sim_z_zprev = tf.exp(sim_z_zprev / temperature)

    # Compute the softmax denominator and contrastive loss
    softmax_denominator = exp_sim_z_zglob + exp_sim_z_zprev
    contrastive_loss = -tf.math.log(exp_sim_z_zglob / softmax_denominator)

    return mu * tf.reduce_mean(contrastive_loss)

def setLayersTraining(model,embedLayerIndex,training):
    for i in range(embedLayerIndex):
#         print(model.layers[i],flush=True)
        model.layers[i].trainable = training

@tf.function
def lossNormFunc(loss, movingAverage,beta):
    meanloss = tf.reduce_mean(loss, axis = 1)
    nextMovingAverage =  (beta * movingAverage) + ((1 - beta) * tf.reduce_mean(tf.math.square(meanloss)))
    normalizationTerm = tf.math.sqrt((nextMovingAverage / (1.0 - beta)) + EPISILON)
    
    normLoss = tf.reduce_mean(meanloss) / normalizationTerm
    
    return normLoss, nextMovingAverage


@tf.function
def difference_model_norm_2_square(local_model, global_model):
    model_difference = tf.nest.map_structure(lambda a, b: a - b,
                                           local_model,
                                           global_model)
    squared_norm = tf.square(tf.linalg.global_norm(model_difference))
    return squared_norm


def fedProxFit(localModel,serverModel,clientDataTrain,clientLabelTrain,mu,optimizer,batchSize,epochs):
    epoch_wise_loss = []
    epoch_wise_acc = []
    
    for epoch in range(epochs):
        indices = tf.range(start=0, limit = clientDataTrain.shape[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        batchIndex = [shuffled_indices[i:i+batchSize] for i in range(0,len(shuffled_indices),batchSize)]
        for index in batchIndex:
            with tf.GradientTape() as tape:
                x = tf.gather(clientDataTrain, index, axis=0)
                outputs = localModel(x,training=True)
                labels = tf.gather(clientLabelTrain, index, axis=0)
                scce_loss = tf.keras.losses.CategoricalCrossentropy()(labels, outputs)
                mu = tf.constant(mu, dtype=tf.float32)
                prox_term =(mu/2)*difference_model_norm_2_square(localModel.trainable_variables, serverModel.trainable_variables)
                fedprox_loss = scce_loss + prox_term
            grads = tape.gradient(fedprox_loss, localModel.trainable_variables)
            optimizer.apply_gradients(zip(grads, localModel.trainable_variables))
    return None, None





def fedProtoFiT(localModel,clientDataTrain,clientLabelTrain,optimizer,batchSize,epochs,globalPrototype,activityCount):
    # agg_protos_label = {activityID: [] for activityID in range(activityCount)}
    agg_protos_label = [[] for _ in range(activityCount)]

    for epoch in range(epochs):
        indices = tf.range(start=0, limit = clientDataTrain.shape[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        batchIndex = [shuffled_indices[i:i+batchSize] for i in range(0,len(shuffled_indices),batchSize)]
        nonOneHotLabels = tf.argmax(clientLabelTrain,axis = -1)

                
        for index in batchIndex:
            with tf.GradientTape() as tape:
                x = tf.gather(clientDataTrain, index, axis=0)
                localPrototype, outputs = localModel(x,training=True)
                labels = tf.gather(clientLabelTrain, index, axis=0)
                cce_loss = tf.keras.losses.CategoricalCrossentropy()(labels, outputs)
                nonOneHotLabels =  tf.argmax(labels,axis = -1)
                prototypeLabel = tf.gather(globalPrototype,nonOneHotLabels)           
                mse_loss = tf.keras.losses.MSE(localPrototype,prototypeLabel)
                fedproto_loss = cce_loss + mse_loss
            grads = tape.gradient(fedproto_loss, localModel.trainable_variables)
            optimizer.apply_gradients(zip(grads, localModel.trainable_variables))

            # if it's the last local epochs, we save the prototypes to send to the server, as done on authors github
            if(epoch == (epochs - 1)):
                for activity in range(activityCount):
                    matches = tf.equal(nonOneHotLabels, activity)
                    matchIndices = tf.where(matches)
                    aggregateClassPrototypes = tf.gather(localPrototype,matchIndices[:,0])
                    if(aggregateClassPrototypes.shape[0] != 0):
                        agg_protos_label[activity].append(aggregateClassPrototypes)

    for activity in range(activityCount):
        if(len(agg_protos_label[activity]) > 1):
            agg_protos_label[activity] = tf.concat(agg_protos_label[activity],axis = 0)
        # elif(len(agg_protos_label[activity]) == 0):
        #     agg_protos_label[activity] = 0
            # to avoid NAN when we average later
    return agg_protos_label
    
def fedProto_Trainer(clientNumber,clientDataTrain,clientLabelTrain,clientDataTest,clientLabelTest,globalPrototype):
    
    clientModelPath = filepath+'clientModels/localModel'+str(clientNumber)+'.h5'

    localModel = model.HART((segment_size,num_input_channels),activityCount)

    if(os.path.exists(clientModelPath)):
        # take the server model for the first time
        localModel.load_weights(clientModelPath)
    else:
        # use old client model, since model is not transmitted
        localModel.load_weights(filepath+'serverWeights.h5')

    localProtoModel = tf.keras.Model(inputs=localModel.inputs, outputs=[localModel.layers[embedLayerIndex].output, localModel.output])

                                
    startTime = time.time()
    foptimizer = tf.keras.optimizers.Adam(clientLearningRate)
    localModel.compile(optimizer=foptimizer,loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])      
    
    agg_protos_label = fedProtoFiT(localProtoModel,clientDataTrain,clientLabelTrain,foptimizer,batch_size,localEpoch,globalPrototype,activityCount)

    fitTime = (time.time() - startTime)/60 

    localModel.save_weights(clientModelPath)

    
    return [clientNumber,
            localModel.get_weights(),
            None,
            personalizationTrainMetrics[1],
            personalizationTrainMetrics[0],
            None,
            None,
            None,
            None,
            fitTime,
            agg_protos_label]



def local_Trainer(clientNumber,clientDataTrain,clientLabelTrain,clientDataTest,clientLabelTest):



    trainCallBacks = []
    checkpoint_filepath = filepath+"bestValModels/bestValcheckpoint"+str(clientNumber)+".h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_acc",
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    )
    trainCallBacks.append(checkpoint_callback)
        
    
    # clientModelPath = filepath+'/clientModels/localModel'+str(clientNumber)+'.h5'
    localModel = model.HART((segment_size,num_input_channels),activityCount)
    localModel.load_weights(filepath+'serverWeights.h5')
    foptimizer = tf.keras.optimizers.Adam(clientLearningRate)
    localModel.compile(optimizer=tf.keras.optimizers.Adam(clientLearningRate),loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])      


    trainHistory = localModel.fit(clientDataTrain, clientLabelTrain,
                                  validation_data = (clientDataTest,clientLabelTest),
                                  batch_size = batch_size, 
                                  epochs = localEpoch,
                                  callbacks=trainCallBacks,
                                  verbose=showTrainVerbose)
        

    localModel.load_weights(checkpoint_filepath)
    personalizationTrainMetrics = localModel.evaluate(clientDataTrain, clientLabelTrain,verbose = showTrainVerbose)
    personalizationTestMetrics = localModel.evaluate(clientDataTest, clientLabelTest,verbose = showTrainVerbose)
    generalizationMetrics = localModel.evaluate(centralTestData, centralTestLabel,verbose = showTrainVerbose)
    
    return [localModel.get_weights(),
            personalizationTrainMetrics[1],
            None,
            None]
