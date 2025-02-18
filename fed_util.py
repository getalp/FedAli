
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
import cvxpy as cvx
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
    global segment_size
    global num_input_channels
    global activityCount
    global showTrainVerbose
    global clientLearningRate
    global batch_size
    global localEpoch
    global centralTestData
    global centralTestLabel
    global GeneralizationTest
    global filepath
    global mu

    
    embedLayerIndex = shared_vars[0].value
    segment_size = shared_vars[1].value
    num_input_channels = shared_vars[2].value
    activityCount = shared_vars[3].value
    showTrainVerbose = shared_vars[4].value
    clientLearningRate = shared_vars[5].value
    batch_size = shared_vars[6].value
    localEpoch = shared_vars[7].value
    centralTestData = tf.reshape(shared_vars[8], (-1, segment_size, num_input_channels))
    centralTestLabel = tf.reshape(shared_vars[9], (-1, activityCount))
    GeneralizationTest = shared_vars[10].value
    filepath = shared_vars[11].value
    mu = shared_vars[12].value
    tf.random.set_seed(1)
    
def set_specific_gpu(ID):  
    gpus_all_physical_list = tf.config.list_physical_devices(device_type='GPU')    
    tf.config.set_visible_devices(gpus_all_physical_list[ID], 'GPU')


def fedAvg_Trainer(clientNumber,clientDataTrain,clientLabelTrain,clientDataTest,clientLabelTest,pretrainModel):

    if(pretrainModel):
        localModel = model.HART_MAE((segment_size,num_input_channels),activityCount)
    else:
        localModel = model.HART((segment_size,num_input_channels),activityCount)
    localModel.load_weights(filepath+'serverWeights.h5')
    startTime = time.time()
    localModel.compile(optimizer=tf.keras.optimizers.Adam(clientLearningRate),loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])      
    trainHistory = localModel.fit(clientDataTrain, clientLabelTrain,batch_size = batch_size, epochs = localEpoch,verbose=showTrainVerbose)
    fitTime = (time.time() - startTime)/60 
    personalizationTestMetrics = localModel.evaluate(clientDataTest, clientLabelTest,verbose = showTrainVerbose)
    generalizationMetrics = [None, None]
    if(GeneralizationTest):
        generalizationMetrics = localModel.evaluate(centralTestData, centralTestLabel,verbose = showTrainVerbose)
    personalizationTrainAcc = np.mean(trainHistory.history['acc'])
    personalizationTrainloss = np.mean(trainHistory.history['loss'])
    print("Client Number " +str(clientNumber)+" Train accuracy "+str(personalizationTrainAcc) + " Personalization Accuracy "+str(personalizationTestMetrics[1]) + " Generalization Accuracy " +str(generalizationMetrics[1]),flush=True )

    gc.collect()

    
    return [clientNumber,
            localModel.get_weights(),
            None,
            personalizationTrainAcc,
            personalizationTrainloss,
            personalizationTestMetrics[1],
            personalizationTestMetrics[0],
            generalizationMetrics[1],
            generalizationMetrics[0],
            fitTime,
            None]

def pairwise(data):
    n = len(data)
    for i in range(n):
        for j in range(i, n):
            yield (data[i], data[j])


def get_head_agg_weight(hList, vList, clientCount, activityCount, projection_dim=192, *args, **kwargs):
    avgHeadWeight = []

    # Precompute pairwise differences and store distances
    hList_tensor = tf.stack(hList)  # Shape: (clientCount, activityCount, projection_dim)
    
    for i in range(clientCount):
        h_ref = hList_tensor[i]  # Reference head
        dist = np.zeros((clientCount, clientCount), dtype=np.float32)

        for j1, j2 in pairwise(range(clientCount)):
            # Efficiently compute pairwise difference using broadcasting
            h_j1 = hList_tensor[j1]  # Shape: (activityCount, projection_dim)
            h_j2 = hList_tensor[j2]  # Shape: (activityCount, projection_dim)

            # Compute the pairwise matrix product and trace in a vectorized manner
            h_diff1 = h_ref - h_j1  # Shape: (activityCount, projection_dim)
            h_diff2 = h_ref - h_j2  # Shape: (activityCount, projection_dim)

            # Use batched matrix multiplication for the outer product of the differences
            h_product = tf.einsum('ik,jk->ij', h_diff1, h_diff2)  # Shape: (projection_dim, projection_dim)
            
            # Compute the distance as the trace of the resulting matrix
            dj12 = tf.linalg.trace(h_product).numpy()  # Scalar

            # Symmetric distance matrix
            dist[j1, j2] = dj12
            dist[j2, j1] = dj12

        # Create p_matrix using the precomputed distance matrix
        p_matrix = np.diag(vList) + dist

        # Perform eigenvalue decomposition on p_matrix
        evals, evecs = np.linalg.eigh(p_matrix)

        # Reconstruct p_matrix from positive eigenvalues
        p_matrix_new = np.zeros_like(p_matrix)
        for ii in range(clientCount):
            if evals[ii] >= 0.01:
                p_matrix_new += evals[ii] * np.outer(evecs[:, ii], evecs[:, ii])

        # Ensure p_matrix is positive semi-definite
        p_matrix = p_matrix_new if np.all(np.linalg.eigvals(p_matrix_new) >= 0) else p_matrix

        # Solve the optimization problem using cvxpy
        if np.all(np.linalg.eigvals(p_matrix) >= 0):
            alphav = cvx.Variable(clientCount)
            obj = cvx.Minimize(cvx.quad_form(alphav, p_matrix))
            prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
            prob.solve()
            alpha = np.clip(alphav.value, 0, None)  # Ensure non-negative solutions
            alpha = [(a if a > 1e-3 else 0) for a in alpha]  # Zero-out small weights
        else:
            # Fallback to local classifier if no valid solution is found
            alpha = [0.0] * clientCount
            alpha[i] = 1.0

        avgHeadWeight.append(alpha)

    return avgHeadWeight


def get_head_agg_weight_optimized(hList, vList, clientCount, activityCount, projection_dim=192, *args, **kwargs):
    avgHeadWeight = []

    # Stack hList tensors into a single tensor
    hList_tensor = tf.stack(hList)  # Shape: (clientCount, activityCount, projection_dim)

    for i in range(clientCount):
        h_ref = hList_tensor[i]  # Reference head

        # Compute the differences between h_ref and all hList tensors
        h_diff = h_ref[None, :, :] - hList_tensor  # Shape: (clientCount, activityCount, projection_dim)

        # Flatten the differences to 2D arrays
        h_diff_flat = tf.reshape(h_diff, (clientCount, -1))  # Shape: (clientCount, activityCount * projection_dim)

        # Compute the distance matrix via matrix multiplication
        dist = tf.matmul(h_diff_flat, h_diff_flat, transpose_b=True).numpy()  # Shape: (clientCount, clientCount)

        # Add vList as a diagonal matrix
        p_matrix = np.diag(vList) + dist

        # Eigenvalue decomposition
        evals, evecs = np.linalg.eigh(p_matrix)

        # Reconstruct p_matrix from positive eigenvalues
        p_matrix_new = sum(
            evals[ii] * np.outer(evecs[:, ii], evecs[:, ii])
            for ii in range(clientCount) if evals[ii] >= 0.01
        )

        # Ensure p_matrix is positive semi-definite
        if np.all(np.linalg.eigvals(p_matrix_new) >= 0):
            p_matrix = p_matrix_new

        # Solve the optimization problem using cvxpy
        if np.all(np.linalg.eigvals(p_matrix) >= 0):
            alphav = cvx.Variable(clientCount)
            obj = cvx.Minimize(cvx.quad_form(alphav, p_matrix))
            constraints = [cvx.sum(alphav) == 1.0, alphav >= 0]
            prob = cvx.Problem(obj, constraints)
            prob.solve()
            alpha = np.clip(alphav.value, 0, None)  # Ensure non-negative solutions
            alpha = [a if a > 1e-3 else 0 for a in alpha]  # Zero-out small weights
        else:
            # Fallback to local classifier if no valid solution is found
            alpha = [0.0] * clientCount
            alpha[i] = 1.0

        avgHeadWeight.append(alpha)

    return avgHeadWeight


def fedPacFit(localModel,clientDataTrain,clientLabelTrain,nonOneHotLabels,optimizer,batchSize,epochs,globalPrototype,activityCount,lamda = 1.0):
    agg_protos_label = [[] for _ in range(activityCount)]
    for epoch in range(epochs):
        indices = tf.range(start=0, limit = clientDataTrain.shape[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        batchIndex = [shuffled_indices[i:i+batchSize] for i in range(0,len(shuffled_indices),batchSize)]

        for index in batchIndex:
            x = tf.gather(clientDataTrain, index, axis=0)
            labels = tf.gather(clientLabelTrain, index, axis=0)
            nonOneHotLabelBatch = tf.gather(nonOneHotLabels, index, axis=0)
            prototypeLabel = tf.gather(globalPrototype,nonOneHotLabelBatch)   

            with tf.GradientTape() as tape:
                localPrototype, outputs = localModel(x,training=True)
                cce_loss = tf.keras.losses.CategoricalCrossentropy()(labels, outputs)
                mse_loss = tf.math.reduce_mean(tf.keras.losses.MSE(localPrototype,prototypeLabel))
                # print("cce" + str(cce_loss),flush = True)
                # print("mse" + str(mse_loss),flush = True)
                fedpac_loss = cce_loss + mse_loss * lamda
            grads = tape.gradient(fedpac_loss, localModel.trainable_variables)
            optimizer.apply_gradients(zip(grads, localModel.trainable_variables))

    return None




def fedPac_Trainer(clientNumber,clientDataTrain,clientLabelTrain,clientDataTest,clientLabelTest,globalPrototype,pretrainModel):
    localPrototypesInit = None



    if(pretrainModel):
        localModel = model.HART_MAE((segment_size,num_input_channels),activityCount)
    else:
        localModel = model.HART((segment_size,num_input_channels),activityCount)
    clientModelPath = filepath+'clientModels/localModel'+str(clientNumber)+'.hkl'
    if(os.path.exists(clientModelPath)):
        # load persoalized moedl
        localModel.set_weights(hkl.load(clientModelPath))
    else:
        # load server model if first time in training round
        localModel.load_weights(filepath+'serverWeights.h5')
    

                                
    startTime = time.time()

    for layers in localModel.layers[:embedLayerIndex]:
        layers.trainable = False
        
    foptimizer = tf.keras.optimizers.Adam(clientLearningRate * 10 )
    localModel.compile(optimizer=foptimizer,loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])     


    trainHistory = localModel.fit(clientDataTrain, clientLabelTrain,batch_size = batch_size, epochs = 1,verbose=showTrainVerbose)
    
    
    # increase the learning rate of the classification heads by a magnitude 
    localProtoModel = tf.keras.Model(inputs=localModel.inputs, outputs=[localModel.layers[embedLayerIndex].output, localModel.output])
    localProtoModel.compile(optimizer=foptimizer,loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])  
    nonOneHotLabels = tf.argmax(clientLabelTrain,axis = -1, output_type=tf.dtypes.int32)
    activeProtoypes = None

    # print(globalPrototype)
    agg_protos_label = [[] for _ in range(activityCount)]

    if globalPrototype is None: 
        localPrototypeInit, _ = localProtoModel.predict(clientDataTrain, batch_size = batch_size,verbose=0)
        for activity in range(activityCount):
            matches = tf.equal(nonOneHotLabels, activity)
            matchIndices = tf.where(matches)
            aggregateClassPrototypes = tf.gather(localPrototypeInit,matchIndices[:,0])
            if aggregateClassPrototypes.shape[0] != 0:
                agg_protos_label[activity].append(aggregateClassPrototypes)
            else:
                agg_protos_label[activity].append(tf.zeros((1, localPrototypeInit.shape[-1]))) 
        localPrototypesInit = [tf.math.reduce_mean(groupPrototypes,axis = 1) for groupPrototypes in agg_protos_label]
        activeProtoypes = np.asarray(localPrototypesInit).squeeze()
    else:
        activeProtoypes = globalPrototype
    for layers in localProtoModel.layers[:embedLayerIndex]:
        layers.trainable = True
    for layers in localProtoModel.layers[embedLayerIndex:]:
        layers.trainable = False
    foptimizer2 = tf.keras.optimizers.Adam(clientLearningRate )
    localProtoModel.compile(optimizer=foptimizer,loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])      

    agg_protos = fedPacFit(localProtoModel,clientDataTrain,clientLabelTrain,nonOneHotLabels,foptimizer2,batch_size,localEpoch,activeProtoypes,activityCount)
    
    class_counts = tf.math.bincount(nonOneHotLabels, minlength=clientLabelTrain.shape[1])
    total_samples = tf.reduce_sum(class_counts)
    classBalance = [tf.cast(cls / total_samples,tf.float32) for cls in class_counts]  
    classBalanceSquare = tf.math.square(classBalance)
    
    localRepresentations, _ = localProtoModel.predict(clientDataTrain, batch_size = batch_size,verbose=0)
    new_agg_protos_label = [[] for _ in range(activityCount)]
    h_ref = [[] for _ in range(activityCount)]
    v = 0
    for activity in range(activityCount):
        matches = tf.equal(nonOneHotLabels, activity)
        matchIndices = tf.where(matches)
        aggregateClassPrototypes = tf.gather(localRepresentations,matchIndices[:,0])
        meanRepresentations = tf.math.reduce_mean(aggregateClassPrototypes,axis = 0 )
        
        activityInstance = aggregateClassPrototypes.shape[0]
    
        if(activityInstance != 0):
            new_agg_protos_label[activity].append(aggregateClassPrototypes)
            h_ref[activity] = (meanRepresentations * classBalance[activity])
            v += classBalance[activity] * tf.linalg.trace(tf.matmul(aggregateClassPrototypes, aggregateClassPrototypes, transpose_a=True) / activityInstance)
            v -= classBalanceSquare[activity] * tf.reduce_sum(tf.square(meanRepresentations))
        else:
                # Handle missing labels by appending a None or zeros vector (based on preference)
            new_agg_protos_label[activity].append(tf.zeros((1, localRepresentations.shape[-1]))) 
            h_ref[activity] = tf.zeros(localRepresentations.shape[-1])

    
    v = v/len(clientDataTrain)
    
    newLocalPrototypes = [tf.math.reduce_mean(groupPrototypes,axis = 1) for groupPrototypes in new_agg_protos_label]
    
    # local_sizes_list = np.unique(nonOneHotLabels,return_counts = True)[1] 

    local_sizes_list = np.zeros(activityCount, dtype=int)
    unique_classes, counts = np.unique(nonOneHotLabels, return_counts=True)
    
    # Assign the counts to the corresponding classes
    for cls, count in zip(unique_classes, counts):
        local_sizes_list[cls] = count

    
    fitTime = (time.time() - startTime)/60 
    
    personalizationTrainMetrics = localModel.evaluate(clientDataTrain, clientLabelTrain,batch_size = batch_size,verbose = showTrainVerbose)
    personalizationTestMetrics = localModel.evaluate(clientDataTest, clientLabelTest,batch_size = batch_size, verbose = showTrainVerbose)
    generalizationMetrics = [None, None]
    if(GeneralizationTest):
        generalizationMetrics = localModel.evaluate(centralTestData, centralTestLabel,batch_size = batch_size, verbose = showTrainVerbose)
    localModel.save_weights(clientModelPath)
    gc.collect()

    
    return [clientNumber,
            localModel.get_weights(),
            (h_ref,v,local_sizes_list),
            personalizationTrainMetrics[1],
            personalizationTrainMetrics[0],
            personalizationTestMetrics[1],
            personalizationTestMetrics[0],
            generalizationMetrics[1],
            generalizationMetrics[0],
            fitTime,
            newLocalPrototypes]



def fedPacProtoAgggregation(local_sizes_list,prototypeList):
    totalClassCount = np.sum(local_sizes_list,axis = 0)
    prototypeWeight = np.expand_dims(local_sizes_list / totalClassCount, axis=-1)
    prototypeList = np.asarray([np.asarray(prototypes).squeeze() for prototypes in prototypeList])
    weightedProtoypes  = prototypeList * prototypeWeight
    newPrototypes = np.sum(weightedProtoypes,axis = 0)
    return newPrototypes

def fedProx_Trainer(clientNumber,clientDataTrain,clientLabelTrain,clientDataTest,clientLabelTest,pretrainModel):

    if(pretrainModel):
        localModel = model.HART_MAE((segment_size,num_input_channels),activityCount)
        localModel.load_weights(filepath+'serverWeights.h5')
        serverModel = model.HART_MAE((segment_size,num_input_channels),activityCount)
        serverModel.load_weights(filepath+'serverWeights.h5')

    else:
        localModel = model.HART((segment_size,num_input_channels),activityCount)
        localModel.load_weights(filepath+'serverWeights.h5')
        serverModel = model.HART((segment_size,num_input_channels),activityCount)
        serverModel.load_weights(filepath+'serverWeights.h5')


    startTime = time.time()
    foptimizer = tf.keras.optimizers.Adam(clientLearningRate)
    localModel.compile(optimizer=foptimizer,loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])      
    _ = fedProxFit(localModel,serverModel,clientDataTrain,clientLabelTrain,mu,foptimizer,batch_size,localEpoch)
    trainHistory = localModel.evaluate(clientDataTrain,clientLabelTrain,verbose = 0)
    fitTime = (time.time() - startTime)/60 
    

    personalizationTestMetrics = localModel.evaluate(clientDataTest, clientLabelTest,verbose = showTrainVerbose)
    generalizationMetrics = [None, None]
    if(GeneralizationTest):
        generalizationMetrics = localModel.evaluate(centralTestData, centralTestLabel,verbose = showTrainVerbose)
    
    personalizationTrainAcc = trainHistory[1]
    personalizationTrainloss = trainHistory[0]
    gc.collect()

    
    return [clientNumber,
            localModel.get_weights(),
            None,
            personalizationTrainAcc,
            personalizationTrainloss,
            personalizationTestMetrics[1],
            personalizationTestMetrics[0],
            generalizationMetrics[1],
            generalizationMetrics[0],
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
    personalizationTestMetrics = localModel.evaluate(clientDataTest, clientLabelTest,verbose = showTrainVerbose)
    generalizationMetrics = [None, None]
    if(GeneralizationTest):
        generalizationMetrics = localModel.evaluate(centralTestData, centralTestLabel,verbose = showTrainVerbose)
    personalizationTrainAcc = np.mean(trainHistory.history['acc'])
    personalizationTrainloss = np.mean(trainHistory.history['loss'])
    print("Client Number " +str(clientNumber)+" Train accuracy "+str(personalizationTrainAcc) + " Personalization Accuracy "+str(personalizationTestMetrics[1]) + " Generalization Accuracy " +str(generalizationMetrics[1]),flush=True )

    localModel.save_weights(filepath+'clientModels/localModel'+str(clientNumber)+'.h5')
    gc.collect()

    # we return local Model, but in reality we can use just return prototype. Returning whole model because the script was build to already extract localPrototype from model weights directly
    return [clientNumber,
            localModel.get_weights(),
            None,
            personalizationTrainAcc,
            personalizationTrainloss,
            personalizationTestMetrics[1],
            personalizationTestMetrics[0],
            generalizationMetrics[1],
            generalizationMetrics[0],
            fitTime,
            None]


def fedAli_global_Trainer(clientNumber,clientDataTrain,clientLabelTrain,clientDataTest,
                          clientLabelTest,prototypeCount,localPrototype,adaptLayerLocation,
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
    personalizationTestMetrics = localModel.evaluate(clientDataTest, clientLabelTest,verbose = showTrainVerbose)
    generalizationMetrics = [None, None]
    if(GeneralizationTest):
        generalizationMetrics = localModel.evaluate(centralTestData, centralTestLabel,verbose = showTrainVerbose)
    personalizationTrainAcc = np.mean(trainHistory.history['acc'])
    personalizationTrainloss = np.mean(trainHistory.history['loss'])
    print("Client Number " +str(clientNumber)+" Train accuracy "+str(personalizationTrainAcc) + " Personalization Accuracy "+str(personalizationTestMetrics[1]) + " Generalization Accuracy " +str(generalizationMetrics[1]),flush=True )
    gc.collect()

    return [clientNumber,
            localModel.get_weights(),
            None,
            personalizationTrainAcc,
            personalizationTrainloss,
            personalizationTestMetrics[1],
            personalizationTestMetrics[0],
            generalizationMetrics[1],
            generalizationMetrics[0],
            fitTime,
            None]




def Moon_Trainer(clientNumber,clientDataTrain,clientLabelTrain,clientDataTest,clientLabelTest,clientPrevModelDir,pretrainModel):

    if(pretrainModel):
        localModel = model.HART_MAE((segment_size,num_input_channels),activityCount)

    else:
        localModel = model.HART((segment_size,num_input_channels),activityCount)
    
    # localModel = model.HART((segment_size,num_input_channels),activityCount)
    foptimizer = tf.keras.optimizers.Adam(clientLearningRate)
    floss = tf.keras.losses.CategoricalCrossentropy()
    localModel.compile(optimizer=foptimizer,loss=floss, metrics=['acc'])    
    localModel.load_weights(clientPrevModelDir)    
    modelFE = extract_intermediate_model_from_base_model(localModel,embedLayerIndex)
    prevModelEmbeeddings = modelFE.predict(clientDataTrain, batch_size = batch_size,verbose=0)
    localModel.load_weights(filepath+'serverWeights.h5')
    serverModelEmbeddings = modelFE.predict(clientDataTrain, batch_size = batch_size,verbose=0)
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
    trainHistory = localModel.evaluate(clientDataTrain,clientLabelTrain,verbose = showTrainVerbose)
    personalizationTrainAcc = trainHistory[1]
    personalizationTrainloss = trainHistory[0]
    personalizationTestMetrics = localModel.evaluate(clientDataTest, clientLabelTest,verbose = showTrainVerbose)
    generalizationMetrics = [None, None]
    if(GeneralizationTest):
        generalizationMetrics = localModel.evaluate(centralTestData, centralTestLabel,verbose = showTrainVerbose)

    localModel.save_weights(clientPrevModelDir)
    gc.collect()

    return [clientNumber,
            localModel.get_weights(),
            None,
            personalizationTrainAcc,
            personalizationTrainloss,
            personalizationTestMetrics[1],
            personalizationTestMetrics[0],
            generalizationMetrics[1],
            generalizationMetrics[0],
            fitTime,
            None]



def MoonFiT(localMOONModel, prevModelEmbeddings, serverModelEmbeddings, clientDataTrain,clientLabelTrain, optimizer, mu=1.0,batch_size = 32, epochs=5, embedLayerIndex = 221,verbose=0):
    print(clientDataTrain.shape, flush = True)
    print(clientLabelTrain.shape, flush = True)
    print(prevModelEmbeddings.shape, flush = True)
    print(serverModelEmbeddings.shape, flush = True)

    dataset = tf.data.Dataset.from_tensor_slices((clientDataTrain, clientLabelTrain, prevModelEmbeddings, serverModelEmbeddings))

    for epoch in range(epochs):
        dataset = dataset.shuffle(len(clientDataTrain)).batch(batch_size,drop_remainder=True)
        for batched_train, batched_label, prev_embeds, server_embeds in dataset:
            Moon_NT_Xent_gradients(localMOONModel, embedLayerIndex, batched_train, batched_label, 
                                   prev_embeds, server_embeds, optimizer, mu=mu)

    
    # for epoch in range(epochs):
    #     indices = tf.range(start=0, limit = clientDataTrain.shape[0], dtype=tf.int32)
    #     shuffled_indices = tf.random.shuffle(indices)
    #     shuffled_train = tf.gather(clientDataTrain, shuffled_indices, axis=0)
    #     shuffled_label = tf.gather(clientLabelTrain, shuffled_indices, axis=0)
    #     for i in range(0, shuffled_train.shape[0] ,batch_size):   
    #         batched_train = shuffled_train[i:i+batch_size]
    #         batched_label = shuffled_label[i:i+batch_size]
    #         Moon_NT_Xent_gradients(localMOONModel,
    #                                embedLayerIndex,
    #                                batched_train,
    #                                batched_label,
    #                                prevModelEmbeeddings[i:i+batch_size],
    #                                serverModelEmbeddings[i:i+batch_size],
    #                                optimizer,
    #                                mu=mu)   
    return None
    
@tf.function
def Moon_NT_Xent_gradients(model, embedLayerIndex, trainData,trainLabel, prevModelEmbeds, serverEmbeds,optimizer, temperature=0.5, mu=1.0,):
    """Compute gradients and update weights based on contrastive loss."""
    with tf.GradientTape() as tape:
        client_features,outputs = model(trainData,training=True)
        contrastive_loss = moon_contrastive(client_features,serverEmbeds,prevModelEmbeds, temperature = temperature, mu = mu)
        cce_loss = tf.keras.losses.CategoricalCrossentropy()(trainLabel, outputs)
        moon_loss = cce_loss + contrastive_loss
    grads = tape.gradient(moon_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return moon_loss

@tf.function
def cosine_similarity(x1, x2):
    """Compute cosine similarity between two tensors."""
    x1_normalized = tf.nn.l2_normalize(x1, axis=1)
    x2_normalized = tf.nn.l2_normalize(x2, axis=1)
    
    return tf.reduce_sum(tf.multiply(x1_normalized, x2_normalized), axis=1)
@tf.function
def moon_contrastive(z, zglob, zprev, temperature=1.0, mu=1.0):
    """Compute contrastive loss using previous and global embeddings."""
    sim_z_zglob = cosine_similarity(z, zglob)
    sim_z_zprev = cosine_similarity(z, zprev)

    exp_sim_z_zglob = tf.exp(sim_z_zglob / temperature)
    exp_sim_z_zprev = tf.exp(sim_z_zprev / temperature)

    softmax_denominator = exp_sim_z_zglob + exp_sim_z_zprev
    contrastive_loss = -tf.math.log(exp_sim_z_zglob / softmax_denominator)

    return mu * tf.reduce_mean(contrastive_loss)

        
def setLayersTraining(model,embedLayerIndex,training):
    for i in range(embedLayerIndex):
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

def fedProxFit(localModel,serverModel,clientDataTrain,clientLabelTrain,fedProxMU,optimizer,batchSize,epochs):
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
                mu = tf.constant(fedProxMU, dtype=tf.float32)
                prox_term =(fedProxMU/2)*difference_model_norm_2_square(localModel.trainable_variables, serverModel.trainable_variables)
                fedprox_loss = scce_loss + prox_term
            grads = tape.gradient(fedprox_loss, localModel.trainable_variables)
            optimizer.apply_gradients(zip(grads, localModel.trainable_variables))
    return None

def fedProtoFiT(localModel,clientDataTrain,clientLabelTrain,optimizer,batchSize,epochs,globalPrototype,activityCount, lamda = 1.0):
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
                prototypeLabel = tf.stop_gradient(tf.gather(globalPrototype,nonOneHotLabels))        
                mse_loss = tf.keras.losses.MSE(localPrototype,prototypeLabel)
                fedproto_loss = cce_loss + (mse_loss * lamda)
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
    
def fedProto_Trainer(clientNumber,clientDataTrain,clientLabelTrain,clientDataTest,clientLabelTest,globalPrototype,pretrainModel):

    if(pretrainModel):
        localModel = model.HART_MAE((segment_size,num_input_channels),activityCount)
    else:
        localModel = model.HART((segment_size,num_input_channels),activityCount)
    
    clientModelPath = filepath+'clientModels/localModel'+str(clientNumber)+'.h5'
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
    
        # loss_acc = localModel.evaluate(clientDataTrain,clientLabelTrain,verbose = 0)
    personalizationTrainMetrics = localModel.evaluate(clientDataTrain, clientLabelTrain,verbose = showTrainVerbose)
    personalizationTestMetrics = localModel.evaluate(clientDataTest, clientLabelTest,verbose = showTrainVerbose)
    generalizationMetrics = [None, None]
    if(GeneralizationTest):
        generalizationMetrics = localModel.evaluate(centralTestData, centralTestLabel,verbose = showTrainVerbose)
    localModel.save_weights(clientModelPath)

    gc.collect()

    return [clientNumber,
            localModel.get_weights(),
            None,
            personalizationTrainMetrics[1],
            personalizationTrainMetrics[0],
            personalizationTestMetrics[1],
            personalizationTestMetrics[0],
            generalizationMetrics[1],
            generalizationMetrics[0],
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
    gc.collect()

    return [localModel.get_weights(),
            personalizationTrainMetrics[1],
            personalizationTestMetrics[1],
            generalizationMetrics[1]]

def fedPer_Trainer(clientNumber,clientDataTrain,clientLabelTrain,clientDataTest,clientLabelTest,pretrainModel):
    clientModelPath = filepath+'/clientModels/localModel'+str(clientNumber)+'.h5'

    if(pretrainModel):
        localModel = model.HART_MAE((segment_size,num_input_channels),activityCount)
    else:
        localModel = model.HART((segment_size,num_input_channels),activityCount)
    
    # localModel = model.HART((segment_size,num_input_channels),activityCount)
    if(os.path.exists(clientModelPath)):
        # use old client model, since model is not transmitted
        localModel.load_weights(clientModelPath)
        personalLayerWeights = copy.deepcopy([personalLayer.get_weights() for personalLayer in localModel.layers[embedLayerIndex+1:]])
        localModel.load_weights(filepath+'serverWeights.h5')
        for layerIdx, layerWeight in enumerate(personalLayerWeights):
            localModel.layers[embedLayerIndex+1+layerIdx].set_weights(layerWeight)
    else:
        # take the server model for the first time
        localModel.load_weights(filepath+'serverWeights.h5')
    
    startTime = time.time()
    localModel.compile(optimizer=tf.keras.optimizers.Adam(clientLearningRate),loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])      
    trainHistory = localModel.fit(clientDataTrain, clientLabelTrain,batch_size = batch_size, epochs = localEpoch,verbose=showTrainVerbose)
    fitTime = (time.time() - startTime)/60 
    personalizationTestMetrics = localModel.evaluate(clientDataTest, clientLabelTest,verbose = showTrainVerbose)
    generalizationMetrics = [None, None]
    if(GeneralizationTest):
        generalizationMetrics = localModel.evaluate(centralTestData, centralTestLabel,verbose = showTrainVerbose)
    personalizationTrainAcc = np.mean(trainHistory.history['acc'])
    personalizationTrainloss = np.mean(trainHistory.history['loss'])
    print("Client Number " +str(clientNumber)+" Train accuracy "+str(personalizationTrainAcc) + " Personalization Accuracy "+str(personalizationTestMetrics[1]) + " Generalization Accuracy " +str(generalizationMetrics[1]),flush=True )
    
    localModel.save_weights(clientModelPath)
    gc.collect()

    return [clientNumber,
            localModel.get_weights(),
            None,
            personalizationTrainAcc,
            personalizationTrainloss,
            personalizationTestMetrics[1],
            personalizationTestMetrics[0],
            generalizationMetrics[1],
            generalizationMetrics[0],
            fitTime,
            None]



def set_evaluator(shared_vars,GPULIST,devID):
    global deviceIndex 
    deviceIndex = devID
    with deviceIndex.get_lock():
        globals()['gpu_id'] = deviceIndex.value
        if(deviceIndex.value >= len(GPULIST)-1):
            deviceIndex.value = -1
        deviceIndex.value += 1
    print(f" client_id: {deviceIndex.value} training on GPU id:{GPULIST[deviceIndex.value]}",flush= True)
    set_specific_gpu(GPULIST[deviceIndex.value])    

    global centralTestData
    global centralTestLabel
    global strategyFL
    global loadPretrain
    global segment_size
    global num_input_channels
    global activityCount


#     global GPUPoolIndex
    centralTestData = shared_vars[0].value
    centralTestLabel = shared_vars[1].value
    strategyFL = shared_vars[2].value
    loadPretrain = shared_vars[3].value


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



def strategy_evaluator(clientNumber,clientDataTest,clientLabelTest):
    print("Loading Client "+str(i))
    if(strategyFL == 'FEDALI'):
        if(loadPretrain):
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
    else:
        if(loadPretrain):
            localModel = model.HART_MAE((segment_size,num_input_channels),activityCount)
        else:
            localModel = model.HART((segment_size,num_input_channels),activityCount)
    
    localModel.set_weights(hkl.load(bestModelPath + "bestModel"+str(i)+".hkl"))
    y_pred = np.argmax(localModel.predict(clientDataTest[i],verbose = showTrainVerbose), axis=-1)
    y_test = np.argmax(clientLabelTest[i], axis=-1)

    weightedPersonalzation = f1_score(y_test, y_pred,average='weighted' )
    microPersonalzation = f1_score(y_test, y_pred,average='micro' )
    macroPersonalzation = f1_score(y_test, y_pred,average='macro' )

    perosonalizationMetric = (weightedPersonalzation,microPersonalzation,macroPersonalzation)
    
    y_pred = np.argmax(localModel.predict(centralTestData,verbose = showTrainVerbose), axis=-1)
    y_test = np.argmax(centralTestLabel, axis=-1)    

    weightedGeneralization = f1_score(y_test, y_pred,average='weighted')
    microGeneralization = f1_score(y_test, y_pred,average='micro')
    macroGeneralization = f1_score(y_test, y_pred,average='macro')

    generalizationMetric = (weightedGeneralization,microGeneralization,macroGeneralization)

    return perosonalizationMetric,generalizationMetric



