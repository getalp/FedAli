#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import mae_model
import alp_model
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, MaxPool1D, Concatenate, Activation, Add, GlobalAveragePooling1D, \
    Dense, BatchNormalization
from tensorflow.keras.regularizers import l2


randomSeed = 1
# tf.random.set_seed(randomSeed)
# def createAndLoadHART_global(algorithm,activityCount,loadPretrain,pretrain_dir, frame_length = 16,enc_embedding_size = 192,convKernels = [3, 7, 15, 31, 31, 31]):

#     patch_layer = mae_model.SensorWiseFrameLayer(frame_length,frame_length)
#     patch_encoder = mae_model.SensorWisePatchEncoder(frame_length,enc_embedding_size,True,0.6)    
#     if(algorithm == "FEDALP"):
#         memoryCount = 1024
#         mae_encoder = alp_model.HART_ALP_localGlobal_encoder(projection_dim = enc_embedding_size,                                 
#                                                      num_heads = 3,
#                                                      filterAttentionHead = 4, 
#                                                      memoryBankSize = memoryCount,
#                                                      convKernels = convKernels)
#     else:
#         mae_encoder = mae_model.HART_encoder(enc_embedding_size,                                                     
#                                          num_heads = 3,
#                                          filterAttentionHead = 4, 
#                                          convKernels = [3, 7, 15, 31, 31, 31])
#     HART_model = tf.keras.Sequential(
#     [
#         layers.Input((128,6)),
#         patch_layer,
#         patch_encoder,
#         mae_encoder,
#         layers.GlobalAveragePooling1D(),
#     ])
#     if(loadPretrain):
#         HART_model.load_weights(pretrain_dir)
#     HART_model = create_classification_model_from_base_model(HART_model,activityCount, model_name = 'server_model')



# def create_classification_model_from_base_model(base_model, output_shape, model_name,dropout_rate = 0.3):
#     intermediate_x = base_model.output
#     x = tf.keras.layers.Dense(1024, activation=tf.nn.swish)(intermediate_x)
#     x = tf.keras.layers.Dropout(dropout_rate)(x)
#     outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
#     model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs, name=model_name)

#     # model.compile(
#     #     optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
#     #     loss=tf.keras.losses.CategoricalCrossentropy(),
#     #     metrics=["accuracy"]
#     # )
#     return model


def createAndLoadHART(activityCount,loadPretrain,pretrain_dir, frame_length = 16,enc_embedding_size = 192,convKernels = [3, 7, 15, 31, 31, 31]):


    if(loadPretrain):
        patch_layer = mae_model.SensorWiseFrameLayer(frame_length,frame_length)
        patch_encoder = mae_model.SensorWisePatchEncoder(frame_length,enc_embedding_size,True,0.6)    
        mae_encoder = mae_model.HART_encoder(enc_embedding_size,                                                     
                                             num_heads = 3,
                                             filterAttentionHead = 4, 
                                             convKernels = [3, 7, 15, 31, 31, 31])
        HART_model = tf.keras.Sequential(
        [
            layers.Input((128,6)),
            patch_layer,
            patch_encoder,
            mae_encoder,
            layers.GlobalAveragePooling1D(),
        ])
        HART_model.load_weights(pretrain_dir)
        HART_model = create_classification_model_from_base_model(HART_model,activityCount, model_name = 'server_model')

    else:
        HART_model = HART((128,6),activityCount)
    return HART_model

def createAndLoadHART_ALP(prototypeCount,activityCount,loadPretrain,pretrain_dir = "", frame_length = 16,enc_embedding_size = 192,convKernels = [3, 7, 15, 31, 31, 31],useGLU = True, influenceFactor = 0.2, singleUpdate = True, matchMethod = 'OT'):
    patch_layer = mae_model.SensorWiseFrameLayer(frame_length,frame_length)
    patch_encoder = mae_model.SensorWisePatchEncoder(frame_length,enc_embedding_size,True,0.6)   

    if(loadPretrain):
        mae_encoder = alp_model.HART_ALP_globalLocal_encoder(prototypeCount = prototypeCount,
                                                    projection_dim = enc_embedding_size,
                                                    num_heads = 3,
                                                    filterAttentionHead = 4, 
                                                    convKernels = convKernels,
                                                    useGLU = useGLU,
                                                    influenceFactor = influenceFactor,
                                                    singleUpdate = singleUpdate,
                                                    matchMethod = matchMethod)
        HART_model = tf.keras.Sequential(
        [
            layers.Input((128,6)),
            patch_layer,
            patch_encoder,
            mae_encoder,
            layers.GlobalAveragePooling1D(),
        ])
        HART_model.load_weights(pretrain_dir)
        HART_model = create_classification_model_from_base_model(HART_model,activityCount, model_name = 'server_model')

    else:
        HART_model = HART_ALP(input_shape = (128,6),
                              activityCount = activityCount,
                              prototypeDecay = 0.96,
                              influenceFactor = influenceFactor, 
                              prototypeCount = prototypeCount,
                              useGLU = useGLU,
                              singleUpdate = singleUpdate,
                              matchMethod = matchMethod)    
    return HART_model


def create_classification_model_from_base_model(base_model, output_shape, model_name,dropout_rate = 0.3):
    intermediate_x = base_model.output
    x = tf.keras.layers.Dense(1024, activation=tf.nn.swish)(intermediate_x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

def ispl_inception_alp(x_shape,
                   n_classes,
                   prototypeCount,
                   filters_number = 64,
                   network_depth=5,
                   use_residual=True,
                   use_bottleneck=True,
                   max_kernel_size=20,
                   learning_rate=0.01,
                   bottleneck_size=32,
                   regularization_rate=0.01,
                   memorySlotSize = 1,
                   decayRate = 0.96,
                   metrics=['accuracy']):
    dim_length = x_shape[0]  # number of samples in a time series
    dim_channels = x_shape[1]  # number of channels
    weightinit = 'lecun_uniform'  # weight initialization

    def inception_module(input_tensor, stride=1, activation='relu'):

        # The  channel number is greater than 1
        if use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(filters=bottleneck_size,
                                     kernel_size=1,
                                     padding='same',
                                     activation=activation,
                                     kernel_initializer=weightinit,

                                     use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_sizes = [max_kernel_size // (2 ** i) for i in range(3)]
        conv_list = []

        for kernel_size in kernel_sizes:
            conv_list.append(Conv1D(filters=filters_number,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    padding='same',
                                    activation=activation,
                                    kernel_initializer=weightinit,
                                    kernel_regularizer=l2(regularization_rate),
                                    use_bias=False)(input_inception))

        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_last = Conv1D(filters=filters_number,
                           kernel_size=1,
                           padding='same',
                           activation=activation,
                           kernel_initializer=weightinit,
                           kernel_regularizer=l2(regularization_rate),
                           use_bias=False)(max_pool_1)

        conv_list.append(conv_last)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        return x

    def shortcut_layer(input_tensor, out_tensor):
        shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]),
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=weightinit,
                            kernel_regularizer=l2(regularization_rate),
                            use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        x = Add()([shortcut_y, out_tensor])
        x = Activation('relu')(x)
        return x

    # Build the actual model:
    input_layer = Input((dim_length, dim_channels))
    x = BatchNormalization()(input_layer)  # Added batchnorm (not in original paper)
    input_res = x

    for depth in range(network_depth):
        # tf.print(x.shape)
        x = inception_module(x)
        if use_residual and depth % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x
        x = AdaptiveLayer(layer = depth, 
           projection_dim = 256,
           memoryBankSize = prototypeCount[depth], 
           memorySlot = memorySlotSize, 
           decay = decayRate)(x) 
        # tf.print(x.shape)
    
    gap_layer = GlobalAveragePooling1D()(x)

    # Final classification layer
    output_layer = Dense(n_classes, activation='softmax',
                         kernel_initializer=weightinit, kernel_regularizer=l2(regularization_rate))(gap_layer)

    # Create model and compile
    m = Model(inputs=input_layer, outputs=output_layer)

    # m.compile(loss=out_loss,
    #           optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
    #           metrics=metrics)

    return m

def ispl_inception(x_shape,
                   n_classes,
                   filters_number = 64,
                   network_depth=5,
                   use_residual=True,
                   use_bottleneck=True,
                   max_kernel_size=20,
                   learning_rate=0.01,
                   bottleneck_size=32,
                   regularization_rate=0.01,
                   metrics=['accuracy']):
    dim_length = x_shape[0]  # number of samples in a time series
    dim_channels = x_shape[1]  # number of channels
    weightinit = 'lecun_uniform'  # weight initialization

    def inception_module(input_tensor, stride=1, activation='relu'):

        # The  channel number is greater than 1
        if use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(filters=bottleneck_size,
                                     kernel_size=1,
                                     padding='same',
                                     activation=activation,
                                     kernel_initializer=weightinit,

                                     use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_sizes = [max_kernel_size // (2 ** i) for i in range(3)]
        conv_list = []

        for kernel_size in kernel_sizes:
            conv_list.append(Conv1D(filters=filters_number,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    padding='same',
                                    activation=activation,
                                    kernel_initializer=weightinit,
                                    kernel_regularizer=l2(regularization_rate),
                                    use_bias=False)(input_inception))

        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_last = Conv1D(filters=filters_number,
                           kernel_size=1,
                           padding='same',
                           activation=activation,
                           kernel_initializer=weightinit,
                           kernel_regularizer=l2(regularization_rate),
                           use_bias=False)(max_pool_1)

        conv_list.append(conv_last)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        return x

    def shortcut_layer(input_tensor, out_tensor):
        shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]),
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=weightinit,
                            kernel_regularizer=l2(regularization_rate),
                            use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        x = Add()([shortcut_y, out_tensor])
        x = Activation('relu')(x)
        return x

    # Build the actual model:
    input_layer = Input((dim_length, dim_channels))
    x = BatchNormalization()(input_layer)  # Added batchnorm (not in original paper)
    input_res = x

    for depth in range(network_depth):
        # tf.print(x.shape)
        x = inception_module(x)
        if use_residual and depth % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x



        # x = MemNodeV4_GLU(layer = depth, 
        #    projection_dim = 256,
        #    generateMemories = generateMemories, 
        #    memoryBankSize = memoryBankSize, 
        #    memorySlot = memorySlotSize, 
        #    decay = decayRate)(x) 
        # tf.print(x.shape)
    
    gap_layer = GlobalAveragePooling1D()(x)

    # Final classification layer
    output_layer = Dense(n_classes, activation='softmax',
                         kernel_initializer=weightinit, kernel_regularizer=l2(regularization_rate))(gap_layer)

    # Create model and compile
    m = Model(inputs=input_layer, outputs=output_layer)

    # m.compile(loss=out_loss,
    #           optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
    #           metrics=metrics)

    return m

class DropPath(layers.Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x,training=None):
        if(training):
            input_shape = tf.shape(x)
            batch_size = input_shape[0]
            rank = x.shape.rank
            shape = (batch_size,) + (1,) * (rank - 1)
            random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
            path_mask = tf.floor(random_tensor)
            output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
            return output
        else:
            return x 

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'drop_prob': self.drop_prob,})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim,**kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = patch + self.position_embedding(positions)
        return encoded
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,})
        return config



    
class SensorWiseMHA(layers.Layer):
    def __init__(self, projectionQuarter, num_heads,startIndex,stopIndex,dropout_rate = 0.0,dropPathRate = 0.0, **kwargs):
        super(SensorWiseMHA, self).__init__(**kwargs)
        self.projectionQuarter = projectionQuarter
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.MHA = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projectionQuarter, dropout = dropout_rate )
        self.startIndex = startIndex
        self.stopIndex = stopIndex
        self.dropPathRate = dropPathRate
        self.DropPath = DropPath(dropPathRate)
    def call(self, inputData, training=None, return_attention_scores = False):
        extractedInput = inputData[:,:,self.startIndex:self.stopIndex]
        if(return_attention_scores):
            MHA_Outputs, attentionScores = self.MHA(extractedInput,extractedInput,return_attention_scores = True )
            return MHA_Outputs , attentionScores
        else:
            MHA_Outputs = self.MHA(extractedInput,extractedInput)
            MHA_Outputs = self.DropPath(MHA_Outputs)
            return MHA_Outputs
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionQuarter': self.projectionQuarter,
            'num_heads': self.num_heads,
            'startIndex': self.startIndex,
            'dropout_rate': self.dropout_rate,
            'stopIndex': self.stopIndex,
            'dropPathRate': self.dropPathRate,})
        return config
def softDepthConv(inputs):
    kernel = inputs[0]
    inputData = inputs[1]
    convOutputs = tf.nn.conv1d(
    inputData,
    kernel,
    stride = 1,
    padding = 'SAME',
    data_format='NCW',)
    return convOutputs




class liteFormer(layers.Layer):
    def __init__(self,startIndex,stopIndex, projectionSize, kernelSize = 16, attentionHead = 3, use_bias=False, dropPathRate = 0.0,dropout_rate = 0,**kwargs):
        super(liteFormer, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.startIndex = startIndex
        self.stopIndex = stopIndex
        self.kernelSize = kernelSize
        self.softmax = tf.nn.softmax
        self.projectionSize = projectionSize
        self.attentionHead = attentionHead 
        self.DropPathLayer = DropPath(dropPathRate)
        self.projectionHalf = projectionSize // 2
    def build(self,inputShape):
        self.depthwise_kernel = [self.add_weight(
            shape=(self.kernelSize,1,1),
            initializer="glorot_uniform",
            trainable=True,
            name="convWeights"+str(_),
            dtype="float32") for _ in range(self.attentionHead)]
        if self.use_bias:
            self.convBias = self.add_weight(
                shape=(self.attentionHead,), 
                initializer="glorot_uniform", 
                trainable=True,  
                name="biasWeights",
                dtype="float32"
            )
        
    def call(self, inputs,training=None):
        formattedInputs = inputs[:,:,self.startIndex:self.stopIndex]
        inputShape = tf.shape(formattedInputs)
        reshapedInputs = tf.reshape(formattedInputs,(-1,self.attentionHead,inputShape[1]))
        if(training):
            for convIndex in range(self.attentionHead):
                self.depthwise_kernel[convIndex].assign(self.softmax(self.depthwise_kernel[convIndex], axis=0))
        convOutputs = tf.convert_to_tensor([tf.nn.conv1d(
            reshapedInputs[:,convIndex:convIndex+1,:],
            self.depthwise_kernel[convIndex],
            stride = 1,
            padding = 'SAME',
            data_format='NCW',) for convIndex in range(self.attentionHead) ])
        convOutputsDropPath = self.DropPathLayer(convOutputs)
        localAttention = tf.reshape(convOutputsDropPath,(-1,inputShape[1],self.projectionSize))
        return localAttention
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'use_bias': self.use_bias,
            'kernelSize': self.kernelSize,
            'startIndex': self.startIndex,
            'stopIndex': self.stopIndex,
            'projectionSize': self.projectionSize,
            'attentionHead': self.attentionHead,})
        return config          

class mixAccGyro(layers.Layer):
    def __init__(self,projectionQuarter,projectionHalf,projection_dim,**kwargs):
        super(mixAccGyro, self).__init__(**kwargs)
        self.projectionQuarter = projectionQuarter
        self.projectionHalf = projectionHalf
        self.projection_dim = projection_dim
        self.projectionThreeFourth = self.projectionHalf+self.projectionQuarter
        self.mixedAccGyroIndex = tf.reshape(tf.transpose(tf.stack(
            [np.arange(projectionQuarter,projectionHalf), np.arange(projectionHalf,projectionHalf + projectionQuarter)])),[-1])
        self.newArrangement = tf.concat((np.arange(0,projectionQuarter),self.mixedAccGyroIndex,np.arange(self.projectionThreeFourth,projection_dim)),axis = 0)
    def call(self, inputs):
        return tf.gather(inputs,self.newArrangement,axis= 2)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionQuarter': self.projectionQuarter,
            'projectionHalf': self.projectionHalf,
            'projection_dim': self.projection_dim,
        })
        return config

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def mlp2(x, hidden_units, dropout_rate):
    x = layers.Dense(hidden_units[0],activation=tf.nn.swish)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_units[1])(x)
    return x

def depthMLP(x, hidden_units, dropout_rate):
    x = layers.Dense(hidden_units[0])(x)
    x = layers.DepthwiseConv1D(3,data_format='channels_first',activation=tf.nn.swish)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_units[1])(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

class SensorPatchesTimeDistributed(layers.Layer):
    def __init__(self, projection_dim,filterCount,patchCount,frameSize = 128, channelsCount = 6,**kwargs):
        super(SensorPatchesTimeDistributed, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.frameSize = frameSize
        self.channelsCount = channelsCount
        self.patchCount = patchCount
        self.filterCount = filterCount
        self.reshapeInputs = layers.Reshape((patchCount, frameSize // patchCount, channelsCount))
        self.kernelSize = (projection_dim//2 + filterCount) // filterCount
        self.accProjection = layers.TimeDistributed(layers.Conv1D(filters = filterCount,kernel_size = self.kernelSize,strides = 1, data_format = "channels_last"))
        self.gyroProjection = layers.TimeDistributed(layers.Conv1D(filters = filterCount,kernel_size = self.kernelSize,strides = 1, data_format = "channels_last"))
        self.flattenTime = layers.TimeDistributed(layers.Flatten())
        assert (projection_dim//2 + filterCount) / filterCount % self.kernelSize == 0
        print("Kernel Size is "+str((projection_dim//2 + filterCount) / filterCount))
#         assert 
    def call(self, inputData):
        inputData = self.reshapeInputs(inputData)
        accProjections = self.flattenTime(self.accProjection(inputData[:,:,:,:3]))
        gyroProjections = self.flattenTime(self.gyroProjection(inputData[:,:,:,3:]))
        Projections = tf.concat((accProjections,gyroProjections),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projection_dim': self.projection_dim,
            'filterCount': self.filterCount,
            'patchCount': self.patchCount,
            'frameSize': self.frameSize,
            'channelsCount': self.channelsCount,})
        return config
    
class SensorPatches(layers.Layer):
    def __init__(self, projection_dim, patchSize,timeStep, **kwargs):
        super(SensorPatches, self).__init__(**kwargs)
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.projection_dim = projection_dim
        self.accProjection = layers.Conv1D(filters = int(projection_dim/2),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.gyroProjection = layers.Conv1D(filters = int(projection_dim/2),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
    def call(self, inputData):

        accProjections = self.accProjection(inputData[:,:,:3])
        gyroProjections = self.gyroProjection(inputData[:,:,3:])
        Projections = tf.concat((accProjections,gyroProjections),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patchSize': self.patchSize,
            'projection_dim': self.projection_dim,
            'timeStep': self.timeStep,})
        return config


class threeSensorPatches(layers.Layer):
    def __init__(self, projection_dim, patchSize,timeStep, **kwargs):
        super(threeSensorPatches, self).__init__(**kwargs)
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.projection_dim = projection_dim
        self.accProjection = layers.Conv1D(filters = int(projection_dim//3),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.gyroProjection = layers.Conv1D(filters = int(projection_dim//3),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.magProjection = layers.Conv1D(filters = int(projection_dim//3),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")

    def call(self, inputData):

        accProjections = self.accProjection(inputData[:,:,:3])
        gyroProjections = self.gyroProjection(inputData[:,:,3:6])
        magProjections = self.magProjection(inputData[:,:,6:])

        Projections = tf.concat((accProjections,gyroProjections,magProjections),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patchSize': self.patchSize,
            'projection_dim': self.projection_dim,
            'timeStep': self.timeStep,})
        return config

        
class fourSensorPatches(layers.Layer):
    def __init__(self, projection_dim, patchSize,timeStep, **kwargs):
        super(fourSensorPatches, self).__init__(**kwargs)
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.projection_dim = projection_dim
        self.accProjection = layers.Conv1D(filters = int(projection_dim/4),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.gyroProjection = layers.Conv1D(filters = int(projection_dim/4),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.magProjection = layers.Conv1D(filters = int(projection_dim/4),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.altProjection = layers.Conv1D(filters = int(projection_dim/4),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")

    def call(self, inputData):

        accProjections = self.accProjection(inputData[:,:,:3])
        gyroProjections = self.gyroProjection(inputData[:,:,3:6])
        magProjection = self.magProjection(inputData[:,:,6:9])
        altProjection = self.altProjection(inputData[:,:,9:])

        Projections = tf.concat((accProjections,gyroProjections,magProjection,altProjection),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patchSize': self.patchSize,
            'projection_dim': self.projection_dim,
            'timeStep': self.timeStep,})
        return config

def extract_intermediate_model_from_base_model(base_model, intermediate_layer=4):
    model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[intermediate_layer].output, name=base_model.name + "_layer_" + str(intermediate_layer))
    return model
        
def HART(input_shape,activityCount, projection_dim = 192,patchSize = 16,timeStep = 16,num_heads = 3,filterAttentionHead = 4, convKernels = [3, 7, 15, 31, 31, 31], mlp_head_units = [1024],dropout_rate = 0.3):
    projectionHalf = projection_dim//2
    projectionQuarter = projection_dim//4
    dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
    transformer_units = [
    projection_dim * 2,
    projection_dim,]  
    inputs = layers.Input(shape=input_shape)
    patches = SensorPatches(projection_dim,patchSize,timeStep)(inputs)
    patchCount = patches.shape[1] 
    encoded_patches = PatchEncoder(patchCount, projection_dim)(patches)
    # Create multiple layers of the Transformer block.
    for layerIndex, kernelLength in enumerate(convKernels):        
        x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
        branch1 = liteFormer(
                          startIndex = projectionQuarter,
                          stopIndex = projectionQuarter + projectionHalf,
                          projectionSize = projectionHalf,
                          attentionHead =  filterAttentionHead, 
                          kernelSize = kernelLength,
                          dropPathRate = dropPathRate[layerIndex],
                          dropout_rate = dropout_rate,
                          name = "liteFormer_"+str(layerIndex))(x1)

                          
        branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x1)

        branch2Gyro = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf ,projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "GyroMHA_"+str(layerIndex))(x1)
        concatAttention = layers.Concatenate(axis=2)((branch2Acc,branch1,branch2Gyro))
        # concatAttention = tf.concat((branch2Acc,branch1,branch2Gyro),axis= 2 )
        x2 = layers.Add()([concatAttention, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp2(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        x3 = DropPath(dropPathRate[layerIndex])(x3)
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    GAP_Ouput = layers.GlobalAveragePooling1D()(representation)
    features = mlp(GAP_Ouput, hidden_units=mlp_head_units, dropout_rate=dropout_rate)
    logits = layers.Dense(activityCount,  activation='softmax')(features)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model



def getLayerIndexByName(model, layername):
    layerIndex = []
    for idx, layer in enumerate(model.layers):
        if layername in layer.name:
            layerIndex.append(idx)
    return layerIndex



def HART_ALP(input_shape,activityCount, projection_dim = 192,
             patchSize = 16,
             timeStep = 16,
             num_heads = 3,
             filterAttentionHead = 4, 
             convKernels = [3, 7, 15, 31, 31, 31], 
             mlp_head_units = [1024],
             dropout_rate = 0.3, 
             prototypeCount = 512,
             prototypeDecay = 0.96,
             influenceFactor = 0.2,
             useGLU = True,
             singleUpdate = False,
             matchMethod = 'OT'):
    projectionHalf = projection_dim//2
    projectionQuarter = projection_dim//4
    dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
    transformer_units = [
    projection_dim * 2,
    projection_dim,]  
    inputs = layers.Input(shape=input_shape)
    patches = SensorPatches(projection_dim,patchSize,timeStep)(inputs)
    patchCount = patches.shape[1] 
    encoded_patches = PatchEncoder(patchCount, projection_dim)(patches)
    # Create multiple layers of the Transformer block.
    for layerIndex, kernelLength in enumerate(convKernels):        
        x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
        x1_alp = alp_model.AlignmentWithProtoype(layer = layerIndex, 
                            matchMethod = matchMethod,
                            projection_dim = projection_dim,
                            memoryBankSize = prototypeCount[layerIndex], 
                            prototypeDecay = prototypeDecay,
                            influenceFactor = influenceFactor)(x1) 
        branch1 = liteFormer(
                          startIndex = projectionQuarter,
                          stopIndex = projectionQuarter + projectionHalf,
                          projectionSize = projectionHalf,
                          attentionHead =  filterAttentionHead, 
                          kernelSize = kernelLength,
                          dropPathRate = dropPathRate[layerIndex],
                          dropout_rate = dropout_rate,
                          name = "liteFormer_"+str(layerIndex))(x1_alp)

                          
        branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x1_alp)
        branch2Gyro = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf ,projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "GyroMHA_"+str(layerIndex))(x1_alp)
        # concatAttention = tf.concat((branch2Acc,branch1,branch2Gyro),axis= 2 )

        concatAttention = layers.Concatenate(axis=2)((branch2Acc,branch1,branch2Gyro))
        x2 = layers.Add()([concatAttention, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp2(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        x3 = DropPath(dropPathRate[layerIndex])(x3)
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    GAP_Ouput = layers.GlobalAveragePooling1D()(representation)
    features = mlp(GAP_Ouput, hidden_units=mlp_head_units, dropout_rate=dropout_rate)
    logits = layers.Dense(activityCount,  activation='softmax')(features)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model


def HART_ALP_GLOBAL(input_shape,
             activityCount, 
             prototypeCount,
             enc_embedding_size = 192, 
             num_heads = 3,
             filterAttentionHead = 4, 
             convKernels = [3, 7, 15, 31, 31, 31], 
             mlp_head_units = 1024,
             dropout_rate = 0.5,
             frame_length = 16,
             decay = 0.96,
             useGlobal = True,
             prototypeDecay = 0.96,
             influenceFactor = 0.2,
             useGLU = True,
             singleUpdate = True):
    patch_layer = mae_model.SensorWiseFrameLayer(frame_length,frame_length)
    patch_encoder = mae_model.SensorWisePatchEncoder(frame_length,enc_embedding_size,True,0.6) 
    mae_alp_encoder = alp_model.HART_ALP_globalLocal_encoder( prototypeCount = prototypeCount,
                                                         projection_dim = enc_embedding_size,                                                     
                                                         num_heads = num_heads,
                                                         filterAttentionHead = filterAttentionHead, 
                                                         convKernels = convKernels,
                                                         prototypeDecay = prototypeDecay,
                                                         influenceFactor = influenceFactor,
                                                         useGLU = useGLU,
                                                         singleUpdate = singleUpdate)
    
    sequentialModel = tf.keras.Sequential(
    [
        layers.InputLayer(input_shape=input_shape),
        patch_layer,
        patch_encoder,
        mae_alp_encoder,
        layers.GlobalAveragePooling1D(),
        layers.Dense(mlp_head_units, activation=tf.nn.swish),
        layers.Dropout(dropout_rate),
        layers.Dense(activityCount,  activation='softmax')
    ])
    
    return sequentialModel





def HART_MAE(input_shape,activityCount, 
             enc_embedding_size = 192, 
             num_heads = 3,
             filterAttentionHead = 4, 
             convKernels = [3, 7, 15, 31, 31, 31], 
             mlp_head_units = 1024,
             dropout_rate = 0.5,
             frame_length = 16):
    patch_layer = mae_model.SensorWiseFrameLayer(frame_length,frame_length)
    patch_encoder = mae_model.SensorWisePatchEncoder(frame_length,enc_embedding_size,True,0.6) 
    mae_alp_encoder = mae_model.HART_encoder(enc_embedding_size,                                                     
                                         num_heads = num_heads,
                                         filterAttentionHead = filterAttentionHead, 
                                         convKernels = convKernels)
    sequentialModel = tf.keras.Sequential(
    [  
        layers.InputLayer(input_shape=input_shape),
        patch_layer,
        patch_encoder,
        mae_alp_encoder,
        layers.GlobalAveragePooling1D(),
        layers.Dense(mlp_head_units, activation=tf.nn.swish),
        layers.Dropout(dropout_rate),
        layers.Dense(activityCount,  activation='softmax')
    ])
    
    return sequentialModel
