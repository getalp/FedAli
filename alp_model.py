#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

from tensorflow.keras import layers
import numpy as np
import model 

randomSeed = 1
# tf.random.set_seed(randomSeed)

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

class GatedLinearUnit(layers.Layer):
    def __init__(self,units,**kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.units = units
        self.sigmoid = tf.keras.activations.sigmoid

    def build(self, input_shape):  # Create the state of the layer (weights)
        self.linear = layers.Dense(self.units * 2)

    def call(self, inputs):
        linearProjection = self.linear(inputs)
        softMaxProjection = self.sigmoid(linearProjection[:,self.units:])
        return tf.multiply(linearProjection[:,:self.units],softMaxProjection)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,})
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

class ClassToken(layers.Layer):
    def __init__(self, hidden_size,**kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.cls_init = tf.random.normal
        self.hidden_size = hidden_size
        self.cls = tf.Variable(
            name="cls",
            initial_value=self.cls_init(shape=(1, 1, self.hidden_size), seed=randomSeed, dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_size': self.hidden_size,})
        return config

class Prompts(layers.Layer):
    def __init__(self, projectionDims,promptCount = 1,**kwargs):
        super(Prompts, self).__init__(**kwargs)
        self.cls_init = tf.random.normal
        self.projectionDims = projectionDims
        self.promptCount = promptCount
        self.prompts = [tf.Variable(
            name="prompt"+str(_),
            initial_value=self.cls_init(shape=(1, 1, self.projectionDims), seed=randomSeed, dtype="float32"),
            trainable=True,
        )  for _ in range(promptCount)]

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        prompt_broadcasted = tf.concat([tf.cast(tf.broadcast_to(promptInits, [batch_size, 1, self.projectionDims]),dtype=inputs.dtype,)for promptInits in self.prompts],1)
        return tf.concat([inputs,prompt_broadcasted], 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionDims': self.projectionDims,
            'promptCount': self.promptCount,})
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


def extract_intermediate_model_from_base_model(base_model, intermediate_layer=4):
    model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[intermediate_layer].output, name=base_model.name + "_layer_" + str(intermediate_layer))
    return model


class AlignmentWithProtoype(layers.Layer):
    """
    A TensorFlow layer for aligning feature projections with local and global prototypes 
    using various matching methods (e.g., Optimal Transport, cosine similarity, or Euclidean distance).
    """
    def __init__(self ,layer = '',matchMethod = 'OT',projection_dim = 192, memoryBankSize = 128, prototypeDecay = 0.96, influenceFactor = 0.5, useGLU =  True, singleUpdate = True, **kwargs):
        super(AlignmentWithProtoype, self).__init__(**kwargs)  
        self.prototypeDecay = prototypeDecay
        self.localPrototypes = tf.Variable(tf.random.normal((memoryBankSize,projection_dim)), 
                                             name = "local_"+str(layer),
                                             trainable= False)
        self.globalPrototypes = tf.Variable(tf.random.normal((memoryBankSize,projection_dim)), 
                                        name = "global_"+str(layer),
                                        trainable= False)
        self.memoryBankSize = memoryBankSize
        self.matchMethod = matchMethod
        self.projection_dim = projection_dim
        self.influenceFactor = influenceFactor
        self.useGLU = useGLU
        self.singleUpdate = singleUpdate

    def build(self,input_shape): 
        """Create layer weights, such as GLU if enabled."""
        if(self.useGLU):
            self.GLU = GatedLinearUnit(self.projection_dim)

    
    @tf.function
    def sinkhorn_matrix(self,out):
        """
        Apply Sinkhorn-Knopp normalization to make the similarity matrix doubly stochastic.
        """
        Q = tf.exp(out / 0.05, name='exp_out')  # Q is K-by-B for consistency with notations from the paper
        Qshape =  tf.cast(tf.shape(Q), tf.float32)
        B = tf.expand_dims(Qshape[1], 0) # number of samples to assign
        K = tf.expand_dims(Qshape[0], 0)  # how many prototypes
        sum_Q = tf.reduce_sum(Q)
        Q /= sum_Q
    
        for it in range(3):
            sum_of_rows = tf.reduce_sum(Q, axis=1, keepdims=True)
            Q /= sum_of_rows
            Q /= K
            Q /= tf.reduce_sum(Q, axis=0, keepdims=True)
            Q /= B
        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q
        
    @tf.function
    def optimal_transport_matching(self,reshapedProjections,memoryPlaceHolder, normalize = True):
        """
        Compute feature-to-prototype assignments using optimal transport.
        """
        projOutput = tf.math.l2_normalize(reshapedProjections, axis=1)
        memoryBankNorm = tf.math.l2_normalize(memoryPlaceHolder, axis=1)
        matmulOutput = tf.linalg.matmul(projOutput,memoryBankNorm,transpose_b = True)
        feature_assignments = self.sinkhorn_matrix(matmulOutput)
        return feature_assignments
        
    @tf.function
    def cossine_similirity_matching(self,A, B):
        """
        Compute pairwise cosine similarity between two sets of embeddings.
        """
        norm_A = tf.norm(A, axis=1, keepdims=True)
        norm_B = tf.norm(B, axis=1, keepdims=True)
        dot_product = tf.matmul(A, tf.transpose(B))
        cosine_sim = dot_product / (tf.matmul(norm_A, tf.transpose(norm_B)) + 1e-7)  # Add epsilon to avoid division by zero
        return cosine_sim

    @tf.function
    def euclidean_distance_matching(self,A, B, normalize=True):
        """
        Compute pairwise Euclidean distances between two sets of embeddings.
        """
        if normalize:
            norm_A = tf.norm(A, axis=1, keepdims=True)
            norm_B = tf.norm(B, axis=1, keepdims=True)
            A = A / (norm_A + 1e-7)
            B = B / (norm_B + 1e-7)
        
        # Efficiently compute pairwise squared Euclidean distances
        A_square = tf.reduce_sum(tf.square(A), axis=1, keepdims=True)  # Shape: (m, 1)
        B_square = tf.reduce_sum(tf.square(B), axis=1, keepdims=True)  # Shape: (n, 1)
        distances = A_square - 2 * tf.matmul(A, B, transpose_b=True) + tf.transpose(B_square)
        
        # Ensure numerical stability by using maximum with zero before taking the square root
        pairwise_distances = tf.sqrt(tf.maximum(distances, 0.0))
        
        return pairwise_distances

    def alignment_process(self,projections,inputShape,training):
        """
        Align input projections with prototypes based on the selected matching method.
        """
        inputShape = tf.shape(projections)
        reshapedProjections = tf.reshape(projections,(-1,inputShape[2] ))

        # Combine local and global prototypes during training, use only local during inference

        if(training):
            combinedPrototypes = tf.concat((self.localPrototypes,self.globalPrototypes), axis = 0)
        else:
            combinedPrototypes = self.localPrototypes
            
        # Perform matching based on the selected method
        if(self.matchMethod == 'euclidean'):
            match_matrix = self.euclidean_distance_matching(reshapedProjections, combinedPrototypes, normalize=True)
        elif(self.matchMethod == 'cossine'):
            match_matrix = self.cossine_similirity_matching(reshapedProjections, combinedPrototypes)
        else:
            match_matrix = self.optimal_transport_matching(reshapedProjections, combinedPrototypes)

        if(training):
            # Prototype updates during training
            if(self.singleUpdate):
                # Hard assignment for single-update strategy
                if(self.matchMethod == 'euclidean'):
                    globalAssignments = tf.argmin(match_matrix[:,self.memoryBankSize:], axis=1)
                    localAssignments =  tf.argmin(match_matrix[:,:self.memoryBankSize], axis=1)
                else:
                    globalAssignments = tf.argmax(match_matrix[:,self.memoryBankSize:], axis=1)
                    localAssignments =  tf.argmax(match_matrix[:,:self.memoryBankSize], axis=1)
    
                memoryAssignments = tf.gather(self.globalPrototypes,globalAssignments)
    
                localMemoryAssignments = tf.gather(self.localPrototypes,localAssignments)
                EMA_Weights = ((self.prototypeDecay * localMemoryAssignments) + ((1 - self.prototypeDecay) * reshapedProjections))
                localPrototypeUpdate = tf.tensor_scatter_nd_update(self.localPrototypes, tf.expand_dims(localAssignments,1), EMA_Weights)
                self.localPrototypes.assign(localPrototypeUpdate)
            else:
                # Bulk update using soft assignments
                if(self.matchMethod == 'euclidean'):
                    globalAssignments = tf.argmin(match_matrix[:,self.memoryBankSize:], axis=1)
                else:
                    globalAssignments = tf.argmax(match_matrix[:,self.memoryBankSize:], axis=1)
        
                memoryAssignments = tf.gather(self.globalPrototypes,globalAssignments)
                
                memorySlotSize = inputShape[0] // self.memoryBankSize
                if(memorySlotSize < 1):
                    memorySlotSize = 1
                match_matrix = tf.transpose(match_matrix[:,:self.memoryBankSize])
                feature_weights,indices =  tf.math.top_k(match_matrix, memorySlotSize , sorted=False)
                normFeatureWeights = tf.math.divide(feature_weights, tf.reduce_sum(feature_weights, axis=1, keepdims=True))
                weightedFeatures = tf.gather(reshapedProjections,indices) * tf.expand_dims(normFeatureWeights, -1)
                aggregatedFeatures = tf.reduce_sum(weightedFeatures,1)    
                EMA_Weights = ((self.prototypeDecay * self.localPrototypes) + ((1 - self.prototypeDecay) * aggregatedFeatures))
                self.localPrototypes.assign(EMA_Weights)

        
        else:
            # Assign embeddings to the closest prototypes during inference
            if(self.matchMethod == 'euclidean'):
                localAssignments = tf.argmin(match_matrix, axis=1)
            else:
                localAssignments = tf.argmax(match_matrix, axis=1)            
            memoryAssignments = tf.gather(combinedPrototypes,localAssignments)
        return memoryAssignments

    def call(self, projections,training=None):
        """
        Forward pass: Align input features with prototypes and produce normalized outputs.
        """
        
        inputShape = tf.shape(projections)
        memoryAssignments = self.alignment_process(projections,inputShape,training)
        memoryAssignments = tf.stop_gradient(memoryAssignments)

        # Apply GLU if enabled, otherwise use raw assignments
        if(self.useGLU):
            gatedAssignments = self.GLU(memoryAssignments)
        else:
            gatedAssignments = memoryAssignments
        formattedAssignments = tf.reshape(gatedAssignments,(-1,inputShape[1],inputShape[2]))

        # Weighted sum of assigned prototype with input projections and normalize
        output = (self.influenceFactor * formattedAssignments ) + ((1 - self.influenceFactor) * projections)
        normedOutput = tf.math.l2_normalize(output,axis = 2)
        return normedOutput 


def HART_ALP_globalLocal_encoder(prototypeCount,
                                projection_dim = 192,
                                num_heads = 3,
                                filterAttentionHead = 4, 
                                convKernels = [3, 7, 15, 31, 31, 31],
                                dropout_rate = 0.1,
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
    inputs = layers.Input((None, projection_dim))
    encoded_patches = inputs

    for layerIndex, kernelLength in enumerate(convKernels):        
        x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
        x2 = AlignmentWithProtoype(layer = layerIndex, 
                            matchMethod = matchMethod,
                            projection_dim = projection_dim,
                            memoryBankSize = prototypeCount[layerIndex], 
                            prototypeDecay = prototypeDecay,
                            influenceFactor = influenceFactor,
                            useGLU = useGLU,
                            singleUpdate = singleUpdate)(x1) 
        branch1 = model.liteFormer(startIndex = projectionQuarter,
                          stopIndex = projectionQuarter + projectionHalf,
                          projectionSize = projectionHalf,
                          attentionHead =  filterAttentionHead, 
                          kernelSize = kernelLength,
                          dropPathRate = dropPathRate[layerIndex],
                          dropout_rate = dropout_rate,
                          name = "liteFormer_"+str(layerIndex))(x2)
        branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x2)
        branch2Gyro = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf ,projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "GyroMHA_"+str(layerIndex))(x2)
        concatAttention = tf.concat((branch2Acc,branch1,branch2Gyro),axis= 2 )
        x2 = layers.Add()([concatAttention, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp2(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        x3 = DropPath(dropPathRate[layerIndex])(x3)
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    return tf.keras.Model(inputs, representation, name="mae_encoder")    




