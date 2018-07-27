import keras
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints

from keras.layers.convolutional import Conv2D
from keras.legacy import interfaces

def Gravity(x, boundaries = [0,1], pressure = 0.5):
    min = boundaries[0]
    max = boundaries[1]
    dist = max - min
    res = min + (1. - np.abs(1. - (x - min) / dist) ** (1. - pressure)) * dist
    return res

    
#tensor values manipulation
class tensor_value():
    def __init__(self, x):
        self.tensor = K.variable(value = x)
        
    def set(self, x):
        K.set_value(self.tensor, x)
        
    def get(self):
        return K.eval(self.tensor)

    
# ---------------
#  Layers
# ---------------
#Pixelwise feature vector normalization layer from "Progressive Growing of GANs" paper
class PixelNorm(Layer): #It will work only if channels are last in order! I have to do something with it.
    def __init__(self, epsilon = 1e-8, **kwargs):
        self.epsilon = epsilon
        super(PixelNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PixelNorm, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        #return x / K.sqrt(K.mean(K.square(x), axis=-1, keepdims=True) + self.epsilon)
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.epsilon)

    def compute_output_shape(self, input_shape):
        return input_shape
        
        
#MiniBatchStddev layer from "Progressive Growing of GANs" paper        
class MiniBatchStddev(Layer): #again position of channels matter!
    def __init__(self, group_size=4, **kwargs):
        self.group_size = group_size
        super(MiniBatchStddev, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MiniBatchStddev, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])# Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1], s[2], 1])             # [N1HW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=-1)    
        
    def compute_output_shape(self, input_shape):
        return (*input_shape[:3], input_shape[3]+1)        
          

#this layer is not complete   
class MinibatchDiscrimination(Layer):
    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
            initializer=self.init,
            name='kernel',
            regularizer=self.W_regularizer,
            trainable=True,
            constraint=self.W_constraint)

        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1]+self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'init': self.init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))        
        

   
# -------------------
#  Constructor
# -------------------
# R - real data
# F - fake noisy data
# L - labels
# N - noise / latent representation


#AE: 
#   DE(EN(R)) -> R
 
#GAN:       
#       Mode 1:
#   D(R) -> 1       {bc}
#   D(DE.(N)) -> 0  {bc}
#   D.(DE(N)) -> 1  {bc}
#
#       Mode 2:
#   D(R), D(DE.(N)) -> 1, 0    {bc}
#   D.(DE(N)) -> 1          {bc}
#
#       Mode 3:
#   D(R), D(DE.(N)), D.(DE(N)) -> 1, 0, 1  {bc}
#
#       Mode S:
#   D(R) -> 1       {logcosh}
#   g = Gravity(D.(DE.(N)), boundaries = [-1,1], pressure = 0.5)
#   D(DE.(N)) -> g  {logcosh}
#   D(F) -> -1      {logcosh}
#   D.(DE(N)) -> 1  {logcosh}

#CGAN:
#   D(R, L) -> 1       {bc}
#   D(DE.(N), L) -> 0  {bc}
#   D.(DE(N), L) -> 1  {bc}

#DiscoGAN:  
#   D(Ra, Rb) -> 1              {mae}
#   D(DE.(Rb), EN.(Ra)) -> -1   {mae}
#   D.(DE(Rb), EN(Ra)) -> 1     {mae}
#   DE(EN(Ra)) -> Ra            {mae}
#   EN(DE(Rb)) -> Rb            {mae}

#AAE:       
#   D(N) -> 1               {bc}
#   D(EN.(R)) -> 0          (bc)
#   x = EN(R)
#   DE(x), D.(x) -> R, 1    {mse, bc}




##############