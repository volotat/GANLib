from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

def Gravity(x, boundaries = [0,1], pressure = 0.5):
    min = boundaries[0]
    max = boundaries[1]
    dist = max - min
    res = min + (1. - np.abs(1. - (x - min) / dist) ** (1. - pressure)) * dist
    return res


    
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
class MiniBatchDiscrimination(Layer):
    def __init__(self, nb_kernel=100,
                 dim_per_kernel=5,
                 trainable=True,
                 **kwargs):
        self.nb_kernel = nb_kernel
        self.dim_per_kernel = dim_per_kernel
        self.trainable = trainable
        super(MiniBatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        self.T = K.variable(np.random.normal(size=(input_shape[1],
                                                   self.nb_kernel*self.dim_per_kernel)),
                            name='MBD_T')
        super(MiniBatchDiscrimination, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        _x = K.dot(x, self.T)
        _x = K.reshape(_x, shape=(-1, self.nb_kernel, self.dim_per_kernel))
        diffs = K.expand_dims(_x, 3) - K.expand_dims(K.permute_dimensions(_x, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), 2)
        _x = K.sum(K.exp(-abs_diffs), 2)
        return K.concatenate([x, _x], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + self.nb_kernel)
        
        


#value storage
class tf_value():
    def __init__(self, ):
        tensor = tf.Variable([0,0])
        
    def set(x):
        zero_tsr 
        tf.assign(zero_tsr,x)
        
    def get()
        return x.eval()
        
    def get_tensor()
        return x
        
        
        
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