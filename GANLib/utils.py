from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

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
    def __init__(self, **kwargs):
        super(PixelNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.epsilon = 1e-8
        super(PixelNorm, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        return x / K.sqrt(K.mean(K.square(x), axis=-1, keepdims=True) + self.epsilon)

    def compute_output_shape(self, input_shape):
        return input_shape
        
        
        
#MiniBatchStddev layer from "Progressive Growing of GANs" paper        
class MiniBatchStddev(Layer): #not sure if it's works correctly yet...
    def __init__(self, group_size=1, **kwargs):
        self.group_size = group_size
        super(MiniBatchStddev, self).__init__(**kwargs)

    def build(self, input_shape):
        #self.input_spec = tf.keras.layers.InputSpec('float32', input_shape)
        super(MiniBatchStddev, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        _, h, w, c =  x.shape #self.input_spec.shape
        # gs = K.maximum(self.group_size, self.input_spec.shape[0])
        gs = self.group_size
        _x = K.reshape(x, (gs, -1, h, w, c))
        _x -= K.mean(_x, axis=0, keepdims=True)
        _x = K.mean(K.square(_x), axis=0)
        _x = K.sqrt(_x + K.epsilon())
        _x = K.sum(_x, axis=[1, 2, 3], keepdims=True)
        
        _x = K.tile(_x, [gs, h, w, 1])
        #_x = tf.tile(_x, [gs, h, w, 1])
        _x = K.concatenate([x, _x], axis=-1)
        return _x

    def compute_output_shape(self, input_shape):
        return (*input_shape[:3], input_shape[3]+1)        
        
        
        
        
        
        
        
        
        
        
        
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


#ProgGAN:   D_1..D_N(R/DE_N..DE_1(N))


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