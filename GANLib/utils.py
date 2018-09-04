import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

'''
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

        
def gradient_control(x, a):
    return K.stop_gradient(x)* (1 - a * 0.99) +  x * (0.01 + a * 0.99)  
    
    
    
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
        return x / K.sqrt(K.mean(K.square(x), axis=-1, keepdims=True) + self.epsilon)

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
'''          
    
# ---------------
#  History
# ---------------

def save_hist_image(hist, file, graphs = (['metric'], ['D loss', 'G loss']), scales = ('log', 'linear')):
    hist_size = hist['hist_size']

    plt.figure(figsize=(14,7))
    plt.subplot(2, 1, 1)
    plt.yscale('log')
    plt.plot(hist['metric'][:hist_size,0], '-',linewidth=0.8, label="metric mean", color='C0')
    plt.plot(hist['metric'][:hist_size,1], '-',linewidth=0.8, label="metric min", color='C0', alpha = 0.5)
    plt.plot(hist['metric'][:hist_size,2], '-',linewidth=0.8, label="metric max", color='C0', alpha = 0.5)
    plt.axhline(hist['best_metric'],linewidth=0.8)
    plt.xlabel('Best result: %f'%hist['best_metric'])
    plt.grid(True)

    
    plt.subplot(2, 1, 2)
    cnt = 0
    for i in range(len(hist)):
        key = list(hist.keys())[i]
        if isinstance(hist[key], np.ndarray) and key != 'metric':
            plt.plot(hist[key][:hist_size,0], '-',linewidth=0.8, label=key, color='C'+str(cnt))
            cnt += 1
    
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    
    plt.savefig(file, format='png')
    plt.close()    