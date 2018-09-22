import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
     
# ---------------
#  Layers
# ---------------

#Pixelwise feature vector normalization layer from "Progressive Growing of GANs" paper
def PixelNorm(x, epsilon = 1e-8): #It will work only if channels are last in order! I have to do something with it.
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)
        
       
#MiniBatchStddev layer from "Progressive Growing of GANs" paper        
def MiniBatchStddev(x, group_size=4): #again position of channels matter!
    group_size = tf.minimum(group_size, tf.shape(x)[0])# Minibatch must be divisible by (or smaller than) group_size.
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