from GANLib import WGAN_GP, plotter, utils

import keras 
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate, add, RepeatVector, UpSampling2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, AveragePooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam, RMSprop, Nadam

import matplotlib.pyplot as plt
import numpy as np

import keras.backend as K

from skimage.measure import block_reduce



#                   Progressively growing of GANs
#   Paper: https://arxiv.org/pdf/1710.10196.pdf

#       Description:
#   Takes as input some dataset and trains the network as usual GAN but progressively 
#   adding layers to generator and discriminator.



#-------------------------------
# Auxiliary functions
#-------------------------------  

def augment(data):
    off_x_p = data.copy()
    off_x_p[:,1:,:,:] = off_x_p[:,:-1,:,:]
    off_x_m = data.copy()
    off_x_p[:,:-1,:,:] = off_x_p[:,1:,:,:]
    data = np.concatenate((data,off_x_p,off_x_m), axis = 0)
    
    off_y_p = data.copy()
    off_y_p[:,:,1:,:] = off_y_p[:,:,:-1,:]
    off_y_m = data.copy()
    off_y_m[:,:,:-1,:] = off_y_m[:,:,1:,:]
    data = np.concatenate((data,off_y_p,off_y_m), axis = 0)
    
    return data

def dynamic_he_scale(x, gain = np.sqrt(2)): 
    #He's normal dynamic weight scaler
    shape = x.shape.as_list()
    fan_in, _ = keras.initializers._compute_fans(shape)
    std = np.sqrt(gain / max(1., fan_in)) 
    x_scale = x * K.constant(std)
    
    return x_scale
   
class Conv2D_sw(Conv2D): #Conv2D layer with dynamically scaled weights
    def __init__(self, filters, kernel_size, **kwargs):
        super(Conv2D_sw, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            **kwargs)
       
    def call(self, inputs):
        outputs = K.conv2d(
                inputs,
                weights_scale_func(self.kernel),
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
                
        if self.activation is not None:
            return self.activation(outputs)
            
        return outputs

class Dense_sw(Dense): #Dense layer with dynamically scaled weights
    def __init__(self, units, **kwargs):
        super(Dense_sw, self).__init__(units = units, **kwargs)
       
    def call(self, inputs):
        output = K.dot(inputs, weights_scale_func(self.kernel))
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
     
           
#-------------------------------
# Define models structure
#-------------------------------      
      
initialization = 'he_normal' #keras.initializers.RandomNormal(0, 1)  
weights_scale_func = dynamic_he_scale

filters = 64
noise_dim = 64 
channels = 3

weights = {}
sheets = 0


def new_sheet(filters, kernel_size, padding, name, pix_norm = True):
    def func(layer):
        w = weights.get(name,None)
        layer = Conv2D_sw(filters, kernel_size, padding=padding, weights = w, name = name, kernel_initializer = initialization)(layer)
        layer = LeakyReLU(alpha=0.2)(layer) 
        if pix_norm: layer = utils.PixelNorm()(layer)
        return layer
    return func
    
def transition_alpha(self):
    return K.minimum(self.epoch.tensor / (self.epochs.tensor/2), 1) 
   
def build_generator(self):
    previous_step = None
    next_step = None

    input_layer = Input(shape=(self.latent_dim,))
    layer = RepeatVector(16)(input_layer)
    layer = Reshape((4, 4, self.latent_dim))(layer)
    
    layer = new_sheet(filters, (4,4), 'same', 'genr_head_0')(layer)
    layer = new_sheet(filters, (3,3), 'same', 'genr_head_1')(layer)
    
    #Growing layers
    for i in range(sheets):
        layer = UpSampling2D(2)(layer)
        if i == sheets-1: previous_step = layer
            
        layer = new_sheet(filters, (3,3), 'same', 'genr_layer_0'+str(i))(layer)
   
    next_step = Conv2D_sw(channels, (1,1), weights = weights.get('to_rgb',None), name = 'to_rgb', kernel_initializer = initialization)(layer) #to RGB
    
    #smooth fading
    if previous_step is not None: 
        previous_step = Conv2D_sw(channels, (1,1), weights = weights.get('to_rgb',None))(previous_step) 
        layer = Lambda(lambda x: x[0] + (x[1] - x[0]) * transition_alpha(self))([previous_step, next_step])
    else:
        layer = next_step
      
    return Model(input_layer, layer)
    
def build_discriminator(self):
    previous_step = None
    next_step = None
    
    input_layer = Input(shape=self.input_shape)
    
    layer = Conv2D_sw(filters, (1,1), weights = weights.get('from_rgb',None), name = 'from_rgb', kernel_initializer = initialization)(input_layer) #from RGB
    layer = LeakyReLU(alpha=0.2)(layer) 
    layer = utils.PixelNorm()(layer)
    
    #Growing layers
    for i in range(sheets, 0, -1):
        layer = new_sheet(filters, (3,3), 'same', 'disc_layer_0'+str(i))(layer)
        layer = AveragePooling2D(2)(layer)

        #smooth fading
        if i == sheets:
            next_step = layer
            
            previous_step = AveragePooling2D(2)(input_layer)
            previous_step = Conv2D_sw(filters, (1,1), weights = weights.get('from_rgb',None))(previous_step) #from RGB
            previous_step = LeakyReLU(alpha=0.2)(previous_step) 
            previous_step = utils.PixelNorm()(previous_step)
        
            layer = Lambda(lambda x: x[0] + (x[1] - x[0]) * transition_alpha(self))([previous_step, next_step])
                
    
    layer = utils.MiniBatchStddev(group_size=4)(layer)
    layer = new_sheet(filters, (3,3), 'same', 'disc_head_0')(layer)
    layer = new_sheet(filters, (4,4), 'valid', 'disc_head_1')(layer)
    
    layer = Flatten()(layer)
    layer = Dense_sw(1, activation=self.disc_activation, kernel_initializer = initialization)(layer)

    return Model(input_layer, layer)
    
#-------------------------------
#  Main code
#-------------------------------  

r, c = 4, 6
sample_noise = np.random.uniform(-1, 1, (r * c, noise_dim))
def sample_images(gen, file):
    gen_imgs = gen.predict([sample_noise])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    gen_imgs = np.clip(gen_imgs,0,1)
    
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            if gen_imgs.shape[-1] == 1: 
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
            else:
                axs[i,j].imshow(gen_imgs[cnt,:,:])
            axs[i,j].axis('off')
            cnt += 1
    
    fig.savefig(file)
    plt.close()   
    
    
# Load the dataset
(dataseta, labelsa), (datasetb, labelsb) = cifar10.load_data()
dataset = np.concatenate((dataseta,datasetb), axis = 0)
labels = np.concatenate((labelsa,labelsb), axis = 0)

indx = np.where(labels == 8)[0] # we choose only one specific domain from the dataset
dataset = dataset[indx]

# Configure input
dataset = (dataset.astype(np.float32) - 127.5) / 127.5
if len(dataset.shape)<4:
    dataset = np.expand_dims(dataset, axis=3)
    

# 6000 examples is not enough, so we augment dataset by shifting it along axis by 1 pixel 
dataset = augment(dataset)
 
epochs_list = [4000, 8000, 16000, 32000]
batch_size_list = [16, 16, 16, 16]  
image_size_list = [4, 8, 16, 32] 

for i in range(len(epochs_list)):    
    epochs = epochs_list[i]
    batch_size = batch_size_list[i]
    
    sz = dataset.shape[1] // image_size_list[i]
    data_set = block_reduce(dataset, block_size=(1, sz, sz, 1), func=np.mean) 
    print(data_set.shape)
    
    # Build and train GAN
    gan = WGAN_GP(data_set.shape[1:], noise_dim, optimizer = Adam(0.0002, 0.5, 0.9, clipnorm = 10))
    gan.generator = build_generator(gan) #define generator model
    gan.discriminator = build_discriminator(gan) #define discriminator model

    def callback():
        sample_images(gan.generator, 'pg_gan.png')
        
    gan.train(data_set, epochs = epochs, batch_size = batch_size, checkpoint_callback = callback, collect_history = False)    
    
    # Save weights of the network
    for l in gan.generator.layers:
        weights[l.name] = l.get_weights() 
        
    for l in gan.discriminator.layers:
        weights[l.name] = l.get_weights() 
            
    sheets += 1