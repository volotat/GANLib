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



#-------------------------------
# Auxiliary functions
#-------------------------------  

def l2_norm(v, eps=1e-12):
    return v / (K.sum(v ** 2) ** 0.5 + eps)
    
def spectral_norm(w, iteration=1):
    #From "Spectral Normalization for GANs" paper 
    #https://arxiv.org/pdf/1802.05957.pdf
    
    w_shape = w.shape.as_list()
    w = K.reshape(w, [-1, w_shape[-1]])

    u = K.truncated_normal([1, w_shape[-1]])
    
    u_hat = u
    v_hat = None
    for i in range(iteration): #power iteration, usually iteration = 1 will be enough
        v_ = K.dot(u_hat, K.transpose(w))
        v_hat = l2_norm(v_)

        u_ = K.dot(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = K.dot(K.dot(v_hat, w), K.transpose(u_hat))
    w_norm = w / sigma

    w_norm = K.reshape(w_norm, w_shape)

    return w_norm

def dynamic_he_scale(x, gain = np.sqrt(2)): 
    #He's normal dynamic weight scaler. Used in paper. Completely not working for me. Probably I'm doing something horribly wrong.
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
      
initialization = 'he_normal' 
filters = 64
noise_dim = 64 
weights_scale_func = spectral_norm #dynamic_he_scale, lambda x: x


layers = 0
weights = {}

channels = 3
transition_alpha = utils.tensor_value(0)


def new_sheet(self, filters, kernel_size, padding, name):
    def func(layer):
        w = weights.get(name,None)
        layer = Conv2D_sw(filters, kernel_size, padding=padding, weights = w, name = name, kernel_initializer = initialization)(layer)
        layer = LeakyReLU(alpha=0.2)(layer) 
        layer = utils.PixelNorm()(layer)
        return layer
    return func
   
def build_generator(self):
    previous_step = None
    next_step = None

    input_layer = Input(shape=(self.latent_dim,))
    layer = RepeatVector(16)(input_layer)
    layer = Reshape((4, 4, self.latent_dim))(layer)
    
    layer = new_sheet(self, filters, (4,4), 'same', 'genr_head_0')(layer)
    layer = new_sheet(self, filters, (3,3), 'same', 'genr_head_1')(layer)
    
    #Growing layers
    for i in range(layers):
        layer = UpSampling2D(2)(layer)
        if i == layers-1: previous_step = layer
            
        layer = new_sheet(self, filters, (3,3), 'same', 'genr_layer_0'+str(i))(layer)
   
    next_step = Conv2D_sw(channels, (1,1), weights = weights.get('to_rgb',None), name = 'to_rgb', kernel_initializer = initialization)(layer) #to RGB
    
    #smooth fading
    if previous_step is not None: 
        previous_step = Conv2D_sw(channels, (1,1), weights = weights.get('to_rgb',None))(previous_step) 
        layer = Lambda(lambda x: x[0] + (x[1] - x[0]) * transition_alpha.tensor)([previous_step, next_step])
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
    for i in range(layers, 0, -1):
        layer = new_sheet(self, filters, (3,3), 'same', 'disc_layer_0'+str(i))(layer)
        layer = AveragePooling2D(2)(layer)

        #smooth fading
        if i == layers:
            next_step = layer
            
            previous_step = AveragePooling2D(2)(input_layer)
            previous_step = Conv2D_sw(filters, (1,1), weights = weights.get('from_rgb',None))(previous_step) #from RGB
            previous_step = LeakyReLU(alpha=0.2)(previous_step) 
            previous_step = utils.PixelNorm()(previous_step)
        
            layer = Lambda(lambda x: x[0] + (x[1] - x[0]) * transition_alpha.tensor)([previous_step, next_step])
                
    
    layer = utils.MiniBatchStddev(group_size=4)(layer)
    layer = new_sheet(self, filters, (3,3), 'same', 'disc_head_0')(layer)
    layer = new_sheet(self, filters, (4,4), 'valid', 'disc_head_1')(layer)
    
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
    
def rebuild(self):
    global layers, inp_shape
    if layers !=0:
        transition_alpha.set(0)
                    
        #copy old weights to new expanded network
        for l in self.generator.layers:
            weights[l.name] = l.get_weights() 
            
        for l in self.discriminator.layers:
            weights[l.name] = l.get_weights() 
    
    
    sz = 2 ** (layers + 2)
    self.input_shape = (sz,sz,channels)
    self.build_models()
    
    
    layers += 1


    
# Load the dataset
(X_train, labels), (_, _) = cifar10.load_data()
indx = np.where(labels == 8)[0]
X_train = X_train[indx]

#Configure input
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
if len(X_train.shape)<4:
    X_train = np.expand_dims(X_train, axis=3)
    
#here has to be a step with data augmentation    
    
def correct_data(self):
    sz = X_train.shape[1] // self.input_shape[0]
    self.data_set = block_reduce(self.data_set, block_size=(1, sz, sz, 1), func=np.mean) 

    
#Build and train GAN
gan = WGAN_GP((4,4, channels), noise_dim)
gan.build_generator = lambda self=gan: build_generator(self)
gan.build_discriminator = lambda self=gan: build_discriminator(self)
gan.rebuild_function = lambda self=gan: rebuild(self)
gan.correct_data = lambda self=gan: correct_data(self)
#gan.build_models() 

def callback():
    sample_images(gan.generator, 'pg_gan_results.png')
    plotter.save_hist_image(gan.history, 'pg_gan_history.png')
    
gan.train(X_train, epochs = [4000, 8000, 16000, 32000], batch_size = [16, 16, 16, 16], checkpoint_callback = callback, validation_split = 0.1)    