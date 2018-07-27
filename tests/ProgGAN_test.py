from GANLib import ProgGAN
from GANLib import plotter
from GANLib import utils

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
import tensorflow as tf
import keras 

from keras.utils import plot_model

init =  'he_normal' #keras.initializers.RandomNormal(0.0, 0.02) #
filt = 64


def l2_norm(v, eps=1e-12):
    return v / (K.sum(v ** 2) ** 0.5 + eps)
    
def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = K.reshape(w, [-1, w_shape[-1]])

    u = K.truncated_normal([1, w_shape[-1]])
    
    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = K.dot(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = K.dot(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = K.dot(K.dot(v_hat, w), K.transpose(u_hat))
    w_norm = w / sigma

    w_norm = K.reshape(w_norm, w_shape)

    return w_norm

def scale_weights(shapes, gain = np.sqrt(2)):
    #He's normal dynamic weight scaler
    fan_in = np.prod(shapes)
    std = gain / np.sqrt(fan_in)
    return K.constant(std)
    
    
weights_scale_func = lambda x: x

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


def new_sheet(self, filters, kernel_size, padding, name, trainable = True):
    def func(layer):
        w = self.weights.get(name,None)
        layer = Conv2D_sw(filters, kernel_size, padding=padding, weights = w, name = name, trainable = trainable, kernel_initializer = init)(layer)
        layer = LeakyReLU(alpha=0.2)(layer) 
        return layer
    return func
   
    


def build_generator(self):
    previous_step = None
    next_step = None

    input_layer = Input(shape=(self.latent_dim,))
    layer = RepeatVector(16)(input_layer)
    layer = Reshape((4, 4, self.latent_dim))(layer)
    
    #layer = Dense_sw(filt * 4 * 4, kernel_initializer = init)(input_layer)
    #layer = LeakyReLU(alpha=0.2)(layer) 
    #layer = Reshape((4, 4, filt))(layer)
    
    layer = new_sheet(self, filt, (4,4), 'same', 'genr_head_0')(layer)
    layer = utils.PixelNorm()(layer)
    layer = new_sheet(self, filt, (3,3), 'same', 'genr_head_1')(layer)
    layer = utils.PixelNorm()(layer)
    
    #Growing layers
    for i in range(self.layers):
        layer = UpSampling2D(2)(layer)
        previous_step = layer
        
        layer = new_sheet(self, filt, (3,3), 'same', 'genr_layer_0'+str(i))(layer)
        layer = utils.PixelNorm()(layer)
        #layer = new_sheet(self, filt, (3,3), 'same', 'genr_layer_1'+str(i))(layer)
        #layer = utils.PixelNorm()(layer)
   
    next_step = Conv2D_sw(self.channels, (1,1), weights = self.weights.get('to_rgb',None), name = 'to_rgb', kernel_initializer = init)(layer) #to RGB
    
    #smooth fading
    if previous_step is not None: 
        previous_step = Conv2D_sw(self.channels, (1,1), weights = self.weights.get('to_rgb',None))(previous_step) 
        
        previous_step = Lambda(lambda x: x * (1 - self.transition_alpha.tensor))(previous_step)
        next_step = Lambda(lambda x: x * self.transition_alpha.tensor)(next_step)
        layer = add([previous_step, next_step])
        #layer = Lambda(lambda x: x[0] + (x[1] - x[0]) * self.transition_alpha.tensor)([previous_step, next_step])
    else:
        layer = next_step
      
    return Model(input_layer, layer)
    
def build_discriminator(self):
    previous_step = None
    next_step = None
    
    input_layer = Input(shape=self.inp_shape)
    layer = input_layer
    
    layer = Conv2D_sw(filt, (1,1), weights = self.weights.get('from_rgb',None), name = 'from_rgb', kernel_initializer = init)(layer) #from RGB
    layer = LeakyReLU(alpha=0.2)(layer) 
    layer = utils.PixelNorm()(layer)
    
    #Growing layers
    for i in range(self.layers, 0, -1):
        layer = new_sheet(self, filt, (3,3), 'same', 'disc_layer_0'+str(i))(layer)
        layer = utils.PixelNorm()(layer)
        #layer = new_sheet(self, filt, (3,3), 'same', 'disc_layer_1'+str(i))(layer)
        layer = AveragePooling2D(2)(layer)

        #smooth fading
        if i == self.layers:
            next_step = layer
            
            previous_step = AveragePooling2D(2)(input_layer)
            previous_step = Conv2D_sw(filt, (1,1), weights = self.weights.get('from_rgb',None))(previous_step) #from RGB
            previous_step = LeakyReLU(alpha=0.2)(previous_step) 
            previous_step = utils.PixelNorm()(previous_step)
        
            previous_step = Lambda(lambda x: x * (1 - self.transition_alpha.tensor))(previous_step)
            next_step = Lambda(lambda x: x * self.transition_alpha.tensor)(next_step)
            layer = add([previous_step, next_step])
            #layer = Lambda(lambda x: x[0] + (x[1] - x[0]) * self.transition_alpha.tensor)([previous_step, next_step])
                
    
    #layer = utils.MiniBatchStddev(group_size=4)(layer)
    layer = new_sheet(self, filt, (3,3), 'same', 'disc_head_0')(layer)
    layer = utils.PixelNorm()(layer)
    layer = new_sheet(self, filt, (4,4), 'valid', 'disc_head_1')(layer)
    layer = utils.PixelNorm()(layer)
    
    layer = Flatten()(layer)
    layer = Dense_sw(1, activation=self.disc_activation, kernel_initializer = init)(layer)

    return Model(input_layer, layer)


       
noise_dim = 100    
r, c = 3, 4
noise = np.random.uniform(-1, 1, (r * c, noise_dim))

def sample_images(gen, file):
    gen_imgs = gen.predict([noise])

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
    fig.savefig(file) #% epoch
    plt.close()   
    
    
img_path = 'ProgGAN'
mode = 'vanilla'
    
# Load the dataset
(X_train, labels), (_, _) = cifar10.load_data()
indx = np.where(labels == 1)[0]
X_train = X_train[indx]

#X_train = np.load('../../Datasets/Faces/face_images_128x128.npy')

#Configure input
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
if len(X_train.shape)<4:
    X_train = np.expand_dims(X_train, axis=3)
    
    
#Run GAN for 20000 iterations
gan = ProgGAN(X_train.shape[1:], noise_dim, mode = mode)
gan.build_generator = lambda self=gan: build_generator(self)
gan.build_discriminator = lambda self=gan: build_discriminator(self)

#optimizer = Adam(0.001, beta_1=0., beta_2=0.99, epsilon=1e-8, clipvalue=0.1)
gan.build_models() 


ind = 0
def callback():
    global ind 
    ind+=1
    path = 'images/'+img_path+'/'
    sample_images(gan.generator, path+'imgs/'+str(ind)+'.png')
    #plotter.save_hist_image(gan.history, path+'_hist.png')
    #[1000, 2000, 3000, 5000, 8000, 13000]
gan.train(X_train, epochs_list = [4000, 8000, 16000, 32000], batch_size_list=[16, 16, 16, 16, 8, 4], checkpoint_callback = callback, validation_split = 0)    