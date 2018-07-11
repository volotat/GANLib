from GANLib import AAE
from GANLib import plotter

from keras.datasets import mnist, fashion_mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam, RMSprop, Nadam

import matplotlib.pyplot as plt
import numpy as np


noise_dim = 100

def build_generator(self):
    input_img = Input(shape=self.input_shape)

    layer = input_img
    layer = Flatten()(layer)
    #layer = GaussianNoise(0.1)(layer)
    
    layer = Dense(128)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    layer = BatchNormalization()(layer)
    
    layer = Dense(128)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    layer = BatchNormalization()(layer)
    
    layer = Dense(784, activation = 'linear')(layer)
    output_img = Reshape((28,28,1))(layer)
    return Model(input_img, output_img)
    
def build_decoder(self):
    input_img = Input(shape=self.input_shape)

    layer = input_img
    layer = Flatten()(layer)
    #layer = GaussianNoise(0.1)(layer)
    
    layer = Dense(128)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    layer = BatchNormalization()(layer)
    
    layer = Dense(128)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    layer = BatchNormalization()(layer)
    
    layer = Dense(784, activation = 'linear')(layer)
    output_img = Reshape((28,28,1))(layer)
    return Model(input_img, output_img)    

def build_discriminator(self):
    input_img = Input(shape=self.input_shape)
    
    layer = input_img
    '''
    layer = Conv2D(32, 1, padding = 'same')(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    layer = Conv2D(16, 3, padding = 'same', strides = 2)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    layer = Conv2D(8, 3, padding = 'same', strides = 2)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    '''
    layer = Flatten()(layer)
    layer = Dense(512)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    layer = Dense(256)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    layer = Dense(64)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    validity = Dense(2, activation = 'linear')(layer)
    
    return Model(input_img, validity)    
     
        


def sample_images(gen, file, dom_set, ind = None):
    r, c = 5, 5
    
    noise = np.random.uniform(-1, 1, (r * c, noise_dim))

    if ind is None:
        gen_imgs = gen.predict([dom_set[:r*c]])
    else:
        gen_imgs = gen.predict([dom_set[:r*c]])[ind]

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

    
img_path = 'AAE'
mode = 'vanilla'
    
# Load the dataset
(mnist_set, labels), (_, _) = mnist.load_data()
mnist_set = (mnist_set.astype(np.float32) - 127.5) / 127.5
mnist_set = np.expand_dims(mnist_set, axis=3)


(fashion_set, labels), (_, _) = fashion_mnist.load_data()
fashion_set = (fashion_set.astype(np.float32) - 127.5) / 127.5
fashion_set = np.expand_dims(fashion_set, axis=3)

set_domain_A = mnist_set  [:128]
set_domain_B = fashion_set[:128]   
'''
ind_a = np.where(labels == 1)[0]
ind_b = np.where(labels == 7)[0]

set_domain_A = X_train[ind_a][:100]
set_domain_B = X_train[ind_b][:100]
'''


#Run GAN for 20000 iterations
gan = AAE(mnist_set.shape[1:], noise_dim, mode = mode)
gan.build_generator = lambda self=gan: build_generator(self)
gan.build_decoder = lambda self=gan: build_decoder(self)
gan.build_discriminator = lambda self=gan: build_discriminator(self)
gan.build_models()

def callback():
    path = 'images/'+img_path+'/conv_'+mode
    sample_images(gan.combined_As, path+'_A.png', set_domain_A)
    sample_images(gan.combined_Bs, path+'_B.png', set_domain_B)
    
    sample_images(gan.generator, path+'_gen.png', set_domain_A)
    #plotter.save_hist_image(gan.history, path+'_hist.png')
    
gan.train(set_domain_A, set_domain_B, epochs=20000, batch_size=64, checkpoint_callback = callback, validation_split = 0.1)    