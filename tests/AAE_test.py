from GANLib import AAE
from GANLib import plotter

from keras.datasets import mnist, fashion_mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate, Lambda
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam, RMSprop, Nadam
import keras.backend as K

import matplotlib.pyplot as plt
import numpy as np


noise_dim = 10

def build_encoder(self):
    input_img = Input(shape=self.input_shape)

    layer = input_img
    layer = Flatten()(layer)
    layer = Dense(256)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    layer = Dense(128)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    latent = Dense(self.latent_dim)(layer)
    
    return Model(input_img, latent)
    
def build_decoder(self):
    input_img = Input(shape=(self.latent_dim,))

    layer = input_img
    layer = Dense(128)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    layer = Dense(256)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    layer = Dense(784)(layer)
    output_img = Reshape((28,28,1))(layer)
    
    return Model(input_img, output_img)    

def build_discriminator(self):
    input_lat = Input(shape=(self.latent_dim,))
    
    layer = input_lat
    layer = Dense(128)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    layer = Dense(64)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    validity = Dense(1, activation = self.disc_activation)(layer)
    
    return Model(input_lat, validity)    
     
        

r, c = 5, 5
noise = np.random.uniform(-1, 1, (r * c, noise_dim))
#noise = np.random.normal(size=(r * c, noise_dim))

def sample_images(gen, file):
    gen_imgs = gen.predict(noise)

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
    
# Load the dataset
(X_train, labels), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
if len(X_train.shape)<4:
    X_train = np.expand_dims(X_train, axis=3)

    
#Run GAN for 20000 iterations
gan = AAE(X_train.shape[1:], noise_dim)
gan.encoder = build_encoder(gan)
gan.decoder = build_decoder(gan)
gan.discriminator = build_discriminator(gan)

def callback():
    path = 'images/'+img_path+'/'
    sample_images(gan.decoder, path+'decoded.png')
    plotter.save_hist_image(gan.history, path+'History.png')
    
gan.train(X_train, epochs=20000, batch_size=64, checkpoint_callback = callback, validation_split = 0.1)    