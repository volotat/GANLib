from GANLib import GAN

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.models import Model

import matplotlib.pyplot as plt
import numpy as np


# Specify models for Generator and Discriminator
def build_generator(self):
    input_lat = Input(shape=(self.latent_dim,))
    
    layer = Dense(128, activation = 'relu')(input_lat)
    layer = Dense(256, activation = 'relu')(layer)
    layer = Dense(784, activation = 'linear')(layer)
    img = Reshape((28,28,1))(layer)

    return Model(input_lat, img)
        
def build_discriminator(self):
    input_img = Input(shape=self.input_shape)
    
    layer = Flatten()(input_img)
    layer = Dense(256, activation = 'relu')(layer)
    layer = Dense(128, activation = 'relu')(layer)
    valid = Dense(1, activation=self.disc_activation)(layer)
    
    return Model(input_img, valid) 
  
# Save examples of generated images to file  
def sample_images(gen, file):
    r, c = 5, 5
    
    noise = np.random.uniform(-1, 1, (r * c, noise_dim))

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
    fig.savefig(file)
    plt.close()    
    
# Load the dataset
(data, _), (_, _) = mnist.load_data()

# Configure input
data = (data.astype(np.float32) - 127.5) / 127.5
if len(data.shape)<4: data = np.expand_dims(data, axis=3)
data_shape = data.shape[1:]
noise_dim = 100


# Build GAN and train it on your data
gan = GAN(data_shape, noise_dim) #define type of Generative model
gan.build_generator = lambda self=gan: build_generator(self) #define generator build function
gan.build_discriminator = lambda self=gan: build_discriminator(self) #define discriminator build function
gan.build_models() #build all necessary models

def callback():
    sample_images(gan.generator, 'simple_gan.png')

gan.train(data, epochs=20000, batch_size=64, checkpoint_callback = callback, collect_history = False) #train GAN for 20000 iterations


