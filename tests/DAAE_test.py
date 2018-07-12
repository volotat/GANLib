from GANLib import DAAE
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


def build_encoder(self):
    input_img = Input(shape=self.a_set_shape)

    layer = input_img
    layer = Flatten()(layer)
    
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
    input_img = Input(shape=self.b_set_shape)

    layer = input_img
    layer = Flatten()(layer)
    
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
    input_a = Input(shape=self.a_set_shape)
    input_b = Input(shape=self.b_set_shape)
    layer = concatenate([input_a, input_b])
    
    layer = Flatten()(layer)
    layer = Dense(512)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    layer = Dense(64)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    validity = Dense(1, activation = 'linear')(layer)
    
    return Model([input_a, input_b], validity)    
     
        
def sample_images(gen, file, dom_set):
    r, c = 5, 5
    
    gen_imgs = gen.predict([dom_set[:r*c]])

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

    
img_path = 'DAAE'
    
# Load the dataset
(mnist_set, labels), (_, _) = mnist.load_data()
mnist_set = (mnist_set.astype(np.float32) - 127.5) / 127.5
mnist_set = np.expand_dims(mnist_set, axis=3)


(fashion_set, labels), (_, _) = fashion_mnist.load_data()
fashion_set = (fashion_set.astype(np.float32) - 127.5) / 127.5
fashion_set = np.expand_dims(fashion_set, axis=3)

set_domain_A = mnist_set  [:128]
set_domain_B = fashion_set[:128]   


#Run GAN for 20000 iterations
gan = DAAE(mnist_set.shape[1:], fashion_set.shape[1:])
gan.build_encoder = lambda self=gan: build_encoder(self)
gan.build_decoder = lambda self=gan: build_decoder(self)
gan.build_discriminator = lambda self=gan: build_discriminator(self)
gan.build_models()

def callback():
    path = 'images/'+img_path+'/'
    
    sample_images(gan.combined_A, path+'A_decoded.png', set_domain_A)
    sample_images(gan.combined_B, path+'B_decoded.png', set_domain_B)
    
    sample_images(gan.encoder, path+'A_encoded.png', set_domain_A)
    sample_images(gan.decoder, path+'B_encoded.png', set_domain_B)
    
    plotter.save_hist_image(gan.history, path+'History.png')
    
gan.train(set_domain_A, set_domain_B, epochs=20000, batch_size=64, checkpoint_callback = callback, validation_split = 0.1)    