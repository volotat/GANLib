from GANLib import CramerGAN
from GANLib import plotter

from keras.datasets import mnist, fashion_mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam, RMSprop, Nadam

import matplotlib.pyplot as plt
import numpy as np

#Do not use BatchNormalization with CramerGAN! It will conflict with the gradient penalty.

class conv_model_28(): 
    def build_generator(self):
        input_lat = Input(shape=(self.latent_dim,))

        layer = input_lat
        layer = Dense(256)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        
        layer = Dense(784)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        
        layer = Reshape((7,7,16))(layer)
        
        #7 -> 14 -> 28
        layer = UpSampling2D(2)(layer)
        layer = Conv2D(8, (3,3), padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer) #14x14x8
        
        layer = UpSampling2D(2)(layer)
        layer = Conv2D(4, (3,3), padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer) #28x28x4
        
        img = Conv2D(1, (1,1), padding='same')(layer)
        return Model(input_lat, img)
        
    def build_discriminator(self):
        input_img = Input(shape=self.input_shape)

        layer = Conv2D(8, (3,3), strides = 2, padding='same')(input_img)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Conv2D(16, (3,3), strides = 2, padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Conv2D(32, (3,3), strides = 2, padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Flatten()(layer)
        validity = Dense(256, activation=self.disc_activation)(layer) #more outputs is better
        return Model(input_img, validity)    
          
class conv_model_32(): 
    def build_generator(self):
        input_lat = Input(shape=(self.latent_dim,))

        layer = input_lat
        layer = Dense(512)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        
        layer = Dense(1024)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        
        layer = Reshape((8,8,16))(layer)
        
        #8 -> 16 -> 32
        layer = UpSampling2D(2)(layer)
        layer = Conv2D(16, (3,3), padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer) #16x16x16
        
        layer = UpSampling2D(2)(layer)
        layer = Conv2D(8, (3,3), padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer) #32x32x8
        
        img = Conv2D(3, (1,1), padding='same')(layer)
        return Model(input_lat, img)
        
    def build_discriminator(self):
        input_img = Input(shape=self.input_shape)

        layer = Conv2D(8, (3,3), strides = 2, padding='same')(input_img)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Conv2D(16, (3,3), strides = 2, padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Conv2D(32, (3,3), strides = 2, padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Flatten()(layer)
        
        validity = Dense(256, activation=self.disc_activation)(layer)
        return Model(input_img, validity)    
        
        
        
tests = { 'dataset':  (mnist,         fashion_mnist, cifar10  ),
          'img_path': ('mnist',       'fashion',     'cifar10'),
          'mode':     ('vanilla',     'vanilla',     'vanilla'),
          'model':    (conv_model_28, conv_model_28, conv_model_32)
        }
        
        
      
noise_dim = 100    

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
    fig.savefig(file) #% epoch
    plt.close()

    
for i in range(len(tests['dataset'])): 
    model = tests['model'][i]  

    # Load the dataset
    (X_train, _), (_, _) = tests['dataset'][i].load_data()

    # Configure input
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    if len(X_train.shape)<4:
        X_train = np.expand_dims(X_train, axis=3)

    #Run GAN for 20000 iterations
    gan = CramerGAN(X_train.shape[1:], noise_dim, mode = tests['mode'][i])
    gan.build_generator = lambda self=gan: model.build_generator(self)
    gan.build_discriminator = lambda self=gan: model.build_discriminator(self)
    gan.build_models()

    def callback():
        path = 'images/CramerGAN/'+tests['img_path'][i]+'/conv_'+tests['mode'][i]
        sample_images(gan.generator, path+'.png')
        plotter.save_hist_image(gan.history, path+'_hist.png')
        
    gan.train(X_train, epochs=20000, batch_size=64, checkpoint_callback = callback, validation_split = 0.1)