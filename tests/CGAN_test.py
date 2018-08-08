from GANLib import CGAN
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

class conv_model_28(): 
    def build_generator(self):
        input_lat = Input(shape=(self.latent_dim,))
        input_lbl = Input(shape=self.label_shape) 

        layer = concatenate([input_lat, input_lbl])
        layer = Dense(256)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = BatchNormalization(momentum=0.8)(layer)
        
        layer = Dense(784)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = BatchNormalization(momentum=0.8)(layer)
        
        layer = Reshape((7,7,16))(layer)
        
        #7 -> 14 -> 28
        layer = UpSampling2D(2)(layer)
        layer = Conv2D(8, (3,3), padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer) #14x14x8
        layer = BatchNormalization(momentum=0.8, axis = -1)(layer)
        
        layer = UpSampling2D(2)(layer)
        layer = Conv2D(4, (3,3), padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer) #28x28x4
        layer = BatchNormalization(momentum=0.8, axis = -1)(layer)
        
        img = Conv2D(1, (1,1), padding='same')(layer)
        return Model([input_lat, input_lbl], img)
        
    def build_discriminator(self):
        input_img = Input(shape=self.input_shape)
        input_lbl = Input(shape=self.label_shape) 

        layer = Conv2D(8, (3,3), strides = 2, padding='same')(input_img)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Conv2D(16, (3,3), strides = 2, padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Conv2D(32, (3,3), strides = 2, padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Flatten()(layer)
        
        layer = concatenate([layer, input_lbl])
        layer = Dense(256)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dense(128)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        
        validity = Dense(1, activation=self.disc_activation)(layer)
        return Model([input_img, input_lbl], validity)    

class dense_model(): 
    def build_generator(self):
        input_lat = Input(shape=(self.latent_dim,))

        layer = input_lat
        
        layer = Dense(128)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = BatchNormalization(momentum=0.8)(layer)
        
        layer = Dense(256)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = BatchNormalization(momentum=0.8)(layer)
        
        layer = Dense(784)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = BatchNormalization(momentum=0.8)(layer)
        
        img = Reshape((28,28,1))(layer)
        return Model(input_lat, img)
        
    def build_discriminator(self):
        input_img = Input(shape=self.input_shape)

        layer = Flatten()(input_img)
        layer = Dense(784)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        
        layer = Dense(256)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        
        layer = Dense(128)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        
        validity = Dense(1, activation=self.disc_activation)(layer)
        
        return Model(input_img, validity) 
          
class conv_model_32(): 
    def build_generator(self):
        input_lat = Input(shape=(self.latent_dim,))

        layer = input_lat
        layer = Dense(512)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = BatchNormalization(momentum=0.8)(layer)
        
        layer = Dense(1024)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = BatchNormalization(momentum=0.8)(layer)
        
        layer = Reshape((8,8,16))(layer)
        
        #8 -> 16 -> 32
        layer = UpSampling2D(2)(layer)
        layer = Conv2D(16, (3,3), padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer) #16x16x16
        layer = BatchNormalization(momentum=0.8, axis = -1)(layer)
        
        layer = UpSampling2D(2)(layer)
        layer = Conv2D(8, (3,3), padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer) #32x32x8
        layer = BatchNormalization(momentum=0.8, axis = -1)(layer)
        
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
        
        layer = Dense(256)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dense(128)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        
        validity = Dense(1, activation=self.disc_activation)(layer)
        return Model(input_img, validity)    
        
        
        
tests = { 'dataset':  (mnist,         mnist,         fashion_mnist, fashion_mnist, cifar10,       cifar10),
          'img_path': ('mnist',       'mnist',       'fashion',     'fashion',     'cifar10',     'cifar10'),
          'mode':     ('vanilla',     'stable',      'vanilla',     'stable',      'vanilla',     'stable'),
          'model':    (conv_model_28, conv_model_28, conv_model_28, conv_model_28, conv_model_32, conv_model_32)
        }
        
        
      
noise_dim = 100    

def sample_images(gen, file):
    r, c = 5, 5
    
    noise = np.random.uniform(-1, 1, (r * c, noise_dim))
    labels = np.zeros((r*c,10))
    for i in range(r):
        labels[i::r, i] = 1.

    gen_imgs = gen.predict([noise, labels])

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
    (X_train, labels), (_, _) = tests['dataset'][i].load_data()
    
    Y_train = np.zeros((X_train.shape[0],10))
    Y_train[np.arange(X_train.shape[0]), labels] = 1.

    # Configure input
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    if len(X_train.shape)<4:
        X_train = np.expand_dims(X_train, axis=3)

    #Run GAN for 20000 iterations
    gan = CGAN(X_train.shape[1:], (10,), noise_dim, mode = tests['mode'][i])
    gan.build_generator = lambda self=gan: model.build_generator(self)
    gan.build_discriminator = lambda self=gan: model.build_discriminator(self)
    gan.build_models()

    def callback():
        path = 'images/CGAN/'+tests['img_path'][i]+'/conv_'+tests['mode'][i]
        sample_images(gan.generator, path+'.png')
        plotter.save_hist_image(gan.history, path+'_hist.png')
        
    gan.train(zip(X_train, Y_train), epochs=20000, batch_size=64, checkpoint_callback = callback, validation_split = 0.1)