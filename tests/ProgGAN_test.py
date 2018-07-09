from GANLib import ProgGAN
from GANLib import plotter

from keras.datasets import mnist, fashion_mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam, RMSprop, Nadam

import matplotlib.pyplot as plt
import numpy as np

       
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
    
    
img_path = 'ProgGAN'
mode = 'vanilla'
    
# Load the dataset
(X_train, _), (_, _) = cifar10.load_data()

# Configure input
X_train = (X_train.astype(np.float32) - 127.5) / 127.5

if len(X_train.shape)<4:
    X_train = np.expand_dims(X_train, axis=3)

#Run GAN for 20000 iterations
gan = ProgGAN(X_train.shape[1:], noise_dim, mode = mode)
#gan.build_generator = lambda self=gan: model.build_generator(self)
#gan.build_discriminator = lambda self=gan: model.build_discriminator(self)
gan.build_models()

def callback():
    path = 'images/'+img_path+'/conv_'+mode
    sample_images(gan.generator, path+'.png')
    plotter.save_hist_image(gan.history, path+'_hist.png')
    
gan.train(X_train, epochs=20000, epochs_grow_rate = 200, batch_size=64, checkpoint_callback = callback, validation_split = 0.1)    