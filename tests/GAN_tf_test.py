from GANLib import GAN_tf

#from keras.datasets import mnist, fashion_mnist, cifar10
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

# G(z)
def generator(x):
    layer = tf.keras.layers.Dense(256)(x)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer)
    #layer = tf.keras.layers.BatchNormalization(momentum=0.8)(layer)
    
    layer = tf.keras.layers.Dense(784)(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer)
    #layer = tf.keras.layers.BatchNormalization(momentum=0.8)(layer)
    
    layer = tf.keras.layers.Reshape((7,7,16))(layer)
    
    #7 -> 14 -> 28
    layer = tf.keras.layers.UpSampling2D(2)(layer)
    layer = tf.keras.layers.Conv2D(8, (3,3), padding='same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) #14x14x8
    #layer = tf.keras.layers.BatchNormalization(momentum=0.8, axis = -1)(layer)
    
    layer = tf.keras.layers.UpSampling2D(2)(layer)
    layer = tf.keras.layers.Conv2D(4, (3,3), padding='same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) #28x28x4
    #layer = tf.keras.layers.BatchNormalization(momentum=0.8, axis = -1)(layer)
    
    layer = tf.keras.layers.Conv2D(1, (1,1), padding='same')(layer)
    
    img = layer #tf.tanh(layer)
    return img

# D(x)
def discriminator(x):
    
    layer = tf.keras.layers.Conv2D(8, (3,3), strides = 2, padding='same')(x)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer)
    layer = tf.keras.layers.Conv2D(16, (3,3), strides = 2, padding='same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer)
    layer = tf.keras.layers.Conv2D(32, (3,3), strides = 2, padding='same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer)
    layer = tf.keras.layers.Flatten()(layer)
    
    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.keras.layers.Dense(256)(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(128)(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    
    validity = tf.keras.layers.Dense(1)(layer)

    return validity
        
tests = { 'dataset':  (tf.keras.datasets.mnist, tf.keras.datasets.fashion_mnist, tf.keras.datasets.cifar10),
          'img_path': ('mnist',       'fashion',     'cifar10')
          #'model':    (conv_model_28, conv_model_28, conv_model_32)
        }
        
        
      
noise_dim = 100    

def sample_images(gen, file):
    r, c = 5, 5
    
    noise = np.random.uniform(-1, 1, (r * c, noise_dim))

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

    
for i in range(len(tests['dataset'])): 
    #model = tests['model'][i]  

    # Load the dataset
    (X_train, _), (_, _) = tests['dataset'][i].load_data()

    # Configure input
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    if len(X_train.shape)<4:
        X_train = np.expand_dims(X_train, axis=3)
    
    #Run GAN for 20000 iterations
    gan = GAN_tf(X_train.shape[1:], noise_dim)
    
    gan.generator = generator
    gan.discriminator = discriminator
   
    def callback():
        path = 'images/GAN/'+tests['img_path'][i]+'/tf_'
        sample_images(gan, path+'.png')
        #gan.save_history_to_image(path+'History.png')
      
    gan.train(X_train, epochs=20000, batch_size=64, checkpoint_callback = callback, collect_history = False)
    