from GANLib import GAN

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# G(z)
def generator(x):
    layer = tf.layers.dense(x, 256)
    layer = tf.nn.leaky_relu(layer,alpha=0.2)
    layer = tf.layers.batch_normalization(layer, momentum=0.8)
    
    layer = tf.layers.dense(layer, 784)
    layer = tf.nn.leaky_relu(layer,alpha=0.2)
    layer = tf.layers.batch_normalization(layer, momentum=0.8)
    
    layer = tf.reshape(layer,[-1,28,28,1])
    img = layer
    return img
        
# D(x)
def discriminator(x):
    layer = x
    
    layer = tf.layers.flatten(layer)
    layer = tf.layers.dense(layer,256)
    layer = tf.nn.leaky_relu(layer, alpha=0.2)
    layer = tf.layers.dense(layer,128)
    layer = tf.nn.leaky_relu(layer, alpha=0.2)
    
    validity = tf.layers.dense(layer,1)

    return validity
  
# Save examples of generated images to file  
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
    fig.savefig(file)
    plt.close()    
    
# Load the dataset
(data, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Configure input
data = (data.astype(np.float32) - 127.5) / 127.5
if len(data.shape)<4: data = np.expand_dims(data, axis=3)
data_shape = data.shape[1:]
noise_dim = 100


# Build GAN and train it on data
gan = GAN(data_shape, noise_dim) #define type of Generative model
gan.generator = generator #define generator model
gan.discriminator = discriminator #define discriminator model

def callback():
    sample_images(gan, 'simple_gan.png')

gan.train(data, epochs=20000, batch_size=64, checkpoint_callback = callback, collect_history = False) #train GAN for 20000 iterations


