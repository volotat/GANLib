from GANLib import GAN_tf

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        return x

# G(z)
def generator(x):
    layer = tf.layers.dense(x, 256)
    layer = tf.nn.leaky_relu(layer,alpha=0.2)
    layer = tf.layers.batch_normalization(layer, momentum=0.8)
    
    layer = tf.layers.dense(layer, 784)
    layer = tf.nn.leaky_relu(layer,alpha=0.2)
    layer = tf.layers.batch_normalization(layer, momentum=0.8)
    
    layer = tf.reshape(layer,[-1,7,7,16])
    
    layer = upscale2d(layer)
    layer = tf.layers.conv2d(layer, 8, (3,3), padding='same')
    layer = tf.nn.leaky_relu(layer, alpha=0.2)
    layer = tf.layers.batch_normalization(layer, momentum=0.8)
    
    layer = upscale2d(layer)
    layer = tf.layers.conv2d(layer, 4, (3,3), padding='same')
    layer = tf.nn.leaky_relu(layer, alpha=0.2)
    layer = tf.layers.batch_normalization(layer, momentum=0.8)
    
    layer = tf.layers.conv2d(layer, 1, (1,1), padding='same')
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
        
tests = { 'dataset':  (tf.keras.datasets.mnist, tf.keras.datasets.fashion_mnist),
          'img_path': ('mnist', 'fashion')
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
        gan.save_history_to_image(path+'History.png')
      
    gan.train(X_train, epochs=20000, batch_size=64, checkpoint_callback = callback)
    