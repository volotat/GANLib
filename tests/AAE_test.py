from GANLib import AAE, distances

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf


noise_dim = 10

def encoder(x):
    layer = tf.layers.flatten(x)
    layer = tf.layers.dense(layer, 256)
    layer = tf.nn.leaky_relu(layer,alpha=0.2)
    layer = tf.layers.batch_normalization(layer, momentum=0.8)
    
    layer = tf.layers.dense(layer, 256)
    layer = tf.nn.leaky_relu(layer,alpha=0.2)
    layer = tf.layers.batch_normalization(layer, momentum=0.8)
    
    layer = tf.layers.dense(layer, 784)
    layer = tf.reshape(layer,[-1,28,28,1])
    img = layer
    return img
    
def decoder(x):
    layer = tf.layers.flatten(x)
    layer = tf.layers.dense(layer, 256)
    layer = tf.nn.leaky_relu(layer,alpha=0.2)
    layer = tf.layers.batch_normalization(layer, momentum=0.8)
    
    layer = tf.layers.dense(layer, 256)
    layer = tf.nn.leaky_relu(layer,alpha=0.2)
    layer = tf.layers.batch_normalization(layer, momentum=0.8)
    
    layer = tf.layers.dense(layer, 784)
    layer = tf.reshape(layer,[-1,28,28,1])
    img = layer
    return img

def discriminator(x, outputs):
    layer = x
    
    layer = tf.layers.flatten(layer)
    layer = tf.layers.dense(layer,256)
    layer = tf.nn.leaky_relu(layer, alpha=0.2)
    layer = tf.layers.dense(layer,128)
    layer = tf.nn.leaky_relu(layer, alpha=0.2)
    
    validity = tf.layers.dense(layer, outputs)

    return validity
        
mnist = tf.keras.datasets.mnist    
tests = { 'dataset':  (mnist, mnist, mnist, mnist, mnist, ),
          'img_name': ('mnist_minmax', 'mnist_cross_entropy', 'mnist_wasserstein', 'mnist_iwasserstein_gp', 'mnist_cramer', ),
          'distance': (distances.minmax, distances.cross_entropy, distances.wasserstein, distances.wasserstein_gp, distances.cramer, ),
          'disc_out': (1, 1, 1, 1, 128, )
        }
        
        
r, c = 5, 5
noise = np.random.uniform(-1, 1, (r * c, noise_dim))

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
    fig.savefig(file)
    plt.close()

    
for i in range(len(tests['dataset'])):
    # Load the dataset
    (X_train, labels), (_, _) = tests['dataset'][i].load_data()

    # Configure input
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    if len(X_train.shape)<4: X_train = np.expand_dims(X_train, axis=3)
    
    #Run GAN for 5000 iterations
    gan = AAE(X_train.shape[1:], noise_dim, distance = tests['distance'][i], n_critic = 3)
    
    gan.encoder = encoder
    gan.decoder = decoder
    gan.discriminator = lambda x: discriminator(x, tests['disc_out'][i])
   
    def callback():
        path = 'images/AAE/tf_'+tests['img_name'][i]
        sample_images(gan, path+'.png')
        gan.save_history_to_image(path+'_history.png')
      
    gan.train(X_train, epochs=5000, batch_size=64, checkpoint_callback = callback)
    