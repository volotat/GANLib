from GANLib import CGAN, distances

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
def generator(z, l):
    layer = tf.concat((z, l), axis = -1)

    layer = tf.layers.dense(layer, 256)
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
def discriminator(x, l, outputs):
    layer = x
    
    layer = tf.layers.flatten(layer)
    layer = tf.concat((layer, l), axis = -1)
    
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
        
noise_dim = 100    

def sample_images(gen, file):
    r, c = 5, 5
    
    noise = np.random.uniform(-1, 1, (r * c, noise_dim))
    labels = np.zeros((r*c,10))
    for i in range(r):
        labels[i::r, i] = 1.

    gen_imgs = gen.predict(noise, labels)

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
    # Load the dataset
    (X_train, labels), (_, _) = tests['dataset'][i].load_data()

    # Configure input
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    if len(X_train.shape)<4:
        X_train = np.expand_dims(X_train, axis=3)
        
    Y_train = np.zeros((X_train.shape[0],10))
    Y_train[np.arange(X_train.shape[0]), labels] = 1.
    
    with tf.Session() as sess:
        #Run GAN for 5000 iterations
        gan = CGAN(sess, [X_train.shape[1:],Y_train.shape[1:]], noise_dim, distance = tests['distance'][i], n_critic = 3)
        
        gan.generator = generator
        gan.discriminator = lambda x, l: discriminator(x, l, tests['disc_out'][i])
       
        def callback():
            path = 'images/CGAN/tf_'+tests['img_name'][i]
            sample_images(gan, path+'.png')
            gan.save_history_to_image(path+'_history.png')
          
        gan.train([X_train, Y_train], epochs=5000, batch_size=64, checkpoint_callback = callback)
    