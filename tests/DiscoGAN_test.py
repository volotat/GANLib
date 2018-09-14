from GANLib import DiscoGAN, distances

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def encoder(x):
    layer = tf.layers.dense(x, 784)
    layer = tf.nn.leaky_relu(layer,alpha=0.2)
    layer = tf.layers.batch_normalization(layer, momentum=0.8)
    
    layer = tf.layers.dense(layer, 784)
    layer = tf.nn.leaky_relu(layer,alpha=0.2)
    layer = tf.layers.batch_normalization(layer, momentum=0.8)
    
    layer = tf.reshape(layer,[-1,28,28,1])
    img = layer
    return img
    
def decoder(x):
    layer = tf.layers.dense(x, 784)
    layer = tf.nn.leaky_relu(layer,alpha=0.2)
    layer = tf.layers.batch_normalization(layer, momentum=0.8)
    
    layer = tf.layers.dense(layer, 784)
    layer = tf.nn.leaky_relu(layer,alpha=0.2)
    layer = tf.layers.batch_normalization(layer, momentum=0.8)
    
    layer = tf.reshape(layer,[-1,28,28,1])
    img = layer
    return img

# D(x)
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
tests = { 'dataset':  (mnist, mnist, mnist, mnist, mnist),
          'img_name': ('mnist_minmax', 'mnist_cross_entropy', 'mnist_wasserstein', 'mnist_iwasserstein_gp', 'mnist_cramer', ),
          'distance': (distances.minmax, distances.cross_entropy, distances.wasserstein, distances.wasserstein_gp, distances.cramer, ),
          'disc_out': (1, 1, 1, 1, 128, )
        }
        

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

    
for i in range(len(tests['dataset'])):
    # Load the dataset
    (mnist_set, labels), (_, _) = tf.keras.datasets.mnist.load_data()
    mnist_set = (mnist_set.astype(np.float32) - 127.5) / 127.5
    mnist_set = np.expand_dims(mnist_set, axis=3)


    (fashion_set, labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    fashion_set = (fashion_set.astype(np.float32) - 127.5) / 127.5
    fashion_set = np.expand_dims(fashion_set, axis=3)

    set_domain_A = mnist_set  [:512]
    set_domain_B = fashion_set[:512]   
    
    #Run GAN for 5000 iterations
    gan = DiscoGAN([set_domain_A.shape[1:], set_domain_B.shape[1:]], distance = tests['distance'][i], n_critic = 3)
    
    gan.generator = generator
    gan.discriminator = lambda x: discriminator(x, tests['disc_out'][i])
    
    gan.encoder = encoder
    gan.decoder = decoder
    gan.discriminator = lambda x: discriminator(x, tests['disc_out'][i])
   
    def callback():
        path = 'images/DiscoGAN/tf_'+tests['img_name'][i]
        sample_images(gan.combined_A, path+'A_decoded.png', set_domain_A)
        sample_images(gan.combined_B, path+'B_decoded.png', set_domain_B)
        
        sample_images(gan.encoder, path+'A_encoded.png', set_domain_A)
        sample_images(gan.decoder, path+'B_encoded.png', set_domain_B)
        
        gan.save_history_to_image(path+'History.png')
      
    gan.train([set_domain_A, set_domain_B], epochs=5000, batch_size=64, checkpoint_callback = callback)
    