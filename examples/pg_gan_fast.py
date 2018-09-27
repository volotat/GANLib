from GANLib import GAN, utils, distances

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

from skimage.measure import block_reduce



#                   Progressive Growing of GANs
#   Paper: https://arxiv.org/pdf/1710.10196.pdf

#       Description:
#   Takes as input some dataset and trains the network as usual GAN but progressively 
#   adding layers to generator and discriminator.

#       Note:
#   This implementation of PG GAN trains much faster than original, but support only 
#   constant amount of filters for all convolution layers.

#-------------------------------
# Auxiliary functions
#-------------------------------  

def augment(data):
    off_x_p = data.copy()
    off_x_p[:,1:,:,:] = off_x_p[:,:-1,:,:]
    off_x_m = data.copy()
    off_x_p[:,:-1,:,:] = off_x_p[:,1:,:,:]
    data = np.concatenate((data,off_x_p,off_x_m), axis = 0)
    
    off_y_p = data.copy()
    off_y_p[:,:,1:,:] = off_y_p[:,:,:-1,:]
    off_y_m = data.copy()
    off_y_m[:,:,:-1,:] = off_y_m[:,:,1:,:]
    data = np.concatenate((data,off_y_p,off_y_m), axis = 0)
    
    return data
    
def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        return x
        
def get_scaled_weight(shape, dtype, partition_info):
    #He's normal dynamic weight scaler
    if len(shape) == 2:
        fan_in = shape[0]
    else:
        receptive_field_size = np.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        
    std = np.sqrt(2 / max(1., fan_in))
    return tf.get_variable("w", shape=shape, initializer=tf.initializers.random_normal(0, 1), dtype = tf.float32) * std
 
#-------------------------------
# Define models structure
#-------------------------------      
      
initialization = get_scaled_weight

filters = 64
noise_dim = 64 
channels = 3

weights = {}
sheets = 0


def new_sheet(filters, kernel_size, padding, name, pix_norm = True):
    def func(layer):
        layer = tf.layers.conv2d(layer, filters, kernel_size, padding=padding, name = name, kernel_initializer = initialization)
        layer = tf.nn.leaky_relu(layer, alpha=0.2) 
        if pix_norm: layer = utils.PixelNorm(layer)
        return layer
    return func
    
def transition_alpha(gan):
    epoch = tf.cast(gan.epoch, tf.float32)
    epochs = tf.cast(gan.epochs, tf.float32)
    a = epoch / (epochs/ 2)
    b = 1
    return tf.minimum(a, b) 
   
def generator(input, gan):
    previous_step = None
    next_step = None

    layer = input
    
    layer = tf.keras.layers.RepeatVector(16)(layer)
    layer = tf.keras.layers.Reshape((4, 4, noise_dim))(layer)
    
    layer = new_sheet(filters, (4,4), 'same', 'genr_head_0')(layer)
    layer = new_sheet(filters, (3,3), 'same', 'genr_head_1')(layer)
    
    #Growing layers
    for i in range(sheets):
        s = image_size_list[i + 1]
        layer = upscale2d(layer)
        if i == sheets-1: previous_step = layer
            
        layer = new_sheet(filters, (3,3), 'same', 'genr_layer_0'+str(i))(layer)
   
    next_step = tf.layers.conv2d(layer, channels, (1,1), name = 'to_rgb_'+str(sheets), kernel_initializer = initialization) #to RGB
    
    #smooth fading
    if previous_step is not None: 
        previous_step = tf.layers.conv2d(previous_step, channels, (1,1), name = 'to_rgb_'+str(sheets - 1)) 
        layer = previous_step + (next_step - previous_step) * transition_alpha(gan)
    else:
        layer = next_step
      
    return layer
    
def discriminator(input, gan):
    previous_step = None
    next_step = None
    
    input_layer = input 
    
    layer = tf.layers.conv2d(input_layer, filters, (1,1), name = 'from_rgb_'+str(sheets), kernel_initializer = initialization) #from RGB
    layer = tf.nn.leaky_relu(layer, alpha=0.2)
    
    #Growing layers
    for i in range(sheets, 0, -1):
        layer = new_sheet(filters, (3,3), 'same', 'disc_layer_0'+str(i), pix_norm = False)(layer)
        layer = tf.layers.average_pooling2d(layer, 2, 2)

        #smooth fading
        if i == sheets:
            next_step = layer
            
            previous_step = tf.layers.average_pooling2d(input_layer, 2, 2)
            previous_step = tf.layers.conv2d(previous_step, filters, (1,1), name = 'from_rgb_'+str(sheets - 1), kernel_initializer = initialization) #from RGB
            previous_step = tf.nn.leaky_relu(previous_step, alpha=0.2)
        
            layer = previous_step + (next_step - previous_step) * transition_alpha(gan)
                
    
    layer = utils.MiniBatchStddev(layer, group_size=4)
    layer = new_sheet(filters, (3,3), 'same', 'disc_head_0', pix_norm = False)(layer)
    layer = new_sheet(filters, (4,4), 'valid', 'disc_head_1', pix_norm = False)(layer)
    
    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.layers.dense(layer, 1, kernel_initializer = initialization)

    return layer
    
#-------------------------------
#  Main code
#-------------------------------  

r, c = 3, 5
sample_noise = np.random.uniform(-1, 1, (r * c, noise_dim))
def sample_images(gen, file):
    gen_imgs = gen.predict(sample_noise, moving_avarage = True)

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
(dataseta, labelsa), (datasetb, labelsb) = tf.keras.datasets.cifar10.load_data()
dataset = np.concatenate((dataseta,datasetb), axis = 0)
labels = np.concatenate((labelsa,labelsb), axis = 0)

indx = np.where(labels == 8)[0] # we choose only one specific domain from the dataset
dataset = dataset[indx]

# Configure input
dataset = (dataset.astype(np.float32) - 127.5) / 127.5
if len(dataset.shape)<4:
    dataset = np.expand_dims(dataset, axis=3)
    

# 6000 examples is not enough, so we augment dataset by shifting it along axis by 1 pixel 
dataset = augment(dataset)
 
epochs_list = [4000, 8000, 16000, 32000]
batch_size_list = [16, 16, 16, 16]  
image_size_list = [4, 8, 16, 32] 

optimizer = tf.train.AdamOptimizer(0.001, 0., 0.99, epsilon = 1e-08) #Hyperparameters for optimizer from paper
with tf.Session() as sess:
    t = time.time()
    dataset_t = tf.Variable(np.zeros_like(dataset), dtype = tf.float32)
    for i in range(len(epochs_list)):    
        epochs = epochs_list[i]
        batch_size = batch_size_list[i]
        
        data_set = sess.run(tf.image.resize_bilinear(dataset_t, (image_size_list[i], image_size_list[i])), feed_dict = {dataset_t: dataset})
        print(data_set.shape)
        
        # Build and train GAN
        gan = GAN(sess, data_set.shape[1:], noise_dim, optimizer = optimizer, distance = distances.wasserstein_gp)
        gan.generator = lambda x: generator(x, gan) #define generator model
        gan.discriminator = lambda x: discriminator(x, gan) #define discriminator model
        
        def callback():
            sample_images(gan, 'pg_gan.png')
            
        gan.train(data_set, epochs = epochs, batch_size = batch_size, checkpoint_callback = callback, collect_history = False)  
        sheets += 1
        
    print('Training complete! Total traning time: %f s'%(time.time() - t))   