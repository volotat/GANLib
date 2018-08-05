from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import Lambda, Reshape, Activation
import os
import numpy as np

import keras.backend as K
import tensorflow as tf

from . import metrics
from . import utils

#                   Cramer GAN
#   Paper: https://arxiv.org/pdf/1705.10743.pdf (the one that implemented)
#   Newer version: https://openreview.net/pdf?id=S1m6h21Cb

#       Description:


class CramerGAN():

    def metric_test(self, set, pred_num = 32):    
        met_arr = np.zeros(pred_num)
        
        n_indx = np.random.choice(set.shape[0],pred_num)
        org_set = set[n_indx]
        
        noise = np.random.uniform(-1, 1, (pred_num, self.latent_dim))
        gen_set = self.generator.predict([noise]) 
        met_arr = metrics.magic_distance(org_set, gen_set)
        return met_arr

    def __init__(self, input_shape, latent_dim = 100, mode = 'vanilla'):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.mode = mode
        
        self.build_discriminator = None
        self.build_generator = None
        
        self.best_model = None
        self.best_metric = 0
        
        self.epoch = 0
        self.history = None
        
        self.lambda_scale = 10
        
    def build_models(self, optimizer = None, path = ''):
        if optimizer is None:
            optimizer = Adam(0.0002, 0.5, 0.9)
            
        self.disc_activation = 'linear'
        
        self.path = path
        if os.path.isfile(path+'/generator.h5') and os.path.isfile(path+'/discriminator.h5'):
            self.generator = load_model(path+'/generator.h5')
            self.discriminator = load_model(path+'/discriminator.h5')
        else:
            if self.build_discriminator is None or self.build_generator is None:
                raise Exception("Model building functions are not defined")
            else:
                # Build generator and discriminator
                self.generator = self.build_generator()
                self.discriminator = self.build_discriminator()
                
                
        G = self.generator
        D = self.discriminator
        
        real_img = Input(shape=self.input_shape)
        noise_a = Input(shape=(self.latent_dim,))
        noise_b = Input(shape=(self.latent_dim,))
        
        genr_img = G([noise_a])
        hxr =  D([real_img])
        hxga = D(G([noise_a]))
        hxgb = D(G([noise_b]))
        
        
        #Define the critic:
        def crit(x, xg_):
            return tf.norm(x - xg_, axis=-1) - tf.norm(x, axis=-1)
        
        #-------------------------------
        # Graph for Generator
        #-------------------------------
        
        self.discriminator.trainable = False
        self.generator.trainable = True 
        
        #Compute the generator loss:
        L_G_tns = Lambda(lambda x: tf.reduce_mean(tf.norm(x[0] - x[1], axis=-1) + tf.norm(x[0] - x[2], axis=-1) - tf.norm(x[1] - x[2], axis=-1)))([hxr, hxga, hxgb])
        
        self.genr_model = Model([real_img, noise_a, noise_b], L_G_tns)
        self.genr_model.compile(loss=utils.ident_loss, optimizer=optimizer)
        
        
        #-------------------------------
        # Graph for the Discriminator
        #-------------------------------
        
        self.discriminator.trainable = True
        self.generator.trainable = False
        
        #compute gradient penalty with respect to weighted average between real and generated images
        def f_ddx(real, genr):
            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * real + (1 - epsilon) * genr
            d_hat = crit(D(x_hat), hxgb)
            
            ddx = tf.gradients(d_hat, x_hat)[0]
            ddx = tf.norm(ddx, axis=1)
            ddx = tf.reduce_mean(tf.square(ddx - 1.0) * self.lambda_scale)
            
            return ddx
        
        #Compute the surrogate generator loss:
        L_S_tns = Lambda(lambda x: tf.reduce_mean(crit(x[0], x[2]) - crit(x[1], x[2])))([hxr, hxga, hxgb])
        #Compute the critic loss:
        L_D_tns = Lambda(lambda x: (f_ddx(x[0], x[1]) - x[2]))([real_img, genr_img, L_S_tns])
        
        self.disc_model = Model([real_img, noise_a, noise_b], L_D_tns)
        self.disc_model.compile(loss=utils.ident_loss, optimizer=optimizer)
        
            
        print('models builded')    
            
    def save(self):
        self.generator.save(self.path+'/generator.h5')
        self.discriminator.save(self.path+'/discriminator.h5')
    
    def train(self, data_set, batch_size=32, epochs=1, verbose=True, checkpoint_range = 100, checkpoint_callback = None, validation_split = 0, save_best_model = False):
        """Trains the model for a given number of epochs (iterations on a dataset).
        # Arguments
            data_set: 
                Numpy array of training data.
            batch_size:
                Number of samples per gradient update.
            epochs: Number of epochs to train the model.
                An epoch is an iteration over batch sized samples of dataset.
            checkpoint_range:
                Range in witch checkpoint callback will be called and history data will be stored.
            verbose: 
                Integer. 0, 1. Verbosity mode.
            checkpoint_callback: List of `keras.callbacks.Callback` instances.
                Callback to apply during training on checkpoint stage.
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples.
            save_best_model:
                Boolean. If True, generator weights will be resigned to best model according to chosen metric.
        # Returns
            A history object. 
        """ 
    
        if 0. < validation_split < 1.:
            split_at = int(data_set.shape[0] * (1. - validation_split))
            train_set = data_set[:split_at]
            valid_set = data_set[split_at:]
        else:
            train_set = data_set
            valid_set = None
    
        # collect statistical info of data
        data_set_std = np.std(data_set,axis = 0)
        data_set_mean = np.mean(data_set,axis = 0)
    
        # target values do not affect the network, so it does not matter what they are ¯\_(ツ)_/¯ 
        dummy = np.zeros((batch_size, 1))

        #mean min max
        max_hist_size = epochs//checkpoint_range + 1
        history = { 'gen_val'    :np.zeros((max_hist_size,3)), 
                    'train_val'  :np.zeros((max_hist_size,3)), 
                    'test_val'   :np.zeros((max_hist_size,3)), 
                    'control_val':np.zeros((max_hist_size,3)), 
                    'metric'     :np.zeros((max_hist_size,3)),
                    'best_metric':0,
                    'hist_size'  :0}
        
        
        
        def get_data():
            # Select a random batch of images
            idx = np.random.randint(0, train_set.shape[0], batch_size)
            imgs = train_set[idx]
            
            # Generate noise for two branch of generated images
            noise_a = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
            noise_b = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
                
            return imgs, noise_a, noise_b
        
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            d_iters = 3
            d_loss = 0
            for _ in range(0, d_iters):
                imgs, noise_a, noise_b = get_data()
                d_loss += self.disc_model.train_on_batch([imgs, noise_a, noise_b], dummy) / d_iters
            
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            imgs, noise_a, noise_b = get_data()
            g_loss = self.genr_model.train_on_batch([imgs, noise_a, noise_b], dummy)

            # Save progress info
            if epoch % checkpoint_range == 0:
                gen_imgs = self.generator.predict([noise_a])
                gen_val = self.discriminator.predict([gen_imgs])
                train_val = self.discriminator.predict([imgs])
                
                if valid_set is not None: 
                    idx = np.random.randint(0, valid_set.shape[0], batch_size)
                    test_val = self.discriminator.predict(valid_set[idx])
                else:
                    test_val = np.zeros(batch_size)
                
                noise = np.random.normal(data_set_mean, data_set_std, (batch_size,)+ self.input_shape)
                cont_val = self.discriminator.predict(noise)
                
                metric = self.metric_test(train_set, 128)
                
                if verbose:
                    print ("%d [D loss: %f] [G loss: %f] [validations TRN: %f, TST: %f] [metric: %f]" % (epoch, d_loss, g_loss, np.mean(train_val), np.mean(test_val), np.mean(metric)))
                
                hist_size = history['hist_size'] = history['hist_size']+1
                history['gen_val']    [hist_size-1] = np.mean(gen_val),  np.min(gen_val),  np.max(gen_val)
                history['train_val']  [hist_size-1] = np.mean(train_val),np.min(train_val),np.max(train_val)
                history['test_val']   [hist_size-1] = np.mean(test_val), np.min(test_val), np.max(test_val)
                history['control_val'][hist_size-1] = np.mean(cont_val), np.min(cont_val), np.max(cont_val) 
                history['metric']     [hist_size-1] = np.mean(metric),   np.min(metric),   np.max(metric)
                
                if (np.mean(metric)*0.98 < self.best_metric or self.best_model == None):
                    self.best_model = self.generator.get_weights()
                    self.best_metric = np.mean(metric)
                    history['best_metric'] = self.best_metric
                    
                self.history = history
                
                if checkpoint_callback is not None:
                    checkpoint_callback()
                
        
        
        if save_best_model:
            self.generator.set_weights(self.best_model)    
            
        self.epoch = epochs
        checkpoint_callback()   
        
        return self.history    