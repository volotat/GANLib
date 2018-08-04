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
#   Paper: https://arxiv.org/pdf/1705.10743.pdf (the one that I trying to implement)
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
            optimizer = Adam(0.0002, 0.5)
            
        if self.mode == 'stable':
            loss = 'logcosh'
            self.disc_activation = 'linear'
        elif self.mode == 'vanilla':
            loss = 'binary_crossentropy'
            self.disc_activation = 'linear'
        else: raise Exception("Mode '" + self.mode+ "' is unknown")
        
    
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
                
                
        self.G = self.generator
        self.D = self.discriminator
        
        real_img = Input(shape=self.input_shape)
        noise_a = Input(shape=(self.latent_dim,))
        noise_b = Input(shape=(self.latent_dim,))
        genr_img = self.G([noise_a])
        
        hxr =  self.D([real_img])
        hxga = self.D(self.G([noise_a]))
        hxgb = self.D(self.G([noise_b]))
        
        
        #-------------------------------
        # Graph for Generator
        #-------------------------------
        
        self.discriminator.trainable = False
        self.generator.trainable = True 
        
        L_G_tns = Lambda(lambda x: tf.norm(x[0] - x[1], axis=-1) + tf.norm(x[0] - x[2], axis=-1) - tf.norm(x[1] - x[2], axis=-1))([hxr, hxga, hxgb])
        L_G_tns = Lambda(lambda x: K.expand_dims(x, axis = -1))(L_G_tns)
        
        self.genr_model = Model([real_img, noise_a, noise_b], L_G_tns)
        self.genr_model.compile(loss=utils.ident_loss, optimizer=optimizer)
        
        
        #-------------------------------
        # Graph for the Discriminator
        #-------------------------------
        
        self.discriminator.trainable = True
        self.generator.trainable = False
        
        def crit(x, xg_):
            return tf.norm(x - xg_, axis=-1) - tf.norm(x, axis=-1)
        
        
        
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * real_img + (1 - epsilon) * genr_img
        d_hat = crit(self.D(x_hat), hxgb)
        
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.norm(ddx, axis=1)
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * self.lambda_scale)
        
        L_D_tns = Lambda(lambda x: (ddx - (crit(x[0], x[2]) - crit(x[1], x[2]))))([hxr, hxga, hxgb])
        L_D_tns = Lambda(lambda x: K.expand_dims(x, axis = -1))(L_D_tns)
        
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
    
        #collect statistical info of data
        data_set_std = np.std(data_set,axis = 0)
        data_set_mean = np.mean(data_set,axis = 0)
    
        # Adversarial ground truths
        #valid = np.ones((batch_size, 1))
        #fake = np.zeros((batch_size, 1))
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
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Select a random batch of images
            idx = np.random.randint(0, train_set.shape[0], batch_size)
            imgs = train_set[idx]
            
            # Generate noise for two branch of generated images
            noise_a = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
            noise_b = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
            
            # ---------------------
            #  Train Discriminator and Generator
            # ---------------------
            
            #target values do not affect the network, so it does not matter what they are ¯\_(ツ)_/¯ 
            d_loss = self.disc_model.train_on_batch([imgs, noise_a, noise_b], dummy)
            g_loss = self.genr_model.train_on_batch([imgs, noise_a, noise_b], dummy)
            
            
            '''
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, train_set.shape[0], batch_size)
            imgs = train_set[idx]

            # Sample noise as generator input
            noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))

            # Generate two branch of new images
            gen_imgs = self.generator.predict([noise])
            
            
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = (d_loss_real + d_loss_fake) / 2
                
            else: raise Exception("Mode '" + self.mode+ "' is unknown")
            
            # ---------------------
            #  Train Discriminator and Generator
            # ---------------------
            
            # Train the generator
            g_loss = self.combined.train_on_batch([noise], valid)
            '''
            # Plot the progress
            if epoch % checkpoint_range == 0:
                print(epoch, d_loss, g_loss)
                '''
                gen_val = self.discriminator.predict([gen_imgs])
                
                #idx = np.random.randint(0, train_set.shape[0], batch_size)
                #train_val = self.discriminator.predict(train_set[idx])
                train_val = self.discriminator.predict([imgs])
                
                if valid_set is not None: 
                    idx = np.random.randint(0, valid_set.shape[0], batch_size)
                    test_val = self.discriminator.predict(valid_set[idx])
                else:
                    test_val = np.zeros(batch_size)
                
                noise = np.random.normal(data_set_mean, data_set_std, (batch_size,)+ self.input_shape)
                cont_val = self.discriminator.predict(noise)
                
                metric = self.metric_test(train_set, 1000)
                
                if verbose:
                    print ("%d [D loss: %f] [G loss: %f] [validations TRN: %f, TST: %f] [metric: %f]" % (epoch, d_loss, g_loss, np.mean(train_val), np.mean(test_val), np.mean(metric)))
                
                hist_size = history['hist_size'] = history['hist_size']+1
                history['gen_val']    [hist_size-1] = np.mean(gen_val),  np.min(gen_val),  np.max(gen_val)
                history['train_val']  [hist_size-1] = np.mean(train_val),np.min(train_val),np.max(train_val)
                history['test_val']   [hist_size-1] = np.mean(test_val), np.min(test_val), np.max(test_val)
                history['control_val'][hist_size-1] = np.mean(cont_val), np.min(cont_val), np.max(cont_val) 
                history['metric']     [hist_size-1] = np.mean(metric),   np.min(metric),   np.max(metric)
                
                if np.mean(metric)*0.98 < self.best_metric or self.best_model == None:
                    self.best_model = self.generator.get_weights()
                    self.best_metric = np.mean(metric)
                    history['best_metric'] = self.best_metric
                    
                self.history = history
                '''
                if checkpoint_callback is not None:
                    checkpoint_callback()
                
        
        
        if save_best_model:
            self.generator.set_weights(self.best_model)    
            
        self.epoch = epochs
        checkpoint_callback()   
        
        return self.history    