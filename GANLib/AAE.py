from keras.layers import Input, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam
import os
import numpy as np

import keras.backend as K

from . import metrics
from . import utils
from .GAN import GAN

import tensorflow as tf

#                   Adversarial Autoencoder
#   Paper: https://arxiv.org/pdf/1511.05644.pdf

#       Description:
#   Attach discriminator to autoencoder in oder to make decoder produce 
#   realistic samples from random noise and make encoder generate  more
#   useful latent representation of data. 

#       To do:
#   Find a way how to split sets into train and test ones
#   Get rid of modes because it does not really help
#   Insert mu, log, Lambda layers inside class

class AAE(GAN):
    def __init__(self, input_shape, latent_dim = 100, **kwargs):
        super(AAE, self).__init__(input_shape, latent_dim , **kwargs)
        
    def set_models_params(self, optimizer):
        if optimizer is None:   
            self.optimizer = Adam(0.0002, 0.5)
        else:
            self.optimizer = optimizer
            
        self.loss = 'mae'
        self.disc_activation = 'linear'    
        
    def build_graph(self):
        TINY = 1e-8
    
        E = self.encoder
        D = self.decoder
        DSC = self.discriminator
        
        real_img = Input(shape=self.input_shape)
        real_lat = Input(shape=(self.latent_dim,))
        genr_lat = E(real_img)
        genr_val = DSC(genr_lat)
        
        self.decoded_encoded = Model(real_img, D(E(real_img)))
        self.decoded_encoded.compile(loss=self.loss, optimizer=self.optimizer)
        
        self.discriminator.trainable = True
        self.encoder.trainable = False
        
        DSC_tns = Lambda(lambda x: -tf.reduce_mean(tf.log(x[0] + TINY) + tf.log(1.0 - x[1] + TINY)) )([real_lat, genr_lat])
        self.disc_model = Model([real_lat, real_img], DSC_tns)
        self.disc_model.compile(loss=utils.ident_loss, optimizer=self.optimizer)
        
        self.discriminator.trainable = False
        self.encoder.trainable = True
        
        GNR_tns = Lambda(lambda x: -tf.reduce_mean(tf.log(x[0] + TINY)))([genr_val])
        self.genr_model = Model([real_img], GNR_tns)
        self.genr_model.compile(loss=utils.ident_loss, optimizer=self.optimizer)
        
    def get_data(self, train_set, batch_size):
        # Select a random batch of images
        idx = np.random.randint(0, train_set.shape[0], batch_size)
        imgs = train_set[idx]
        
        # Generate noise for two branch of generated images
        noise_a = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
        noise_b = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
            
        return imgs, noise_a, noise_b    
        
    def train_on_batch(self, batch_size):
        #This values will be used in a way that do no affect the network
        self.dummy = np.zeros((batch_size, 1))
        
        TINY = 1e-8
        #everywhere clipped by norm 10
        
        #reconst_trainer
        #enc_vars + dec_vars // self.rec_loss
        #self.rec_loss = tf.reduce_mean( - self.output_dist.logli(self.x_in, x_dist_info))
        #mse?
        
        # ---------------------
        #  Reconstruction
        # ---------------------
        
        # Select a random batch of images
        idx = np.random.randint(0, self.train_set.shape[0], batch_size)
        imgs = self.train_set[idx]
        
        # Train the encoder-decoder model as usual autoencoder
        self.decoded_encoded.train_on_batch([imgs], imgs)
        
        
        # ---------------------
        #  Regularization
        # ---------------------
        
        #discriminator_trainer
        #dis_vars // self.dis_loss
        #self.dis_loss = -tf.reduce_mean(tf.log(real_d + TINY) + tf.log(1. - fake_d + TINY))
        real_lats = np.random.normal(size=(batch_size, self.latent_dim))
        d_loss = self.disc_model.train_on_batch([real_lats, imgs], self.dummy)
        
        #generator_trainer
        #enc_vars // self.gen_loss
        #self.gen_loss = -tf.reduce_mean(tf.log(fake_d + TINY))
        g_loss = self.genr_model.train_on_batch([imgs], self.dummy)
        
        return d_loss, g_loss
        
    def build_models(self, optimizer = None, files = None, custom_objects = None):
        self.set_models_params(optimizer)
        
        loaded = False
        if files is not None:
            # Try to load models
            try:
                '''
                self.generator = load_model(files[0], custom_objects=custom_objects)
                self.discriminator = load_model(files[1], custom_objects=custom_objects)
                '''
                print('not yet')
                loaded = True
                print('models loaded')  
            except IOError as e:
                warnings.warn("Files cannot be opened. Models will be rebuilded instead!")
                
        if not loaded:
            # Build models
            if self.build_discriminator is None or self.build_encoder is None or self.build_decoder is None:
                raise Exception("Model building functions are not defined")
            else:   
                self.encoder = self.build_encoder()
                self.decoder = self.build_decoder()
                self.discriminator = self.build_discriminator()
            
            print('models builded')  
            
        self.build_graph()