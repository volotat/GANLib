from keras.layers import Input, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam
import os
import numpy as np

import keras.backend as K

from . import metrics
from . import utils
from .GAN import GAN

#                   Adversarial Autoencoder
#   Paper: https://arxiv.org/pdf/1511.05644.pdf

#       Description:
#   Attach discriminator to autoencoder in oder to make decoder produce 
#   realistic samples from random noise and make encoder generate  more
#   useful latent representation of data. 

#       To do:
#   Do not store control value

class AAE(GAN):
    def __init__(self, input_shape, latent_dim = 100, **kwargs):
        super(AAE, self).__init__(input_shape, latent_dim , **kwargs)
        
    def set_models_params(self):
        if self.optimizer is None: self.optimizer = Adam(0.0002, 0.5, clipnorm = 10)
            
        self.loss = 'mse'
        self.disc_activation = 'sigmoid'    
        
    def build_graph(self):
        TINY = 1e-8
    
        E = self.encoder
        D = self.decoder
        DSC = self.discriminator
        
        real_img = Input(shape=self.input_shape)
        real_lat = Input(shape=(self.latent_dim,))
        real_val = DSC(real_lat) 
        genr_val = DSC(E(real_img))
        
        self.encoded_decoded = Model(real_img, D(E(real_img)))
        self.encoded_decoded.compile(loss=self.loss, optimizer=self.optimizer)
        
        self.discriminator.trainable = True
        self.encoder.trainable = False
        
        DSC_tns = Lambda(lambda x: -K.mean(K.log(x[0] + TINY) + K.log(1.0 - x[1] + TINY)) )([real_val, genr_val])
        self.disc_model = Model([real_lat, real_img], DSC_tns)
        self.disc_model.compile(loss=utils.ident_loss, optimizer=self.optimizer)
        
        self.discriminator.trainable = False
        self.encoder.trainable = True
        
        GNR_tns = Lambda(lambda x: -K.mean(K.log(x[0] + TINY)))([genr_val])
        self.genr_model = Model([real_img], GNR_tns)
        self.genr_model.compile(loss=utils.ident_loss, optimizer=self.optimizer)

    def prepare_data(self, data_set, validation_split, batch_size):
        super(AAE, self).prepare_data(data_set, validation_split, batch_size)
        
        #This values will be used in a way that do no affect the network
        self.dummy = np.zeros((batch_size, 1))
        
    def train_on_batch(self, batch_size):
        # ---------------------
        #  Reconstruction
        # ---------------------
        
        # Select a random batch of images
        idx = np.random.randint(0, self.train_set.shape[0], batch_size)
        imgs = self.train_set[idx]
        
        # Train the encoder-decoder model as usual autoencoder
        self.m_loss = self.encoded_decoded.train_on_batch([imgs], imgs)
        
        # ---------------------
        #  Regularization
        # ---------------------
        
        #train discriminator to recognize distributions
        #real_lats = np.random.normal(size=(batch_size, self.latent_dim))
        real_lats = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
        d_loss = self.disc_model.train_on_batch([real_lats, imgs], self.dummy)
        
        #train encode to produce right distribution
        g_loss = self.genr_model.train_on_batch([imgs], self.dummy)
        
        return d_loss, g_loss
        
    def build_models(self, files = None, custom_objects = None):
        self.set_models_params()
        
        loaded = False
        if files is not None:
            # Try to load models
            try:
                self.encoder = load_model(files[0], custom_objects=custom_objects)
                self.decoder = load_model(files[1], custom_objects=custom_objects)
                self.discriminator = load_model(files[2], custom_objects=custom_objects)
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
        
    def save(self, files):
        self.encoder.save(files[0])
        self.decoder.save(files[1])
        self.discriminator.save(files[2])
        
    def test_network(self, batch_size):
        idx = np.random.randint(0, self.train_set.shape[0], batch_size)
        imgs = self.train_set[idx]
        
        gen_lats = self.encoder.predict([imgs])
        real_lats = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
    
        gen_val = self.discriminator.predict([gen_lats])
        train_val = self.discriminator.predict([real_lats])
        
        
        if self.valid_set is not None: 
            idx = np.random.randint(0, self.valid_set.shape[0], batch_size)
            val_imgs = self.valid_set[idx]
            val_gen_lats = self.encoder.predict([val_imgs])
            test_val = self.discriminator.predict([val_gen_lats])
        else:
            test_val = np.zeros(batch_size)
        
        cont_val = np.zeros(1)
        
        metric = self.m_loss    
        return gen_val, train_val, test_val, cont_val, metric