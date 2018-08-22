from keras.layers import Input, Lambda, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
import os
import numpy as np

import keras.backend as K

from .. import metrics
from .. import utils
from .GAN import GAN

class XGAN(GAN):
    def __init__(self, input_shape, latent_dim = 100, **kwargs):
        super(XGAN, self).__init__(input_shape, latent_dim , **kwargs)
        
    def set_models_params(self):
        if self.optimizer is None: self.optimizer = Adam(0.0002, 0.5, clipnorm = 10)
            
        self.models = ['identificator', 'generator']
        self.loss = 'mse'
        self.disc_activation = 'sigmoid'    
        self.indexes_weights = None
        
    def build_graph(self):
        TINY = 1e-8
    
        I = self.identificator
        G = self.generator
       
        inputs = I.inputs
        
        self.encoded_decoded = Model(inputs, G(I(inputs)))
        self.encoded_decoded.compile(loss=self.loss, optimizer=self.optimizer)
        
        '''
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
        '''

    def prepare_data(self, data_set, validation_split, batch_size):
        super(XGAN, self).prepare_data(data_set, validation_split, batch_size)
        
        #self.data_size = self.train_set.shape[0]
        #self.eye = np.eye(self.data_size)
        #This values will be used in a way that do no affect the network
        self.dummy = np.zeros((batch_size, 1))
        
        
        
        
    def train_on_batch(self, batch_size):
        # ---------------------
        #  Reconstruction
        # ---------------------
        
        # Select a random batch of images
        idx = np.random.randint(0, self.train_set[0].shape[0], batch_size)
        
        #poss = self.train_set[0][idx]
        indx = self.train_set[0][idx]
        imgs = self.train_set[1][idx]
        
        # Train the encoder-decoder model as usual autoencoder
        self.m_loss = self.encoded_decoded.train_on_batch([indx], imgs)
        '''
        # ---------------------
        #  Regularization
        # ---------------------
        
        #train discriminator to recognize distributions
        #real_lats = np.random.normal(size=(batch_size, self.latent_dim))
        real_lats = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
        d_loss = self.disc_model.train_on_batch([real_lats, imgs], self.dummy)
        
        #train encode to produce right distribution
        g_loss = self.genr_model.train_on_batch([imgs], self.dummy)
        '''
        d_loss = 0
        g_loss = self.m_loss
        return d_loss, g_loss
        
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