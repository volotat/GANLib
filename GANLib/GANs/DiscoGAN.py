from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import os
import numpy as np


from .. import metrics
from .. import utils
from .GAN import GAN

#                   DiscoGAN
#   Paper: https://arxiv.org/pdf/1703.05192.pdf

#       Description:
#   Takes as input two sets from different domains and by finding correlations 
#   between them encode samples from one domain to another and backwards. 
#   In limit it suppose to find one to one bijection mapping between this sets.

#       Note:
#   Discriminator works in a bit different way that it described in the paper 
#   simply because it shows better performance in my experiments.

#       To do:
#   Find a way how to split sets into train and test ones
#   Define metric and make results of metric_test store in history


class DiscoGAN(GAN):
    def __init__(self, input_shape, latent_dim = 100, **kwargs):
        super(DiscoGAN, self).__init__(input_shape, latent_dim , **kwargs)
        
    def set_models_params(self):
        if self.optimizer is None: self.optimizer = Adam(0.0002, 0.5, clipnorm = 10)
            
        self.models = ['encoder', 'decoder', 'discriminator']
        self.loss = 'mse'
        self.disc_activation = 'sigmoid'    
        
    def build_graph(self):
        input_img = self.encoder.inputs
        encod_img = self.decoder.inputs
        
        self.combined_A = Model(input_img, self.decoder(self.encoder(input_img)))
        self.combined_A.compile(loss='mae', optimizer=self.optimizer)
        
        self.combined_B = Model(encod_img, self.encoder(self.decoder(encod_img)))
        self.combined_B.compile(loss='mae', optimizer=self.optimizer)
        
        
        self.discriminator.compile(loss='mae', optimizer=self.optimizer)
        
        self.discriminator.trainable = False
        self.combined = Model(encod_img + input_img, self.discriminator([self.decoder(encod_img), self.encoder(input_img)]))
        self.combined.compile(loss='mae', optimizer=self.optimizer)

    def prepare_data(self, data_set, validation_split, batch_size):
        super(DiscoGAN, self).prepare_data(data_set, validation_split, batch_size)
        
        self.domain_A_set = self.train_set[0]
        self.domain_B_set = self.train_set[1]
        
        #This values will be used in a way that do no affect the network
        self.dummy = np.zeros((batch_size, 1))
        
    def train_on_batch(self, batch_size):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx_a = np.random.randint(0, self.domain_A_set.shape[0], batch_size)
        idx_b = np.random.randint(0, self.domain_B_set.shape[0], batch_size)
        domain_A_samples = self.domain_A_set[idx_a]
        domain_B_samples = self.domain_B_set[idx_b]
        
        # Generate new images
        gen_b = self.encoder.predict([domain_A_samples])
        gen_a = self.decoder.predict([domain_B_samples])
        
        d_loss_real = self.discriminator.train_on_batch([domain_A_samples, domain_B_samples], self.valid)
        d_loss_fake = self.discriminator.train_on_batch([gen_a, gen_b], self.fake)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        # ---------------------
        #  Train decoder and encoder
        # ---------------------
        self.combined.train_on_batch([domain_B_samples, domain_A_samples], self.valid)

        d_loss = 0
        a_loss = self.combined_A.train_on_batch([domain_A_samples], [domain_A_samples])
        b_loss = self.combined_B.train_on_batch([domain_B_samples], [domain_B_samples])
        g_loss = (a_loss + b_loss) / 2
        
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
        
        metric = self.m_loss    
        return {'metric': metric, 'gen_val': gen_val, 'train_val': train_val, 'test_val': test_val}