from keras.layers import Input, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam
import os
import numpy as np

import keras.backend as K

from . import metrics
from . import utils
from .GAN import GAN

#       Wasserstein GAN with Gradient Penalty
#   Paper: https://arxiv.org/pdf/1704.00028.pdf 

#       Description:
#   More stable approach to Wasserstein GAN training that get rid of 
#   weights clipping that necessary for usual Wasserstein GAN

class WGAN_GP(GAN):
    def __init__(self, input_shape, latent_dim = 100, **kwargs):
        super(WGAN_GP, self).__init__(input_shape, latent_dim , **kwargs)
        
    def set_models_params(self):
        if self.optimizer is None: self.optimizer = Adam(0.0002, 0.5, 0.9)
            
        self.loss = utils.ident_loss
        self.disc_activation = 'linear'    
        
    def build_graph(self):
        lambda_scale = 10
    
        G = self.generator
        D = self.discriminator
        
        real_img = Input(shape=self.input_shape)
        noise = Input(shape=(self.latent_dim,))
        genr_img = G([noise])
        
        
        
        def norm(x, axis):
            return K.sqrt(K.sum(K.square(x), axis=axis))
            
        #compute gradient penalty with respect to weighted average between real and generated images    
        def f_ddx(genr, real):
            epsilon = K.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * real + (1 - epsilon) * genr
            
            ddx = K.gradients(D(x_hat), x_hat)[0]
            ddx = norm(ddx, axis=1)
            ddx = K.mean(K.square(ddx - 1.0) * lambda_scale)
            
            return ddx
            
        #-------------------------------
        # Graph for Generator
        #-------------------------------
        
        self.discriminator.trainable = False
        self.generator.trainable = True 
        
        #Compute the generator loss:
        L_G_tns = Lambda(lambda x: K.mean(-x[0]))([D(genr_img)])
        
        self.genr_model = Model([noise], L_G_tns)
        self.genr_model.compile(loss=self.loss, optimizer=self.optimizer)
        
        
        #-------------------------------
        # Graph for the Discriminator
        #-------------------------------
        
        self.discriminator.trainable = True
        self.generator.trainable = False 
        
        #Compute the critic loss:
        L_D_tns = Lambda(lambda x: K.mean(x[0] - x[1] + f_ddx(x[2], x[3])))([D(genr_img), D(real_img), genr_img, real_img])
        
        self.disc_model = Model([real_img, noise], L_D_tns)
        self.disc_model.compile(loss=self.loss, optimizer=self.optimizer)
        
    def get_data(self, train_set, batch_size):
        # Select a random batch of images
        idx = np.random.randint(0, train_set.shape[0], batch_size)
        imgs = train_set[idx]
        
        # Generate noise for two branch of generated images
        noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
            
        return imgs, noise  
        
    def prepare_data(self, data_set, validation_split, batch_size):
        super(WGAN_GP, self).prepare_data(data_set, validation_split, batch_size)
        
        #This values will be used in a way that do no affect the network
        self.dummy = np.zeros((batch_size, 1))
        
    def train_on_batch(self, batch_size):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        d_iters = 3
        d_loss = 0
        for _ in range(0, d_iters):
            imgs, noise = self.get_data(self.train_set, batch_size)
            d_loss += self.disc_model.train_on_batch([imgs, noise], self.dummy) / d_iters
        
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        imgs, noise = self.get_data(self.train_set, batch_size)
        g_loss = self.genr_model.train_on_batch([noise], self.dummy)
        
        return d_loss, g_loss