from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import Lambda, Reshape, Activation
import os
import numpy as np

import keras.backend as K

from . import metrics
from . import utils
from .GAN import GAN

#                   Cramer GAN
#   Paper: https://arxiv.org/pdf/1705.10743.pdf (the one that implemented)
#   Newer version: https://openreview.net/pdf?id=S1m6h21Cb

#       Description:
#   Cramer distance is a different approach to measure distance between probabilities 
#   that has some advantages over Wasserstein metric.


class CramerGAN(GAN):
    def __init__(self, input_shape, latent_dim = 100, **kwargs):
        super(CramerGAN, self).__init__(input_shape, latent_dim , **kwargs)
        
    def set_models_params(self, optimizer):
        if optimizer is None:   
            self.optimizer = Adam(0.0002, 0.5, 0.9)
        else:
            self.optimizer = optimizer
            
        self.loss = utils.ident_loss
        self.disc_activation = 'linear'    
        
    def build_graph(self):
        lambda_scale = 10
    
        G = self.generator
        D = self.discriminator
        
        real_img = Input(shape=self.input_shape)
        noise_a = Input(shape=(self.latent_dim,))
        noise_b = Input(shape=(self.latent_dim,))
        
        genr_img = G([noise_a])
        hxr =  D([real_img])
        hxga = D(G([noise_a]))
        hxgb = D(G([noise_b]))
        
        def norm(x, axis):
            return K.sqrt(K.sum(K.square(x), axis=axis)) #, keepdims=True
            
        #Define the critic:    
        def crit(x, xg_):
            return norm(x - xg_, axis=-1) - norm(x, axis=-1)
        
        #-------------------------------
        # Graph for Generator
        #-------------------------------
        
        self.discriminator.trainable = False
        self.generator.trainable = True 
        
        #Compute the generator loss:
        L_G_tns = Lambda(lambda x: K.mean(norm(x[0] - x[1], axis=-1) + norm(x[0] - x[2], axis=-1) - norm(x[1] - x[2], axis=-1)))([hxr, hxga, hxgb])
        
        self.genr_model = Model([real_img, noise_a, noise_b], L_G_tns)
        self.genr_model.compile(loss=self.loss, optimizer=self.optimizer)
        
        
        #-------------------------------
        # Graph for the Discriminator
        #-------------------------------
        
        self.discriminator.trainable = True
        self.generator.trainable = False
        
        #compute gradient penalty with respect to weighted average between real and generated images
        def f_ddx(real, genr):
            epsilon = K.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * real + (1 - epsilon) * genr
            d_hat = crit(D(x_hat), hxgb)
            
            ddx = K.gradients(d_hat, x_hat)[0]
            ddx = norm(ddx, axis=1)
            ddx = K.mean(K.square(ddx - 1.0) * lambda_scale)
            
            return ddx
        
        #Compute the surrogate generator loss:
        L_S_tns = Lambda(lambda x: K.mean(crit(x[0], x[2]) - crit(x[1], x[2])))([hxr, hxga, hxgb])
        #Compute the critic loss:
        L_D_tns = Lambda(lambda x: (f_ddx(x[0], x[1]) - x[2]))([real_img, genr_img, L_S_tns])
        
        self.disc_model = Model([real_img, noise_a, noise_b], L_D_tns)
        self.disc_model.compile(loss=self.loss, optimizer=self.optimizer)
        
    def get_data(self, train_set, batch_size):
        # Select a random batch of images
        idx = np.random.randint(0, train_set.shape[0], batch_size)
        imgs = train_set[idx]
        
        # Generate noise for two branch of generated images
        noise_a = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
        noise_b = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
            
        return imgs, noise_a, noise_b    
        
    def train_on_batch(self, batch_size):
        # target values do not affect the network, so it does not matter what they are ¯\_(ツ)_/¯ 
        self.dummy = np.zeros((batch_size, 1))
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        d_iters = 3
        d_loss = 0
        for _ in range(0, d_iters):
            imgs, noise_a, noise_b = self.get_data(self.train_set, batch_size)
            d_loss += self.disc_model.train_on_batch([imgs, noise_a, noise_b], self.dummy) / d_iters
        
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        imgs, noise_a, noise_b = self.get_data(self.train_set, batch_size)
        g_loss = self.genr_model.train_on_batch([imgs, noise_a, noise_b], self.dummy)
        
        return d_loss, g_loss