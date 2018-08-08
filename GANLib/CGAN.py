from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import os
import numpy as np


from . import metrics
from . import utils
from .GAN import GAN

#                   Conditional Generative Adversarial Network
#   Paper: https://arxiv.org/pdf/1411.1784.pdf

#       Description:
#   Takes as input dataset with it class labels and learn to generate samples 
#   similar to original dataset specified by some given labels.

class CGAN(GAN):
    def __init__(self, input_shape, label_shape, latent_dim = 100, **kwargs):
        super(CGAN, self).__init__(input_shape, latent_dim , **kwargs)
        self.label_shape = label_shape
        
    def set_models_params(self, optimizer):
        if optimizer is None:   
            self.optimizer = Adam(0.0002, 0.5, 0.9)
        else:
            self.optimizer = optimizer
            
        self.loss = 'binary_crossentropy'
        self.disc_activation = 'sigmoid'

    def build_graph(self):
        self.discriminator.compile(loss=self.loss, optimizer=self.optimizer)
        
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=self.label_shape)
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=self.loss, optimizer=self.optimizer)
        
    def train_on_batch(self, train_set, batch_size):
        train_set_data = train_set[0]
        train_set_labels = train_set[1]
    
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, train_set_data.shape[0], batch_size)
        imgs, labels = train_set_data[idx], train_set_labels[idx]

        # Sample noise as generator input
        noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))

        # Generate new images
        gen_imgs = self.generator.predict([noise, labels])
        
        d_loss_real = self.discriminator.train_on_batch([imgs,labels], valid)
        d_loss_fake = self.discriminator.train_on_batch([gen_imgs,labels], fake)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        # Train the generator
        g_loss = self.combined.train_on_batch([noise, labels], valid)
        
        return d_loss, g_loss