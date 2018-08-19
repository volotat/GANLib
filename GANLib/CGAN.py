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
    def metric_test(self, set_data, set_labels, pred_num = 32):    
        met_arr = np.zeros(pred_num)
        
        n_indx = np.random.choice(set_data.shape[0],pred_num)
        labels = set_labels[n_indx]
        org_set = set_data[n_indx]
        
        noise = np.random.uniform(-1, 1, (pred_num, self.latent_dim))
        gen_set = self.generator.predict([noise,labels]) 
        met_arr = metrics.magic_distance(org_set, gen_set)
        return met_arr

    def __init__(self, input_shapes, latent_dim = 100, **kwargs):
        super(CGAN, self).__init__(input_shapes[0], latent_dim , **kwargs)
        self.label_shape = input_shapes[1]
        
    def set_models_params(self):
        if self.optimizer is None: self.optimizer = Adam(0.0002, 0.5, 0.9)
            
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
     
    def prepare_data(self, data_set, validation_split, batch_size):
        if 0. < validation_split < 1.:
            split_at = int(data_set[0].shape[0] * (1. - validation_split))
            self.train_set_data = data_set[0][:split_at]
            self.valid_set_data = data_set[0][split_at:]
            
            self.train_set_labels = data_set[1][:split_at]
            self.valid_set_labels = data_set[1][split_at:]
        else:
            self.train_set_data = data_set[0]
            self.train_set_labels = data_set[1]
            self.valid_set_data = None
            self.valid_set_labels = None
    
        #collect statistical info of data
        self.data_set_std = np.std(data_set[0],axis = 0)
        self.data_set_mean = np.mean(data_set[0],axis = 0)
        
        self.label_set_std = np.std(data_set[1],axis = 0)
        self.label_set_mean = np.mean(data_set[1],axis = 0)
    
        # Adversarial ground truths
        out_shape = self.discriminator.output_shape
        self.valid = np.ones((batch_size,) + out_shape[1:])
        self.fake = np.zeros((batch_size,) + out_shape[1:])
        #self.valid = np.ones((batch_size, 1))
        #self.fake = np.zeros((batch_size, 1))
     
    def train_on_batch(self, batch_size):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, self.train_set_data.shape[0], batch_size)
        imgs, labels = self.train_set_data[idx], self.train_set_labels[idx]

        # Sample noise as generator input
        noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))

        # Generate new images
        gen_imgs = self.generator.predict([noise, labels])
        
        d_loss_real = self.discriminator.train_on_batch([imgs,labels], self.valid)
        d_loss_fake = self.discriminator.train_on_batch([gen_imgs,labels], self.fake)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        # Train the generator
        g_loss = self.combined.train_on_batch([noise, labels], self.valid)
        
        return d_loss, g_loss
        
    def test_network(self, batch_size):
        idx = np.random.randint(0, self.train_set_data.shape[0], batch_size)
        imgs, labels = self.train_set_data[idx], self.train_set_labels[idx]
        
        noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
        gen_imgs = self.generator.predict([noise, labels])
        gen_val = self.discriminator.predict([gen_imgs, labels])
        train_val = self.discriminator.predict([imgs, labels])
        
        if self.valid_set_data is not None and self.valid_set_labels is not None: 
            idx = np.random.randint(0, self.valid_set_data.shape[0], batch_size)
            test_val = self.discriminator.predict([self.valid_set_data[idx], self.valid_set_labels[idx]])
        else:
            test_val = np.zeros(batch_size)
        
        noise_as_data = np.random.normal(self.data_set_mean, self.data_set_std, (batch_size,)+ self.input_shape)
        #noise_as_labels = np.random.normal(self.label_set_mean, self.label_set_std, (batch_size,)+ self.label_shape)
        data_cont_val  = self.discriminator.predict([noise_as_data, labels])
        #label_cont_val = self.discriminator.predict([imgs, noise_as_labels])
        cont_val = data_cont_val
        
        metric = self.metric_test(self.train_set_data, self.train_set_labels, batch_size)   
        
        return gen_val, train_val, test_val, cont_val, metric