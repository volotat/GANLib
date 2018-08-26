from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import os
import numpy as np

from .. import metrics
from .. import utils

#                   Generative Adversarial Network
#   Paper: https://arxiv.org/pdf/1406.2661.pdf

#       Description:
#   Takes as input some dataset and by adversarial training two different 
#   networks (generator and discriminator) learn to generate samples 
#   that very similar to given dataset from random noise.

#       To do:
#   Rewrite history collector in a way that each module might use its own collection of values to store

class GAN(object):
    def metric_test(self, set, pred_num = 32):    
        met_arr = np.zeros(pred_num)
        
        n_indx = np.random.choice(set.shape[0],pred_num)
        org_set = set[n_indx]
        
        noise = np.random.uniform(-1, 1, (pred_num, self.latent_dim))
        gen_set = self.generator.predict([noise]) 
        met_arr = metrics.magic_distance(org_set, gen_set)
        return met_arr

    def __init__(self, input_shape, latent_dim = 100, optimizer = None):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        self.best_model = None
        self.best_metric = 0
        
        self.history = None
        
        self.epoch = utils.tensor_value(0)
        self.epochs = utils.tensor_value(0)
        
        self.optimizer = optimizer
        self.set_models_params()
        
        
    def set_models_params(self):
        if self.optimizer is None: self.optimizer = Adam(0.0002, 0.5)
        
        self.models = ['generator', 'discriminator']
        self.loss = 'binary_crossentropy'
        self.disc_activation = 'sigmoid'
        
    def build_graph(self):
        self.discriminator.compile(loss=self.loss, optimizer=self.optimizer)
    
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        img = self.generator([noise])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise], valid)
        self.combined.compile(loss=self.loss, optimizer=self.optimizer)
        
        self.discriminator.trainable = True
        
        
    def prepare_data(self, data_set, validation_split, batch_size):
        if 0. < validation_split < 1.:
            split_at = int(data_set.shape[0] * (1. - validation_split))
            self.train_set = data_set[:split_at]
            self.valid_set = data_set[split_at:]
        else:
            self.train_set = data_set
            self.valid_set = None
    
        # Adversarial ground truths
        if hasattr(self, 'discriminator'):
            #out_shape = self.discriminator.output_shape
            #self.valid = np.ones((batch_size,) + out_shape[1:])
            #self.fake = np.zeros((batch_size,) + out_shape[1:])
            self.valid = np.ones((batch_size, 1))
            self.fake = np.zeros((batch_size, 1))
        
        
    def train_on_batch(self, batch_size):
        # Select a random batch of images
        idx = np.random.randint(0, self.train_set.shape[0], batch_size)
        imgs = self.train_set[idx]
    
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample noise as generator input
        noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))

        # Generate new images
        gen_imgs = self.generator.predict([noise])
        
        d_loss_real = self.discriminator.train_on_batch(imgs, self.valid)
        d_loss_fake = self.discriminator.train_on_batch(gen_imgs, self.fake)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        # Train the generator
        g_loss = self.combined.train_on_batch([noise], self.valid)
        
        return d_loss, g_loss
        
        
    def build_models(self, files = None, custom_objects = None):
        for model in self.models:
            if not hasattr(self, model): raise Exception("%s are not defined!"%(model))
            
        self.build_graph()
      
      
    def test_network(self, batch_size):
        noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
        gen_imgs = self.generator.predict([noise])
        gen_val = self.discriminator.predict([gen_imgs])
        
        idx = np.random.randint(0, self.train_set.shape[0], batch_size)
        imgs = self.train_set[idx]
        train_val = self.discriminator.predict([imgs])
        
        if self.valid_set is not None: 
            idx = np.random.randint(0, self.valid_set.shape[0], batch_size)
            test_val = self.discriminator.predict(self.valid_set[idx])
        else:
            test_val = np.zeros(batch_size)
        
        metric = self.metric_test(self.train_set, batch_size)    
        return {'metric': metric, 'gen_val': gen_val, 'train_val': train_val, 'test_val': test_val}
        
    
    def train(self, data_set, batch_size=32, epochs=1, verbose=True, checkpoint_range = 100, checkpoint_callback = None, validation_split = 0, save_best_model = False, collect_history = True):
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
            collect_history:
                Boolean. If True, all training history will store into 'history' object. Sometimes it might be computationally expensive.
        # Returns
            A history object. 
        """ 

        #mean min max
        max_hist_size = epochs//checkpoint_range + 1
        history = { 'best_metric':0,
                    'hist_size'  :0}
                    
        self.epoch.set(0)
        self.epochs.set(epochs)
        
        # Build Network
       
        self.prepare_data(data_set, validation_split, batch_size)
        self.build_models()
        
        # Train Network
        for epoch in range(epochs):
            self.epoch.set(epoch)
            
            d_loss, g_loss = self.train_on_batch(batch_size)

            # Save history
            if epoch % checkpoint_range == 0:
                if not collect_history:
                    if verbose: print('%d [D loss: %f] [G loss: %f]' % (epoch, d_loss, g_loss))
                else:
                    dict_of_vals = self.test_network(128)
                    dict_of_vals['D loss'] = d_loss
                    dict_of_vals['G loss'] = g_loss
                    
                    hist_size = history['hist_size'] = history['hist_size']+1
                    metric = np.mean(dict_of_vals['metric'])
                    
                    for k, v in dict_of_vals.items():
                        if k not in history:
                            history[k] = np.zeros((max_hist_size,3))
                        
                        history[k][hist_size-1] = np.mean(v),  np.min(v),  np.max(v)
                    
                    if verbose: print ("%d [D loss: %f] [G loss: %f] [%s: %f]" % (epoch, d_loss, g_loss, 'metric', metric))
                    
                    if metric*0.98 < self.best_metric or self.best_model == None:
                        #self.best_model = self.generator.get_weights()
                        self.best_metric = metric
                        history['best_metric'] = self.best_metric
                        
                    self.history = history
                
                if checkpoint_callback is not None:
                    checkpoint_callback()
        
        if save_best_model:
            self.generator.set_weights(self.best_model)    
            
        self.epoch.set(epochs)
        checkpoint_callback()   
        
        return self.history   

    def save_history_to_image(file):
        utils.save_hist_image(self.history, file, graphs = (['metric'], ['D loss', 'G loss']), scales = ('log', 'linear'))
        
   