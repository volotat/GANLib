from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import os
import numpy as np


from . import metrics
from . import utils

#                   Conditional Generative Adversarial Network
#   Paper: https://arxiv.org/pdf/1411.1784.pdf

#       Description:
#   Takes as input dataset with it class labels and learn to generate samples 
#   similar to original dataset specified by some given labels.

class CGAN():
    def metric_test(self, set_data, set_labels, pred_num = 32):    
        met_arr = np.zeros(pred_num)
        
        n_indx = np.random.choice(set_data.shape[0],pred_num)
        labels = set_labels[n_indx]
        org_set = set_data[n_indx]
        
        noise = np.random.uniform(-1, 1, (pred_num, self.latent_dim))
        gen_set = self.generator.predict([noise,labels]) 
        met_arr = metrics.magic_distance(org_set, gen_set)
        return met_arr

    def __init__(self, input_shape, label_shape, latent_dim = 100, mode = 'vanilla'):
        # Input shape
        self.input_shape = input_shape
        self.label_shape = label_shape
        self.latent_dim = latent_dim
        self.mode = mode
        
        self.build_discriminator = None
        self.build_generator = None
        
        self.best_model = None
        self.best_metric = 0
        
        self.epoch = 0
        self.history = None
        
    def build_models(self, optimizer = None, path = ''):
        if optimizer is None:
            optimizer = Adam(0.0002, 0.5)
            
        if self.mode == 'stable':
            loss = 'logcosh'
            self.disc_activation = 'linear'
        elif self.mode == 'vanilla':
            loss = 'binary_crossentropy'
            self.disc_activation = 'sigmoid'
        else: raise Exception("Mode '" + self.mode+ "' is unknown")
    
        if os.path.isfile(path+'/generator.h5') and os.path.isfile(path+'/discriminator.h5'):
            self.generator = load_model(path+'/generator.h5')
            self.discriminator = load_model(path+'/discriminator.h5')
        else:
            if self.build_discriminator is None or self.build_generator is None:
                raise Exception("Model building functions are not defined")
            else:
                # Build and compile the discriminator
                self.discriminator = self.build_discriminator()
                self.discriminator.compile(loss=loss, optimizer=optimizer)

                # Build the generator
                self.generator = self.build_generator()

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
        self.combined.compile(loss=loss, optimizer=optimizer)
            
        print('models builded')   
            
    def save(self):
        self.generator.save('generator.h5')
        self.discriminator.save('discriminator.h5')
      
    def train(self, data_set, label_set, batch_size=32, epochs=1, verbose=1, checkpoint_range = 100, checkpoint_callback = None, validation_split = 0, save_best_model = False):
        """Trains the model for a given number of epochs (iterations on a dataset).
        # Arguments
            data_set: 
                Numpy array of training data.
            data_labels: 
                Numpy array of labels assigned with data_set   
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
            train_set_data = data_set[:split_at]
            valid_set_data = data_set[split_at:]
            
            train_set_labels = label_set[:split_at]
            valid_set_labels = label_set[split_at:]
        else:
            train_set_data = data_set
            train_set_labels = label_set
            valid_set_data = None
            valid_set_labels = None
    
        #collect statistical info of data
        data_set_std = np.std(data_set,axis = 0)
        data_set_mean = np.mean(data_set,axis = 0)
        
        label_set_std = np.std(label_set,axis = 0)
        label_set_mean = np.mean(label_set,axis = 0)
    
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        #mean min max
        max_hist_size = epochs//checkpoint_range + 1
        history = { 'gen_val'           :np.zeros((max_hist_size,3)), 
                    'train_val'         :np.zeros((max_hist_size,3)), 
                    'test_val'          :np.zeros((max_hist_size,3)), 
                    'data_control_val'  :np.zeros((max_hist_size,3)), 
                    'label_control_val' :np.zeros((max_hist_size,3)), 
                    'metric'            :np.zeros((max_hist_size,3)),
                    'best_metric'       :0,
                    'hist_size'         :0}
        
        for epoch in range(epochs):
            self.epoch = epoch
            
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
            
            if self.mode == 'stable':
                trash_imgs = imgs.copy()
                trash_labels = labels.copy()
                trash_imgs[:batch_size//2] = np.random.normal(data_set_mean, data_set_std, (batch_size//2,) + self.input_shape)
                trash_labels[batch_size//2:] = np.random.normal(label_set_mean, label_set_std, (batch_size//2,) + self.label_shape)
            
                #trash_imgs = np.random.normal(train_set_mean, train_set_std, (batch_size,) + self.input_shape)

                # Validate how good generated images looks like
                val = self.discriminator.predict([gen_imgs,labels])
                crit = utils.Gravity(val, boundaries = [-1,1])
                
                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([imgs,labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs,labels], crit)
                d_loss_trsh = self.discriminator.train_on_batch([trash_imgs, trash_labels], -valid)
                d_loss = (d_loss_real + d_loss_fake + d_loss_trsh) / 3
            elif self.mode == 'vanilla':
                d_loss_real = self.discriminator.train_on_batch([imgs,labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs,labels], fake)
                d_loss = (d_loss_real + d_loss_fake) / 2
                
            else: raise Exception("Mode '" + self.mode+ "' is unknown")
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Train the generator
            g_loss = self.combined.train_on_batch([noise, labels], valid)

            # Plot the progress
            if epoch % checkpoint_range == 0:
                gen_val = self.discriminator.predict([gen_imgs, labels])
                train_val = self.discriminator.predict([imgs, labels])
                
                if valid_set_data is not None and valid_set_labels is not None: 
                    idx = np.random.randint(0, valid_set_data.shape[0], batch_size)
                    test_val = self.discriminator.predict([valid_set_data[idx], valid_set_labels[idx]])
                else:
                    test_val = np.zeros(batch_size)
                
                noise_as_data = np.random.normal(data_set_mean, data_set_std, (batch_size,)+ self.input_shape)
                noise_as_labels = np.random.normal(label_set_mean, label_set_std, (batch_size,)+ self.label_shape)
                data_cont_val  = self.discriminator.predict([noise_as_data, labels])
                label_cont_val = self.discriminator.predict([imgs, noise_as_labels])
                
                metric = self.metric_test(train_set_data, train_set_labels, 1000)
                print ("%d [D loss: %f] [G loss: %f] [validations TRN: %f, TST: %f] [metric: %f]" % (epoch, d_loss, g_loss, np.mean(train_val), np.mean(test_val), np.mean(metric)))
                
                hist_size = history['hist_size'] = history['hist_size']+1
                history['gen_val']    [hist_size-1] = np.mean(gen_val),  np.min(gen_val),  np.max(gen_val)
                history['train_val']  [hist_size-1] = np.mean(train_val),np.min(train_val),np.max(train_val)
                history['test_val']   [hist_size-1] = np.mean(test_val), np.min(test_val), np.max(test_val)
                history['data_control_val'] [hist_size-1] = np.mean(data_cont_val), np.min(data_cont_val), np.max(data_cont_val) 
                history['label_control_val'][hist_size-1] = np.mean(label_cont_val), np.min(label_cont_val), np.max(label_cont_val) 
                history['metric']     [hist_size-1] = np.mean(metric),   np.min(metric),   np.max(metric)
                
                if np.mean(metric)*0.98 < self.best_metric or self.best_model == None:
                    self.best_model = self.generator.get_weights()
                    self.best_metric = np.mean(metric)
                    history['best_metric'] = self.best_metric
                    
                self.history = history
                
                if checkpoint_callback is not None:
                    checkpoint_callback()
        
        
        
        if save_best_model:
            self.generator.set_weights(self.best_model)    
            
        self.epoch = epochs
        checkpoint_callback()   
        
        return self.history   