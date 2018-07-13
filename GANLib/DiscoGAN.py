from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import os
import numpy as np


from . import metrics

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


class DiscoGAN():
    '''
    def metric_test(self, set, pred_num = 32):    
        met_arr = np.zeros(pred_num)
        
        n_indx = np.random.choice(set.shape[0],pred_num)
        org_set = set[n_indx]
        
        noise = np.random.uniform(-1, 1, (pred_num, self.latent_dim))
        gen_set = self.encoder.predict([noise]) 
        met_arr = metrics.magic_distance(org_set, gen_set)
        return met_arr
    '''
    def __init__(self, a_set_shape, b_set_shape):
        self.a_set_shape = a_set_shape
        self.b_set_shape = b_set_shape
        
        self.build_discriminator = None
        self.build_encoder = None
        self.build_decoder = None
        
        self.best_model = None
        self.best_metric = 0
        
        self.epoch = 0
        self.history = None
     
    
    def build_models(self, optimizer = None, path = ''):
        if optimizer is None:
            optimizer = Adam(0.0002, 0.5)
        
        if os.path.isfile(path+'/encoder.h5') and os.path.isfile(path+'/discriminator.h5'):
            pass
            #self.encoder = load_model(path+'/encoder.h5')
            #self.discriminator = load_model(path+'/discriminator.h5')
        else:
            if self.build_discriminator is None or self.build_encoder is None or self.build_decoder is None:
                raise Exception("Model building functions are not defined")
            else:
                # Build and compile the discriminator
                self.discriminator = self.build_discriminator()
                self.discriminator.compile(loss='mae', optimizer=optimizer)

                # Build the encoder
                self.encoder = self.build_encoder()
                self.decoder = self.build_decoder()
        
        # The encoder takes noise and the target label as input
        # and generates the corresponding digit of that label
        input_img = Input(shape=self.a_set_shape)
        encod_img = Input(shape=self.b_set_shape)
        
        self.combined_A = Model(input_img, self.decoder(self.encoder(input_img)))
        self.combined_A.compile(loss='mae', optimizer=optimizer)
        
        self.combined_B = Model(encod_img, self.encoder(self.decoder(encod_img)))
        self.combined_B.compile(loss='mae', optimizer=optimizer)
        
        self.discriminator.trainable = False
        self.combined = Model([encod_img, input_img], self.discriminator([self.decoder(encod_img), self.encoder(input_img)]))
        self.combined.compile(loss='mae', optimizer=optimizer)
        print('models builded')    
            
    def save(self):
        self.encoder.save('encoder.h5')
        self.discriminator.save('discriminator.h5')
    
    def train(self, domain_A_set, domain_B_set , batch_size=32, epochs=1, verbose=1, checkpoint_range = 100, checkpoint_callback = None, validation_split = 0, save_best_model = False):
        """Trains the model for a given number of epochs (iterations on a dataset).
        # Arguments
            domain_A_set, domain_B_set: 
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
                Boolean. If True, encoder weights will be resigned to best model according to chosen metric.
        # Returns
            A history object. 
        """ 
    
        '''
        if 0. < validation_split < 1.:
            split_at = int(data_set.shape[0] * (1. - validation_split))
            train_set = data_set[:split_at]
            valid_set = data_set[split_at:]
        else:
            train_set = data_set
            valid_set = None
        '''
        #collect statistical info of data
        a_set_std = np.std(domain_A_set,axis = 0)
        a_set_mean = np.mean(domain_A_set,axis = 0)
        
        b_set_std = np.std(domain_B_set,axis = 0)
        b_set_mean = np.mean(domain_B_set,axis = 0)
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        #mean min max
        max_hist_size = epochs//checkpoint_range + 1
        history = { 'gen_val'    :np.zeros((max_hist_size,3)), 
                    'train_val'  :np.zeros((max_hist_size,3)), 
                    #'test_val'   :np.zeros((max_hist_size,3)), 
                    'control_val':np.zeros((max_hist_size,3)), 
                    'metric'     :np.zeros((max_hist_size,3)),
                    'best_metric':0,
                    'hist_size'  :0}
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx_a = np.random.randint(0, domain_A_set.shape[0], batch_size)
            idx_b = np.random.randint(0, domain_B_set.shape[0], batch_size)
            domain_A_samples = domain_A_set[idx_a]
            domain_B_samples = domain_B_set[idx_b]
            
            # Generate new images
            gen_b = self.encoder.predict([domain_A_samples])
            gen_a = self.decoder.predict([domain_B_samples])
            
            d_loss_fake = self.discriminator.train_on_batch([gen_a, gen_b], fake)
            d_loss_real = self.discriminator.train_on_batch([domain_A_samples, domain_B_samples], valid)
            d_loss = (d_loss_real + d_loss_fake) / 2
            
            # ---------------------
            #  Train decoder and encoder
            # ---------------------
            self.combined.train_on_batch([domain_B_samples, domain_A_samples], valid)
            
            a_loss = self.combined_A.train_on_batch([domain_A_samples], [domain_A_samples])
            b_loss = self.combined_B.train_on_batch([domain_B_samples], [domain_B_samples])
            g_loss = (a_loss + b_loss) / 2
            
            # Plot the progress
            if epoch % checkpoint_range == 0:
                gen_val = self.discriminator.predict([gen_a, gen_b])
                train_val = self.discriminator.predict([domain_A_samples, domain_B_samples])
                
                a_like_noise = np.random.normal(a_set_std, a_set_std, (batch_size,)+ self.a_set_shape)
                b_like_noise = np.random.normal(b_set_std, b_set_std, (batch_size,)+ self.b_set_shape)
                cont_val = self.discriminator.predict([a_like_noise, b_like_noise])
                
                '''
                if valid_set is not None: 
                    idx = np.random.randint(0, valid_set.shape[0], batch_size)
                    test_val = self.discriminator.predict(valid_set[idx])
                else:
                    test_val = np.zeros(batch_size)
                '''
                test_val = np.zeros(1)
                metric = np.zeros(1) #self.metric_test(train_set, 1000)
                print ("%d [D loss: %f] [G loss: %f] [validations TRN: %f, TST: %f] [metric: %f]" % (epoch, d_loss, g_loss, np.mean(train_val), np.mean(test_val), np.mean(metric))) 
                
                hist_size = history['hist_size'] = history['hist_size']+1
                history['gen_val']    [hist_size-1] = np.mean(gen_val),  np.min(gen_val),  np.max(gen_val)
                history['train_val']  [hist_size-1] = np.mean(train_val),np.min(train_val),np.max(train_val)
                #history['test_val']   [hist_size-1] = np.mean(test_val), np.min(test_val), np.max(test_val)
                history['control_val'][hist_size-1] = np.mean(cont_val), np.min(cont_val), np.max(cont_val) 
                #history['metric']     [hist_size-1] = np.mean(metric),   np.min(metric),   np.max(metric)
                
                if np.mean(metric)*0.98 < self.best_metric or self.best_model == None:
                    self.best_model = self.encoder.get_weights()
                    self.best_metric = np.mean(metric)
                    history['best_metric'] = self.best_metric
                    
                self.history = history
                
                if checkpoint_callback is not None:
                    checkpoint_callback()
        
        
        
        if save_best_model:
            self.encoder.set_weights(self.best_model)    
            
        self.epoch = epochs
        checkpoint_callback()   
        
        return self.history    