from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import os
import numpy as np

from skimage.measure import block_reduce

from functools import partial
import keras.backend as K
from keras.layers.merge import _Merge



from . import metrics
from . import utils

#                   Progressively growing GAN
#   Paper: https://arxiv.org/pdf/1710.10196.pdf

#       Description:
#   Takes as input some dataset and trains the network as usual GAN but progressively 
#   adding layers to generator and discriminator.

#       Notes:
#   Pixelwise normalization, minibatch stddev and other "tricks" from paper realized 
#   outside of the class in utils module. The reason is they are more related to internal 
#   model structure than overall algorithm and GANLib allows specified whatever model you want.

#       To do:
#   In original paper all weights remains trainable, but I need to make this optional
#   Need to restrict data shape to power of 2
#   Make channels on layers became smaller while growing or make it optional
#   Update train comment
#   Need a way to save models and continue training after load
#   Add He's initialization to trainable layers and improved Wasserstein loss to models
 
 
class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        shape = K.shape(inputs[0])
        weights = K.random_uniform((shape[0], 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1]) 
 
class ProgGAN():
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
        
    def metric_test(self, set, pred_num = 32):    
        met_arr = np.zeros(pred_num)
        
        n_indx = np.random.choice(set.shape[0],pred_num)
        org_set = set[n_indx]
        
        noise = np.random.uniform(-1, 1, (pred_num, self.latent_dim))
        gen_set = self.generator.predict([noise]) 
        scale = org_set.shape[1] // gen_set.shape[1]
        gen_set = gen_set.repeat(scale, axis = 1).repeat(scale, axis = 2)
        
        met_arr = metrics.magic_distance(org_set, gen_set)
        return met_arr   

    def __init__(self, input_shape, latent_dim = 100, mode = 'vanilla'):
        self.input_shape = input_shape
        self.channels = input_shape[-1]
        self.latent_dim = latent_dim
        self.mode = mode
        
        self.build_discriminator_layer = None
        self.build_generator_layer = None
        
        self.best_model = None
        self.best_metric = 0
        
        self.epoch = 0
        self.history = None
        
        self.layers = 0
        sz = 2 ** (self.layers + 2)
        self.inp_shape = (sz,sz,self.channels)
        
        self.weights = {}
        self.transition_alpha = utils.tensor_value(0)
        '''
        self.genr_head_weights = None
        self.disc_head_weights = None
        
        self.genr_weights = []
        self.disc_weights = []
        '''
        

    def build_models(self, optimizer = None, path = ''):
        if optimizer is None:
            #optimizer = Adam(0.0002, 0.5) 
            optimizer = Adam(0.0005, beta_1=0.5, beta_2=0.9) #, clipvalue=0.1
            
        if self.mode == 'stable':
            loss = 'logcosh'
            self.disc_activation = 'linear'
        elif self.mode == 'vanilla':
            loss = 'binary_crossentropy'
            #self.disc_activation = 'sigmoid'
            self.disc_activation = 'linear'
        else: raise Exception("Mode '" + self.mode+ "' is unknown")
    
        self.path = path
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

        '''
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
        self.combined.compile(loss=loss, optimizer=optimizer)
        '''

        #-------------------------------
        # Graph for the Discriminator
        #-------------------------------

        # Freeze generator's layers while training discriminator
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.inp_shape)

        # Noise input
        z_disc = Input(shape=(100,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.discriminator(fake_img)
        valid = self.discriminator(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.discriminator(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.discriminator_model = Model(inputs=[real_img, z_disc], outputs=[valid, fake, validity_interpolated])
        self.discriminator_model.compile(loss=['mse',
                                              'mse',
                                              partial_gp_loss],
                                        optimizer=optimizer)
        #-------------------------------
        # Graph for Generator
        #-------------------------------

        # For the generator we freeze the discriminator's layers
        self.discriminator.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.discriminator(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss='mse', optimizer=optimizer)


        
        print('models builded')        
    
    def save(self):
        self.generator.save(self.path+'/generator.h5')
        self.discriminator.save(self.path+'/discriminator.h5')
    
    def train(self, data_set, batch_size_list=[32], epochs_list=[1], verbose=True, checkpoint_range = 100, checkpoint_callback = None, validation_split = 0, save_best_model = False):
        """Trains the model for a given number of epochs (iterations on a dataset).
        # Arguments
            data_set: 
                Numpy array of training data.
            batch_size:
                Number of samples per gradient update.
            epochs: Number of epochs to train the model.
                An epoch is an iteration over batch sized samples of dataset.
            grow_epochs: List of epochs indexes in witch networks will grow 
                new layers if it's possible
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
        
        data_set_org = data_set.copy()
        
        def setup():
            sz = data_set_org.shape[1] // self.inp_shape[0]
            data_set = block_reduce(data_set_org, block_size=(1, sz, sz, 1), func=np.mean)
        
            if 0. < validation_split < 1.:
                split_at = int(data_set.shape[0] * (1. - validation_split))
                train_set = data_set[:split_at]
                valid_set = data_set[split_at:]
            else:
                train_set = data_set
                valid_set = None
        
            #collect statistical info of data
            data_set_std = np.std(data_set,axis = 0)
            data_set_mean = np.mean(data_set,axis = 0)
            
            return train_set, valid_set, data_set_std, data_set_mean
    
        train_set, valid_set, data_set_std, data_set_mean = setup()
    
        

        #mean min max
        tot_num_of_epochs = np.sum(np.array(epochs_list))
        max_hist_size = tot_num_of_epochs//checkpoint_range + 1
        history = { 'gen_val'    :np.zeros((max_hist_size,3)), 
                    'train_val'  :np.zeros((max_hist_size,3)), 
                    'test_val'   :np.zeros((max_hist_size,3)), 
                    'control_val':np.zeros((max_hist_size,3)), 
                    'metric'     :np.zeros((max_hist_size,3)),
                    'best_metric':0,
                    'hist_size'  :0}
        
        pr_w_gen = None
        pr_w_disc = None
                        
        for i in range(len(epochs_list)):
            epochs = epochs_list[i]
            batch_size = batch_size_list[i]
            
            # Adversarial ground truths
            '''
            out_shape = self.discriminator.output_shape
            valid = np.ones((batch_size,) + out_shape[1:])
            fake = np.zeros((batch_size,) + out_shape[1:])
            '''
            
            valid = -np.ones((batch_size, 1))
            fake =  np.ones((batch_size, 1))
            dummy = np.zeros((batch_size, 1))
        
            for epoch in range(epochs):
                self.epoch = epoch
                
                a = self.transition_alpha.get()    
                self.transition_alpha.set(min(epoch / float(epochs//2), 1))   

                # Select a random batch of images
                idx = np.random.randint(0, train_set.shape[0], batch_size)
                imgs = train_set[idx]
                
                '''
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Sample noise as generator input
                noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))

                # Generate new images
                gen_imgs = self.generator.predict([noise])
                
                if self.mode == 'stable':
                    trash_imgs = np.random.normal(data_set_mean, data_set_std, (batch_size,) + self.inp_shape)

                    # Validate how good generated images looks like
                    val = self.discriminator.predict([gen_imgs])
                    crit = utils.Gravity(val, boundaries = [-1,1])
                    
                    # Train the discriminator
                    d_loss_real = self.discriminator.train_on_batch([imgs], valid)
                    d_loss_fake = self.discriminator.train_on_batch([gen_imgs], crit)
                    d_loss_trsh = self.discriminator.train_on_batch([trash_imgs], -valid)
                    d_loss = (d_loss_real + d_loss_fake + d_loss_trsh) / 3
                    
                elif self.mode == 'vanilla':
                    d_loss_real = self.discriminator.train_on_batch(imgs, valid) #valid
                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake) #fake
                    d_loss = (d_loss_real + d_loss_fake) / 2
                    
                else: raise Exception("Mode '" + self.mode+ "' is unknown") 
                
                # ---------------------
                #  Train Generator
                # ---------------------
                
                # Train the generator
                g_loss = self.combined.train_on_batch([noise], valid) #valid
                '''
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                #for _ in range(5):
                # Sample noise as generator input
                noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.discriminator_model.train_on_batch([imgs, noise], [valid, fake, dummy])

                # ---------------------
                #  Train Generator
                # ---------------------

                g_loss = self.generator_model.train_on_batch(noise, valid)
                    
                # Plot the progress
                if epoch % checkpoint_range == 0:
                    print('Epoch: %d, losses: %s:%s' % (epoch, d_loss, g_loss))
                    '''
                    gen_val = self.discriminator.predict([gen_imgs])
                    
                    #idx = np.random.randint(0, train_set.shape[0], batch_size)
                    #train_val = self.discriminator.predict(train_set[idx])
                    train_val = self.discriminator.predict([imgs])
                    
                    if valid_set is not None: 
                        idx = np.random.randint(0, valid_set.shape[0], batch_size)
                        test_val = self.discriminator.predict(valid_set[idx])
                    else:
                        test_val = np.zeros(batch_size)
                    
                    noise = np.random.normal(data_set_mean, data_set_std, (batch_size,)+ self.inp_shape)
                    cont_val = self.discriminator.predict(noise)
                    
                    metric = self.metric_test(data_set, 1000)
                    if verbose:
                        print ("%d [D loss: %f] [G loss: %f] [validations TRN: %f, TST: %f] [metric: %f]" % (epoch, d_loss, g_loss, np.mean(train_val), np.mean(test_val), np.mean(metric)))
                    
                    hist_size = history['hist_size'] = history['hist_size']+1
                    history['gen_val']    [hist_size-1] = np.mean(gen_val),  np.min(gen_val),  np.max(gen_val)
                    history['train_val']  [hist_size-1] = np.mean(train_val),np.min(train_val),np.max(train_val)
                    history['test_val']   [hist_size-1] = np.mean(test_val), np.min(test_val), np.max(test_val)
                    history['control_val'][hist_size-1] = np.mean(cont_val), np.min(cont_val), np.max(cont_val) 
                    history['metric']     [hist_size-1] = np.mean(metric),   np.min(metric),   np.max(metric)
                    
                    if np.mean(metric)*0.98 < self.best_metric or self.best_model == None:
                        self.best_model = self.generator.get_weights()
                        self.best_metric = np.mean(metric)
                        history['best_metric'] = self.best_metric
                        
                    self.history = history
                    '''
                    if checkpoint_callback is not None:
                        checkpoint_callback()
        
        
        
            # ---------------------
            # Grow Network
            # ---------------------
            
            if self.inp_shape != self.input_shape:
                self.transition_alpha.set(0)
                
                for l in self.generator.layers:
                    self.weights[l.name] = l.get_weights()
                    
                for l in self.discriminator.layers:
                    self.weights[l.name] = l.get_weights()   
                
                self.layers += 1
                sz = 2 ** (self.layers + 2)
                self.inp_shape = (sz,sz,self.channels)
                self.build_models()
                
                train_set, valid_set, data_set_std, data_set_mean = setup()
        
        
        if save_best_model:
            self.generator.set_weights(self.best_model)    
            
        self.epoch = epochs
        checkpoint_callback()   
        
        return self.history   