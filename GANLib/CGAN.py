from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import os
import numpy as np


from . import metrics


class CGAN():
    def metric_test(self, X_train, Y_train, pred_num = 32):    
        met_arr = np.zeros(pred_num)
        
        n_indx = np.random.choice(X_train.shape[0],pred_num)
        input = Y_train[n_indx]
        output = X_train[n_indx]
        
        noise = np.random.uniform(-1, 1, (pred_num, self.latent_dim))
        new_arr = self.generator.predict([noise,input]) 
        met_arr = self.metric(output, new_arr)
        return met_arr

    def __init__(self, input_shape, label_shape, latent_dim = 100):
        # Input shape
        self.input_shape = input_shape
        self.label_shape = label_shape
        #self.num_classes = 10
        self.input_dim = np.prod(input_shape)
        self.label_dim = np.prod(label_shape)
        self.latent_dim = latent_dim

        self.optimizer = Adam(0.0002, 0.5)
        
        self.best_model = None
        self.best_metric = 0
        
        self.epoch = 0
        self.history = None
        
    def build_models(self):
        if os.path.isfile('generator.h5') and os.path.isfile('discriminator.h5'):
            self.generator = load_model('generator.h5')
            self.discriminator = load_model('discriminator.h5')
        else:
            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss=['binary_crossentropy'],
                optimizer=self.optimizer,
                metrics=['accuracy'])

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
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=self.optimizer)
            
        print('models builded')    
            
    def save(self):
        self.generator.save('generator.h5')
        self.discriminator.save('discriminator.h5')
      
    def train(self, X_train, y_train, epochs, batch_size=128, sample_interval=100, validation_split = 500, checkpoint = lambda:True):
        checkpoint_range = 100
        c_r = checkpoint_range
    
        #collect statistical info of data
        X_train_std = np.std(X_train,axis = 0)
        X_train_mean = np.mean(X_train,axis = 0)
        
        y_train_std = np.std(y_train,axis = 0)
        y_train_mean = np.mean(y_train,axis = 0)
    
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        history = { 'gen_val':np.zeros(epochs//c_r+1), 
                    'train_val':np.zeros((epochs//c_r+1,batch_size)), 
                    'test_val':np.zeros((epochs//c_r+1,batch_size)), 
                    'control_val_x':np.zeros(epochs//c_r+1), 
                    'control_val_y':np.zeros(epochs//c_r+1), 
                    'metric':np.zeros((epochs//c_r+1,1000))}
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(validation_split, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            #noise = np.random.normal(0, 0.3, (batch_size, self.latent_dim))
            noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            val = self.discriminator.predict([gen_imgs, labels])
            crit = 1. - np.abs(1. - val) ** 0.5
            
            trash_batch_0 = imgs.copy()
            trash_batch_1 = labels.copy()
            trash_batch_0[:batch_size//2] = np.random.normal(X_train_mean, X_train_std, (batch_size//2,) + self.input_shape)
            trash_batch_1[batch_size//2:] = np.random.normal(y_train_mean, y_train_std, (batch_size//2,) + self.label_shape)
            
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], crit)
            d_loss_trsh = self.discriminator.train_on_batch([trash_batch_0, trash_batch_1], fake)
            d_loss = np.add(d_loss_real, d_loss_fake) / 2
            
            #clip_value = 10
            #for l in self.discriminator.layers:
            #    weights = l.get_weights()
            #    weights = [np.clip(w, -clip_value, clip_value) for w in weights]
            #    l.set_weights(weights)
            
            
            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            idx = np.random.randint(validation_split, X_train.shape[0], batch_size)
            sampled_labels = y_train[idx] #np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
            #if epoch%16 == 0:
            #    g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
            #else:
            #    g_loss = 0

            # Plot the progress
            if epoch % c_r == 0:
                gen_val = np.mean(self.discriminator.predict([gen_imgs, labels]))
                
                idx = np.random.randint(validation_split, X_train.shape[0], batch_size)
                train_val = self.discriminator.predict([X_train[idx], y_train[idx]])
                
                idx = np.random.randint(0, validation_split, batch_size)
                test_val = self.discriminator.predict([X_train[idx], y_train[idx]])
                
                idx = np.random.randint(validation_split, X_train.shape[0], batch_size)
                noise = np.random.normal(y_train_mean, y_train_std, (batch_size,)+ self.label_shape)
                cont_val_x = np.mean(self.discriminator.predict([X_train[idx],noise])) #green
                
                
                idx = np.random.randint(validation_split, y_train.shape[0], batch_size)
                noise = np.random.normal(X_train_mean, X_train_std, (batch_size,)+ self.input_shape)
                cont_val_y = np.mean(self.discriminator.predict([noise, y_train[idx]])) #red
                
                metric = self.metric_test(X_train, y_train, 1000)
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [validations TRN: %f, TST: %f] [metric: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss, np.mean(train_val), np.mean(test_val), np.mean(metric)))
                
                history['gen_val'][epoch//c_r] = gen_val
                history['train_val'][epoch//c_r] = train_val.reshape(batch_size)
                history['test_val'][epoch//c_r] = test_val.reshape(batch_size)
                history['control_val_x'][epoch//c_r] = cont_val_x
                history['control_val_y'][epoch//c_r] = cont_val_y
                history['metric'][epoch//c_r] = metric
                
                if np.mean(metric)*0.98 < self.best_metric or self.best_model == None:
                    self.best_model = self.generator.get_weights()
                    self.best_metric = np.mean(metric)
            
                self.history = history
                
                checkpoint()
            # If at save interval => save generated image samples
            #if epoch % sample_interval == 0:
            #    self.sample_images(epoch)
            #    self.save_hist_image(history,epoch)
                
        #self.generator.set_weights(self.best_model)    
        #self.sample_images(epochs)
        #return history    