import tensorflow as tf
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

class DiscoGAN(GAN):
    def __init__(self, input_shapes, latent_dim = 100, **kwargs):
        super(DiscoGAN, self).__init__(input_shapes, latent_dim , **kwargs)
        self.input_shape_a = input_shapes[0]
        self.input_shape_b = input_shapes[1]
        
    def set_models_params(self):
        if self.optimizer is None: self.optimizer = tf.train.AdamOptimizer(0.001, 0.5, epsilon = 1e-07)
        if self.distance is None: self.distance = distances.minmax
            
        self.models = ['encoder', 'decoder', 'discriminator']

    def build_graph(self):
        
        def ENC(x):
            with tf.variable_scope('ENC', reuse=tf.AUTO_REUSE) as scope:
                res = self.encoder(x)
            return res
            
        def DEC(x):
            with tf.variable_scope('DEC', reuse=tf.AUTO_REUSE) as scope:
                res = self.decoder(x)
            return res
            
        def D(x):
            with tf.variable_scope('D', reuse=tf.AUTO_REUSE) as scope:
                logits = self.discriminator(x)
            return logits
        
        self.enc_input = tf.placeholder(tf.float32, shape=(None,) + self.input_shape_a)
        self.dec_input = tf.placeholder(tf.float32, shape=(None,) + self.input_shape_b)
        
        #self.disc_input_a = tf.placeholder(tf.float32, shape=(None,) + self.input_shape_a)
        #self.disc_input_b = tf.placeholder(tf.float32, shape=(None,) + self.input_shape_b)
        
        self.t_encode_a = ENC(self.enc_input)
        self.t_encode_b = DEC(self.dec_input)
        
        encoder_loss = tf.reduce_mean(tf.squared_difference(self.enc_input, DEC(ENC(self.enc_input))))
        decoder_loss = tf.reduce_mean(tf.squared_difference(self.dec_input, ENC(DEC(self.dec_input))))
        
        self.autoencode_loss = 0.5 * (encoder_loss + decoder_loss)
        self.autoencode_vars = tf.trainable_variables('ENC') + tf.trainable_variables('DEC')
        #print(self.autoencode_vars)
        #exit()
        self.autoencode_train = self.optimizer.minimize(self.autoencode_loss, var_list=self.autoencode_vars)
        
        
        '''
        self.genr = G(self.genr_input, self.genr_label)
        logit_real = D(self.disc_input, self.disc_label)
        logit_fake = D(self.genr, self.genr_label)
        
        real = self.disc_input
        fake = self.genr
        
        dist = self.distance(
            optimizer = self.optimizer, 
            logits = [logit_real, logit_fake], 
            examples = [real, fake], 
            models = [G, D],
            vars = [tf.trainable_variables('G'), tf.trainable_variables('D')],
            gan = self
            )
            
        self.train_genr, self.train_disc = dist.get_train_sessions() 
        self.genr_loss, self.disc_loss = dist.get_losses()
        '''
        self.sess.run(tf.global_variables_initializer())
    
    def prepare_data(self, data_set, validation_split, batch_size):
        '''
        super(DiscoGAN, self).prepare_data(data_set, validation_split, batch_size)
        '''
        self.domain_A_set = data_set[0]
        self.domain_B_set = data_set[1]
        
     
    def encode_a(self, data_domain_a):  
        imgs = self.sess.run(self.t_encode_a, feed_dict = {self.enc_input: data_domain_a})
        return imgs 
        
    def encode_b(self, data_domain_b):  
        imgs = self.sess.run(self.t_encode_b, feed_dict = {self.dec_input: data_domain_b})
        return imgs 
        
    def decode_a(self, data_encoded_a):  
        imgs = self.sess.run(self.t_encode_b, feed_dict = {self.enc_input: data_encoded_a})
        return imgs 
        
    def decode_b(self, data_encoded_b):  
        imgs = self.sess.run(self.t_encode_a, feed_dict = {self.dec_input: data_encoded_b})
        return imgs 
     
    def train_on_batch(self, batch_size):
        '''
        for j in range(self.n_critic):
            # Select a random batch of images
            idx = np.random.randint(0, self.train_set_data.shape[0], batch_size)
            imgs = self.train_set_data  [idx]
            lbls = self.train_set_labels[idx]
        
            # Sample noise as generator input
            noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
            self.sess.run(self.train_disc, feed_dict={self.disc_input: imgs, self.disc_label: lbls, self.genr_label: lbls, self.genr_input: noise})
            
        noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
        self.sess.run([self.train_genr], feed_dict={self.disc_input: imgs, self.disc_label: lbls, self.genr_label: lbls, self.genr_input: noise})
        
        d_loss, g_loss = self.sess.run([self.disc_loss, self.genr_loss], feed_dict={self.disc_input: imgs, self.disc_label: lbls, self.genr_label: lbls, self.genr_input: noise})
        '''
        
        # Select a random batch of images
        idx_a = np.random.randint(0, self.domain_A_set.shape[0], batch_size)
        idx_b = np.random.randint(0, self.domain_B_set.shape[0], batch_size)
        domain_A_samples = self.domain_A_set[idx_a]
        domain_B_samples = self.domain_B_set[idx_b]
        
        self.sess.run(self.autoencode_train, feed_dict={self.enc_input: domain_A_samples, self.dec_input: domain_B_samples}) 
        self.m_loss = self.sess.run(self.autoencode_loss, feed_dict={self.enc_input: domain_A_samples, self.dec_input: domain_B_samples})
        
        d_loss = 0
        g_loss = 0
        return d_loss, g_loss
        
    def test_network(self, batch_size):
        metric = self.m_loss   
        
        return {'metric': metric}