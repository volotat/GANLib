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

class DiscoGAN(GAN):
    def __init__(self, sess, input_shapes, latent_dim = 100, **kwargs):
        super(DiscoGAN, self).__init__(sess, input_shapes, latent_dim , **kwargs)
        self.input_shape_a = input_shapes[0]
        self.input_shape_b = input_shapes[1]
        
    def set_models_params(self):
        if self.optimizer is None: self.optimizer = tf.train.AdamOptimizer(0.0002, 0.5, epsilon = 1e-07)
        if self.distance is None: self.distance = distances.minmax
            
        self.models = ['encoder', 'decoder', 'discriminator_a', 'discriminator_b']

    def build_graph(self):
        
        def ENC(x):
            with tf.variable_scope('ENC', reuse=tf.AUTO_REUSE) as scope:
                res = self.encoder(x)
            return res
            
        def DEC(x):
            with tf.variable_scope('DEC', reuse=tf.AUTO_REUSE) as scope:
                res = self.decoder(x)
            return res
            
        def Da(x):
            with tf.variable_scope('Da', reuse=tf.AUTO_REUSE) as scope:
                logits = self.discriminator_a(x)
            return logits
            
        def Db(x):
            with tf.variable_scope('Db', reuse=tf.AUTO_REUSE) as scope:
                logits = self.discriminator_b(x)
            return logits
        
        self.enc_input = tf.placeholder(tf.float32, shape=(None,) + self.input_shape_a)
        self.dec_input = tf.placeholder(tf.float32, shape=(None,) + self.input_shape_b)
        
        self.disc_a_input = tf.placeholder(tf.float32, shape=(None,) + self.input_shape_a)
        self.disc_b_input = tf.placeholder(tf.float32, shape=(None,) + self.input_shape_b)
        
        self.t_encode_a = ENC(self.enc_input)
        self.t_encode_b = DEC(self.dec_input)
        
        encoder_loss = tf.reduce_mean(tf.squared_difference(self.enc_input, DEC(ENC(self.enc_input))))
        decoder_loss = tf.reduce_mean(tf.squared_difference(self.dec_input, ENC(DEC(self.dec_input))))
        
        self.autoencode_loss = 0.5 * (encoder_loss + decoder_loss)
        self.autoencode_vars = tf.trainable_variables('ENC') + tf.trainable_variables('DEC')
        self.autoencode_train = self.optimizer.minimize(self.autoencode_loss, var_list=self.autoencode_vars)
        

        # Domain a GAN
        genr = ENC(self.enc_input)
        logit_real = Db(self.disc_b_input)
        logit_fake = Db(genr)
        
        real = self.disc_b_input
        fake = genr
        
        dist_a = self.distance(
            optimizer = self.optimizer, 
            logits = [logit_real, logit_fake], 
            examples = [real, fake], 
            models = [ENC, Db],
            inputs = [self.enc_input, self.disc_b_input],
            vars = [tf.trainable_variables('ENC'), tf.trainable_variables('Db')],
            gan = self
            )
            
        self.train_genr_a, self.train_disc_a = dist_a.get_train_sessions() 
        self.genr_loss_a, self.disc_loss_a = dist_a.get_losses()
        
        
        # Domain b GAN
        genr = DEC(self.dec_input)
        logit_real = Da(self.disc_a_input)
        logit_fake = Da(genr)
        
        real = self.disc_a_input
        fake = genr
        
        dist_b = self.distance(
            optimizer = self.optimizer, 
            logits = [logit_real, logit_fake], 
            examples = [real, fake], 
            models = [DEC, Da],
            inputs = [self.dec_input, self.disc_a_input],
            vars = [tf.trainable_variables('DEC'), tf.trainable_variables('Da')],
            gan = self
            )
            
        self.train_genr_b, self.train_disc_b = dist_b.get_train_sessions() 
        self.genr_loss_b, self.disc_loss_b = dist_b.get_losses()
        
        
        self.genr_loss, self.disc_loss = 0.5 * (self.genr_loss_a + self.genr_loss_b), 0.5 *(self.disc_loss_a + self.disc_loss_b) 
        
        self.sess.run(tf.global_variables_initializer())
    
    def prepare_data(self, data_set, validation_split, batch_size):
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
        # ----------------------
        # Train GAN part
        # ----------------------
        
        for j in range(self.n_critic):
            # Select a random batch of images
            idx_a = np.random.randint(0, self.domain_A_set.shape[0], batch_size)
            idx_b = np.random.randint(0, self.domain_B_set.shape[0], batch_size)
            domain_A_samples = self.domain_A_set[idx_a]
            domain_B_samples = self.domain_B_set[idx_b]
        
            feed_dict={self.disc_b_input: domain_B_samples, self.enc_input: domain_A_samples,
                       self.disc_a_input: domain_A_samples, self.dec_input: domain_B_samples}
                       
            self.sess.run([self.train_disc_a, self.train_disc_b], feed_dict=feed_dict)
            
        self.sess.run([self.train_genr_a, self.train_genr_b], feed_dict=feed_dict)
        
        d_loss, g_loss = self.sess.run([self.disc_loss, self.genr_loss], feed_dict=feed_dict)
        
        
        # ----------------------
        # Train autoencoder part
        # ----------------------
        
        # Select a random batch of images
        idx_a = np.random.randint(0, self.domain_A_set.shape[0], batch_size)
        idx_b = np.random.randint(0, self.domain_B_set.shape[0], batch_size)
        domain_A_samples = self.domain_A_set[idx_a]
        domain_B_samples = self.domain_B_set[idx_b]
        
        self.sess.run(self.autoencode_train, feed_dict={self.enc_input: domain_A_samples, self.dec_input: domain_B_samples}) 
        self.m_loss = self.sess.run(self.autoencode_loss, feed_dict={self.enc_input: domain_A_samples, self.dec_input: domain_B_samples})
        
        return d_loss, g_loss
        
    def test_network(self, batch_size):
        metric = self.m_loss   
        
        return {'metric': metric}