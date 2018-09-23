import tensorflow as tf
import numpy as np


from .. import metrics
from .. import utils
from .GAN import GAN

#                   Adversarial Autoencoder
#   Paper: https://arxiv.org/pdf/1511.05644.pdf

#       Description:
#   Attach discriminator to autoencoder in oder to make decoder produce 
#   realistic samples from random noise and make encoder generate  more
#   useful latent representation of data. 

class AAE(GAN):
    def __init__(self, sess, input_shape, latent_dim = 100, **kwargs):
        super(AAE, self).__init__(sess, input_shape, latent_dim , **kwargs)
        
    def set_models_params(self):
        if self.optimizer is None: self.optimizer = tf.train.AdamOptimizer(0.0002, 0.5, epsilon = 1e-07)
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
        
        self.enc_input = tf.placeholder(tf.float32, shape=(None,) + self.input_shape)
        self.dec_input = tf.placeholder(tf.float32, shape=(None, self.latent_dim))
        self.disc_input = tf.placeholder(tf.float32, shape=(None, self.latent_dim))
        
        self.autoencode_loss = tf.reduce_mean(tf.squared_difference(self.enc_input, DEC(ENC(self.enc_input))))
        self.autoencode_vars = tf.trainable_variables('ENC') + tf.trainable_variables('DEC')
        self.autoencode_train = self.optimizer.minimize(self.autoencode_loss, var_list=self.autoencode_vars)
        
        self.dec = DEC(self.dec_input)

        # Domain a GAN
        genr = ENC(self.enc_input)
        logit_real = D(self.disc_input)
        logit_fake = D(genr)
        
        real = self.disc_input
        fake = genr
        
        dist_a = self.distance(
            optimizer = self.optimizer, 
            logits = [logit_real, logit_fake], 
            examples = [real, fake], 
            models = [ENC, D],
            inputs = [self.enc_input, self.disc_input],
            vars = [tf.trainable_variables('ENC'), tf.trainable_variables('D')],
            gan = self
            )
            
        self.train_genr, self.train_disc = dist_a.get_train_sessions() 
        self.genr_loss, self.disc_loss = dist_a.get_losses()
                
        self.sess.run(tf.global_variables_initializer())
     
    def predict(self, noise):  
        imgs = self.sess.run(self.dec, feed_dict = {self.dec_input: noise})
        return imgs 
     
    def train_on_batch(self, batch_size):
        for j in range(self.n_critic):
            # Select a random batch of images
            idx = np.random.randint(0, self.train_set.shape[0], batch_size)
            imgs = self.train_set[idx]
        
            # Sample noise as generator input
            noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
            self.sess.run(self.train_disc, feed_dict={self.disc_input: noise, self.enc_input: imgs})
                        
        noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
        self.sess.run([self.train_genr], feed_dict={self.disc_input: noise, self.enc_input: imgs})
        
        d_loss, g_loss = self.sess.run([self.disc_loss, self.genr_loss], feed_dict={self.disc_input: noise, self.enc_input: imgs})
        
        # train Autoencoder
        self.sess.run(self.autoencode_train, feed_dict={self.enc_input: imgs}) 
        self.m_loss = self.sess.run(self.autoencode_loss, feed_dict={self.enc_input: imgs})
        
        return d_loss, g_loss
        
    def test_network(self, batch_size):
        metric = self.m_loss
        
        return {'metric': metric}