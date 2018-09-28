import tensorflow as tf
import numpy as np


from .. import metrics
from .. import utils
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
        gen_set = self.predict(noise,labels) 
        met_arr = metrics.magic_distance(org_set, gen_set)
        return met_arr

    def __init__(self, sess, input_shapes, latent_dim = 100, **kwargs):
        super(CGAN, self).__init__(sess, input_shapes[0], latent_dim , **kwargs)
        self.label_shape = input_shapes[1]
        
    def set_models_params(self):
        if self.optimizer is None: self.optimizer = tf.train.AdamOptimizer(0.001, 0.5, epsilon = 1e-07)
        if self.distance is None: self.distance = distances.minmax
            
        self.models = ['generator', 'discriminator']

    def build_graph(self):
        
        def G(x, l):
            with tf.variable_scope('G', reuse=tf.AUTO_REUSE) as scope:
                res = self.generator(x, l)
            return res
            
        def D(x, l):
            with tf.variable_scope('D', reuse=tf.AUTO_REUSE) as scope:
                logits = self.discriminator(x, l)
            return logits
        
        self.genr_input = tf.placeholder(tf.float32, shape=(None, self.latent_dim))
        self.genr_label = tf.placeholder(tf.float32, shape=(None,) + self.label_shape)
        
        self.disc_input = tf.placeholder(tf.float32, shape=(None,) + self.input_shape)
        self.disc_label = tf.placeholder(tf.float32, shape=(None,) + self.label_shape)
        
        
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
            inputs = [self.genr_input, self.disc_input],
            vars = [tf.trainable_variables('G'), tf.trainable_variables('D')],
            gan = self
            )
            
        self.train_genr, self.train_disc = dist.get_train_sessions() 
        self.genr_loss, self.disc_loss = dist.get_losses()
        
        self.sess.run(tf.global_variables_initializer())
     
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
     
    def predict(self, noise, labels):  
        imgs = self.sess.run(self.genr, feed_dict = {self.genr_input: noise, self.genr_label: labels})
        return imgs 
     
    def train_on_batch(self, batch_size):
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
        return d_loss, g_loss
        
    def test_network(self, batch_size):
        metric = self.metric_test(self.train_set_data, self.train_set_labels, batch_size)   
        
        return {'metric': metric}