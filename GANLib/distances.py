import numpy as np
import tensorflow as tf


def gradient_penalty(real, fake, discriminator, lambda_scale = 10.):
    shape = tf.concat((tf.shape(real)[0:1], tf.tile([1], [real.shape.ndims - 1])), axis=0)
    epsilon = tf.random_uniform(shape, minval=0., maxval=1.)
    X_hat = real + epsilon * (fake - real)
    X_hat.set_shape(real.get_shape().as_list())
    
    D_X_hat = discriminator(X_hat)
    grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
    red_idx = np.arange(1, X_hat.shape.ndims)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    
    return gradient_penalty * lambda_scale    

    
# ---------------
# Optimization Distances
# ---------------     

class distance(object):
    def __init__(self, optimizer = None, logits = [None, None], examples = [None, None], models = [None, None], vars = [None, None]):
        self.optimizer = optimizer
        
        self.logit_real = logits[0]
        self.logit_fake = logits[1]
        
        self.real = examples[0]
        self.fake = examples[1]
        
        self.G = models[0]
        self.D = models[1]
        
        self.genr_vars = vars[0]
        self.disc_vars = vars[1]
    
    def get_train_sessions(self):
        pass
    
    def get_losses(self):
        pass


# losses from the original paper: https://arxiv.org/pdf/1406.2661.pdf        
class minmax(distance):
    def __init__(self, **kwargs):
        super(minmax, self).__init__(**kwargs)
        
        eps = 1e-7
        disc_real = tf.maximum(tf.nn.sigmoid(self.logit_real), eps)
        disc_fake = tf.maximum(tf.nn.sigmoid(self.logit_fake), eps)
        
        self.disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1 - disc_fake))
        self.genr_loss = tf.reduce_mean(1 - tf.log(disc_fake))
        
        self.train_genr = self.optimizer.minimize(self.genr_loss, var_list=self.genr_vars) 
        self.train_disc = self.optimizer.minimize(self.disc_loss, var_list=self.disc_vars)

    def get_train_sessions(self):
        return self.train_genr, self.train_disc
    
    def get_losses(self):
        return self.genr_loss, self.disc_loss

# practically the same losses as original but written in a different way
class cross_entropy(distance):
    def __init__(self, **kwargs):
        super(cross_entropy, self).__init__(**kwargs)        
        
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_real, labels=tf.ones_like(self.logit_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.zeros_like(self.logit_fake)))
        self.disc_loss = (d_loss_real + d_loss_fake) / 2.
        
        self.genr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.ones_like(vlogit_fake)))
        
        self.train_genr = self.optimizer.minimize(self.genr_loss, var_list=self.genr_vars) 
        self.train_disc = self.optimizer.minimize(self.disc_loss, var_list=self.disc_vars)
        
    def get_train_sessions(self):
        return self.train_genr, self.train_disc
    
    def get_losses(self):
        return self.genr_loss, self.disc_loss
        
        
# losses from "Wasserstein GAN" :https://arxiv.org/pdf/1701.07875.pdf        
class wasserstein(distance):
    def __init__(self, **kwargs):
        super(wasserstein, self).__init__(**kwargs)
        
        self.disc_loss = tf.reduce_mean(self.logit_fake) - tf.reduce_mean(self.logit_real)
        self.genr_loss = -tf.reduce_mean(self.logit_fake)  
        
        self.train_genr = self.optimizer.minimize(self.genr_loss, var_list=self.genr_vars) 
        self.train_disc = self.optimizer.minimize(self.disc_loss, var_list=self.disc_vars)
        
        self.disc_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.disc_vars]
    
    def get_train_sessions(self):
        return self.train_genr, tf.group(self.disc_clip, self.train_disc)
    
    def get_losses(self):
        return self.genr_loss, self.disc_loss

# losses from "Improved Training of Wasserstein GANs" :https://arxiv.org/pdf/1704.00028.pdf
class wasserstein_gp(distance):
    def __init__(self, **kwargs):
        super(wasserstein_gp, self).__init__(**kwargs)
        
        gp = gradient_penalty(self.real, self.fake, self.D)
    
        self.disc_loss = tf.reduce_mean(self.logit_fake) - tf.reduce_mean(self.logit_real) + gp
        self.genr_loss = -tf.reduce_mean(self.logit_fake) 
        
        self.train_genr = self.optimizer.minimize(self.genr_loss, var_list=self.genr_vars) 
        self.train_disc = self.optimizer.minimize(self.disc_loss, var_list=self.disc_vars)
    
    def get_train_sessions(self):
        return self.train_genr, self.train_disc
    
    def get_losses(self):
        return self.genr_loss, self.disc_loss

# not yet        
class cramer(distance):
    def __init__(self, **kwargs):
        super(cramer, self).__init__(**kwargs)        
        
        
    def get_train_sessions(self):
        return self.train_genr, self.train_disc
    
    def get_losses(self):
        return self.genr_loss, self.disc_loss        
