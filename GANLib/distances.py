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

def minmax(logit_real, logit_fake, real, fake, G, D): #losses from the original paper: https://arxiv.org/pdf/1406.2661.pdf
    eps = 1e-7
    disc_real = tf.maximum(tf.nn.sigmoid(logit_real), eps)
    disc_fake = tf.maximum(tf.nn.sigmoid(logit_fake), eps)
    
    disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1 - disc_fake))
    genr_loss = tf.reduce_mean(1 - tf.log(disc_fake))
    return disc_loss, genr_loss
          
def cross_entropy(logit_real, logit_fake, real, fake, G, D): #practically the same losses as original but written in a different way
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_real, labels=tf.ones_like(logit_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_fake, labels=tf.zeros_like(logit_fake)))
    disc_loss = (d_loss_real + d_loss_fake) / 2.
    
    genr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_fake, labels=tf.ones_like(logit_fake)))
    return disc_loss, genr_loss
           
def wasserstein_gp(logit_real, logit_fake, real, fake, G, D): #losses from "Improved Training of Wasserstein GANs" :https://arxiv.org/pdf/1704.00028.pdf
    gp = gradient_penalty(real, fake, D)
    
    disc_loss = tf.reduce_mean(logit_fake) - tf.reduce_mean(logit_real) + gp
    genr_loss = -tf.reduce_mean(logit_fake)  
      
    return disc_loss, genr_loss   
