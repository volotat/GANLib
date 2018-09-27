# GANLib
Tensorflow based GANs library

### Dependencies
Tensorflow 1.10.1  
Numpy 1.14.2  
Matplotlib 2.1.1  

### Examples
Simple GAN: [simple_gan.py](https://github.com/Mylittlerapture/GANLib/blob/master/examples/simple_gan.py)  
Progressive Growing of GANs: [pg_gan.py](https://github.com/Mylittlerapture/GANLib/blob/master/examples/pg_gan.py)  
And slightly faster version of it: [pg_gan_fast.py](https://github.com/Mylittlerapture/GANLib/blob/master/examples/pg_gan_fast.py)

### Available GANs
Vanilla GAN. https://arxiv.org/pdf/1406.2661.pdf  
Conditional GAN. https://arxiv.org/pdf/1411.1784.pdf  
Disco GAN. https://arxiv.org/pdf/1703.05192.pdf  
Adversarial Autoencoder. https://arxiv.org/pdf/1511.05644.pdf

### Optimization Distances
Minmax (Original GAN optimization distance). https://arxiv.org/pdf/1406.2661.pdf  
Crossentropy (Same as original but written in terms of cross entropy cost function)  
Wasserstein (Earth mover's distance applyed to GAN) https://arxiv.org/pdf/1701.07875.pdf  
Wasserstein GP (Improved version of Wasserstein distance) https://arxiv.org/pdf/1704.00028.pdf  
Cramer (Most advanced, but difficult to calculate distance) https://openreview.net/pdf?id=S1m6h21Cb  !!! discriminator output layer should have more that one neuron, and bigger the number the better !!!  
