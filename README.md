# GANLib

An attempt of creating easy-to-use GAN library on top of keras. Created rather for educational purposes and for personal usage.


Also going to dedicate this project as part of [#100DaysofMLCode](https://github.com/llSourcell/100_Days_of_ML_Code) challenge to get some additional motivation. Here will be the log of the progress: [100DMLCLog.md](100DMLCLog.md)

#### Installing:
WARNING! Keep in mind this is a very early stage of the project. Many things might not work as expected!
```sh
git clone --recursive https://github.com/Mylittlerapture/GANLib GANLib
cd GANLib
pip3 install -e .
```


#### Example:
Import all necessary modules
```python
from GANLib import GAN

from keras.layers import Input, Dense, Reshape, Flatten
from keras.models import Model
```

Specify models for Generator and Discriminator
```python
def build_generator(self):
    input_lat = Input(shape=(self.latent_dim,))
    
    layer = Dense(256, activation = 'relu')(input_lat)
    layer = Dense(784, activation = 'linear')(layer)
    img = Reshape((28,28,1))(layer)

    return Model(input_lat, img)
    
def build_discriminator(self):
    input_img = Input(shape=self.input_shape)
    
    layer = Flatten()(input_img)
    layer = Dense(256, activation = 'relu')(layer)
    layer = Dense(128, activation = 'relu')(layer)
    valid = Dense(1, activation='sigmoid')(layer)
    
    return Model(input_img, valid)  
```

Build GAN and train it on your data 
```python
gan = GAN(data_shape, noise_dim) #define type of Generative model
gan.build_generator = lambda self=gan: build_generator(self) #define generator build function
gan.build_discriminator = lambda self=gan: build_discriminator(self) #define discriminator build function
gan.build_models() #build all necessery models
gan.train(data, epochs=20000, batch_size=64, validation_split = 0.1) #train GAN for 20000 iterations
```

Generate new samples
```python
noise = np.random.uniform(-1, 1, (gen_batch_size, gan.latent_dim))
gen_imgs = gan.generator.predict([noise])
```


#### Current progress for each module:
GAN: completed  
CGAN: completed  
ProgGAN: completed
DiscoGAN: works as supposed to, but do not store all necessary history  
AAE: in progress
