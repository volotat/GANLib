# GANLib

An attempt of crating easy-to-use GAN library on top of keras. Created rather for educational purposes and for personal usage.


Also going to dedicate this project as part of [#100DaysofMLCode](https://github.com/llSourcell/100_Days_of_ML_Code) challenge to get some additional motivation. Here will be the log of the progress: [100DMLCLog.md](100DMLCLog.md)



#### Example:
```python
gan = GAN(data_shape, noise_dim) #define type of Generative model
gan.build_generator = lambda self=gan: build_generator(self) #define generator build function
gan.build_discriminator = lambda self=gan: build_discriminator(self) #define discriminator build function
gan.build_models() #build all necessery models
gan.train(data, epochs=20000, batch_size=64, validation_split = 0.1) #train GAN for 20000 iterations
```
