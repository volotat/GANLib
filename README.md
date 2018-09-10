# GANLib

An attempt of creating easy-to-use GAN library on top of keras. Created rather for educational purposes and for personal usage. 
Right now I'm in process of switching to tensorflow backend on separate branch. As long as all functionality of main branch will be recreated, tensorflow backend will become a main one.


Also going to dedicate this project as part of [#100DaysofMLCode](https://github.com/llSourcell/100_Days_of_ML_Code) challenge to get some additional motivation. Here will be the log of the progress: [100DMLCLog.md](100DMLCLog.md)

### Dependencies
Keras 2.2.0  
Numpy 1.14.2  
Matplotlib 2.1.1  

### Installing
WARNING! Keep in mind this is a very early stage of the project. Many things might not work as expected! (That said, feel free to open issue, ask a question or make any suggestions.)
```sh
git clone --recursive https://github.com/Mylittlerapture/GANLib GANLib
cd GANLib
pip3 install -e .
```


### Examples
Simple GAN: [simple_gan.py](https://github.com/Mylittlerapture/GANLib/blob/master/examples/simple_gan.py)  
Progressive Growing of GANs: [pg_gan.py](https://github.com/Mylittlerapture/GANLib/blob/master/examples/pg_gan.py)


### Current progress for each module
AAE: completed  
CGAN: completed  
CramerGAN: completed  
GAN: completed  
WGAP GP: completed  
DiscoGAN: completed 

