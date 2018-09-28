from __future__ import absolute_import


from . import distances
from . import metrics
from . import utils

from .GANs.GAN import GAN #main class
from .GANs.AAE import AAE
from .GANs.CGAN import CGAN
from .GANs.DiscoGAN import DiscoGAN
<<<<<<< HEAD
from .GANs.GAN import GAN
from .GANs.WGAN_GP import WGAN_GP
=======
>>>>>>> Tensorflow

__version__ = '0.0.6'