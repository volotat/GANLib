from __future__ import absolute_import


from . import distances
from . import metrics
from . import utils

from .GANs.GAN import GAN #main class
from .GANs.AAR import AAE
from .GANs.CGAN import CGAN
from .GANs.DiscoGAN import DiscoGAN

__version__ = '0.0.6'