from __future__ import absolute_import


from . import metrics
from . import plotter
from . import utils

from .GANs.AAE import AAE
from .GANs.CGAN import CGAN
from .GANs.CramerGAN import CramerGAN
from .GANs.DiscoGAN import DiscoGAN
from .GANs.GAN import GAN
from .GANs.WGAN_GP import WGAN_GP

from .GANs.XGAN import XGAN

__version__ = '0.0.5'