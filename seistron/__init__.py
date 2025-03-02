"""Top-level package for seistron."""

from importlib.metadata import version
__version__ = version("seistron")

__author__ = """Earl Patrick Bellinger"""
__email__ = 'earl.bellinger@yale.edu'

from .data import *
from .emulate import *
from .hpc import *
from .models import *
from .sample import *
from .visualize import *
from .utils import *
