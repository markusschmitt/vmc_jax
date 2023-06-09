from . import global_defs
from . import nets
from . import operator
from . import util
from . import mpi_wrapper
from . import vqs
from . import sampler
from . import stats

from .version import __version__
from .global_defs import set_pmap_devices

from jax.config import config
config.update("jax_enable_x64", True)
