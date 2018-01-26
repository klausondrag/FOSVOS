from itertools import product

from util.logger import get_logger

log = get_logger(__file__)

_n_offline_optimizer = 3
_n_online_optimizer = 4

# (offline_optimizer, online_optimizer)
variants = list(product(range(_n_offline_optimizer), range(_n_online_optimizer)))
log.info(variants)
