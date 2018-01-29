from itertools import product

from util.logger import get_logger

log = get_logger(__file__)

_n_offline_optimizer_1 = 3
_n_online_optimizer_1 = 4
variants = list(product(range(_n_offline_optimizer_1),
                        range(_n_online_optimizer_1)))

_n_offline_optimizer_2 = 10
variants += list(product(range(_n_offline_optimizer_1, _n_offline_optimizer_2),
                         range(_n_online_optimizer_1)))

_n_online_optimizer_2 = 7
variants += list(product(range(_n_offline_optimizer_2),
                         range(_n_online_optimizer_1, _n_online_optimizer_2)))

log.info(list(enumerate(variants)))
