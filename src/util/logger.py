import logging
from colorlog import ColoredFormatter

_formatter = ColoredFormatter(
    "%(asctime)s %(log_color)s%(levelname)-8s%(reset)s [%(name)s]%(reset)s %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }
)


def get_logger(module_name, log_level=logging.INFO):
    """
    Get a logger for this module.
    :param module_name: the module name
    :param log_level: messages less severe than this level get omitted
    :return: the logger
    """
    logger = logging.getLogger(module_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_formatter)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


if __name__ == '__main__':
    log = get_logger('main', log_level=logging.DEBUG)

    log.debug('debug text')
    log.info('info text')
    log.warning('warning text')
    log.error('error text')
    log.critical('critical text')
