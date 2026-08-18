import logging
import os
from datetime import datetime

def setup_logging(log_file=None, log_level=logging.INFO, console=True, reset_file=True):

    log_dir = os.path.dirname(log_file) if log_file else 'logs'
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    if reset_file and os.path.exists(log_file):
        try:
            os.remove(log_file)
        except FileNotFoundError:
            pass

    logger = logging.getLogger()
    logger.setLevel(log_level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def get_logger(name=__name__):
    return logging.getLogger(name)
