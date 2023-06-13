import os
import logging
from logging import handlers


def build_logger(log_name, log_path, enable_outfile=True, level=logging.INFO, when='D', back_count=0):
    """
    :param log_name: Log name
    :param level: Log level
    :param when: Flip interval (S,M,H,D,W,midnight)
    :param back_count:  The number of backup files. If it exceeds this value, it will be deleted automatically
    :return: logger
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_file_path = os.path.join(log_path, log_name)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s: %(message)s')

    # output to console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if enable_outfile:
        # output to file
        fh = logging.handlers.TimedRotatingFileHandler(
            filename=log_file_path,
            when=when,
            backupCount=back_count,
            encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
