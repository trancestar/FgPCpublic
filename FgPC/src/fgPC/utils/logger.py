import logging
import subprocess

def init_logger(save_dir, logfile, log_level=logging.INFO):

    log_file = save_dir + "/" + logfile + ".log"

    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

    logger.info(f"-------------------------------")
    logger.info(f"--------Start logging ---------")
    logger.info(f"-------------------------------\n")
    logger.info(f"Current git hash: {commit_hash}")

    return logger