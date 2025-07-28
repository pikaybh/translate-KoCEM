import logging, os
from datetime import datetime

from dotenv import load_dotenv


load_dotenv()
log_dir = os.getenv('LOG_DIR', 'logs')


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Sets up and returns a logger for the project.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    os.makedirs(log_dir, exist_ok=True)
    prefix = f"{name}_" if name else ""
    file_handler = logging.FileHandler(
        os.path.join(log_dir, prefix+'{}.log'.format(datetime.now().strftime('%Y%m%d'))), 
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logging.getLogger().handlers.clear()
    
    return logger


__all__ = ['setup_logger']