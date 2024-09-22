import logging
import os

os.makedirs('logs', exist_ok=True)

def setup_logger():
    logger = logging.getLogger("DocumentProcessingApp")
    logger.setLevel(logging.DEBUG)  

 
    if not logger.handlers:

        fh = logging.FileHandler('logs/app.log', encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


logger = setup_logger()
