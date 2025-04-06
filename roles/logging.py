import os
import logging

def get_logger(
    log_dir: str,
    device_id: str, 
    role: str
):
    # Create a custom logger
    logger = logging.getLogger("myapp")
    logger.setLevel(logging.DEBUG)  # Set global logging level

    # Formatter for both file and console
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(f"{log_dir}/{device_id}_{role}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # You can use a different level if desired
    console_handler.setFormatter(formatter)

    # Attach handlers
    if not logger.handlers:  # Avoid duplicate handlers on rerun
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger