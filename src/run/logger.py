import logging

def create_logger(
        log_file: str, 
        console_log: bool = True,
        logger_name: str = "Experiment"
        )-> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.hadlers = []

    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if console_log:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    logger.info(f"Logger intialized. Writing to {log_file}")
    return logger



def create_toy_logger(logger_name: str = "ToyLogger") -> logging.Logger:
    """Create a simple logger that prints logs to the console in a formatted style."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    # Example usage
    logger = create_toy_logger()
    logger.info("This is an info log message.")
    logger.debug("This is a debug message (won't show at INFO level).")
    logger.warning("This is a warning message.")
