import logging


def get_logger(name: str, debug: bool):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(stream_handler)
    return logger
