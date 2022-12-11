import logging
import sys

get_trace = getattr(sys, 'gettrace', None)
debug = False
if get_trace():
    print('Program runs in Debug mode')
    debug = True


def get_logger(name: str):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(stream_handler)
    return logger
