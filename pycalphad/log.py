"""
The log module handles setup for logging errors, debug messages and warnings.
"""

#pylint: disable=C0103
import logging
logger = logging.getLogger('pycalphad')
_h = logging.StreamHandler()
fmt = '%(name)s %(levelname)s %(asctime)s [%(funcName)s %(lineno)d] %(message)s'
_f = logging.Formatter(fmt)
_h.setFormatter(_f)
logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False

def debug_mode():
    "Set logger level to log debug messages."
    logger.setLevel(logging.DEBUG)
