import logging
from typing import Any, Iterable, Literal

logger = logging.getLogger(__name__)

LevelType = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']



class IdentityChain:
    """
    A wrapper chain that logs the input and returns it as output.
    """
    def __init__(self, 
        logger=logger, 
        level= 'DEBUG'
        ):
        self.logger = logger
        self.level = level
    
    def __call__(self, *args) -> Any | Iterable:
        if self.level == 'DEBUG':
            self.logger.debug(f"IdentityChain input: {args}")
        elif self.level == 'INFO':
            self.logger.info(f"IdentityChain input: {args}")
        elif self.level == 'WARNING':
            self.logger.warning(f"IdentityChain input: {args}")
        elif self.level == 'ERROR':
            self.logger.error(f"IdentityChain input: {args}")
        return args
    


__all__ = ["IdentityChain"]