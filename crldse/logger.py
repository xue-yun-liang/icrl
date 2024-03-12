import logging

class logger:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._logger = logging.getLogger(__name__)
            cls._instance.setup_logger()
        return cls._instance
    
    def setup_logger(self, log_file='test_log.log'):
        """a setup func for crldse's basic debug module"""
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def get_logger(self):
        """a get func for crldse's basic debug module.
        
        Paramters
            None
        Return
            python logging func
        
        """
        return self._logger

logger = Logger()
