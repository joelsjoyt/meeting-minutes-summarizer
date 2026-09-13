""""
This file acts as the entrypoint for the app
"""

import logging
from util.utils import setup_logger, download_required_models
from views.app_view import run_app_ui


def main():
    root = logging.getLogger()
    setup_logger(root)
    
    logger = logging.getLogger(__name__)
    logger.info("Application has started")

    if download_required_models():
        run_app_ui()
    
if __name__ == "__main__":
    main()
    





    
