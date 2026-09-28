# Structured logging: what production actually looks at 

# Print statements disappear the moment your terminal closes. In production, logs are how you debug a pipeline that ran unattended at 3am and failed.

import logging
import sys
import json
from datetime import datetime, timezone


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure logging once at pipeline startup.
    Every module imports logging.getLogger(__name__) and inherits this config.
    """
    logger = logging.getLogger("churn_pipeline")
    logger.setLevel(getattr(logging, level))
    
    if logger.handlers:
        # Guard against duplicate handles if this is called twice
        # (happens easily in notebooks or whem a module is re-imported)
        return logger
    
        
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logging()

# Called ONCE at import time. Every other module does:
# logger = logging.getLogger("churn_pipeline")
# and automatically gets this exact configuration — no re-setup needed.
