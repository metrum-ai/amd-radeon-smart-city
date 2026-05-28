# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Application entry point.

Run with::

    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

Or for development::

    uvicorn main:app --reload
"""

import logging
import os

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from smart_city.api.app import create_app  # noqa: E402

import yaml  # noqa: E402
from smart_city.core.config import load_config  # noqa: E402

_config = None
try:
    _config = load_config()
except (OSError, ValueError, TypeError, ImportError, yaml.YAMLError) as _exc:
    logging.getLogger(__name__).warning(
        "Could not load config (%s); starting without config.",
        _exc,
        exc_info=True,
    )

app = create_app(config=_config)
