#!/bin/bash
cd "$(dirname "$0")"
PYTHONPATH=. celery -A evil_celery worker --loglevel=info
