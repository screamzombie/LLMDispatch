#!/bin/bash
cd "$(dirname "$0")"
conda activate EvilMagic
# 确保 PYTHONPATH 是项目根目录
PYTHONPATH=. celery -A evil_celery worker --loglevel=info