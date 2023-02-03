#!/usr/bin/env bash
celery -A AnalysisEngine worker -P solo --loglevel=info --concurrency=1
