# -*- coding: utf-8 -*-
import os

from kombu import Queue
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AnalysisEngine.settings')

BROKER_URL = 'redis://redis:6379/0'
CELERY_RESULT_BACKEND = 'redis://redis:6379/0'

app = Celery('WorkerEngine', broker=BROKER_URL, backend=CELERY_RESULT_BACKEND)

app.conf.update(
    accept_content=["json", "pickle"],
    task_serializer="json",
    result_serializer="pickle"
)
app.autodiscover_tasks()
app.conf.task_queues = (
    Queue('django', 'WebAnalyzer', routing_key='analysis_tasks'),
)
app.conf.timezone = 'Asia/Seoul'

app.conf.beat_schedule = {
    'delete-old-database': {
        'task': 'WebAnalyzer.beats.delete_old_database',
        'schedule': 86400,
        'args': (1),
    },
}