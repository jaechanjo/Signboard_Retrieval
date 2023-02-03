# -*- coding: utf-8 -*-
import cv2
from AnalysisEngine.celerys import app
from celery.signals import worker_init, worker_process_init
from billiard import current_process
from utils import logging


@worker_init.connect
def model_load_info(**__):
    print(logging.i("===================="))
    print(logging.s("Worker Analyzer Initialize"))
    print(logging.s("===================="))

@worker_process_init.connect
def module_load_init(**__):
    global model

    # TODO:
    #   - Add your model
    #   - You can use worker_index if you need to get and set gpu_id
    #       - ex) gpu_id = worker_index % TOTAL_GPU_NUMBER
    from main import SuperGlue
    model = SuperGlue()


@app.task
def analyzer_by_image(db_path, query_path, db_det, query_det):
    result = model.inference(db_path, query_path, db_det, query_det)
    return result


# For development version
print(logging.i("===================="))
print(logging.s("Development"))
print(logging.s("===================="))
module_load_init()
