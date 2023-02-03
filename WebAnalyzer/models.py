# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from rest_framework import exceptions

from AnalysisEngine.celerys import app
from WebAnalyzer.utils import filename
from django.db.models import JSONField
import ast
import cv2

class ImageModel(models.Model):
    db_image = models.ImageField(upload_to=filename.default)
    query_image = models.ImageField(upload_to=filename.default)
    db_json = models.JSONField()
    query_json = models.JSONField()
    token = models.AutoField(primary_key=True)
    uploaded_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    result = JSONField(null=True)

    def save(self, *args, **kwargs):
        super(ImageModel, self).save(*args, **kwargs)
        try:
            task = app.send_task(
                name='WebAnalyzer.tasks.analyzer_by_image',
                args=[self.db_image.path, self.query_image.path, self.db_json, self.query_json],
                exchange='WebAnalyzer',
                routing_key='analysis_tasks',
            )
            self.result = task.get()
        except:
            raise exceptions.ValidationError("Error occurred in celery task. Please contact administrator.")

        super(ImageModel, self).save()