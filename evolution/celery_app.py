from celery import Celery

app = Celery('tasks',
             broker='redis://172.18.0.2:6379/0',
             backend='redis://172.18.0.2:6379/0',
             task_serializer='json',
             result_serializer='json',
             accept_content=['application/json'])
