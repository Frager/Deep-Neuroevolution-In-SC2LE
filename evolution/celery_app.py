from celery import Celery


# rabbitmq:
app = Celery('tasks',
             broker='amqp://chris:master@192.168.99.100:32770/vchris',
             backend='amqp://chris:master@192.168.99.100:32770/vchris',
             task_serializer='json',
             result_serializer='json',
             accept_content=['application/json'])

# # redis:
# app = Celery('tasks')
# app.conf.BROKER_URL = 'redis://192.168.99.100:32768/0'
# app.conf.CELERY_RESULT_BACKEND = "redis"
# app.conf.CELERY_REDIS_HOST = "192.168.99.100"
# app.conf.CELERY_REDIS_PORT = 32768
# app.conf.CELERY_REDIS_DB = 0

# app.conf.update(
#     CELERY_TASK_SERIALIZER='json',
#     CELERY_RESULT_SERIALIZER='json',
#     CELERY_ACCEPT_CONTENT=['json'])
