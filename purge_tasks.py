from celery_app import app as app

app.control.purge()
