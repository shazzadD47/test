#! /bin/bash

APP=${APP:-"AI"}

if [ "${APP,,}" = "ai" ]; then
    echo "Running AI"
    alembic upgrade head
    gunicorn --forwarded-allow-ips "*" -k uvicorn.workers.UvicornWorker -c gunicorn/gunicorn_conf.py "app.main:app"
elif [ "${APP,,}" = "celery" ]; then
    echo "Running Celery"
    celery -A app.core worker --loglevel=info
else
    echo "Invalid APP. Must be either 'AI' or 'CELERY'"
fi
