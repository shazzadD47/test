import multiprocessing
import os

host = os.getenv("HOST", "0.0.0.0")  # nosec B104
port = os.getenv("PORT", "8000")

workers_per_core_str = os.getenv("WORKERS_PER_CORE", "1")
max_workers_str = os.getenv("MAX_WORKERS", None)
web_concurrency_str = os.getenv("WEB_CONCURRENCY", None)

cores = multiprocessing.cpu_count()
workers_per_core = int(workers_per_core_str)
default_web_concurrency = workers_per_core * cores + 1

if web_concurrency_str:
    web_concurrency = int(web_concurrency_str)
    assert web_concurrency > 0  # nosec B101
else:
    web_concurrency = max(default_web_concurrency, 2)

    if max_workers_str:
        max_workers = int(max_workers_str)
        web_concurrency = min(web_concurrency, max_workers)

graceful_timeout_str = os.getenv("GRACEFUL_TIMEOUT", "600")
timeout_str = os.getenv("TIMEOUT", "600")
keep_alive_str = os.getenv("KEEP_ALIVE", "600")

# Gunicorn config variables
loglevel = os.getenv("LOG_LEVEL", "INFO")
workers = web_concurrency
bind = os.getenv("BIND", f"{host}:{port}")
worker_tmp_dir = "/dev/shm"  # nosec B108
graceful_timeout = int(graceful_timeout_str)
timeout = int(timeout_str)
keepalive = int(keep_alive_str)
