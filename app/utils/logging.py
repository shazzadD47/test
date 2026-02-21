import atexit
import os
from logging.config import ConvertingDict, ConvertingList, valid_ident
from logging.handlers import QueueHandler, QueueListener
from queue import Full, Queue

DEFAULT_LOG_QUEUE_MAXSIZE = int(os.getenv("LOG_QUEUE_MAXSIZE", "10000"))


def _resolve_handlers(handlers):
    if not isinstance(handlers, ConvertingList):
        return handlers

    return [handlers[i] for i in range(len(handlers))]


def _resolve_queue(queue: Queue):
    if not isinstance(queue, ConvertingDict):
        return queue

    if "__resolved_value__" in queue:
        return queue["__resolved_value__"]

    class_name = queue.pop("class")
    class_ = queue.configurator.resolve(class_name)

    props = queue.pop(".", None)
    kwargs = {k: queue[k] for k in queue if valid_ident(k)}

    result = class_(**kwargs)

    if props:
        for name, value in props.items():
            setattr(result, name, value)

    queue["__resolved_value__"] = result

    return result


class QueueListenerHandler(QueueHandler):
    def __init__(
        self,
        handlers,
        respect_handler_level=False,
        auto_run=True,
        queue=Queue(DEFAULT_LOG_QUEUE_MAXSIZE),
    ):
        queue = _resolve_queue(queue)
        super().__init__(queue)

        handlers = _resolve_handlers(handlers)
        self._listener = QueueListener(
            self.queue, *handlers, respect_handler_level=respect_handler_level
        )

        if auto_run:
            self.start()
            atexit.register(self.stop)

    def start(self):
        self._listener.start()

    def stop(self):
        self._listener.stop()

    def emit(self, record):
        try:
            return super().emit(record)
        except Full:
            return
