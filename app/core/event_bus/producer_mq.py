import json

import pika
from celery.utils.log import get_task_logger
from pika.exceptions import AMQPError

from app.configs import settings
from app.logging import logger

celery_logger = get_task_logger("delineate.event_bus.producer_mq")


class ProducerMQ:
    """
    Synchronous RabbitMQ producer class. Sends messages to a queue.

    Usage:
    ```python
    # Example usage with context manager
    def main():
        message = {"task": "process_task", "x": 10, "y": 20}

        # Using the context manager to handle connection automatically
        with ProducerMQ(host='localhost', queue_name='test_queue') as producer:
            producer.send_message(message)

    # Run the producer example
    if __name__ == "__main__":
        main()
    ```
    """

    def __init__(self, queue_name: str = None):
        """
        Initialize connection parameters for RabbitMQ producer.

        :param host: RabbitMQ server host (default: 'localhost')
        :param queue_name: Name of the queue to send messages to (default: 'test_queue')
        """
        self.uri = settings.RMQ_URL
        self.queue_name = queue_name or settings.RMQ_QUEUE
        self.connection = None
        self.channel = None

    def _connect(self):
        """
        Synchronously connect to RabbitMQ and create a channel.
        """
        try:
            parameters = pika.URLParameters(self.uri)
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.queue_name, durable=True)
            logger.info(f"[RMQ] Connected to RabbitMQ queue named '{self.queue_name}'")
        except AMQPError as e:
            logger.exception(f"[RMQ] RabbitMQ connection error: {e}")
            raise

    def send_message(self, message: dict):
        """
        Sends a message to the RabbitMQ queue synchronously.

        :param message: The message to be sent (must be a dictionary)
        """
        try:
            if self.channel is None:
                self._connect()

            message_body = json.dumps(message)
            self.channel.basic_publish(
                exchange="",
                routing_key=self.queue_name,
                body=message_body,
                properties=pika.BasicProperties(
                    delivery_mode=2
                ),  # Make message persistent
            )

            logger.info(f"[RMQ] Sent message: {message}")
            celery_logger.info(f"[RMQ] Sent message: {message}")
        except AMQPError as e:
            logger.exception(f"[RMQ] RabbitMQ Message error: {e}")
            celery_logger.exception(f"[RMQ] RabbitMQ Message error: {e}")

    def _close_connection(self):
        """
        Synchronously close the RabbitMQ connection.
        """
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("[RMQ] Connection closed.")

    def __enter__(self):
        """
        Enter the context manager, ensuring connection is established.
        """
        self._connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager, ensuring connection is closed.
        """
        self._close_connection()
