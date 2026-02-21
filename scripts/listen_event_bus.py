#!/usr/bin/env python3
"""
RabbitMQ Event Bus Listener

Connects to the RabbitMQ event bus and logs all incoming events.
Uses the same connection settings as the app (RMQ_URL, RMQ_QUEUE from .env).

Usage:
    python scripts/listen_event_bus.py
    python scripts/listen_event_bus.py --queue BACKEND_V3_GCP_DEV
    python scripts/listen_event_bus.py --log-file events.log
"""

import argparse
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone

import pika
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("event_bus_listener")


def on_message(channel, method, properties, body):
    """Callback invoked for each message received from the queue."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    try:
        payload = json.loads(body)
        pattern = payload.get("pattern", "UNKNOWN")
        data = payload.get("data", {})

        logger.info(
            "[%s] Event: %s\n" "  Payload: %s",
            timestamp,
            pattern,
            json.dumps(data, indent=2, default=str),
        )
    except json.JSONDecodeError:
        logger.warning(
            "[%s] Non-JSON message received:\n  %s",
            timestamp,
            body.decode(errors="replace"),
        )

    channel.basic_ack(delivery_tag=method.delivery_tag)


def main():
    parser = argparse.ArgumentParser(description="Listen to the RabbitMQ event bus")
    parser.add_argument(
        "--queue",
        default=os.getenv("RMQ_QUEUE"),
        help="Queue name to listen on (default: RMQ_QUEUE from .env)",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("RMQ_URL"),
        help="RabbitMQ connection URL (default: RMQ_URL from .env)",
    )
    parser.add_argument(
        "--log-file",
        help="Optional file path to also write logs to",
    )
    args = parser.parse_args()

    if not args.url:
        logger.error("RMQ_URL not set. Provide --url or set RMQ_URL in .env")
        sys.exit(1)
    if not args.queue:
        logger.error("RMQ_QUEUE not set. Provide --queue or set RMQ_QUEUE in .env")
        sys.exit(1)

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    logger.info("Connecting to RabbitMQ...")
    logger.info("Queue: %s", args.queue)

    parameters = pika.URLParameters(args.url)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue=args.queue, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=args.queue, on_message_callback=on_message)

    def shutdown(signum, frame):
        logger.info("Shutting down listener...")
        channel.stop_consuming()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("Listening for events (Ctrl+C to stop)...")

    try:
        channel.start_consuming()
    finally:
        connection.close()
        logger.info("Connection closed.")


if __name__ == "__main__":
    main()
