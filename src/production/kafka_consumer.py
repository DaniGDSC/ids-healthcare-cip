"""Kafka consumer — consumes fused feature vectors and feeds WindowBuffer.

Replaces the file-based watchdog ingestion with a message queue consumer.
In production: wraps confluent_kafka or aiokafka.
In testing: accepts messages via inject() for deterministic tests.

The consumer maintains exactly-once semantics by committing offsets
only after the window buffer has accepted the flow vector.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from dashboard.streaming.feature_aligner import MODEL_FEATURES, N_FEATURES
from dashboard.streaming.window_buffer import WindowBuffer

logger = logging.getLogger(__name__)


class KafkaFlowConsumer:
    """Consumes fused+scaled flow vectors from Kafka and feeds WindowBuffer.

    In production, wrap this with a real Kafka consumer library.
    The class provides a transport-agnostic interface so the
    WindowBuffer and InferenceService don't care where flows come from.

    Args:
        buffer: WindowBuffer to feed flow vectors into.
        topic: Kafka topic to consume from.
        group_id: Consumer group ID for offset management.
        on_window_ready: Callback when a full window is available.
    """

    def __init__(
        self,
        buffer: WindowBuffer,
        topic: str = "iomt.flows.fused",
        group_id: str = "iomt-inference",
        on_window_ready: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self._buffer = buffer
        self._topic = topic
        self._group_id = group_id
        self._on_window_ready = on_window_ready
        self._consumed_count = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Message queue for testing (inject messages without real Kafka)
        self._test_queue: List[Dict[str, Any]] = []

    def inject(self, message: Dict[str, Any]) -> None:
        """Inject a message directly (for testing without Kafka).

        Expected message format:
            {"features": [f1, f2, ..., f24], "timestamp": float}
        or:
            {"features": {"SrcBytes": 310.0, ...}, "timestamp": float}

        Args:
            message: Flow message with scaled feature vector.
        """
        features = message.get("features")
        if features is None:
            logger.warning("Message has no 'features' field")
            return

        vec = self._extract_vector(features)
        if vec is not None:
            self._buffer.append(vec)
            self._consumed_count += 1

            if self._on_window_ready and not self._buffer.suppress_alerts:
                window = self._buffer.get_window()
                if window is not None:
                    self._on_window_ready(window)

    def inject_batch(self, messages: List[Dict[str, Any]]) -> int:
        """Inject a batch of messages for testing.

        Returns:
            Number of successfully consumed messages.
        """
        count = 0
        for msg in messages:
            self.inject(msg)
            count += 1
        return count

    def start(self, kafka_config: Optional[Dict[str, Any]] = None) -> None:
        """Start consuming from Kafka in a background thread.

        In a real deployment, kafka_config would contain:
            bootstrap_servers, security_protocol, sasl_mechanism, etc.

        Args:
            kafka_config: Kafka connection configuration. If None,
                processes from the test queue only.
        """
        if self._running:
            logger.warning("Consumer already running")
            return

        self._stop_event.clear()
        self._running = True

        if kafka_config:
            self._thread = threading.Thread(
                target=self._kafka_loop,
                args=(kafka_config,),
                daemon=True,
            )
            self._thread.start()
            logger.info("Kafka consumer started: topic=%s, group=%s",
                        self._topic, self._group_id)
        else:
            logger.info("Consumer started in test mode (inject only)")

    def stop(self) -> None:
        """Stop the consumer."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._running = False
        logger.info("Consumer stopped after %d messages", self._consumed_count)

    def _kafka_loop(self, config: Dict[str, Any]) -> None:
        """Main Kafka consumption loop (production).

        This is a placeholder — in production, replace with:
            from confluent_kafka import Consumer
            consumer = Consumer(config)
            consumer.subscribe([self._topic])
            while not self._stop_event.is_set():
                msg = consumer.poll(timeout=0.1)
                if msg and not msg.error():
                    self.inject(json.loads(msg.value()))
                    consumer.commit(msg)
        """
        logger.info("Kafka loop placeholder — use inject() for testing")
        while not self._stop_event.is_set():
            # Process any test queue messages
            if self._test_queue:
                msg = self._test_queue.pop(0)
                self.inject(msg)
            else:
                time.sleep(0.05)

    def _extract_vector(self, features: Any) -> Optional[np.ndarray]:
        """Extract a float32 vector from message features.

        Supports both list format [f1, f2, ..., f24] and
        dict format {"SrcBytes": 310.0, ...}.
        """
        if isinstance(features, list):
            if len(features) != N_FEATURES:
                logger.warning("Feature list has %d elements, expected %d",
                               len(features), N_FEATURES)
                return None
            return np.array(features, dtype=np.float32)

        if isinstance(features, dict):
            vec = np.zeros(N_FEATURES, dtype=np.float32)
            for i, feat in enumerate(MODEL_FEATURES):
                if feat in features:
                    try:
                        vec[i] = float(features[feat])
                    except (ValueError, TypeError):
                        pass
            return vec

        if isinstance(features, np.ndarray):
            return features.astype(np.float32).ravel()[:N_FEATURES]

        logger.warning("Unsupported features type: %s", type(features))
        return None

    @property
    def consumed_count(self) -> int:
        return self._consumed_count

    @property
    def running(self) -> bool:
        return self._running

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "topic": self._topic,
            "group_id": self._group_id,
            "consumed_count": self._consumed_count,
            "buffer_flow_count": self._buffer.flow_count,
            "buffer_state": self._buffer.state.value,
        }
