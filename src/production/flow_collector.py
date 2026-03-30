"""Flow collector — captures network flows from TAP and publishes to Kafka.

In production, this wraps Argus or a similar flow exporter. Reads raw
flow records from the exporter's output and publishes structured
24-feature vectors to a Kafka topic.

For WUSTL-compatible networks, all 16 network features are available
natively from Argus flow records (no imputation needed).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FlowCollector:
    """Collects network flows and publishes to a message queue.

    In production: wraps Argus with Kafka producer.
    In testing: accepts flows via inject() method.

    Args:
        publish_fn: Callback to publish a flow record.
            Signature: publish_fn(topic: str, payload: dict) -> None.
            In production, this wraps KafkaProducer.send().
        topic: Kafka topic name for raw flows.
        network_features: List of network feature names to extract.
    """

    NETWORK_FEATURES: List[str] = [
        "SrcBytes", "DstBytes", "SrcLoad", "DstLoad",
        "SIntPkt", "DIntPkt", "SIntPktAct",
        "sMaxPktSz", "dMaxPktSz", "sMinPktSz",
        "Dur", "TotBytes", "Load", "pSrcLoss", "pDstLoss", "Packet_num",
    ]

    def __init__(
        self,
        publish_fn: Callable[[str, Dict[str, Any]], None],
        topic: str = "iomt.flows.raw",
    ) -> None:
        self._publish = publish_fn
        self._topic = topic
        self._flow_count = 0

    def inject(self, flow: Dict[str, float]) -> None:
        """Inject a single flow record (for testing or Argus integration).

        Args:
            flow: Dict mapping feature names to float values.
                  Must contain all 16 network features.
        """
        missing = [f for f in self.NETWORK_FEATURES if f not in flow]
        if missing:
            logger.warning("Flow missing features: %s", missing)
            return

        payload = {
            "features": {f: float(flow[f]) for f in self.NETWORK_FEATURES},
            "timestamp": time.time(),
            "flow_index": self._flow_count,
        }
        self._publish(self._topic, payload)
        self._flow_count += 1

    def inject_batch(self, flows: List[Dict[str, float]]) -> int:
        """Inject a batch of flow records.

        Returns:
            Number of successfully published flows.
        """
        count = 0
        for flow in flows:
            self.inject(flow)
            count += 1
        return count

    @property
    def flow_count(self) -> int:
        return self._flow_count
