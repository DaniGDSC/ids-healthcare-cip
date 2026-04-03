"""mTLS configuration for inter-service communication.

Provides SSL context factory for mutual TLS between:
  - Kafka producer/consumer
  - SIEM syslog (TLS)
  - SMTP relay
  - FastAPI inter-service calls
  - Dashboard WebSocket

In production: load certs from /etc/iomt-ids/certs/.
In testing: dry_run mode skips cert loading.
"""

from __future__ import annotations

import logging
import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TLSConfig:
    """TLS/mTLS certificate configuration.

    Args:
        ca_cert: Path to CA certificate (PEM).
        server_cert: Path to server certificate (PEM).
        server_key: Path to server private key (PEM).
        client_cert: Path to client certificate for mTLS (PEM).
        client_key: Path to client private key for mTLS (PEM).
        min_version: Minimum TLS version (default TLSv1.2).
        verify_hostname: Whether to verify server hostname.
    """

    ca_cert: Optional[Path] = None
    server_cert: Optional[Path] = None
    server_key: Optional[Path] = None
    client_cert: Optional[Path] = None
    client_key: Optional[Path] = None
    min_version: str = "TLSv1.2"
    verify_hostname: bool = True

    def create_server_context(self) -> ssl.SSLContext:
        """Create SSL context for server-side (FastAPI, WebSocket)."""
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = self._parse_version()

        if self.server_cert and self.server_key:
            ctx.load_cert_chain(str(self.server_cert), str(self.server_key))
        if self.ca_cert:
            ctx.load_verify_locations(str(self.ca_cert))
            ctx.verify_mode = ssl.CERT_REQUIRED  # mTLS: require client cert

        logger.info("Server TLS context created (min=%s, mTLS=%s)",
                     self.min_version, self.ca_cert is not None)
        return ctx

    def create_client_context(self) -> ssl.SSLContext:
        """Create SSL context for client-side (Kafka, SIEM, SMTP)."""
        ctx = ssl.create_default_context()
        ctx.minimum_version = self._parse_version()
        ctx.check_hostname = self.verify_hostname

        if self.ca_cert:
            ctx.load_verify_locations(str(self.ca_cert))
        if self.client_cert and self.client_key:
            ctx.load_cert_chain(str(self.client_cert), str(self.client_key))

        logger.info("Client TLS context created (min=%s, mTLS=%s)",
                     self.min_version, self.client_cert is not None)
        return ctx

    def _parse_version(self) -> ssl.TLSVersion:
        versions = {
            "TLSv1.2": ssl.TLSVersion.TLSv1_2,
            "TLSv1.3": ssl.TLSVersion.TLSv1_3,
        }
        return versions.get(self.min_version, ssl.TLSVersion.TLSv1_2)

    def to_kafka_config(self) -> Dict[str, Any]:
        """Generate Kafka SSL configuration dict."""
        config: Dict[str, Any] = {}
        if self.ca_cert:
            config["ssl.ca.location"] = str(self.ca_cert)
        if self.client_cert:
            config["ssl.certificate.location"] = str(self.client_cert)
        if self.client_key:
            config["ssl.key.location"] = str(self.client_key)
        if config:
            config["security.protocol"] = "SSL"
        return config

    def get_status(self) -> Dict[str, Any]:
        return {
            "min_version": self.min_version,
            "ca_cert": str(self.ca_cert) if self.ca_cert else None,
            "server_cert_present": self.server_cert is not None,
            "client_cert_present": self.client_cert is not None,
            "mtls_enabled": self.client_cert is not None and self.ca_cert is not None,
        }


def load_from_directory(cert_dir: str | Path) -> TLSConfig:
    """Load TLS config from a standard certificate directory.

    Expected structure:
        cert_dir/
            ca.pem          — CA certificate
            server.pem      — Server certificate
            server-key.pem  — Server private key
            client.pem      — Client certificate (mTLS)
            client-key.pem  — Client private key (mTLS)
    """
    d = Path(cert_dir)
    return TLSConfig(
        ca_cert=d / "ca.pem" if (d / "ca.pem").exists() else None,
        server_cert=d / "server.pem" if (d / "server.pem").exists() else None,
        server_key=d / "server-key.pem" if (d / "server-key.pem").exists() else None,
        client_cert=d / "client.pem" if (d / "client.pem").exists() else None,
        client_key=d / "client-key.pem" if (d / "client-key.pem").exists() else None,
    )
