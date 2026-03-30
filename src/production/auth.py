"""Dashboard authentication — role-based access control.

Supports three modes:
  1. 'open' — no authentication (research/demo, current default)
  2. 'local' — username/password with role mapping (testing/pilot)
  3. 'ldap' — LDAP/Active Directory integration (production)

Roles match the dashboard's 5-role system:
  IT Security Analyst, Clinical IT Administrator,
  Attending Physician, Hospital Manager, Regulatory Auditor
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ROLES = [
    "IT Security Analyst",
    "Clinical IT Administrator",
    "Attending Physician",
    "Hospital Manager",
    "Regulatory Auditor",
]


class AuthSession:
    """Authenticated user session."""

    __slots__ = ("username", "role", "token", "created_at", "expires_at")

    def __init__(self, username: str, role: str, ttl_seconds: int = 28800) -> None:
        self.username = username
        self.role = role
        self.token = secrets.token_urlsafe(32)
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds

    @property
    def is_valid(self) -> bool:
        return time.time() < self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "username": self.username,
            "role": self.role,
            "token": self.token[:8] + "...",
            "valid": self.is_valid,
        }


class AuthProvider:
    """Role-based authentication provider.

    Args:
        mode: 'open', 'local', or 'ldap'.
        users: Local user database (for 'local' mode).
            Format: {"username": {"password_hash": "sha256hex", "role": "..."}}
        ldap_config: LDAP connection config (for 'ldap' mode).
    """

    def __init__(
        self,
        mode: str = "open",
        users: Optional[Dict[str, Dict[str, str]]] = None,
        ldap_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._mode = mode
        self._users = users or {}
        self._ldap_config = ldap_config or {}
        self._sessions: Dict[str, AuthSession] = {}

    def authenticate(self, username: str, password: str) -> Optional[AuthSession]:
        """Authenticate a user and create a session.

        Args:
            username: User identifier.
            password: Plaintext password (hashed immediately, never stored).

        Returns:
            AuthSession if authenticated, None otherwise.
        """
        if self._mode == "open":
            session = AuthSession(username, ROLES[0])
            self._sessions[session.token] = session
            return session

        if self._mode == "local":
            return self._auth_local(username, password)

        if self._mode == "ldap":
            return self._auth_ldap(username, password)

        return None

    def validate_token(self, token: str) -> Optional[AuthSession]:
        """Validate a session token.

        Returns:
            AuthSession if valid and not expired, None otherwise.
        """
        session = self._sessions.get(token)
        if session and session.is_valid:
            return session
        if session and not session.is_valid:
            del self._sessions[token]
        return None

    def logout(self, token: str) -> bool:
        """Invalidate a session."""
        if token in self._sessions:
            del self._sessions[token]
            return True
        return False

    def _auth_local(self, username: str, password: str) -> Optional[AuthSession]:
        """Authenticate against local user database."""
        user = self._users.get(username)
        if not user:
            logger.warning("Auth failed: unknown user %s", username)
            return None

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if not hmac.compare_digest(password_hash, user.get("password_hash", "")):
            logger.warning("Auth failed: bad password for %s", username)
            return None

        role = user.get("role", ROLES[0])
        if role not in ROLES:
            logger.warning("Auth failed: invalid role %s for %s", role, username)
            return None

        session = AuthSession(username, role)
        self._sessions[session.token] = session
        logger.info("Auth success: %s as %s", username, role)
        return session

    def _auth_ldap(self, username: str, password: str) -> Optional[AuthSession]:
        """Authenticate against LDAP/Active Directory.

        In production, replace this with ldap3 library calls:
            from ldap3 import Server, Connection, ALL
            server = Server(self._ldap_config["host"], get_info=ALL)
            conn = Connection(server, user=f"cn={username},...", password=password)
            if conn.bind():
                role = self._resolve_ldap_role(conn, username)
                ...
        """
        logger.info("LDAP auth placeholder for %s — use local mode for testing", username)
        return None

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password for local user database storage."""
        return hashlib.sha256(password.encode()).hexdigest()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def active_sessions(self) -> int:
        # Clean expired
        expired = [t for t, s in self._sessions.items() if not s.is_valid]
        for t in expired:
            del self._sessions[t]
        return len(self._sessions)

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": self._mode,
            "active_sessions": self.active_sessions,
            "registered_users": len(self._users) if self._mode == "local" else "N/A",
        }
