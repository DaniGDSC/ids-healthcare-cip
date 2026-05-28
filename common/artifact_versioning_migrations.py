"""Schema migration registry (Sprint 6 / Tầng 3.5).

Each entry maps ``(artifact_name, from_version, to_version)`` to a
function that transforms a loaded artifact payload in-place. The
version gate isn't a migration runner — it's a *detector*. When a
loaded artifact's version is older than the registry expected:

  - the consumer can ``apply_migration_chain(payload, name, from_v,
    to_v)`` to walk the chain of registered migrations and arrive at
    the current shape, or
  - the consumer can refuse to read the file and prompt a regen.

For now the registry is empty — no migrations have been needed yet
because every schema bump so far happened during initial development
(before any external consumer started reading the files). The first
real migration will land here when a downstream pipeline starts
reading an artifact format we then change.

The empty state is intentional, not an oversight; tests pin the
shape so a future migration drops in alongside its registry row.
"""
from __future__ import annotations

from typing import Callable


# (artifact_name, from_version, to_version) → migrator function
# Migrator signature: ``def f(payload: dict) -> dict``
MIGRATIONS: dict[tuple[str, str, str], Callable[[dict], dict]] = {}


def apply_migration_chain(
    payload: dict,
    artifact_name: str,
    from_version: str,
    to_version: str,
) -> dict:
    """Walk the registered migration chain from ``from_version`` to
    ``to_version``.

    Raises ``ValueError`` when no path is registered. Returns the
    transformed payload unchanged when ``from_version == to_version``.
    """
    if from_version == to_version:
        return payload

    # Greedy walk: find the next registered step from current version
    current = from_version
    transformed = payload
    seen: set[str] = {current}
    while current != to_version:
        candidates = [
            (key, fn) for key, fn in MIGRATIONS.items()
            if key[0] == artifact_name and key[1] == current
        ]
        if not candidates:
            raise ValueError(
                f"No migration registered for {artifact_name} from "
                f"{current!r} (target {to_version!r})"
            )
        # Prefer the step that gets us closest to target — for now,
        # just take the first registered (single-line chain is the
        # common case). When migrations branch, extend this with a
        # BFS.
        key, fn = candidates[0]
        next_version = key[2]
        if next_version in seen:
            raise ValueError(
                f"Migration cycle detected at {artifact_name}:{next_version}"
            )
        transformed = fn(transformed)
        seen.add(next_version)
        current = next_version
    return transformed


__all__ = ["MIGRATIONS", "apply_migration_chain"]
