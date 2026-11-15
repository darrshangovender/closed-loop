"""Shared test configuration.

* Hypothesis keeps nothing in the tree (no example database, cache in the temp folder) and has
  no wall-clock deadline (the laptop that runs these tests is often busy).
* Tests marked ``f2`` need the three sibling repositories. They skip with a clear message when a
  sibling is not installed. ``find_spec`` does not import anything, so the check costs nothing.
* The whole suite runs with the network shut: a connection to, or a name lookup of, anything but the
  local machine raises ``NetworkBlocked``. That is how the suite proves that nothing in it, the
  optional LLM adapter included, reaches a remote service (see ``tests/proposers``).
"""

import importlib.util
import os
import socket
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from closed_loop.eval.tuned import PROVISIONAL, use_tuned

# Hypothesis writes a constants cache even with the example database off; keep it out of the tree.
os.environ.setdefault(
    "HYPOTHESIS_STORAGE_DIRECTORY", str(Path(tempfile.gettempdir()) / "closed-loop-hypothesis")
)

try:
    from hypothesis import settings
except ImportError:  # hypothesis is a dev extra
    pass
else:
    settings.register_profile("closed_loop", database=None, deadline=None)
    settings.load_profile("closed_loop")

SIBLING_MODULES = ("asset_health", "maintenance_agent", "plant_copilot")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    missing = [m for m in SIBLING_MODULES if importlib.util.find_spec(m) is None]
    if not missing:
        return
    skip = pytest.mark.skip(
        reason=f"f2 test needs the sibling repositories; not installed: {', '.join(missing)}"
    )
    for item in items:
        if "f2" in item.keywords:
            item.add_marker(skip)


class NetworkBlocked(RuntimeError):
    """A test tried to reach a machine other than this one."""


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _is_local(address: Any) -> bool:
    if isinstance(address, str | bytes):  # a Unix socket path
        return True
    host = address[0] if isinstance(address, tuple) and address else address
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    return host in _LOCAL_HOSTS or (isinstance(host, str) and host.startswith("127."))


@pytest.fixture(autouse=True, scope="session")
def _network_is_blocked() -> Iterator[None]:
    """Refuse every non-local connection, datagram and name lookup for the whole session."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    real_getaddrinfo = socket.getaddrinfo

    def refuse(what: str, target: Any) -> NetworkBlocked:
        return NetworkBlocked(f"network access is blocked in the test suite: {what} {target!r}")

    def connect(self: socket.socket, address: Any) -> None:
        if not _is_local(address):
            raise refuse("connect to", address)
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        if not _is_local(address):
            raise refuse("connect to", address)
        return real_connect_ex(self, address)

    def sendto(self: socket.socket, data: Any, *rest: Any) -> int:
        if not _is_local(rest[-1]):
            raise refuse("send to", rest[-1])
        return real_sendto(self, data, *rest)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host is not None and not _is_local((host,)):
            raise refuse("look up", host)
        return real_getaddrinfo(host, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", connect)
        patch.setattr(socket.socket, "connect_ex", connect_ex)
        patch.setattr(socket.socket, "sendto", sendto)
        patch.setattr(socket, "getaddrinfo", getaddrinfo)
        yield


# Tests written and pinned before ``benchmarks/tuned.json`` existed (P12a, P6a) ran on the
# provisional defaults. The file is final now and moves the arms (B3's k, the planner's delta), so
# those tests say so: the pins stay what they were and nothing about the final values is claimed.
@pytest.fixture(scope="module")
def provisional_defaults_module() -> Iterator[None]:
    """The provisional tuned values for the whole module (``pytestmark = usefixtures(...)``)."""
    with use_tuned(PROVISIONAL):
        yield


@pytest.fixture
def provisional_defaults() -> Iterator[None]:
    """The provisional tuned values for one test (``@pytest.mark.usefixtures(...)``)."""
    with use_tuned(PROVISIONAL):
        yield
