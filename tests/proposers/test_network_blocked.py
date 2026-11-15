"""The suite runs with the network shut (``tests/conftest.py``): a test cannot reach a remote host.

This is how the repository proves that the optional LLM adapter, or anything else, never talks to a
service while the tests run. Local sockets stay open (an asyncio loop needs a socket pair).
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from tests.conftest import NetworkBlocked


class TestRemoteIsRefused:
    def test_create_connection_to_a_name(self) -> None:
        with pytest.raises(NetworkBlocked, match="look up"):
            socket.create_connection(("example.com", 80), timeout=1)

    def test_connect_to_an_address(self) -> None:
        with (
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s,
            pytest.raises(NetworkBlocked, match="connect to"),
        ):
            s.connect(("93.184.216.34", 80))

    def test_connect_ex_to_an_address(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, pytest.raises(NetworkBlocked):
            s.connect_ex(("93.184.216.34", 80))

    def test_a_datagram_to_a_remote_address(self) -> None:
        with (
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s,
            pytest.raises(NetworkBlocked, match="send to"),
        ):
            s.sendto(b"x", ("8.8.8.8", 53))

    def test_a_datagram_with_flags_to_a_remote_address(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s, pytest.raises(NetworkBlocked):
            s.sendto(b"x", 0, ("8.8.8.8", 53))

    def test_a_name_lookup(self) -> None:
        with pytest.raises(NetworkBlocked, match="look up"):
            socket.getaddrinfo("example.org", 443)

    def test_the_standard_library_clients_inherit_the_block(self) -> None:
        import urllib.request

        with pytest.raises(Exception) as caught:
            urllib.request.urlopen("http://example.com/", timeout=1)
        assert "network access is blocked" in repr(caught.value) or isinstance(
            caught.value, NetworkBlocked
        )

    def test_the_error_is_a_runtime_error_that_names_the_target(self) -> None:
        with pytest.raises(RuntimeError, match="example.net"):
            socket.getaddrinfo("example.net", 80)


class TestLocalStaysOpen:
    def test_loopback_connection_works(self) -> None:
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
                conn, _ = server.accept()
                with conn:
                    client.sendall(b"ping")
                    assert conn.recv(4) == b"ping"

    def test_localhost_resolves(self) -> None:
        assert socket.getaddrinfo("localhost", 80)

    def test_a_socket_pair_works(self) -> None:
        a, b = socket.socketpair()
        with a, b:
            a.sendall(b"x")
            assert b.recv(1) == b"x"

    def test_the_guard_is_installed_once_for_the_session(self) -> None:
        patched: Any = socket.socket.connect
        assert patched.__name__ == "connect" and patched.__module__ == "tests.conftest"
