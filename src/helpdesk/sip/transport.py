"""UDP transport for SIP signalling."""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import Callable, Optional, Tuple

from .messages import SipMessage, SipParseError, parse

log = logging.getLogger(__name__)


def detect_local_ip(peer_host: str, peer_port: int = 5060) -> str:
    """Local address that would be used to reach the PBX.

    Connecting a UDP socket sends nothing but makes the kernel pick the right
    source address, which is what must go into Contact and SDP.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((peer_host, peer_port))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


class SipTransport(asyncio.DatagramProtocol):
    def __init__(self, on_message: Callable[[SipMessage, Tuple[str, int]], None], trace: bool = False) -> None:
        self.on_message = on_message
        self.trace = trace
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.local_addr: Tuple[str, int] = ("0.0.0.0", 0)

    def connection_made(self, transport) -> None:  # type: ignore[override]
        self.transport = transport
        self.local_addr = transport.get_extra_info("sockname")

    def datagram_received(self, data: bytes, addr) -> None:  # type: ignore[override]
        stripped = data.strip()
        if not stripped:
            # Keepalive ping (CRLF) from the PBX -- answer with CRLF.
            if self.transport:
                self.transport.sendto(b"\r\n", addr)
            return
        try:
            message = parse(data)
        except SipParseError as exc:
            log.warning("dropping malformed SIP datagram from %s: %s", addr, exc)
            return
        if self.trace:
            log.debug("<<< %s:%s\n%s", addr[0], addr[1], data.decode("utf-8", "replace"))
        try:
            self.on_message(message, addr)
        except Exception:  # pragma: no cover - never let one message kill the loop
            log.exception("error handling SIP message %s", message)

    def error_received(self, exc) -> None:  # type: ignore[override]
        log.debug("SIP socket error: %s", exc)

    def send(self, message: SipMessage, addr: Tuple[str, int]) -> None:
        if self.transport is None:
            log.error("SIP transport not ready, dropping %s", message)
            return
        data = message.encode()
        if self.trace:
            log.debug(">>> %s:%s\n%s", addr[0], addr[1], data.decode("utf-8", "replace"))
        self.transport.sendto(data, addr)

    def close(self) -> None:
        if self.transport:
            self.transport.close()
            self.transport = None


async def create_transport(
    bind_host: str,
    bind_port: int,
    on_message: Callable[[SipMessage, Tuple[str, int]], None],
    trace: bool = False,
) -> SipTransport:
    loop = asyncio.get_running_loop()
    protocol = SipTransport(on_message, trace=trace)
    await loop.create_datagram_endpoint(lambda: protocol, local_addr=(bind_host, bind_port))
    return protocol
