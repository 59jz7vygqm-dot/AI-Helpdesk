"""RTP send/receive for one audio leg.

Design notes that matter for latency:

* Playout is a 20 ms tick driven by an absolute clock, not cumulative sleeps, so
  it cannot drift.
* The receive path hands frames straight to the session; there is no de-jitter
  delay on input because the recogniser tolerates reordering far better than the
  caller tolerates waiting.
* The output queue is a deque of int16 samples that ``clear()`` empties, which is
  what makes barge-in instant.
"""

from __future__ import annotations

import asyncio
import logging
import random
import struct
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple

import numpy as np

from ..audio.codec import decode as codec_decode
from ..audio.codec import encode as codec_encode
from ..audio.codec import silence_byte

log = logging.getLogger(__name__)

RTP_VERSION = 2
RTP_HEADER_LEN = 12


@dataclass
class RtpHeader:
    payload_type: int
    sequence: int
    timestamp: int
    ssrc: int
    marker: bool = False

    @staticmethod
    def parse(data: bytes) -> Optional[Tuple["RtpHeader", bytes]]:
        if len(data) < RTP_HEADER_LEN:
            return None
        b0, b1, seq, ts, ssrc = struct.unpack("!BBHII", data[:RTP_HEADER_LEN])
        if (b0 >> 6) != RTP_VERSION:
            return None
        csrc_count = b0 & 0x0F
        has_extension = bool(b0 & 0x10)
        has_padding = bool(b0 & 0x20)
        offset = RTP_HEADER_LEN + 4 * csrc_count
        if has_extension:
            if len(data) < offset + 4:
                return None
            ext_words = struct.unpack("!H", data[offset + 2 : offset + 4])[0]
            offset += 4 + 4 * ext_words
        if offset > len(data):
            return None
        payload = data[offset:]
        if has_padding and payload:
            pad = payload[-1]
            if 0 < pad <= len(payload):
                payload = payload[:-pad]
        header = RtpHeader(
            payload_type=b1 & 0x7F,
            sequence=seq,
            timestamp=ts,
            ssrc=ssrc,
            marker=bool(b1 & 0x80),
        )
        return header, payload

    def encode(self) -> bytes:
        b0 = RTP_VERSION << 6
        b1 = (0x80 if self.marker else 0) | (self.payload_type & 0x7F)
        return struct.pack(
            "!BBHII", b0, b1, self.sequence & 0xFFFF, self.timestamp & 0xFFFFFFFF, self.ssrc
        )


class DtmfCollector:
    """RFC 2833 telephone-event decoder.

    Reports each digit once, on the first packet of the event, so DTMF feels as
    immediate as it does on a normal phone.
    """

    def __init__(self) -> None:
        self._active_timestamp: Optional[int] = None

    def feed(self, timestamp: int, payload: bytes) -> Optional[str]:
        if len(payload) < 4:
            return None
        event = payload[0]
        end = bool(payload[1] & 0x80)
        if end:
            if self._active_timestamp == timestamp:
                self._active_timestamp = None
            return None
        if self._active_timestamp == timestamp:
            return None
        self._active_timestamp = timestamp
        table = "0123456789*#ABCD"
        return table[event] if event < len(table) else None


class RtpSession(asyncio.DatagramProtocol):
    """One bidirectional RTP stream.

    ``on_audio`` receives 8 kHz int16 arrays as they arrive (20 ms each).
    ``enqueue`` accepts 8 kHz int16 audio to play towards the caller.
    """

    def __init__(
        self,
        encoding: str,
        payload_type: int,
        *,
        on_audio: Callable[[np.ndarray], None],
        on_dtmf: Optional[Callable[[str], None]] = None,
        dtmf_payload: Optional[int] = 101,
        frame_ms: int = 20,
        sample_rate: int = 8000,
        send_silence: bool = True,
    ) -> None:
        self.encoding = encoding
        self.payload_type = payload_type
        self.dtmf_payload = dtmf_payload
        self.on_audio = on_audio
        self.on_dtmf = on_dtmf
        self.frame_ms = frame_ms
        self.sample_rate = sample_rate
        self.samples_per_frame = sample_rate * frame_ms // 1000
        self.send_silence = send_silence

        self.transport: Optional[asyncio.DatagramTransport] = None
        self.remote: Optional[Tuple[str, int]] = None
        #: learn the real source address; symmetric RTP beats trusting the SDP
        self.lock_remote_to_source = True
        self._remote_locked = False

        self._out: Deque[np.ndarray] = deque()
        self._out_samples = 0
        self._seq = random.randint(0, 0xFFFF)
        self._timestamp = random.randint(0, 0x7FFFFFFF)
        self._ssrc = random.randint(0, 0x7FFFFFFF)
        self._marker_pending = True
        self._sender_task: Optional[asyncio.Task] = None
        self._closed = False
        self._dtmf = DtmfCollector()

        self._silence_payload = bytes([silence_byte(encoding)]) * self.samples_per_frame

        self.packets_sent = 0
        self.packets_received = 0
        self.bytes_received = 0
        self.last_receive_time: float = 0.0
        self._last_seq: Optional[int] = None
        self.packets_lost = 0

    # ---- asyncio protocol ---------------------------------------------
    def connection_made(self, transport) -> None:  # type: ignore[override]
        self.transport = transport

    def datagram_received(self, data: bytes, addr) -> None:  # type: ignore[override]
        parsed = RtpHeader.parse(data)
        if parsed is None:
            return
        header, payload = parsed

        if self.lock_remote_to_source and not self._remote_locked:
            # Symmetric RTP: many PBXes behind NAT send from a port other than
            # the one advertised in SDP, and only that port will accept our audio.
            if self.remote != addr:
                log.info("RTP remote locked to %s:%s (SDP said %s)", addr[0], addr[1], self.remote)
            self.remote = addr
            self._remote_locked = True

        self.packets_received += 1
        self.bytes_received += len(payload)
        self.last_receive_time = time.monotonic()

        if self._last_seq is not None:
            gap = (header.sequence - self._last_seq) & 0xFFFF
            if 1 < gap < 1000:
                self.packets_lost += gap - 1
        self._last_seq = header.sequence

        if self.dtmf_payload is not None and header.payload_type == self.dtmf_payload:
            digit = self._dtmf.feed(header.timestamp, payload)
            if digit and self.on_dtmf:
                self.on_dtmf(digit)
            return

        if header.payload_type != self.payload_type:
            return
        if not payload:
            return
        try:
            pcm = codec_decode(payload, self.encoding)
        except Exception:  # pragma: no cover - malformed payload
            return
        self.on_audio(pcm)

    def error_received(self, exc) -> None:  # type: ignore[override]
        log.debug("RTP socket error: %s", exc)

    # ---- playout ------------------------------------------------------
    def enqueue(self, pcm: np.ndarray) -> None:
        arr = np.asarray(pcm, dtype=np.int16)
        if arr.size:
            self._out.append(arr)
            self._out_samples += arr.size

    def queued_ms(self) -> int:
        return int(self._out_samples * 1000 / self.sample_rate)

    def clear_playout(self) -> None:
        """Drop everything pending -- used for barge-in."""
        self._out.clear()
        self._out_samples = 0
        self._marker_pending = True

    def is_playing(self) -> bool:
        return self._out_samples > 0

    def _take_frame(self) -> Optional[np.ndarray]:
        need = self.samples_per_frame
        if self._out_samples == 0:
            return None
        chunks = []
        got = 0
        while got < need and self._out:
            head = self._out[0]
            take = min(need - got, head.size)
            chunks.append(head[:take])
            if take == head.size:
                self._out.popleft()
            else:
                self._out[0] = head[take:]
            got += take
            self._out_samples -= take
        frame = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if frame.size < need:
            # Pad the tail of an utterance rather than sending a short packet.
            frame = np.pad(frame, (0, need - frame.size))
        return frame

    async def _send_loop(self) -> None:
        interval = self.frame_ms / 1000.0
        next_tick = time.monotonic()
        while not self._closed:
            next_tick += interval
            delay = next_tick - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.2:
                # Fell badly behind (GPU stall, GC pause): resync instead of
                # bursting a backlog of packets at the caller.
                next_tick = time.monotonic()
            if self._closed:
                break
            try:
                self._emit_one()
            except Exception:  # pragma: no cover - keep the call alive
                log.exception("RTP send tick failed")

    def _emit_one(self) -> None:
        if self.transport is None or self.remote is None:
            return
        frame = self._take_frame()
        if frame is None:
            if not self.send_silence:
                return
            payload = self._silence_payload
            marker = False
        else:
            payload = codec_encode(frame, self.encoding)
            marker = self._marker_pending
            self._marker_pending = False

        header = RtpHeader(
            payload_type=self.payload_type,
            sequence=self._seq,
            timestamp=self._timestamp,
            ssrc=self._ssrc,
            marker=marker,
        )
        self.transport.sendto(header.encode() + payload, self.remote)
        self._seq = (self._seq + 1) & 0xFFFF
        self._timestamp = (self._timestamp + self.samples_per_frame) & 0xFFFFFFFF
        self.packets_sent += 1

    def start(self) -> None:
        if self._sender_task is None:
            self._sender_task = asyncio.ensure_future(self._send_loop())

    def set_remote(self, host: str, port: int) -> None:
        self.remote = (host, port)
        self._remote_locked = False

    async def close(self) -> None:
        self._closed = True
        if self._sender_task:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except (asyncio.CancelledError, Exception):  # noqa: B014
                pass
            self._sender_task = None
        if self.transport:
            self.transport.close()
            self.transport = None


async def open_rtp_session(
    local_host: str,
    port_range: Tuple[int, int],
    session: RtpSession,
) -> int:
    """Bind an even port from the configured range (RFC 3550 convention)."""
    loop = asyncio.get_running_loop()
    low, high = port_range
    start = low if low % 2 == 0 else low + 1
    candidates = list(range(start, high, 2))
    random.shuffle(candidates)
    last_error: Optional[Exception] = None
    for port in candidates:
        try:
            await loop.create_datagram_endpoint(lambda: session, local_addr=(local_host, port))
            return port
        except OSError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"no free RTP port in {low}-{high}: {last_error}")
