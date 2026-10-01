"""SDP offer/answer for a narrowband audio leg.

Deliberately narrow: G.711 plus RFC 2833 DTMF.  Every PBX speaks it, it needs no
transcoding on the PBX side, and 8 kHz keeps the per-frame cost tiny.  Wideband
would only help if the whole path to the caller were wideband, which over the
PSTN it is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: static payload types we understand (RFC 3551)
STATIC_CODECS = {0: "PCMU", 8: "PCMA"}
CODEC_PAYLOADS = {"PCMU": 0, "PCMA": 8}


@dataclass
class MediaDescription:
    media: str = "audio"
    port: int = 0
    transport: str = "RTP/AVP"
    payloads: List[int] = field(default_factory=list)
    rtpmap: Dict[int, str] = field(default_factory=dict)
    attributes: List[str] = field(default_factory=list)
    direction: str = "sendrecv"
    ptime: int = 20


@dataclass
class SessionDescription:
    address: str = "0.0.0.0"
    session_name: str = "helpdesk"
    media: List[MediaDescription] = field(default_factory=list)

    @property
    def audio(self) -> Optional[MediaDescription]:
        for m in self.media:
            if m.media == "audio":
                return m
        return None


def parse_sdp(body: bytes) -> SessionDescription:
    text = body.decode("utf-8", errors="replace")
    session = SessionDescription()
    current: Optional[MediaDescription] = None
    session_direction: Optional[str] = None

    for raw in re.split(r"\r\n|\n", text):
        line = raw.strip()
        if len(line) < 2 or line[1] != "=":
            continue
        kind, value = line[0], line[2:]

        if kind == "c":
            parts = value.split()
            if len(parts) >= 3:
                addr = parts[2].split("/")[0]
                if current is None:
                    session.address = addr
                else:
                    current.attributes.append(f"c-addr:{addr}")
        elif kind == "s":
            session.session_name = value
        elif kind == "m":
            parts = value.split()
            current = MediaDescription(media=parts[0] if parts else "audio")
            if len(parts) >= 2:
                try:
                    current.port = int(parts[1])
                except ValueError:
                    current.port = 0
            if len(parts) >= 3:
                current.transport = parts[2]
            for payload in parts[3:]:
                try:
                    current.payloads.append(int(payload))
                except ValueError:
                    continue
            if session_direction:
                current.direction = session_direction
            session.media.append(current)
        elif kind == "a":
            attr = value.strip()
            lowered = attr.lower()
            if lowered in ("sendrecv", "sendonly", "recvonly", "inactive"):
                if current is None:
                    session_direction = lowered
                    for m in session.media:
                        m.direction = lowered
                else:
                    current.direction = lowered
                continue
            if current is not None:
                current.attributes.append(attr)
                match = re.match(r"rtpmap:(\d+)\s+([^/]+)/(\d+)", attr, re.IGNORECASE)
                if match:
                    current.rtpmap[int(match.group(1))] = match.group(2).upper()
                match = re.match(r"ptime:(\d+)", attr, re.IGNORECASE)
                if match:
                    current.ptime = int(match.group(1))
    return session


def media_address(session: SessionDescription, media: MediaDescription) -> str:
    """Media-level c= line wins over the session-level one (RFC 4566 §5.7)."""
    for attr in media.attributes:
        if attr.startswith("c-addr:"):
            return attr.split(":", 1)[1]
    return session.address


def select_codec(media: MediaDescription, preference: List[str]) -> Optional[tuple]:
    """Pick the first codec we prefer that the offer actually contains.

    Returns ``(payload_type, encoding_name)``.
    """
    available: Dict[str, int] = {}
    for payload in media.payloads:
        name = media.rtpmap.get(payload) or STATIC_CODECS.get(payload)
        if name in ("PCMU", "PCMA") and name not in available:
            available[name] = payload
    for want in preference:
        want = want.upper()
        if want in available:
            return available[want], want
    return None


def find_dtmf_payload(media: MediaDescription) -> Optional[int]:
    for payload, name in media.rtpmap.items():
        if name.upper() in ("TELEPHONE-EVENT", "TELEPHONE_EVENT"):
            return payload
    return None


def build_sdp(
    address: str,
    port: int,
    encoding: str,
    *,
    dtmf_payload: Optional[int] = 101,
    session_id: Optional[int] = None,
    ptime: int = 20,
    direction: str = "sendrecv",
) -> bytes:
    import time

    sid = session_id or int(time.time())
    payload = CODEC_PAYLOADS[encoding.upper()]
    payload_list = [str(payload)]
    if dtmf_payload is not None:
        payload_list.append(str(dtmf_payload))

    lines = [
        "v=0",
        f"o=- {sid} {sid} IN IP4 {address}",
        "s=helpdesk",
        f"c=IN IP4 {address}",
        "t=0 0",
        f"m=audio {port} RTP/AVP {' '.join(payload_list)}",
        f"a=rtpmap:{payload} {encoding.upper()}/8000",
    ]
    if dtmf_payload is not None:
        lines.append(f"a=rtpmap:{dtmf_payload} telephone-event/8000")
        lines.append(f"a=fmtp:{dtmf_payload} 0-16")
    lines.append(f"a=ptime:{ptime}")
    lines.append(f"a={direction}")
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")
