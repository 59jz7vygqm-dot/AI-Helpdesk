"""Minimal SIP message parsing and building (RFC 3261 subset).

Only what a registering endpoint needs: REGISTER, inbound INVITE/ACK/BYE/CANCEL,
OPTIONS keepalives, re-INVITE, and outbound REFER for the transfer.  Keeping this
in-process (instead of binding pjsua2) means every RTP frame stays in Python with
no cross-language hop, and the image needs no native build.
"""

from __future__ import annotations

import random
import re
import string
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

CRLF = "\r\n"

#: compact header forms (RFC 3261 §7.3.3) mapped to their long names
COMPACT_HEADERS = {
    "i": "Call-ID",
    "m": "Contact",
    "e": "Content-Encoding",
    "l": "Content-Length",
    "c": "Content-Type",
    "f": "From",
    "s": "Subject",
    "k": "Supported",
    "t": "To",
    "v": "Via",
    "r": "Refer-To",
    "b": "Referred-By",
    "x": "Session-Expires",
}

#: headers that may legitimately appear more than once
MULTI_HEADERS = {"via", "record-route", "route", "contact", "supported", "allow"}


#: headers whose conventional spelling is not simple title-case.  The RFC says
#: header names are case-insensitive, but enough deployed stacks compare them
#: literally that it is not worth the risk.
SPECIAL_CASE = {
    "call-id": "Call-ID",
    "cseq": "CSeq",
    "www-authenticate": "WWW-Authenticate",
    "min-se": "Min-SE",
    "mime-version": "MIME-Version",
    "rseq": "RSeq",
    "rack": "RAck",
    "p-asserted-identity": "P-Asserted-Identity",
    "p-preferred-identity": "P-Preferred-Identity",
    "refer-to": "Refer-To",
    "referred-by": "Referred-By",
    "user-agent": "User-Agent",
    "max-forwards": "Max-Forwards",
    "record-route": "Record-Route",
    "session-expires": "Session-Expires",
    "retry-after": "Retry-After",
    "x-reason": "X-Reason",
}

#: emission order for readability and for stacks that expect the usual layout
HEADER_ORDER = [
    "via",
    "max-forwards",
    "from",
    "to",
    "call-id",
    "cseq",
    "contact",
    "route",
    "record-route",
]


def _canonical(name: str) -> str:
    key = name.strip().lower()
    if key in COMPACT_HEADERS:
        return COMPACT_HEADERS[key]
    if key in SPECIAL_CASE:
        return SPECIAL_CASE[key]
    return "-".join(part.capitalize() for part in key.split("-"))


def random_token(length: int = 10) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def new_branch() -> str:
    # RFC 3261 requires the magic cookie so proxies recognise the transaction id.
    return "z9hG4bK" + random_token(16)


def new_call_id(host: str) -> str:
    return f"{uuid.uuid4().hex}@{host}"


def new_tag() -> str:
    return random_token(12)


@dataclass
class SipMessage:
    is_request: bool
    method: str = ""
    uri: str = ""
    status: int = 0
    reason: str = ""
    version: str = "SIP/2.0"
    headers: Dict[str, List[str]] = field(default_factory=dict)
    body: bytes = b""

    # ---- header access -------------------------------------------------
    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        values = self.headers.get(name.lower())
        return values[0] if values else default

    def get_all(self, name: str) -> List[str]:
        return list(self.headers.get(name.lower(), ()))

    def set(self, name: str, value: str) -> None:
        self.headers[name.lower()] = [value]

    def add(self, name: str, value: str) -> None:
        self.headers.setdefault(name.lower(), []).append(value)

    # ---- derived fields ------------------------------------------------
    @property
    def call_id(self) -> str:
        return (self.get("call-id") or "").strip()

    @property
    def cseq_number(self) -> int:
        raw = self.get("cseq") or "0"
        try:
            return int(raw.split()[0])
        except (ValueError, IndexError):
            return 0

    @property
    def cseq_method(self) -> str:
        raw = (self.get("cseq") or "").split()
        return raw[1].upper() if len(raw) > 1 else ""

    @property
    def from_tag(self) -> Optional[str]:
        return parse_tag(self.get("from", "") or "")

    @property
    def to_tag(self) -> Optional[str]:
        return parse_tag(self.get("to", "") or "")

    def top_via_branch(self) -> Optional[str]:
        via = self.get("via")
        if not via:
            return None
        match = re.search(r"branch=([^;,\s]+)", via)
        return match.group(1) if match else None

    def encode(self) -> bytes:
        if self.is_request:
            start = f"{self.method} {self.uri} {self.version}"
        else:
            start = f"{self.version} {self.status} {self.reason}"
        lines = [start]
        def rank(key: str) -> tuple:
            try:
                return (0, HEADER_ORDER.index(key), key)
            except ValueError:
                return (1, 0, key)

        ordered = sorted(self.headers.items(), key=lambda kv: rank(kv[0]))
        for key, values in ordered:
            label = _canonical(key)
            for value in values:
                lines.append(f"{label}: {value}")
        head = CRLF.join(lines) + CRLF + CRLF
        return head.encode("utf-8") + self.body

    def __str__(self) -> str:  # pragma: no cover - debug helper
        if self.is_request:
            return f"{self.method} {self.uri}"
        return f"{self.status} {self.reason}"


class SipParseError(ValueError):
    pass


def parse(data: bytes) -> SipMessage:
    split = data.split(b"\r\n\r\n", 1)
    if len(split) == 1:
        # Be lenient: some stacks emit bare LF line endings.
        split = data.split(b"\n\n", 1)
    head = split[0].decode("utf-8", errors="replace")
    body = split[1] if len(split) > 1 else b""

    lines = [line for line in re.split(r"\r\n|\n", head) if line != ""]
    if not lines:
        raise SipParseError("empty message")

    # Unfold continuation lines (a leading space continues the previous header).
    unfolded: List[str] = [lines[0]]
    for line in lines[1:]:
        if line[:1] in (" ", "\t") and len(unfolded) > 1:
            unfolded[-1] += " " + line.strip()
        else:
            unfolded.append(line)

    start = unfolded[0].split(None, 2)
    msg = SipMessage(is_request=not unfolded[0].upper().startswith("SIP/"))
    if msg.is_request:
        if len(start) < 3:
            raise SipParseError(f"bad request line: {unfolded[0]!r}")
        msg.method, msg.uri, msg.version = start[0].upper(), start[1], start[2]
    else:
        if len(start) < 2:
            raise SipParseError(f"bad status line: {unfolded[0]!r}")
        msg.version = start[0]
        try:
            msg.status = int(start[1])
        except ValueError as exc:
            raise SipParseError(f"bad status code: {start[1]!r}") from exc
        msg.reason = start[2] if len(start) > 2 else ""

    for line in unfolded[1:]:
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        key = _canonical(name).lower()
        value = value.strip()
        if key in MULTI_HEADERS:
            # Comma-separated values are equivalent to repeated headers, but
            # splitting naively would break URIs containing commas in params.
            for part in _split_header_list(value):
                msg.headers.setdefault(key, []).append(part)
        else:
            msg.headers.setdefault(key, []).append(value)

    declared = msg.get("content-length")
    if declared is not None:
        try:
            want = int(declared)
            if 0 <= want <= len(body):
                body = body[:want]
        except ValueError:
            pass
    msg.body = body
    return msg


def _split_header_list(value: str) -> List[str]:
    """Split a comma-separated header value, respecting <> and "" quoting."""
    parts: List[str] = []
    depth = 0
    in_quotes = False
    current: List[str] = []
    for ch in value:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == "<" and not in_quotes:
            depth += 1
        elif ch == ">" and not in_quotes:
            depth = max(0, depth - 1)
        if ch == "," and depth == 0 and not in_quotes:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts or [value]


def parse_tag(header_value: str) -> Optional[str]:
    match = re.search(r";\s*tag=([^;\s]+)", header_value)
    return match.group(1) if match else None


def parse_uri(header_value: str) -> str:
    """Extract the bare URI from a name-addr or addr-spec header value."""
    match = re.search(r"<([^>]+)>", header_value)
    raw = match.group(1) if match else header_value.strip()
    # Strip any uri-parameters that are not part of the target address.
    return raw.split(";")[0].strip()


def parse_display_name(header_value: str) -> str:
    value = header_value.strip()
    match = re.match(r'^\s*"([^"]*)"', value)
    if match:
        return match.group(1)
    match = re.match(r"^\s*([^<]+)<", value)
    if match:
        return match.group(1).strip()
    return ""


def uri_user(uri: str) -> str:
    """The user part of a SIP URI, i.e. the caller number."""
    body = uri.split(":", 1)[1] if ":" in uri else uri
    body = body.split(";")[0]
    if "@" in body:
        return body.split("@", 1)[0]
    return body


def uri_host(uri: str) -> str:
    body = uri.split(":", 1)[1] if ":" in uri else uri
    body = body.split(";")[0]
    hostport = body.split("@", 1)[1] if "@" in body else body
    return hostport.split(":")[0]


def parse_params(value: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for part in value.split(";")[1:]:
        if "=" in part:
            key, _, val = part.partition("=")
            params[key.strip().lower()] = val.strip().strip('"')
        elif part.strip():
            params[part.strip().lower()] = ""
    return params


def build_response(
    request: SipMessage,
    status: int,
    reason: str,
    *,
    to_tag: Optional[str] = None,
    contact: Optional[str] = None,
    body: bytes = b"",
    content_type: Optional[str] = None,
    extra: Optional[Iterable[Tuple[str, str]]] = None,
) -> SipMessage:
    """Build a response, copying the headers RFC 3261 §8.2.6.2 requires."""
    resp = SipMessage(is_request=False, status=status, reason=reason)
    for via in request.get_all("via"):
        resp.add("Via", via)
    for rr in request.get_all("record-route"):
        resp.add("Record-Route", rr)
    resp.set("From", request.get("from", "") or "")

    to_value = request.get("to", "") or ""
    if to_tag and not parse_tag(to_value):
        to_value = f"{to_value};tag={to_tag}"
    resp.set("To", to_value)
    resp.set("Call-ID", request.call_id)
    resp.set("CSeq", request.get("cseq", "") or "")
    if contact:
        resp.set("Contact", contact)
    if extra:
        for name, value in extra:
            resp.add(name, value)
    if body:
        resp.set("Content-Type", content_type or "application/sdp")
    resp.set("Content-Length", str(len(body)))
    resp.body = body
    return resp
