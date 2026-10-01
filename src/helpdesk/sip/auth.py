"""HTTP Digest authentication for SIP (RFC 3261 §22, RFC 2617/7616).

PBXes challenge REGISTER and sometimes INVITE/REFER.  MD5 and MD5-sess are what
Asterisk, FreeSWITCH and 3CX actually send; SHA-256 is accepted too since RFC
7616 allows it and newer builds offer it.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional

from .messages import parse_params


def _hash(algorithm: str):
    algo = (algorithm or "MD5").upper()
    if algo.startswith("SHA-256"):
        return hashlib.sha256
    if algo.startswith("SHA-512-256"):
        return lambda data=b"": hashlib.new("sha512_256", data)
    return hashlib.md5


@dataclass
class Challenge:
    scheme: str
    realm: str
    nonce: str
    algorithm: str = "MD5"
    qop: str = ""
    opaque: str = ""
    stale: bool = False
    header: str = "WWW-Authenticate"

    @property
    def response_header(self) -> str:
        return (
            "Proxy-Authorization"
            if self.header.lower() == "proxy-authenticate"
            else "Authorization"
        )


def parse_challenge(value: str, header: str = "WWW-Authenticate") -> Optional[Challenge]:
    value = value.strip()
    if not value:
        return None
    scheme, _, rest = value.partition(" ")
    if scheme.lower() != "digest":
        return None
    params: Dict[str, str] = {}
    for match in re.finditer(r'(\w[\w-]*)\s*=\s*(?:"([^"]*)"|([^,\s]+))', rest):
        params[match.group(1).lower()] = match.group(2) if match.group(2) is not None else match.group(3)
    if "realm" not in params or "nonce" not in params:
        return None
    return Challenge(
        scheme="Digest",
        realm=params["realm"],
        nonce=params["nonce"],
        algorithm=params.get("algorithm", "MD5"),
        qop=params.get("qop", ""),
        opaque=params.get("opaque", ""),
        stale=params.get("stale", "false").lower() == "true",
        header=header,
    )


class DigestAuth:
    """Stateful credential holder; tracks the nonce count per challenge."""

    def __init__(self, username: str, password: str, auth_username: Optional[str] = None) -> None:
        self.username = auth_username or username
        self.password = password
        self._nc = 0
        self._cnonce = os.urandom(8).hex()

    def reset_nonce(self) -> None:
        self._nc = 0
        self._cnonce = os.urandom(8).hex()

    def authorization(self, challenge: Challenge, method: str, uri: str, body: bytes = b"") -> str:
        hashfn = _hash(challenge.algorithm)

        def h(data: str) -> str:
            return hashfn(data.encode("utf-8")).hexdigest()

        ha1 = h(f"{self.username}:{challenge.realm}:{self.password}")
        if challenge.algorithm.upper().endswith("-SESS"):
            ha1 = h(f"{ha1}:{challenge.nonce}:{self._cnonce}")

        qop_options = [opt.strip().lower() for opt in challenge.qop.split(",") if opt.strip()]
        use_qop = "auth-int" if "auth-int" in qop_options and "auth" not in qop_options else (
            "auth" if "auth" in qop_options else ""
        )

        if use_qop == "auth-int":
            ha2 = h(f"{method}:{uri}:{hashfn(body).hexdigest()}")
        else:
            ha2 = h(f"{method}:{uri}")

        parts = [
            f'username="{self.username}"',
            f'realm="{challenge.realm}"',
            f'nonce="{challenge.nonce}"',
            f'uri="{uri}"',
        ]
        if use_qop:
            self._nc += 1
            nc = f"{self._nc:08x}"
            response = h(f"{ha1}:{challenge.nonce}:{nc}:{self._cnonce}:{use_qop}:{ha2}")
            parts += [f'response="{response}"', f"qop={use_qop}", f"nc={nc}", f'cnonce="{self._cnonce}"']
        else:
            response = h(f"{ha1}:{challenge.nonce}:{ha2}")
            parts.append(f'response="{response}"')

        if challenge.algorithm:
            parts.append(f"algorithm={challenge.algorithm}")
        if challenge.opaque:
            parts.append(f'opaque="{challenge.opaque}"')
        return "Digest " + ", ".join(parts)


def challenge_from_response(message) -> Optional[Challenge]:
    for header in ("www-authenticate", "proxy-authenticate"):
        value = message.get(header)
        if value:
            canonical = "Proxy-Authenticate" if header == "proxy-authenticate" else "WWW-Authenticate"
            challenge = parse_challenge(value, canonical)
            if challenge:
                return challenge
    return None
