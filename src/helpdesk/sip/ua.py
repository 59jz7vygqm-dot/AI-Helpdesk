"""SIP user agent: registers with the PBX, answers calls, transfers them away.

Scope is an endpoint, not a proxy: one registration, inbound calls only, plus an
outbound REFER so the PBX can move a call the bot cannot handle to a human.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import messages as sipmsg
from . import sdp as sdplib
from .auth import Challenge, DigestAuth, challenge_from_response
from .messages import SipMessage, build_response
from .rtp import RtpSession, open_rtp_session
from .transport import SipTransport, create_transport, detect_local_ip

log = logging.getLogger(__name__)

USER_AGENT = "ai-helpdesk/1.0"

#: touched while the registration is alive so the container healthcheck can see it
HEARTBEAT_PATH = os.environ.get("HELPDESK_HEARTBEAT", "/tmp/helpdesk-registered")


class CallState(enum.Enum):
    INCOMING = "incoming"
    RINGING = "ringing"
    ANSWERED = "answered"
    TRANSFERRING = "transferring"
    ENDED = "ended"


class TransferError(RuntimeError):
    pass


class TransferMethod(enum.Enum):
    #: SIP REFER: the clean way, but a managed PBX may have allow_transfer=no
    REFER = "refer"
    #: DTMF feature code (Asterisk features.conf blindxfer), which survives
    #: allow_transfer=no because the PBX, not the endpoint, performs the transfer
    DTMF = "dtmf"
    #: try REFER, fall back to DTMF when the PBX refuses it
    AUTO = "auto"


@dataclass
class SipAccount:
    username: str
    password: str
    domain: str
    server_host: str
    server_port: int = 5060
    auth_username: Optional[str] = None
    display_name: str = "AI Helpdesk"
    register_expires: int = 300
    bind_host: str = "0.0.0.0"
    bind_port: int = 5060
    #: public/LAN address announced in Contact and SDP; auto-detected when empty
    advertise_host: str = ""
    rtp_port_range: Tuple[int, int] = (16000, 16200)
    codec_preference: List[str] = field(default_factory=lambda: ["PCMA", "PCMU"])
    trace: bool = False


@dataclass
class Call:
    """One answered dialog plus its media."""

    call_id: str
    local_tag: str
    remote_tag: Optional[str]
    from_header: str
    to_header: str
    remote_target: str
    route_set: List[str]
    caller_number: str
    caller_name: str
    dialled_number: str
    invite: SipMessage
    source: Tuple[str, int]

    state: CallState = CallState.INCOMING
    rtp: Optional[RtpSession] = None
    local_cseq: int = 1
    started_at: float = field(default_factory=time.monotonic)
    on_dtmf: Optional[Callable[[str], None]] = None
    on_audio: Optional[Callable[[np.ndarray], None]] = None
    on_end: Optional[Callable[[str], None]] = None
    end_reason: str = ""
    _ended = False

    @property
    def active(self) -> bool:
        return self.state in (CallState.ANSWERED, CallState.TRANSFERRING)

    @property
    def duration(self) -> float:
        return time.monotonic() - self.started_at


class SipUserAgent:
    """Registers one account and dispatches inbound calls to a handler."""

    def __init__(
        self,
        account: SipAccount,
        on_incoming_call: Callable[[Call], Awaitable[None]],
        *,
        max_concurrent_calls: int = 1,
    ) -> None:
        self.account = account
        self.on_incoming_call = on_incoming_call
        self.max_concurrent_calls = max_concurrent_calls

        self.auth = DigestAuth(account.username, account.password, account.auth_username)
        self.transport: Optional[SipTransport] = None
        self.local_ip = account.advertise_host or ""
        self.registered = False
        self.calls: Dict[str, Call] = {}

        self._server_addr: Tuple[str, int] = (account.server_host, account.server_port)
        self._register_cseq = 0
        self._register_call_id = ""
        self._register_tag = sipmsg.new_tag()
        self._register_challenge: Optional[Challenge] = None
        self._register_task: Optional[asyncio.Task] = None
        self._stopping = False
        #: in-flight non-INVITE client transactions awaiting a final response
        self._pending: Dict[str, asyncio.Future] = {}
        self._handled_invites: Dict[str, float] = {}
        self._call_tasks: Dict[str, asyncio.Task] = {}

    # ---- lifecycle -----------------------------------------------------
    async def start(self) -> None:
        acct = self.account
        if not self.local_ip:
            self.local_ip = detect_local_ip(acct.server_host, acct.server_port)
        log.info("SIP local address %s:%s -> PBX %s:%s", self.local_ip, acct.bind_port, acct.server_host, acct.server_port)
        self.transport = await create_transport(
            acct.bind_host, acct.bind_port, self._on_message, trace=acct.trace
        )
        self._register_call_id = sipmsg.new_call_id(self.local_ip)
        self._register_task = asyncio.ensure_future(self._register_loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._register_task:
            self._register_task.cancel()
            try:
                await self._register_task
            except asyncio.CancelledError:
                pass
        for call in list(self.calls.values()):
            try:
                await self.hangup(call, reason="shutdown")
            except Exception:  # pragma: no cover
                log.debug("hangup during shutdown failed", exc_info=True)
        if self.registered:
            with contextlib.suppress(OSError):
                os.unlink(HEARTBEAT_PATH)
            try:
                await self._send_register(expires=0)
            except Exception:  # pragma: no cover
                log.debug("de-registration failed", exc_info=True)
        if self.transport:
            self.transport.close()

    @property
    def contact(self) -> str:
        acct = self.account
        return f'"{acct.display_name}" <sip:{acct.username}@{self.local_ip}:{acct.bind_port}>'

    @property
    def contact_uri(self) -> str:
        return f"sip:{self.account.username}@{self.local_ip}:{self.account.bind_port}"

    # ---- registration --------------------------------------------------
    async def _register_loop(self) -> None:
        backoff = 2
        while not self._stopping:
            try:
                expires = await self._send_register(expires=self.account.register_expires)
                self.registered = True
                self._touch_heartbeat()
                backoff = 2
                # Refresh well before expiry so a lost packet does not unregister us.
                sleep_for = max(30, int(expires * 0.75))
                log.info("registered as %s, refreshing in %ss", self.account.username, sleep_for)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.registered = False
                sleep_for = backoff
                backoff = min(60, backoff * 2)
                log.error("registration failed (%s), retrying in %ss", exc, sleep_for)
            await asyncio.sleep(sleep_for)

    def _touch_heartbeat(self) -> None:
        try:
            with open(HEARTBEAT_PATH, "w", encoding="utf-8") as handle:
                handle.write(f"{time.time():.0f} {self.account.username}\n")
        except OSError:  # pragma: no cover - read-only /tmp
            log.debug("could not write heartbeat to %s", HEARTBEAT_PATH, exc_info=True)

    async def _send_register(self, expires: int) -> int:
        acct = self.account
        self._register_cseq += 1
        request = self._build_request(
            "REGISTER",
            f"sip:{acct.domain}",
            call_id=self._register_call_id,
            cseq=self._register_cseq,
            from_header=f'"{acct.display_name}" <sip:{acct.username}@{acct.domain}>;tag={self._register_tag}',
            to_header=f"<sip:{acct.username}@{acct.domain}>",
        )
        request.set("Contact", f"{self.contact};expires={expires}")
        request.set("Expires", str(expires))
        if self._register_challenge:
            request.set(
                self._register_challenge.response_header,
                self.auth.authorization(self._register_challenge, "REGISTER", f"sip:{acct.domain}"),
            )

        response = await self._transact(request)
        if response.status in (401, 407):
            challenge = challenge_from_response(response)
            if not challenge:
                raise RuntimeError(f"{response.status} without a usable challenge")
            self._register_challenge = challenge
            self.auth.reset_nonce()
            self._register_cseq += 1
            retry = self._build_request(
                "REGISTER",
                f"sip:{acct.domain}",
                call_id=self._register_call_id,
                cseq=self._register_cseq,
                from_header=f'"{acct.display_name}" <sip:{acct.username}@{acct.domain}>;tag={self._register_tag}',
                to_header=f"<sip:{acct.username}@{acct.domain}>",
            )
            retry.set("Contact", f"{self.contact};expires={expires}")
            retry.set("Expires", str(expires))
            retry.set(
                challenge.response_header,
                self.auth.authorization(challenge, "REGISTER", f"sip:{acct.domain}"),
            )
            response = await self._transact(retry)

        if response.status // 100 != 2:
            raise RuntimeError(f"REGISTER rejected: {response.status} {response.reason}")

        granted = expires
        contact = response.get("contact")
        if contact:
            params = sipmsg.parse_params(contact)
            if "expires" in params and params["expires"].isdigit():
                granted = int(params["expires"])
        header_expires = response.get("expires")
        if header_expires and header_expires.isdigit():
            granted = int(header_expires)
        return granted or expires

    # ---- request plumbing ---------------------------------------------
    def _build_request(
        self,
        method: str,
        uri: str,
        *,
        call_id: str,
        cseq: int,
        from_header: str,
        to_header: str,
        route_set: Optional[List[str]] = None,
        branch: Optional[str] = None,
    ) -> SipMessage:
        msg = SipMessage(is_request=True, method=method, uri=uri)
        msg.set(
            "Via",
            f"SIP/2.0/UDP {self.local_ip}:{self.account.bind_port};rport;branch={branch or sipmsg.new_branch()}",
        )
        msg.set("Max-Forwards", "70")
        msg.set("From", from_header)
        msg.set("To", to_header)
        msg.set("Call-ID", call_id)
        msg.set("CSeq", f"{cseq} {method}")
        msg.set("User-Agent", USER_AGENT)
        for route in route_set or []:
            msg.add("Route", route)
        msg.set("Content-Length", "0")
        return msg

    async def _transact(self, request: SipMessage, timeout: float = 8.0) -> SipMessage:
        """Send a non-INVITE request and wait for its final response.

        Retransmits on timer E/F intervals, which matters on a lossy LAN.
        """
        assert self.transport is not None
        branch = request.top_via_branch() or sipmsg.new_branch()
        key = f"{branch}:{request.method}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[key] = future
        try:
            delay = 0.5
            waited = 0.0
            self.transport.send(request, self._server_addr)
            while True:
                try:
                    return await asyncio.wait_for(asyncio.shield(future), timeout=delay)
                except asyncio.TimeoutError:
                    waited += delay
                    if waited >= timeout:
                        raise asyncio.TimeoutError(f"no response to {request.method}")
                    delay = min(4.0, delay * 2)
                    self.transport.send(request, self._server_addr)
        finally:
            self._pending.pop(key, None)

    # ---- message dispatch ---------------------------------------------
    def _on_message(self, message: SipMessage, addr: Tuple[str, int]) -> None:
        if message.is_request:
            self._on_request(message, addr)
        else:
            self._on_response(message, addr)

    def _on_response(self, message: SipMessage, addr: Tuple[str, int]) -> None:
        if message.status < 200:
            return  # provisional; nothing here waits on 1xx
        branch = message.top_via_branch()
        key = f"{branch}:{message.cseq_method}"
        future = self._pending.get(key)
        if future and not future.done():
            future.set_result(message)

    def _on_request(self, message: SipMessage, addr: Tuple[str, int]) -> None:
        method = message.method
        handler = {
            "INVITE": self._on_invite,
            "ACK": self._on_ack,
            "BYE": self._on_bye,
            "CANCEL": self._on_cancel,
            "OPTIONS": self._on_options,
            "INFO": self._on_info,
            "NOTIFY": self._on_notify,
            "UPDATE": self._on_update,
        }.get(method)
        if handler is None:
            self._respond(message, addr, 501, "Not Implemented")
            return
        handler(message, addr)

    def _respond(self, request: SipMessage, addr: Tuple[str, int], status: int, reason: str, **kwargs) -> None:
        assert self.transport is not None
        extra = list(kwargs.pop("extra", []) or [])
        extra.append(("User-Agent", USER_AGENT))
        response = build_response(request, status, reason, extra=extra, **kwargs)
        self.transport.send(response, addr)

    # ---- inbound call --------------------------------------------------
    def _on_invite(self, message: SipMessage, addr: Tuple[str, int]) -> None:
        call_id = message.call_id
        existing = self.calls.get(call_id)

        if existing and existing.remote_tag == message.from_tag:
            # Re-INVITE inside an established dialog (hold, codec change, or a
            # retransmission).  Answer with our current media and keep going.
            self._answer_reinvite(existing, message, addr)
            return

        now = time.monotonic()
        self._handled_invites = {k: v for k, v in self._handled_invites.items() if now - v < 32}
        if call_id in self._handled_invites:
            return  # retransmission of an INVITE we already declined

        if len(self.calls) >= self.max_concurrent_calls:
            log.info("rejecting call %s: %d call(s) already active", call_id, len(self.calls))
            self._handled_invites[call_id] = now
            self._respond(message, addr, 486, "Busy Here", to_tag=sipmsg.new_tag())
            return

        from_header = message.get("from", "") or ""
        to_header = message.get("to", "") or ""
        from_uri = sipmsg.parse_uri(from_header)
        contact = message.get("contact")
        remote_target = sipmsg.parse_uri(contact) if contact else from_uri

        # Record-Route is top-down in the request; the route set for our
        # responses/in-dialog requests is the reverse (RFC 3261 §12.1.1).
        route_set = list(reversed(message.get_all("record-route")))

        call = Call(
            call_id=call_id,
            local_tag=sipmsg.new_tag(),
            remote_tag=message.from_tag,
            from_header=from_header,
            to_header=to_header,
            remote_target=remote_target,
            route_set=route_set,
            caller_number=sipmsg.uri_user(from_uri),
            caller_name=sipmsg.parse_display_name(from_header),
            dialled_number=sipmsg.uri_user(sipmsg.parse_uri(to_header)),
            invite=message,
            source=addr,
        )
        self.calls[call_id] = call
        log.info(
            "incoming call %s from %s (%s) to %s",
            call_id,
            call.caller_number,
            call.caller_name or "no name",
            call.dialled_number,
        )

        self._respond(message, addr, 100, "Trying")
        self._call_tasks[call_id] = asyncio.ensure_future(self._run_call(call))

    async def _run_call(self, call: Call) -> None:
        try:
            await self.on_incoming_call(call)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("call handler crashed for %s", call.call_id)
            try:
                await self.hangup(call, reason="handler-error")
            except Exception:  # pragma: no cover
                pass
        finally:
            self._call_tasks.pop(call.call_id, None)

    def ring(self, call: Call) -> None:
        if call.state is not CallState.INCOMING:
            return
        call.state = CallState.RINGING
        self._respond(
            call.invite,
            call.source,
            180,
            "Ringing",
            to_tag=call.local_tag,
            contact=self.contact,
        )

    async def answer(
        self,
        call: Call,
        *,
        on_audio: Callable[[np.ndarray], None],
        on_dtmf: Optional[Callable[[str], None]] = None,
        on_sent: Optional[Callable[[Optional[np.ndarray]], None]] = None,
    ) -> RtpSession:
        """Send 200 OK with our SDP answer and start media."""
        if call.state not in (CallState.INCOMING, CallState.RINGING):
            raise RuntimeError(f"cannot answer call in state {call.state}")

        offer = sdplib.parse_sdp(call.invite.body)
        audio = offer.audio
        if audio is None:
            self._respond(call.invite, call.source, 488, "Not Acceptable Here", to_tag=call.local_tag)
            raise RuntimeError("INVITE carried no audio media")

        chosen = sdplib.select_codec(audio, self.account.codec_preference)
        if chosen is None:
            self._respond(call.invite, call.source, 488, "Not Acceptable Here", to_tag=call.local_tag)
            raise RuntimeError(f"no common codec; offer had {audio.payloads}")
        payload_type, encoding = chosen
        dtmf_payload = sdplib.find_dtmf_payload(audio)

        call.on_audio = on_audio
        call.on_dtmf = on_dtmf
        rtp = RtpSession(
            encoding,
            payload_type,
            on_audio=on_audio,
            on_dtmf=on_dtmf,
            on_sent=on_sent,
            dtmf_payload=dtmf_payload,
        )
        local_port = await open_rtp_session(
            self.account.bind_host if self.account.bind_host != "0.0.0.0" else "0.0.0.0",
            self.account.rtp_port_range,
            rtp,
        )
        remote_host = sdplib.media_address(offer, audio)
        rtp.set_remote(remote_host, audio.port)
        rtp.start()
        call.rtp = rtp

        body = sdplib.build_sdp(
            self.local_ip,
            local_port,
            encoding,
            dtmf_payload=dtmf_payload,
        )
        self._respond(
            call.invite,
            call.source,
            200,
            "OK",
            to_tag=call.local_tag,
            contact=self.contact,
            body=body,
            content_type="application/sdp",
            extra=[("Allow", "INVITE, ACK, BYE, CANCEL, OPTIONS, INFO, NOTIFY, REFER, UPDATE")],
        )
        call.state = CallState.ANSWERED
        call.started_at = time.monotonic()
        log.info(
            "answered %s with %s (local rtp %s, remote %s:%s, dtmf pt %s)",
            call.call_id,
            encoding,
            local_port,
            remote_host,
            audio.port,
            dtmf_payload,
        )
        return rtp

    def _answer_reinvite(self, call: Call, message: SipMessage, addr: Tuple[str, int]) -> None:
        if call.rtp is None:
            self._respond(message, addr, 491, "Request Pending", to_tag=call.local_tag)
            return
        offer = sdplib.parse_sdp(message.body)
        audio = offer.audio
        if audio is not None and audio.port:
            host = sdplib.media_address(offer, audio)
            if (host, audio.port) != call.rtp.remote:
                call.rtp.set_remote(host, audio.port)
        on_hold = audio is not None and audio.direction in ("sendonly", "inactive")
        local_port = call.rtp.transport.get_extra_info("sockname")[1] if call.rtp.transport else 0
        body = sdplib.build_sdp(
            self.local_ip,
            local_port,
            call.rtp.encoding,
            dtmf_payload=call.rtp.dtmf_payload,
            direction="recvonly" if on_hold else "sendrecv",
        )
        contact = message.get("contact")
        if contact:
            call.remote_target = sipmsg.parse_uri(contact)
        log.info("re-INVITE on %s (%s)", call.call_id, "hold" if on_hold else "resume")
        self._respond(
            message,
            addr,
            200,
            "OK",
            to_tag=call.local_tag,
            contact=self.contact,
            body=body,
            content_type="application/sdp",
        )

    def _on_ack(self, message: SipMessage, addr: Tuple[str, int]) -> None:
        log.debug("ACK for %s", message.call_id)

    def _on_bye(self, message: SipMessage, addr: Tuple[str, int]) -> None:
        call = self.calls.get(message.call_id)
        self._respond(message, addr, 200, "OK", to_tag=call.local_tag if call else sipmsg.new_tag())
        if call:
            log.info("caller hung up %s after %.1fs", call.call_id, call.duration)
            asyncio.ensure_future(self._finish_call(call, "remote-bye"))

    def _on_cancel(self, message: SipMessage, addr: Tuple[str, int]) -> None:
        call = self.calls.get(message.call_id)
        self._respond(message, addr, 200, "OK", to_tag=sipmsg.new_tag())
        if call and call.state in (CallState.INCOMING, CallState.RINGING):
            # The INVITE transaction must be closed with 487 (RFC 3261 §9.2).
            self._respond(call.invite, call.source, 487, "Request Terminated", to_tag=call.local_tag)
            log.info("caller cancelled %s before answer", call.call_id)
            asyncio.ensure_future(self._finish_call(call, "cancelled"))

    def _on_options(self, message: SipMessage, addr: Tuple[str, int]) -> None:
        self._respond(
            message,
            addr,
            200,
            "OK",
            to_tag=sipmsg.new_tag(),
            extra=[
                ("Allow", "INVITE, ACK, BYE, CANCEL, OPTIONS, INFO, NOTIFY, REFER, UPDATE"),
                ("Accept", "application/sdp"),
            ],
        )

    def _on_info(self, message: SipMessage, addr: Tuple[str, int]) -> None:
        call = self.calls.get(message.call_id)
        self._respond(message, addr, 200, "OK", to_tag=call.local_tag if call else sipmsg.new_tag())
        # Some PBXes signal DTMF out-of-band as application/dtmf-relay.
        if call and call.on_dtmf and (message.get("content-type") or "").lower().startswith("application/dtmf"):
            text = message.body.decode("utf-8", "replace")
            for line in text.splitlines():
                if line.lower().startswith("signal="):
                    digit = line.split("=", 1)[1].strip()
                    if digit:
                        call.on_dtmf(digit[0])

    def _on_update(self, message: SipMessage, addr: Tuple[str, int]) -> None:
        call = self.calls.get(message.call_id)
        if call and call.rtp and message.body:
            self._answer_reinvite(call, message, addr)
            return
        self._respond(message, addr, 200, "OK", to_tag=call.local_tag if call else sipmsg.new_tag())

    def _on_notify(self, message: SipMessage, addr: Tuple[str, int]) -> None:
        """NOTIFY carries the progress of a REFER we sent."""
        call = self.calls.get(message.call_id)
        self._respond(message, addr, 200, "OK", to_tag=call.local_tag if call else sipmsg.new_tag())
        event = (message.get("event") or "").lower()
        if not event.startswith("refer"):
            return
        body = message.body.decode("utf-8", "replace").strip()
        state = sipmsg.parse_params(message.get("subscription-state") or "").get("reason", "")
        log.info("REFER progress on %s: %s (%s)", message.call_id, body.splitlines()[:1], state or "pending")
        if call and call.state is CallState.TRANSFERRING:
            first_line = body.split("\r\n")[0] if body else ""
            if " 2" in first_line:
                # Transfer accepted by the target; the PBX will take the call away.
                asyncio.ensure_future(self._finish_call(call, "transferred"))
            elif first_line and any(code in first_line for code in (" 4", " 5", " 6")):
                log.warning("transfer of %s failed at the target: %s", call.call_id, first_line)
                call.state = CallState.ANSWERED
                call.end_reason = "transfer-rejected"

    # ---- outbound in-dialog requests ----------------------------------
    def _dialog_request(self, call: Call, method: str) -> SipMessage:
        call.local_cseq += 1
        # Roles swap for requests we originate: our To is the caller's From.
        from_header = f"{self.contact.split(' <')[0]} <sip:{self.account.username}@{self.account.domain}>;tag={call.local_tag}"
        to_header = call.from_header
        if call.remote_tag and not sipmsg.parse_tag(to_header):
            to_header = f"{to_header};tag={call.remote_tag}"
        return self._build_request(
            method,
            call.remote_target,
            call_id=call.call_id,
            cseq=call.local_cseq,
            from_header=from_header,
            to_header=to_header,
            route_set=call.route_set,
        )

    def _dialog_target(self, call: Call) -> Tuple[str, int]:
        """Where to actually send in-dialog requests.

        With a route set the first route decides; otherwise the remote target.
        Falling back to the PBX address is the safe choice for a registered
        endpoint behind NAT.
        """
        if call.route_set:
            uri = sipmsg.parse_uri(call.route_set[0])
        else:
            uri = call.remote_target
        host = sipmsg.uri_host(uri)
        port = 5060
        hostport = uri.split("@")[-1] if "@" in uri else uri.split(":", 1)[-1]
        if ":" in hostport:
            tail = hostport.rsplit(":", 1)[-1].split(";")[0]
            if tail.isdigit():
                port = int(tail)
        if not host:
            return self._server_addr
        return (host, port)

    async def hangup(self, call: Call, reason: str = "") -> None:
        if call.state in (CallState.ENDED,):
            return
        if call.state in (CallState.INCOMING, CallState.RINGING):
            self._respond(call.invite, call.source, 603, "Decline", to_tag=call.local_tag)
            await self._finish_call(call, reason or "declined")
            return
        request = self._dialog_request(call, "BYE")
        if reason:
            request.set("X-Reason", reason[:120])
        target = self._dialog_target(call)
        assert self.transport is not None
        try:
            self.transport.send(request, target)
        except Exception:  # pragma: no cover
            log.debug("sending BYE failed", exc_info=True)
        log.info("hung up %s (%s) after %.1fs", call.call_id, reason or "no reason", call.duration)
        await self._finish_call(call, reason or "local-bye")

    async def transfer(
        self,
        call: Call,
        target_number: str,
        *,
        method: TransferMethod = TransferMethod.AUTO,
        timeout: float = 6.0,
        feature_code: str = "##",
        dtmf_delay_ms: int = 700,
        dtmf_terminator: str = "",
        dtmf_settle_s: float = 4.0,
    ) -> str:
        """Hand the call to a human.  Returns the method that succeeded.

        On a PBX you administer yourself, REFER is the right answer.  On a
        managed Asterisk it may be switched off per endpoint
        (``allow_transfer=no``), and then only the PBX's own feature code works --
        hence AUTO, which tries REFER first and falls back.
        """
        if method is TransferMethod.REFER:
            await self.transfer_refer(call, target_number, timeout=timeout)
            return "refer"
        if method is TransferMethod.DTMF:
            await self.transfer_dtmf(
                call, target_number, feature_code=feature_code,
                delay_ms=dtmf_delay_ms, terminator=dtmf_terminator, settle_s=dtmf_settle_s,
            )
            return "dtmf"

        try:
            await self.transfer_refer(call, target_number, timeout=timeout)
            return "refer"
        except TransferError as exc:
            log.warning("REFER refused (%s); falling back to the DTMF feature code", exc)
            await self.transfer_dtmf(
                call, target_number, feature_code=feature_code,
                delay_ms=dtmf_delay_ms, terminator=dtmf_terminator, settle_s=dtmf_settle_s,
            )
            return "dtmf"

    async def transfer_dtmf(
        self,
        call: Call,
        target_number: str,
        *,
        feature_code: str = "##",
        delay_ms: int = 700,
        terminator: str = "",
        settle_s: float = 4.0,
    ) -> None:
        """Blind transfer by dialling the PBX's in-call feature code.

        Asterisk watches the audio stream for the ``blindxfer`` sequence from
        features.conf, then collects the destination as further DTMF.  The
        endpoint never signals a transfer, so this works where REFER is denied --
        provided the channel was dialled with the ``t`` option, which FreePBX
        sets for extensions by default.
        """
        if not call.active:
            raise TransferError(f"call is {call.state.value}, cannot transfer")
        if call.rtp is None:
            raise TransferError("no media; cannot send DTMF")
        if call.rtp.dtmf_payload is None:
            raise TransferError(
                "the PBX offered no telephone-event payload, so DTMF transfer is "
                "impossible; ask the provider to enable REFER instead"
            )

        call.state = CallState.TRANSFERRING
        rtp = call.rtp
        # Anything still queued would be interleaved with the tones.
        rtp.clear_playout()

        log.info(
            "DTMF transfer of %s: feature code %r then %s",
            call.call_id, feature_code, target_number,
        )
        rtp.send_dtmf(feature_code)
        await self._await_dtmf(rtp, timeout=3.0)
        # Give the PBX time to recognise the code and start collecting digits.
        await asyncio.sleep(delay_ms / 1000)
        rtp.send_dtmf(target_number + terminator)
        await self._await_dtmf(rtp, timeout=6.0)

        # A successful blind transfer ends our leg: wait for the BYE.  If it
        # never comes the feature code was not accepted and the caller is still
        # on the line with us.
        deadline = time.monotonic() + settle_s
        while time.monotonic() < deadline:
            if call.state is CallState.ENDED or call.call_id not in self.calls:
                log.info("DTMF transfer of %s accepted by the PBX", call.call_id)
                return
            await asyncio.sleep(0.1)
        call.state = CallState.ANSWERED
        raise TransferError(
            f"PBX did not act on the feature code {feature_code!r} within {settle_s}s "
            "(wrong code, or in-call transfers are disabled for this extension)"
        )

    @staticmethod
    async def _await_dtmf(rtp: RtpSession, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while rtp.dtmf_pending() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)

    async def transfer_refer(self, call: Call, target_number: str, timeout: float = 6.0) -> None:
        """Blind transfer via SIP REFER.

        REFER hands the call to the PBX dialplan, so the target can be an
        extension, a queue, or an external number.
        """
        if not call.active:
            raise TransferError(f"call is {call.state.value}, cannot transfer")

        target = target_number.strip()
        refer_to = target if target.startswith("sip:") else f"sip:{target}@{self.account.domain}"

        request = self._dialog_request(call, "REFER")
        request.set("Refer-To", f"<{refer_to}>")
        request.set("Referred-By", f"<sip:{self.account.username}@{self.account.domain}>")
        request.set("Contact", self.contact)
        target_addr = self._dialog_target(call)

        assert self.transport is not None
        branch = request.top_via_branch()
        key = f"{branch}:REFER"
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[key] = future
        call.state = CallState.TRANSFERRING
        try:
            self.transport.send(request, target_addr)
            try:
                response = await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError as exc:
                call.state = CallState.ANSWERED
                raise TransferError(f"no response to REFER within {timeout}s") from exc

            if response.status in (401, 407):
                challenge = challenge_from_response(response)
                if challenge:
                    self.auth.reset_nonce()
                    retry = self._dialog_request(call, "REFER")
                    retry.set("Refer-To", f"<{refer_to}>")
                    retry.set("Referred-By", f"<sip:{self.account.username}@{self.account.domain}>")
                    retry.set("Contact", self.contact)
                    retry.set(
                        challenge.response_header,
                        self.auth.authorization(challenge, "REFER", call.remote_target),
                    )
                    retry_branch = retry.top_via_branch()
                    retry_key = f"{retry_branch}:REFER"
                    retry_future: asyncio.Future = loop.create_future()
                    self._pending[retry_key] = retry_future
                    try:
                        self.transport.send(retry, target_addr)
                        response = await asyncio.wait_for(retry_future, timeout=timeout)
                    finally:
                        self._pending.pop(retry_key, None)

            if response.status // 100 != 2:
                call.state = CallState.ANSWERED
                raise TransferError(f"REFER rejected: {response.status} {response.reason}")

            log.info("REFER to %s accepted for %s (%s)", refer_to, call.call_id, response.status)
        finally:
            self._pending.pop(key, None)

    async def _finish_call(self, call: Call, reason: str) -> None:
        if call._ended:
            return
        call._ended = True
        call.state = CallState.ENDED
        call.end_reason = call.end_reason or reason
        if call.rtp:
            await call.rtp.close()
            call.rtp = None
        self.calls.pop(call.call_id, None)
        if call.on_end:
            try:
                call.on_end(call.end_reason)
            except Exception:  # pragma: no cover
                log.debug("on_end callback failed", exc_info=True)
