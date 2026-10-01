"""End-to-end signalling test against a fake PBX.

Drives the real code path: REGISTER with a digest challenge, inbound INVITE,
200 OK with SDP, ACK, RTP both ways, DTMF, REFER transfer and the closing NOTIFY.
"""

from __future__ import annotations

import asyncio
import os
import re
import socket
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from helpdesk.sip import messages as sipmsg
from helpdesk.sip.rtp import RtpHeader
from helpdesk.sip.ua import Call, SipAccount, SipUserAgent


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakePbx(asyncio.DatagramProtocol):
    def __init__(self) -> None:
        self.transport = None
        self.received: list = []
        self.ua_addr = None
        self.register_count = 0
        self.authorized_register = asyncio.Event()
        self.got_200_for_invite = asyncio.Event()
        self.invite_answer = None
        self.refer_seen = asyncio.Event()
        self.refer_target = ""
        self.call_id = "pbxcall-1@fake"
        self.from_tag = "pbxtag1"
        self.rtp_port = free_port()
        self.rtp_sock = None
        self.rtp_from_ua = 0
        self.bye_seen = asyncio.Event()

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        self.ua_addr = addr
        msg = sipmsg.parse(data)
        self.received.append(msg)
        if msg.is_request:
            self._handle_request(msg, addr)
        else:
            self._handle_response(msg, addr)

    # -- requests from the UA
    def _handle_request(self, msg, addr):
        if msg.method == "REGISTER":
            self.register_count += 1
            if not msg.get("authorization"):
                resp = sipmsg.build_response(
                    msg, 401, "Unauthorized", to_tag="pbx-reg",
                    extra=[("WWW-Authenticate",
                            'Digest realm="fake-pbx", nonce="abc123nonce", qop="auth", algorithm=MD5')],
                )
                self.transport.sendto(resp.encode(), addr)
                return
            auth = msg.get("authorization")
            assert 'username="900"' in auth, auth
            assert "response=" in auth and "nc=" in auth, auth
            expires = msg.get("expires", "0")
            resp = sipmsg.build_response(
                msg, 200, "OK", to_tag="pbx-reg",
                contact=f"{msg.get('contact')}", extra=[("Expires", expires)],
            )
            self.transport.sendto(resp.encode(), addr)
            if expires != "0":
                self.authorized_register.set()
        elif msg.method == "REFER":
            self.refer_target = sipmsg.parse_uri(msg.get("refer-to") or "")
            resp = sipmsg.build_response(msg, 202, "Accepted", to_tag=self.from_tag)
            self.transport.sendto(resp.encode(), addr)
            self.refer_seen.set()
        elif msg.method == "BYE":
            resp = sipmsg.build_response(msg, 200, "OK", to_tag=self.from_tag)
            self.transport.sendto(resp.encode(), addr)
            self.bye_seen.set()
        elif msg.method == "OPTIONS":
            self.transport.sendto(sipmsg.build_response(msg, 200, "OK", to_tag="x").encode(), addr)

    # -- responses to our INVITE
    def _handle_response(self, msg, addr):
        if msg.cseq_method == "INVITE" and msg.status == 200:
            self.invite_answer = msg
            self.got_200_for_invite.set()
            ack = sipmsg.SipMessage(is_request=True, method="ACK",
                                    uri=sipmsg.parse_uri(msg.get("contact") or ""))
            ack.set("Via", f"SIP/2.0/UDP 127.0.0.1:{self.port};branch={sipmsg.new_branch()}")
            ack.set("From", msg.get("from"))
            ack.set("To", msg.get("to"))
            ack.set("Call-ID", msg.call_id)
            ack.set("CSeq", f"{msg.cseq_number} ACK")
            ack.set("Content-Length", "0")
            self.transport.sendto(ack.encode(), addr)

    @property
    def port(self):
        return self.transport.get_extra_info("sockname")[1]

    def send_invite(self, addr):
        sdp = (
            "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=fake\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\n"
            f"m=audio {self.rtp_port} RTP/AVP 8 0 101\r\n"
            "a=rtpmap:8 PCMA/8000\r\na=rtpmap:0 PCMU/8000\r\n"
            "a=rtpmap:101 telephone-event/8000\r\na=ptime:20\r\na=sendrecv\r\n"
        ).encode()
        inv = sipmsg.SipMessage(is_request=True, method="INVITE", uri="sip:900@127.0.0.1")
        inv.set("Via", f"SIP/2.0/UDP 127.0.0.1:{self.port};branch={sipmsg.new_branch()}")
        inv.set("From", f'"Max Mustermann" <sip:4915112345@fake-pbx>;tag={self.from_tag}')
        inv.set("To", "<sip:900@fake-pbx>")
        inv.set("Call-ID", self.call_id)
        inv.set("CSeq", "1 INVITE")
        inv.set("Contact", "<sip:4915112345@127.0.0.1:%d>" % self.port)
        inv.set("Record-Route", "<sip:127.0.0.1:%d;lr>" % self.port)
        inv.set("Content-Type", "application/sdp")
        inv.set("Content-Length", str(len(sdp)))
        inv.body = sdp
        self.transport.sendto(inv.encode(), addr)

    def send_notify(self, addr, body="SIP/2.0 200 OK"):
        n = sipmsg.SipMessage(is_request=True, method="NOTIFY", uri="sip:900@127.0.0.1")
        n.set("Via", f"SIP/2.0/UDP 127.0.0.1:{self.port};branch={sipmsg.new_branch()}")
        n.set("From", f'<sip:4915112345@fake-pbx>;tag={self.from_tag}')
        n.set("To", f"<sip:900@fake-pbx>;tag={self.answer_to_tag}")
        n.set("Call-ID", self.call_id)
        n.set("CSeq", "2 NOTIFY")
        n.set("Event", "refer")
        n.set("Subscription-State", "terminated;reason=noresource")
        n.set("Content-Type", "message/sipfrag")
        n.set("Content-Length", str(len(body)))
        n.body = body.encode()
        self.transport.sendto(n.encode(), addr)

    @property
    def answer_to_tag(self):
        return sipmsg.parse_tag(self.invite_answer.get("to"))


async def wait_for(predicate, timeout=5.0, what="condition"):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


async def main() -> int:
    loop = asyncio.get_running_loop()
    pbx_port = free_port()
    ua_port = free_port()

    pbx = FakePbx()
    await loop.create_datagram_endpoint(lambda: pbx, local_addr=("127.0.0.1", pbx_port))

    account = SipAccount(
        username="900", password="secret", domain="fake-pbx",
        server_host="127.0.0.1", server_port=pbx_port,
        bind_host="127.0.0.1", bind_port=ua_port, advertise_host="127.0.0.1",
        rtp_port_range=(20000, 20100), codec_preference=["PCMA", "PCMU"],
    )

    audio_in: list = []
    dtmf_in: list = []
    call_box: dict = {}
    answered = asyncio.Event()

    async def on_call(call: Call):
        call_box["call"] = call
        agent.ring(call)
        await agent.answer(call, on_audio=audio_in.append, on_dtmf=dtmf_in.append)
        answered.set()

    agent = SipUserAgent(account, on_call)
    await agent.start()

    await asyncio.wait_for(pbx.authorized_register.wait(), 5)
    await wait_for(lambda: agent.registered, 5, "registered flag")
    print(f"PASS register (challenge+retry, {pbx.register_count} REGISTERs)")

    pbx.send_invite((("127.0.0.1"), ua_port))
    await asyncio.wait_for(answered.wait(), 5)
    await asyncio.wait_for(pbx.got_200_for_invite.wait(), 5)

    answer_sdp = pbx.invite_answer.body.decode()
    m = re.search(r"m=audio (\d+) RTP/AVP ([\d ]+)", answer_sdp)
    ua_rtp_port = int(m.group(1))
    print(f"PASS answer: codec payloads {m.group(2).strip()!r}, ua rtp port {ua_rtp_port}")
    assert "PCMA/8000" in answer_sdp, answer_sdp
    call = call_box["call"]
    assert call.caller_number == "4915112345", call.caller_number
    assert call.caller_name == "Max Mustermann", call.caller_name
    assert call.dialled_number == "900", call.dialled_number
    print(f"PASS caller id: {call.caller_name} / {call.caller_number} -> {call.dialled_number}")

    # RTP: send 10 frames of A-law tone, expect them decoded into on_audio
    rtp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtp_sock.bind(("127.0.0.1", pbx.rtp_port))
    rtp_sock.setblocking(False)
    seq, ts = 1, 0
    for _ in range(10):
        payload = bytes([0x2A]) * 160
        hdr = RtpHeader(payload_type=8, sequence=seq, timestamp=ts, ssrc=42).encode()
        rtp_sock.sendto(hdr + payload, ("127.0.0.1", ua_rtp_port))
        seq += 1
        ts += 160
    # DTMF digit 7
    rtp_sock.sendto(
        RtpHeader(payload_type=101, sequence=seq, timestamp=ts, ssrc=42).encode()
        + bytes([7, 10, 0, 160]),
        ("127.0.0.1", ua_rtp_port),
    )
    await asyncio.sleep(0.4)
    assert len(audio_in) >= 10, f"only got {len(audio_in)} inbound frames"
    assert audio_in[0].size == 160, audio_in[0].size
    print(f"PASS inbound rtp: {len(audio_in)} frames x {audio_in[0].size} samples")
    assert dtmf_in == ["7"], dtmf_in
    print(f"PASS dtmf: {dtmf_in}")

    # Outbound pacing: drain whatever idle fill accumulated, then measure a
    # clean window.  Exactly 20 ms per packet is what keeps the caller's
    # jitter buffer from adding delay of its own.
    def drain():
        n = 0
        try:
            while True:
                rtp_sock.recvfrom(2048)
                n += 1
        except BlockingIOError:
            return n

    drain()
    tone = (np.sin(2 * np.pi * 440 * np.arange(2400) / 8000) * 8000).astype(np.int16)
    call.rtp.enqueue(tone)
    t0 = asyncio.get_running_loop().time()
    await asyncio.sleep(0.30)
    elapsed = asyncio.get_running_loop().time() - t0
    frames, seqs, non_silent = 0, [], 0
    try:
        while True:
            data, _ = rtp_sock.recvfrom(2048)
            parsed = RtpHeader.parse(data)
            if not parsed:
                continue
            hdr, payload = parsed
            if hdr.payload_type != 8 or len(payload) != 160:
                continue
            frames += 1
            seqs.append(hdr.sequence)
            if len(set(payload)) > 1:
                non_silent += 1
    except BlockingIOError:
        pass

    expected = elapsed / 0.020
    assert abs(frames - expected) <= 3, f"pacing off: {frames} frames in {elapsed*1000:.0f}ms (expect ~{expected:.0f})"
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), "sequence numbers not monotonic"
    assert non_silent >= 10, f"tone did not reach the wire ({non_silent} non-silent frames)"
    print(f"PASS outbound pacing: {frames} frames in {elapsed*1000:.0f}ms "
          f"(~{elapsed*1000/max(frames,1):.1f}ms apart), {non_silent} carrying audio")

    # The decoded tone must match what we queued, i.e. the A-law round trip
    # through the real send path is sane.
    from helpdesk.audio.codec import decode as _dec
    drain()
    call.rtp.clear_playout()
    await asyncio.sleep(0.05)
    drain()
    probe = (np.sin(2 * np.pi * 440 * np.arange(800) / 8000) * 8000).astype(np.int16)
    call.rtp.enqueue(probe)
    await asyncio.sleep(0.15)
    got = []
    try:
        while True:
            data, _ = rtp_sock.recvfrom(2048)
            parsed = RtpHeader.parse(data)
            if parsed and parsed[0].payload_type == 8:
                got.append(_dec(parsed[1], "PCMA"))
    except BlockingIOError:
        pass
    rebuilt = np.concatenate(got)[: probe.size]
    corr = float(np.corrcoef(rebuilt.astype(float), probe[: rebuilt.size].astype(float))[0, 1])
    assert corr > 0.99, f"audio corrupted on the wire (corr={corr:.4f})"
    print(f"PASS audio integrity through send path (corr={corr:.4f})")

    # Barge-in: clearing the queue must take effect immediately
    call.rtp.enqueue(tone)
    assert call.rtp.queued_ms() > 100
    call.rtp.clear_playout()
    assert call.rtp.queued_ms() == 0 and not call.rtp.is_playing()
    print("PASS barge-in clears playout")

    # Transfer via REFER, then the PBX confirms with NOTIFY
    ended: list = []
    call.on_end = ended.append
    await agent.transfer(call, "4930999888")
    await asyncio.wait_for(pbx.refer_seen.wait(), 5)
    assert pbx.refer_target == "sip:4930999888@fake-pbx", pbx.refer_target
    print(f"PASS refer: Refer-To {pbx.refer_target}")
    pbx.send_notify(("127.0.0.1", ua_port))
    await asyncio.sleep(0.3)
    assert ended == ["transferred"], ended
    assert call.call_id not in agent.calls
    print(f"PASS transfer completed via NOTIFY, call cleaned up (reason={ended[0]})")

    rtp_sock.close()
    await agent.stop()
    print("PASS shutdown (de-register sent)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
