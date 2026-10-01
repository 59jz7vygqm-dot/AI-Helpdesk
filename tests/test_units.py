"""Unit checks for the pure functions: codecs, SIP parsing, auth, text shaping."""

from __future__ import annotations

import hashlib
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from helpdesk.audio import codec
from helpdesk.audio.vad import Endpointer, EndpointerConfig, EnergyVad, VadEvent
from helpdesk.config import apply_env_overrides, DEFAULTS
from helpdesk.kb.index import split_markdown, tokenize
from helpdesk.llm.agent import Action, MarkerFilter
from helpdesk.sip import messages as sipmsg
from helpdesk.sip import sdp as sdplib
from helpdesk.sip.auth import DigestAuth, parse_challenge
from helpdesk.sip.rtp import DtmfCollector, RtpHeader
from helpdesk.tts.text import SentenceStreamer, split_sentences, spoken_form

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILED += 1
        print(f"  FAIL {name}{' -- ' + detail if detail else ''}")


print("codec")
tone = (np.sin(2 * np.pi * 440 * np.arange(8000) / 8000) * 20000).astype(np.int16)
for enc in ("PCMU", "PCMA"):
    back = codec.decode(codec.encode(tone, enc), enc)
    # G.711 is logarithmic: ~2-3% error at high amplitude is correct behaviour.
    err = float(np.max(np.abs(tone.astype(int) - back.astype(int)))) / 20000
    check(f"{enc} round trip within 4%", err < 0.04, f"rel err {err:.3f}")
check("ulaw idle byte decodes to 0", codec.ULAW_DECODE[codec.ULAW_SILENCE] == 0)
check("8k->16k doubles length", codec.resample(tone, 8000, 16000).size == 16000)
check("22.05k->8k ratio", abs(codec.resample(np.zeros(22050, np.int16), 22050, 8000).size - 8000) <= 1)
check("resample keeps int16", codec.resample(tone, 8000, 16000).dtype == np.int16)
check("empty resample is safe", codec.resample(np.zeros(0, np.int16), 8000, 16000).size == 0)

print("sip parsing")
raw = (
    b"INVITE sip:900@pbx SIP/2.0\r\n"
    b"Via: SIP/2.0/UDP 10.0.0.5:5060;branch=z9hG4bKabc;rport\r\n"
    b"Via: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bKouter\r\n"
    b'From: "Max Mustermann" <sip:4915112345@pbx>;tag=aaa\r\n'
    b"t: <sip:900@pbx>\r\n"
    b"i: call-1@pbx\r\n"
    b"CSeq: 7 INVITE\r\n"
    b"Record-Route: <sip:p1;lr>, <sip:p2;lr>\r\n"
    b"Content-Length: 5\r\n\r\nv=0\r\nEXTRA-IGNORED"
)
msg = sipmsg.parse(raw)
check("method", msg.method == "INVITE")
check("compact headers expanded", msg.call_id == "call-1@pbx" and msg.get("to") == "<sip:900@pbx>")
check("multiple Via preserved", len(msg.get_all("via")) == 2)
check("comma-separated Record-Route split", len(msg.get_all("record-route")) == 2, str(msg.get_all("record-route")))
check("top branch", msg.top_via_branch() == "z9hG4bKabc")
check("body truncated to Content-Length", msg.body == b"v=0\r\n", repr(msg.body))
check("display name", sipmsg.parse_display_name(msg.get("from")) == "Max Mustermann")
check("caller number", sipmsg.uri_user(sipmsg.parse_uri(msg.get("from"))) == "4915112345")
check("uri host", sipmsg.uri_host("sip:900@pbx.local:5062;transport=udp") == "pbx.local")
check("cseq", (msg.cseq_number, msg.cseq_method) == (7, "INVITE"))
folded = sipmsg.parse(b"OPTIONS sip:a SIP/2.0\r\nSubject: one\r\n  two\r\nContent-Length: 0\r\n\r\n")
check("header folding", folded.get("subject") == "one two", folded.get("subject"))
resp = sipmsg.build_response(msg, 200, "OK", to_tag="t1", contact="<sip:900@1.2.3.4>")
text = resp.encode().decode()
check("response copies both Via", text.count("Via:") == 2)
check("response spells Call-ID correctly", "Call-ID:" in text and "CSeq:" in text)
check("to tag added", ";tag=t1" in text)
check("lenient on bare LF", sipmsg.parse(b"OPTIONS sip:a SIP/2.0\nCall-ID: x\n\n").call_id == "x")

print("digest auth")
ch = parse_challenge('Digest realm="pbx", nonce="n1", qop="auth", algorithm=MD5')
header = DigestAuth("900", "pw").authorization(ch, "REGISTER", "sip:pbx")
fields = dict(re.findall(r'(\w+)="?([^",]+)"?', header))
ha1 = hashlib.md5(b"900:pbx:pw").hexdigest()
ha2 = hashlib.md5(b"REGISTER:sip:pbx").hexdigest()
expect = hashlib.md5(f"{ha1}:n1:{fields['nc']}:{fields['cnonce']}:auth:{ha2}".encode()).hexdigest()
check("qop=auth response", fields["response"] == expect)
ch2 = parse_challenge('Digest realm="r", nonce="n"')
h2 = DigestAuth("u", "p").authorization(ch2, "INVITE", "sip:x@y")
f2 = dict(re.findall(r'(\w+)="?([^",]+)"?', h2))
ha1b = hashlib.md5(b"u:r:p").hexdigest()
ha2b = hashlib.md5(b"INVITE:sip:x@y").hexdigest()
check("no-qop response", f2["response"] == hashlib.md5(f"{ha1b}:n:{ha2b}".encode()).hexdigest())
check("nonce count increments", "nc=00000001" in header)
check("proxy challenge picks the right header",
      parse_challenge('Digest realm="r", nonce="n"', "Proxy-Authenticate").response_header == "Proxy-Authorization")
check("non-digest scheme ignored", parse_challenge('Basic realm="r"') is None)

print("sdp")
offer = sdplib.parse_sdp(
    b"v=0\r\nc=IN IP4 10.0.0.5\r\nm=audio 14002 RTP/AVP 8 0 101\r\n"
    b"a=rtpmap:8 PCMA/8000\r\na=rtpmap:0 PCMU/8000\r\n"
    b"a=rtpmap:101 telephone-event/8000\r\na=sendrecv\r\n"
)
audio = offer.audio
check("port parsed", audio.port == 14002)
check("prefers PCMA when asked", sdplib.select_codec(audio, ["PCMA", "PCMU"]) == (8, "PCMA"))
check("prefers PCMU when asked", sdplib.select_codec(audio, ["PCMU", "PCMA"]) == (0, "PCMU"))
check("dtmf payload found", sdplib.find_dtmf_payload(audio) == 101)
no_common = sdplib.parse_sdp(b"v=0\r\nc=IN IP4 1.1.1.1\r\nm=audio 1 RTP/AVP 9\r\na=rtpmap:9 G722/8000\r\n")
check("no common codec returns None", sdplib.select_codec(no_common.audio, ["PCMA", "PCMU"]) is None)
hold = sdplib.parse_sdp(b"v=0\r\nc=IN IP4 1.1.1.1\r\nm=audio 1 RTP/AVP 8\r\na=sendonly\r\n")
check("hold direction detected", hold.audio.direction == "sendonly")
answer = sdplib.build_sdp("9.9.9.9", 40000, "PCMA").decode()
check("answer is well formed", "m=audio 40000 RTP/AVP 8 101" in answer and "a=ptime:20" in answer)

print("rtp")
hdr = RtpHeader(payload_type=8, sequence=65535, timestamp=1, ssrc=5, marker=True)
parsed, payload = RtpHeader.parse(hdr.encode() + b"\xd5" * 160)
check("header round trip", (parsed.sequence, parsed.marker, len(payload)) == (65535, True, 160))
check("rejects short datagram", RtpHeader.parse(b"\x80\x08") is None)
check("rejects wrong version", RtpHeader.parse(b"\x00" * 20) is None)
dtmf = DtmfCollector()
check("dtmf reported once", (dtmf.feed(1, bytes([3, 10, 0, 160])), dtmf.feed(1, bytes([3, 10, 0, 160]))) == ("3", None))
check("dtmf hash digit", dtmf.feed(2, bytes([11, 10, 0, 160])) == "#")

print("endpointer")
ep = Endpointer(EndpointerConfig(sample_rate=8000), EnergyVad())
rng = np.random.default_rng(3)
events = []
for _ in range(30):
    events.append(ep.push(rng.normal(0, 50, 160).astype(np.int16)))
for _ in range(40):
    events.append(ep.push(rng.normal(0, 6000, 160).astype(np.int16)))
for _ in range(40):
    events.append(ep.push(rng.normal(0, 50, 160).astype(np.int16)))
seq = [e.value for e in events if e is not VadEvent.NONE]
check("exactly one start/speculative/end", seq == ["speech_start", "speculative_end", "speech_end"], str(seq))
check("utterance captured with pre-roll", ep.utterance_ms() > 800, f"{ep.utterance_ms()} ms")

print("agent markers")
def run_filter(tokens):
    f = MarkerFilter()
    out = "".join(f.feed(t) for t in tokens) + f.flush()
    return out, f.action

out, action = run_filter(["Ich verbinde ", "Sie. ", "[WEITER", "LEITEN", "]"])
check("marker split across tokens never leaks", "WEITER" not in out and action is Action.TRANSFER, out)
out, action = run_filter(["Tschuess. [AUFLEGEN]"])
check("hangup marker", action is Action.HANGUP and "AUFLEGEN" not in out)
out, action = run_filter(["Siehe [Punkt 3] oben"])
check("ordinary brackets survive", out == "Siehe [Punkt 3] oben" and action is Action.NONE, out)
out, _ = run_filter(["Abgeschnitten [WEITERL"])
check("unterminated marker stripped", "[" not in out, out)

print("spoken text")
check("abbreviations expanded", spoken_form("ca. 50 % inkl. MwSt.") == "circa 50 Prozent inklusive Mehrwertsteuer")
check("markdown stripped", spoken_form("**Wichtig**: `code`") == "Wichtig: code")
check("sentence split keeps abbreviations",
      split_sentences("Das sind z.B. 3 Dinge. Und Dr. Meier kommt.")
      == ["Das sind z.B. 3 Dinge.", "Und Dr. Meier kommt."])
streamer = SentenceStreamer()
chunks = []
for token in "Guten Tag, hier ist der Service. Wie kann ich helfen?".split(" "):
    chunks += list(streamer.feed(token + " "))
chunks += streamer.flush()
check("first chunk released at a clause boundary", chunks[0].endswith(",") or chunks[0].endswith("."), str(chunks))
check("streamer loses nothing", "".join(chunks).replace(" ", "") ==
      "GutenTag,hieristderService.Wiekannichhelfen?".replace(" ", ""), str(chunks))

print("knowledge")
chunks = split_markdown("# A\ntext a\n\n## B\ntext b\n", "f.md")
check("splits on headings", len(chunks) == 2 and chunks[1].title == "B")
long_doc = "# T\n" + ("satz. " * 400)
check("long section is chunked", len(split_markdown(long_doc, "f.md", max_chars=300)) > 3)
check("identifier kept and split", set(tokenize("Fehler E-512")) >= {"e-512", "512"})
check("stopwords removed", "der" not in tokenize("der Drucker"))

print("config")
# Capture before overriding, so the check does not depend on the default value.
original_silence = DEFAULTS["vad"]["end_silence_ms"]
os.environ["HELPDESK_VAD_END_SILENCE_MS"] = "333"
os.environ["HELPDESK_TTS_PIPER_USE_CUDA"] = "yes"
os.environ["HELPDESK_SIP_CODEC_PREFERENCE"] = "PCMU,PCMA"
merged = apply_env_overrides(DEFAULTS)
check("int override", merged["vad"]["end_silence_ms"] == 333)
check("nested bool override", merged["tts"]["piper"]["use_cuda"] is True)
check("list override", merged["sip"]["codec_preference"] == ["PCMU", "PCMA"])
check("defaults not mutated", DEFAULTS["vad"]["end_silence_ms"] == original_silence,
      f"DEFAULTS changed from {original_silence}")

print()
print(f"{PASSED} checks passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
