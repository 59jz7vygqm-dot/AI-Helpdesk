# AI Helpdesk — SIP-Telefonagent mit lokaler KI

Registriert sich als Nebenstelle an einer PBX, nimmt Anrufe an und führt ein
Gespräch: Spracherkennung, Antwort aus einer eigenen Wissensdatenbank,
Sprachausgabe. Man kann ihn jederzeit unterbrechen. Wenn er nicht weiterhelfen
kann, sagt er das und leitet auf eine andere Rufnummer weiter — per SIP REFER,
und falls die PBX das verbietet, über deren eigenen Feature-Code.

Alles läuft lokal. Kein Cloud-Dienst, kein Audio verlässt den Server.

```
Anrufer ──SIP/RTP── PBX ──SIP/RTP── Container ─┬─ faster-whisper (GPU)   Erkennung
                                               ├─ Ollama         (GPU)   Antwort
                                               ├─ Qwen3-TTS      (GPU)   Stimme
                                               └─ Wissens-DB  (Markdown) Inhalte
```

---

## Die wichtigsten Punkte zuerst — ehrlich

**Unterbrechen: ja, jederzeit.** Sobald der Anrufer zu reden anfängt, bricht der
Agent mitten im Wort ab — die Wiedergabe wird verworfen, die Synthese gestoppt
und zugehört. Zwei Dinge machen das erst benutzbar: sein **eigenes Echo** löst es
nicht aus (er vergleicht, was reinkommt, mit dem, was er gerade gesendet hat), und
ein eingeworfenes **„mhm" oder „alles klar" lässt ihn weiterreden** statt die
Antwort von vorn zu beginnen — er merkt sich, was der Anrufer noch nicht gehört
hat. Beides ist getestet.

**Latenz: ja, das geht.** Die Pause zwischen „Anrufer hört auf zu reden" und
„Agent fängt an zu reden" liegt mit Piper bei **600–750 ms** und mit der
natürlichen Stimme bei etwa **1,0 s**. Beides ist am Telefon unauffällig. Erreicht wird das mit drei Tricks: die Erkennung startet
**vor** Ende der Sprechpause, die Antwort wird satzweise synthetisiert während das
Modell noch generiert, und feste Sätze wie die Begrüßung sind vorgerendert. Wo die
Zeit hingeht, steht pro Antwort im Log — du musst nicht raten.

**Stimme: in zwei Stufen.** Standard ist Piper — nicht weil es am besten klingt,
sondern weil es garantiert baut und läuft. Damit bringst du den Anruf erst
komplett zum Laufen; die gute Stimme ist danach ein Konfigurationsschritt.

| Stimme | Erstes Audio | VRAM | Klingt | Aufwand |
|---|---|---|---|---|
| **Piper** (Standard) | ~30–60 ms | 0 (CPU) | verständlich, hörbar synthetisch | im Image enthalten |
| Chatterbox | ~300–500 ms | ~3 GB | natürlich, klonbar, MIT | Image mit `TTS_PROFILE=quality` |
| Qwen3-TTS | ~150–400 ms | ~4–8 GB | am besten, Apache-2.0 | eigener Container, siehe unten |

Warum Qwen3-TTS nicht im Image ist, obwohl es das beste wäre: sein PyPI-Paket
verlangt **Python ≥ 3.13**, die CUDA-Basis-Images liefern 3.10. Und Chatterbox
pinnt `torch==2.6.0` exakt, verträgt sich also nicht mit einem selbst gewählten
torch. Beides sind reale Abhängigkeitskonflikte, keine Vermutungen — deshalb ist
der Standardpfad bewusst der, der ohne Überraschungen durchläuft.

Was zur Menschlichkeit genauso viel beiträgt wie die Stimme — und alles eingebaut
ist: dass man ihn **jederzeit unterbrechen** kann, dass sein eigenes Echo das
**nicht fälschlich** auslöst, dass ein eingeworfenes „mhm" ihn **weiterreden**
lässt statt neu anzufangen, dass er **kurz** antwortet, und dass er Abkürzungen
ausspricht statt zu buchstabieren.

---

## Latenzbudget

Gemessen ab dem Moment, in dem der Anrufer aufhört zu sprechen:

| Stufe | Dauer | Anmerkung |
|---|---|---|
| Sprechpause abwarten | **420 ms** | der größte Posten, einstellbar |
| Spracherkennung | 0–100 ms | läuft spekulativ schon vorher an |
| Wissenssuche | ~15 ms | numpy-Skalarprodukt, keine Datenbank |
| LLM bis erstes Token | 120–200 ms | 7B Q4 auf einer L4 |
| TTS bis erstes Audio | 30–60 ms | Piper; Chatterbox 300–500 ms |
| **Summe** | **~600–750 ms** | mit Chatterbox ~900 ms – 1,1 s |

Die eine Stellschraube, die wirklich zählt, ist `vad.end_silence_ms`. Runter auf
300 ms fühlt sich spürbar flotter an, aber der Agent fängt an, Leute zu
unterbrechen, die mitten im Satz Luft holen. 420 ms ist der Kompromiss, mit dem
ich anfangen würde. Jeder Turn wird geloggt, damit du nicht raten musst:

```
turn 3: response 612 ms (asr 71*, kb 14, llm_ttft 183, tts 44) utterance 2180 ms
```

Das `*` heißt: die spekulative Erkennung hat gegriffen, die ASR-Zeit war gratis.

---

## VRAM: fertige Profile

In `config/profiles/` liegen drei vollständige Konfigurationen. Eine davon über
`config/config.yaml` kopieren:

| Profil | VRAM | Stimme | Wofür |
|---|---|---|---|
| **`demo-single-gpu.yaml`** | **~7 GB** | **Piper** | **hier anfangen** — läuft im Standard-Image |
| `12gb-shared.yaml` | ~7 GB | Piper | GPU wird mit anderem geteilt |
| `quality-chatterbox.yaml` | ~10 GB | Chatterbox | natürliche Stimme, Image mit `TTS_PROFILE=quality` |

```bash
cp config/profiles/demo-single-gpu.yaml config/config.yaml
```

`config/config.yaml` is yours and is gitignored, so `git pull` never touches your
settings. The profiles and `config.example.yaml` are the tracked copies.

`demo-single-gpu.yaml` ist der Startpunkt für „eine freie L4": gute Stimme,
kleines Sprachmodell, Embeddings auf der CPU — also kein zweites Ollama-Modell
zum Herunterladen.

Alle Profile nutzen ein 7B-Sprachmodell. Am Telefon hört der Anrufer die Stimme,
nicht die Modellgröße: die Antworten sind kurz und stehen in der
Wissensdatenbank, ein 14B formuliert sie selten besser, kostet aber 150–250 ms
bei *jeder* Antwort. Wenn du es vergleichen willst, ist es eine Zeile:
`llm.model: "qwen3:14b"` plus `think: false`.

Wichtig: `keep_alive: "-1"` lässt das Sprachmodell dauerhaft im VRAM. Ohne das
lädt Ollama es nach ein paar Minuten Ruhe aus — und der nächste Anrufer wartet
zwanzig Sekunden.

Und: `think: false`. Modelle wie `qwen3:14b` „denken" sonst erst mehrere Sekunden
still nach, bevor das erste Wort kommt. Am Telefon ist das totes Schweigen.

Wichtig: `keep_alive: "-1"` in der Config lässt das Modell dauerhaft im VRAM.
Ohne das lädt Ollama es nach ein paar Minuten Ruhe aus — und der nächste Anrufer
wartet 20 Sekunden.

**Ein Anruf gleichzeitig** (`max_concurrent_calls: 1`). Bei einer GPU ist das
die ehrliche Zahl: zwei parallele Anrufe würden sich die GPU teilen und wären
beide langsam. Weitere Anrufe werden mit `486 Busy Here` abgewiesen, die PBX
kann sie dann auf die Weiterleitungsnummer schicken.

### Bessere Stimme — der zweite Schritt

**Stufe 1: Chatterbox** (natürlich, klonbar, MIT-Lizenz). Image neu bauen und
umstellen:

```bash
TTS_PROFILE=quality docker compose build     # ~15 Min, zieht torch
cp config/profiles/quality-chatterbox.yaml config/config.yaml
docker compose up -d
docker compose exec helpdesk python3 /app/scripts/try_pipeline.py "Test"
```

Eigene Stimme klonen — 10–20 Sekunden klare Aufnahme nach `./voices/` legen:

```yaml
tts:
  backend: chatterbox
  chatterbox:
    reference_audio: /models/piper/meine-stimme.wav
```

**Stufe 2: Qwen3-TTS** (bestes Deutsch, Apache-2.0, Klonen aus 3 Sekunden). Es
kann nicht ins Image, weil sein PyPI-Paket Python 3.13 braucht. Der saubere Weg
ist ein eigener Container, der es als OpenAI-kompatiblen TTS-Dienst anbietet —
das Backend dafür ist schon eingebaut:

```yaml
tts:
  backend: openai
  openai:
    base_url: "http://127.0.0.1:8880/v1"
    response_format: pcm        # Pflicht: alles andere kostet Latenz
    sample_rate: 24000
```

Damit ist auch jeder andere TTS-Server nutzbar (Kokoro, XTTS, LocalAI). Den
Qwen3-Dienst selbst liefert dieses Repo noch nicht mit — sag Bescheid, wenn du
ihn brauchst.

Rechtlich, weil es praktisch relevant ist: eine fremde Stimme zu klonen braucht
deren Einverständnis.

## Installation

### 1. Voraussetzungen

Auf dem Host: Docker mit NVIDIA Container Toolkit, und Ollama läuft bereits.

```bash
nvidia-smi -L                    # UUID der L4 notieren
curl -s localhost:11434/api/tags # Ollama erreichbar?
```

### 1b. Mehrere GPUs: beide Seiten auf dieselbe freie Karte pinnen

Überspringen, wenn der Server nur eine GPU hat. Sonst ist das der Schritt, an dem
es sonst scheitert: **Ollama nimmt sich ohne Pinning GPU 0** — und wenn die voll
ist, läuft das Modell auf der CPU oder stirbt mit „out of memory".

```bash
nvidia-smi    # welche Karte ist frei? Index notieren, hier Beispiel 4
```

Container (in `.env`):

```bash
GPU_ID=4
```

Ollama (systemd-Override, sonst wirkt es nicht):

```bash
sudo systemctl edit ollama
```

Diese zwei Zeilen eintragen:

```ini
[Service]
Environment="CUDA_VISIBLE_DEVICES=4"
```

Dann:

```bash
sudo systemctl restart ollama
nvidia-smi --id=4           # nach dem ersten Anruf muss hier Ollama auftauchen
```

`./scripts/preflight.sh` prüft beides und meckert, wenn nur eine Seite gepinnt ist.

**Docker 29 und CDI:** Neuere Docker-Versionen wählen GPUs über CDI aus. Falls
`--gpus device=4` nicht geht, `--device nvidia.com/gpu=4` aber schon (preflight
sagt dir das), dann mit dem Overlay starten:

```bash
docker compose -f docker-compose.yml -f docker-compose.cdi.yml up -d --build
```

Fehlt die CDI-Spec ganz, hilft:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Profil wählen und Modelle holen

```bash
git clone <dieses-repo> && cd AI-Helpdesk
cp config/profiles/16gb-quality.yaml config/config.yaml
./scripts/download_models.sh     # Ollama-Modelle + Piper als Fallback-Stimme
```

`download_models.sh` holt die deutsche Piper-Stimme nach `./voices/` — ohne die
startet der Container nicht. Whisper lädt beim ersten Start selbst (~1,6 GB, ins
`/models`-Volume, also einmalig). Das Preflight prüft die Stimme mit.

### 3. Zugangsdaten

```bash
cp .env.example .env
nano .env        # SIP-Daten und TRANSFER_NUMBER eintragen
```

Die PBX braucht dafür eine normale Nebenstelle/einen SIP-Account — genau so
angelegt wie für ein Tischtelefon.

### 4. Wissensdatenbank füllen

Markdown-Dateien in `knowledge/`. Die Beispiele dort löschen und eigene Inhalte
rein — `knowledge/README.md` erklärt, wie man schreibt, damit der Agent die
richtige Stelle findet. Der Index baut sich beim Start automatisch neu, sobald
sich der Text ändert.

### 5. Starten

```bash
docker compose up -d --build
docker compose logs -f
```

Erwartete Ausgabe:

```
loading ASR model large-v3-turbo (cuda, int8_float16)
ASR model ready in 4.2s
loading Piper voice /models/piper/de_DE-thorsten-high.onnx (cuda=False)
knowledge base built: 14 chunks from /app/knowledge in 2.1s
warmup complete in 12.4s
registered as 900, refreshing in 225s
helpdesk ready: extension 900 on 192.168.1.10, transfers go to 200
```

Dann die Nummer anrufen.

---

## Vor dem ersten Anruf testen

Der nützlichste Befehl im Repo — das ganze Gehirn ohne Telefon, und der Weg, die
Stimme zu beurteilen, bevor jemand anruft:

```bash
docker compose exec helpdesk python3 /app/scripts/try_pipeline.py "Mein Drucker zeigt E-512"
```

Zeigt die Antwort, die Zeiten jeder Stufe, welche Wissensquellen getroffen
wurden, ob weitergeleitet würde — und schreibt das Audio als WAV, so wie der
Anrufer es hört (8 kHz). Damit lässt sich die Wissensdatenbank und der Ton
durchprobieren, ohne jedes Mal anzurufen. Ohne Argument wird es interaktiv.

Die Testsuite läuft ohne GPU, ohne PBX, ohne Netz:

```bash
./scripts/run_tests.sh
```

Sie prüft gegen eine simulierte PBX den kompletten Signalisierungsablauf
(REGISTER mit Digest-Auth, Anruf annehmen, Codec-Aushandlung, DTMF, REFER-
Weiterleitung), das 20-ms-Timing des Audioversands, die Audiointegrität, und auf
Session-Ebene Begrüßung, Gesprächsablauf, Unterbrechung, Weiterleitung inklusive
Fehlerfall und die Stille-Überwachung.

---

## Managed Asterisk: was du beim Anbieter klären musst

Dass die PBX fremdverwaltet ist, ist kein Hindernis — der Agent ist aus Sicht der
PBX eine völlig normale Nebenstelle, genau wie ein Tischtelefon. Aber **ein**
Punkt hängt vom Anbieter ab, und den solltest du vorher klären statt beim ersten
Anruf zu rätseln.

**Die eine wichtige Frage:** Erlaubt die Nebenstelle SIP REFER?
Konkret: steht bei dem Endpoint `allow_transfer = no` (PJSIP) bzw.
`allowtransfer=no` (chan_sip)? Falls ja, lehnt die PBX jeden Weiterleitungs-
versuch per REFER ab — genau wie bei einem Tischtelefon, dessen Transfer-Taste
dann auch nicht funktioniert.

Deshalb gibt es zwei Wege, und standardmäßig probiert der Agent beide:

```yaml
dialog:
  transfer_method: auto             # REFER zuerst, bei Ablehnung Feature-Code
  transfer_dtmf_feature_code: "##"  # Asterisks blindxfer aus features.conf
```

1. **SIP REFER** — der saubere Weg. Die PBX übernimmt den Anruf und wählt das
   Ziel nach ihrem Wählplan, also funktioniert Nebenstelle, Warteschlange oder
   externe Nummer gleichermaßen.
2. **DTMF-Feature-Code** — der Fallback. Der Agent wählt mitten im Gespräch die
   Transfer-Tastenfolge aus Asterisks `features.conf` und danach die Zielnummer,
   exakt wie ein Mensch, der „##200" drückt. Das funktioniert auch bei
   `allow_transfer = no`, weil hier die PBX die Weiterleitung macht und das
   Endgerät nichts signalisiert.

**Was du den Anbieter fragen solltest** — drei Sätze reichen:

> 1. Ist für die Nebenstelle `allow_transfer` aktiviert (SIP REFER erlaubt)?
>    Wenn nein: bitte für diese eine Nebenstelle aktivieren.
> 2. Falls nicht möglich: welche DTMF-Sequenz ist als `blindxfer` in
>    `features.conf` konfiguriert, und ist der Dial mit der Option `t` gesetzt,
>    sodass die angerufene Seite weiterleiten darf?
> 3. Welche Codecs sind für die Nebenstelle erlaubt? G.711 (`alaw`) genügt.

Für den Feature-Code gibt es keinen verlässlichen Standardwert: Asterisk pur
verwendet üblicherweise `#1`, FreePBX häufig `##`. Deshalb rät das Programm
nicht, sondern nimmt den konfigurierten Wert. Wenn der Fallback greift, aber die
PBX nicht reagiert, steht das als klarer Fehler im Log:

```
REFER refused (403 Forbidden); falling back to the DTMF feature code
PBX did not act on the feature code '##' within 4.0s
(wrong code, or in-call transfers are disabled for this extension)
```

Falls beides nicht geht, bleibt als Notlösung: den Anbieter bitten, eine
Rufumleitung bei Besetzt auf die Zielnummer zu legen, und `max_concurrent_calls`
so zu nutzen, dass weitere Anrufe mit `486 Busy Here` abgewiesen werden. Das ist
hässlich, aber es verliert keinen Anrufer.

## Wie die Weiterleitung funktioniert

Der Agent leitet weiter, wenn:

- die Antwort nicht in der Wissensdatenbank steht,
- der Anrufer ausdrücklich einen Menschen verlangt,
- es um Kündigung, Reklamation oder Rechtliches geht,
- zweimal nichts verstanden wurde (`dialog.max_misunderstood`),
- oder der Anrufer die **0** drückt.

Der Agent sagt vorher einen Satz an und wartet, bis der zu Ende gespielt ist.
Scheitert die Weiterleitung auf beiden Wegen, entschuldigt er sich und legt auf,
statt den Anrufer in Stille hängen zu lassen. Die Zielrufnummer nennt er nie.

Das Modell signalisiert die Weiterleitung mit dem Marker `[WEITERLEITEN]` in
seinem Text. Der wird aus dem Audio herausgefiltert, auch wenn er mitten im
Token-Stream aufgeteilt ankommt — getestet. Marker statt Tool-Calling, weil das
mit jedem Ollama-Modell funktioniert und keinen zweiten Modellaufruf kostet,
bevor das erste Wort gesprochen werden kann.

---

## Konfiguration

Alles in `config/config.yaml`, kommentiert. Jeder Wert ist per Umgebungsvariable
überschreibbar (`HELPDESK_<SEKTION>_<SCHLÜSSEL>`, auch verschachtelt):

```bash
HELPDESK_VAD_END_SILENCE_MS=350
HELPDESK_TTS_BACKEND=chatterbox
HELPDESK_LLM_MODEL=qwen2.5:3b-instruct-q4_K_M
```

Was man am ehesten anfasst:

| Schlüssel | Wirkung |
|---|---|
| `vad.end_silence_ms` | Reaktionszeit vs. Leute-ins-Wort-fallen |
| `vad.barge_in_ms` | wie leicht man den Agenten unterbrechen kann |
| `vad.echo_attenuation_db` | runter, wenn Echo den Agenten unterbricht |
| `dialog.transfer_method` | `auto`, `refer` oder `dtmf` (siehe oben) |
| `tts.backend` | `piper` (schnell) oder `chatterbox` (natürlich) |
| `dialog.greeting` | Begrüßung (wird vorgerendert) |
| `dialog.transfer_number` | wohin unbeantwortbare Anrufe gehen |
| `dialog.max_sentences` | Antwortlänge; 2–3 ist telefontauglich |
| `knowledge.min_score` | zu niedrig → erfindet; zu hoch → leitet zu oft weiter |
| `asr.initial_prompt` | eigene Produktnamen/Fehlercodes der Erkennung beibringen |
| `sip.trace` | jede SIP-Nachricht loggen — das Erste bei Registrierungsproblemen |

---

## Fehlersuche

**Registrierung schlägt fehl** — `sip.trace: true` setzen und Logs ansehen. Meist
falsches Passwort, oder die PBX erwartet einen separaten Auth-Namen
(`sip.auth_username`). Bei `403 Forbidden` lässt die PBX die IP des Containers
nicht zu.

**Verbindung steht, aber kein Ton** — fast immer Docker-Netzwerk. RTP handelt
seine Ports im SDP aus, und Dockers NAT schreibt die nicht um. Deshalb steht
`network_mode: host` in der Compose-Datei. Wenn du bridged betreiben musst, muss
der komplette RTP-Bereich veröffentlicht und `sip.advertise_host` gesetzt sein.

**Einseitiger Ton** — der Container schickt Audio an die Adresse aus dem SDP.
Hinter NAT lernt er die echte Gegenstelle automatisch aus eintreffendem RTP
(symmetrisches RTP). Hilft das nicht, `sip.advertise_host` auf die LAN-IP des
Hosts setzen.

**Der Agent antwortet erst nach Sekunden** — prüfen, ob das Modell im VRAM
geblieben ist (`nvidia-smi`, und `keep_alive: "-1"`). Wenn das erste Token
dauert, ist das Modell zu groß oder es wird in den RAM ausgelagert. Auf einem
Server mit mehreren GPUs ist die häufigste Ursache, dass Ollama nicht gepinnt ist
und auf einer vollen Karte in den RAM ausgelagert wurde — siehe Schritt 1b.

**`out of memory` beim Start** — Container und Ollama liegen auf verschiedenen
Karten, oder auf einer belegten. `nvidia-smi` zeigt, wer wo wie viel hält;
`GPU_ID` und `CUDA_VISIBLE_DEVICES` müssen auf dieselbe freie Karte zeigen.

**Er fällt mir ins Wort** — `vad.end_silence_ms` hoch (500–600).
**Er reagiert zu träge** — denselben Wert runter (300–350).

**Er unterbricht sich selbst** — das wäre Echo, das als Sprache des Anrufers
durchgeht. `vad.echo_attenuation_db` von 12 auf 8 senken (strenger) oder
`vad.barge_in_ms` hoch. Im Log steht, wie viele Frames der Echo-Schutz abgewiesen
hat.

**Man kann ihn nicht unterbrechen** — umgekehrt: `vad.barge_in_ms` runter
(180–220) und `vad.echo_attenuation_db` hoch (16–20), oder zum Prüfen
`vad.echo_guard: false`.

**Weiterleitung scheitert** — siehe den Abschnitt zur managed Asterisk oben. Das
Log nennt immer, welcher Weg versucht wurde und warum er scheiterte.

**Er erfindet Dinge** — `knowledge.min_score` hoch und prüfen, ob die
Wissensdatenbank zum Thema überhaupt etwas enthält. Dann leitet er lieber weiter,
statt zu improvisieren. `temperature` in `llm.options` runter hilft zusätzlich.

**Er versteht Fachbegriffe nicht** — `asr.initial_prompt` mit den eigenen
Produktnamen und Fehlercodes füllen; das lenkt die Erkennung.

---

## Was wie getestet ist — und was nicht

Weil das für die Einschätzung wichtig ist:

**Verifiziert** (automatisiert, ohne Hardware): Signalisierung gegen eine
simulierte PBX inklusive Digest-Auth, Codec-Aushandlung, REFER-Weiterleitung und
**dem DTMF-Fallback, wenn die PBX REFER mit 403 ablehnt**; G.711-Kodierung gegen
den Standard; DTMF-Senden nach RFC 2833 (Paketstruktur und Rückdekodierung);
20,0 ms gemessener Sendetakt; Audiointegrität über den echten Sendepfad
(Korrelation 0,9999); auf Gesprächsebene Begrüßung, Unterbrechung,
**Echo-Abwehr bei −18 dB**, **Fortsetzen nach „mhm"**, Weiterleitung samt
Fehlerfall, Stille-Überwachung und DTMF-Null.

**Noch nicht verifiziert**, weil mir dafür die Hardware fehlt: der Lauf gegen
eine echte PBX, die tatsächliche Latenz auf deiner L4, und wie Chatterbox auf
Deutsch klingt. Die Zahlen für die Modellstufen sind veröffentlichte Benchmarks
für diese Hardwareklasse, keine Messung auf deinem Server.

Eine Einschränkung, die ich benennen muss: **die Backends für Chatterbox und
Qwen3-TTS sind gegen die dokumentierte API geschrieben, nicht gegen eine laufende
Installation.** Der Code geht defensiv mit Abweichungen um und nennt im Fehlerfall
den Ausweg, aber beim ersten Umstellen kann Nacharbeit nötig sein. Prüfen lässt
sich das mit `scripts/try_pipeline.py`, ohne einen einzigen Anruf. Piper ist
dagegen vollständig im Image und der garantierte Pfad.

Der erste echte Anruf ist der eigentliche Test — `sip.trace: true` dabei
anlassen.

Der SIP-Stack ist selbst geschrieben, statt pjsua2 zu binden. Grund: die
Audioframes bleiben in Python, ohne Sprachwechsel im 20-ms-Takt, und das Image
braucht keinen C-Build. Preis dafür: er kennt bewusst nur, was eine
registrierte Nebenstelle braucht — kein TCP/TLS, kein SRTP, kein IPv6, nur G.711.
Für eine lokale PBX im LAN ist das genau der richtige Umfang.

---

## Projektstruktur

```
src/helpdesk/
  sip/          SIP und RTP: Nachrichten, Digest-Auth, SDP, Medien, User Agent
  audio/        G.711-Codec, Resampling, Sprachaktivität und Endpunkterkennung
  asr/          faster-whisper mit Halluzinationsfilter
  llm/          Ollama- und OpenAI/vLLM-Streaming, der Dialogagent
  tts/          Piper, Chatterbox, Qwen3-TTS, OpenAI-kompatibel, Satzaufteilung
  kb/           Markdown-Chunking, Embeddings, hybride Suche
  session.py    Gesprächsablauf: der Latenzpfad
  app.py        Verdrahtung und Start
config/         Konfiguration, kommentiert, plus drei VRAM-Profile
knowledge/      Wissensdatenbank (Markdown)
scripts/        Preflight, Modelldownload, Pipeline-Test, Tests, Healthcheck
tests/          simulierte PBX, Session-, Verdrahtungs- und Unit-Tests
```
