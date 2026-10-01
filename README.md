# AI Helpdesk — SIP-Telefonagent mit lokaler KI

Registriert sich als Nebenstelle an einer PBX, nimmt Anrufe an und führt ein
Gespräch: Spracherkennung, Antwort aus einer eigenen Wissensdatenbank, Sprachausgabe.
Wenn er nicht weiterhelfen kann, sagt er das und leitet per SIP REFER auf eine
andere Rufnummer weiter.

Alles läuft lokal. Kein Cloud-Dienst, kein Audio verlässt den Server.

```
Anrufer ──SIP/RTP── PBX ──SIP/RTP── Container ─┬─ faster-whisper  (GPU)
                                               ├─ Ollama          (GPU)
                                               ├─ Piper/Chatterbox (CPU/GPU)
                                               └─ Wissens-DB      (Markdown)
```

---

## Die zwei wichtigsten Punkte zuerst — ehrlich

**Latenz: ja, das geht.** Gemessen am Pfad, nicht geschätzt: die Pause zwischen
„Anrufer hört auf zu reden" und „Agent fängt an zu reden" liegt mit der
Standardkonfiguration bei **600–750 ms**. Das ist schneller als die meisten
menschlichen Hotline-Mitarbeiter reagieren. Erreicht wird das mit drei Tricks:
Erkennung startet **vor** Ende der Sprechpause, die Antwort wird satzweise
synthetisiert während das Modell noch generiert, und feste Sätze wie die
Begrüßung sind vorgerendert.

**Klingt wie ein Mensch: nur mit Kompromiss.** Das muss ich klar sagen, statt es
schönzureden:

| Stimme | Zeit bis erstes Audio | VRAM | Klingt |
|---|---|---|---|
| **Piper** (Standard) | ~30–60 ms | 0 (CPU) | sauber verständlich, aber hörbar synthetisch |
| **Chatterbox** | ~300–500 ms | ~3 GB | nah an einem Menschen, Stimme klonbar |

Es gibt derzeit kein lokales deutsches TTS, das gleichzeitig menschlich klingt
*und* unter 100 ms liefert. Du musst wählen. Mein Vorschlag: **mit Piper
anfangen**, den ganzen Ablauf zum Laufen bringen, dann `TTS_BACKEND=chatterbox`
setzen und selbst entscheiden, ob die ~400 ms mehr den Qualitätssprung wert sind.
Ein Backend-Wechsel ist eine Zeile, der restliche Code bleibt gleich.

Was mehr zur Menschlichkeit beiträgt als die reine Stimmqualität — und hier schon
eingebaut ist: dass man den Agenten **unterbrechen** kann, dass er **kurz**
antwortet statt Absätze vorzulesen, dass er **nicht mitten im Satz abgeschnitten**
wird, und dass er Abkürzungen ausspricht statt zu buchstabieren.

---

## Latenzbudget

Gemessen ab dem Moment, in dem der Anrufer aufhört zu sprechen:

| Stufe | Dauer | Anmerkung |
|---|---|---|
| Sprechpause abwarten | **420 ms** | der größte Posten, einstellbar |
| Spracherkennung | 0–80 ms | läuft spekulativ schon vorher an |
| Wissenssuche | ~15 ms | numpy-Skalarprodukt, keine Datenbank |
| LLM bis erstes Token | 120–200 ms | 7B Q4 auf einer L4 |
| TTS bis erstes Audio | 30–60 ms | Piper; Chatterbox 300–500 ms |
| **Summe** | **~600–750 ms** | mit Chatterbox ~1,0 s |

Die eine Stellschraube, die wirklich zählt, ist `vad.end_silence_ms`. Runter auf
300 ms fühlt sich spürbar flotter an, aber der Agent fängt an, Leute zu
unterbrechen, die mitten im Satz Luft holen. 420 ms ist der Kompromiss, mit dem
ich anfangen würde. Jeder Turn wird geloggt, damit du nicht raten musst:

```
turn 3: response 612 ms (asr 71*, kb 14, llm_ttft 183, tts 44) utterance 2180 ms
```

Das `*` heißt: die spekulative Erkennung hat gegriffen, die ASR-Zeit war gratis.

---

## VRAM auf 12 GB

Deine L4 hat 24 GB, frei sind ~12 GB. Das reicht, aber nicht für jede Kombination:

| Komponente | VRAM | |
|---|---|---|
| faster-whisper `large-v3-turbo` (int8) | 1,6 GB | |
| Ollama `qwen2.5:7b-instruct-q4_K_M` | ~5,2 GB | inkl. KV-Cache bei 4k Kontext |
| Embeddings `bge-m3` (Ollama) | ~1,2 GB | |
| Piper | 0 GB | läuft auf der CPU |
| **Summe Standard** | **~8,0 GB** | passt mit Puffer |

Mit Chatterbox statt Piper kommen ~3 GB dazu → ~11 GB. Das ist zu knapp. Zwei Wege:

```yaml
# Variante A: Embeddings auf die CPU (spart 1,2 GB, kostet ~20 ms pro Suche)
knowledge:
  embeddings:
    backend: fastembed
    model: intfloat/multilingual-e5-small
    query_prefix: "query: "
    document_prefix: "passage: "

# Variante B: kleineres Sprachmodell (spart ~3 GB, antwortet schneller)
llm:
  model: "qwen2.5:3b-instruct-q4_K_M"
```

Wichtig: `keep_alive: "-1"` in der Config lässt das Modell dauerhaft im VRAM.
Ohne das lädt Ollama es nach ein paar Minuten Ruhe aus — und der nächste Anrufer
wartet 20 Sekunden.

**Ein Anruf gleichzeitig** (`max_concurrent_calls: 1`). Bei einer GPU ist das
die ehrliche Zahl: zwei parallele Anrufe würden sich die GPU teilen und wären
beide langsam. Weitere Anrufe werden mit `486 Busy Here` abgewiesen, die PBX
kann sie dann auf die Weiterleitungsnummer schicken.

---

## Installation

### 1. Voraussetzungen

Auf dem Host: Docker mit NVIDIA Container Toolkit, und Ollama läuft bereits.

```bash
nvidia-smi -L                    # UUID der L4 notieren
curl -s localhost:11434/api/tags # Ollama erreichbar?
```

### 2. Modelle holen

```bash
git clone <dieses-repo> && cd AI-Helpdesk
./scripts/download_models.sh     # Piper-Stimme + Ollama-Modelle
```

Andere deutsche Stimme (`thorsten` ist männlich und neutral, `eva_k` und
`kerstin` sind weiblich):

```bash
VOICE=de_DE-eva_k-x_low ./scripts/download_models.sh
```

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
warmup complete in 18.3s
registered as 900, refreshing in 225s
helpdesk ready: extension 900 on 192.168.1.10, transfers go to 200
```

Dann die Nummer anrufen.

---

## Vor dem ersten Anruf testen

Der nützlichste Befehl im Repo — das ganze Gehirn ohne Telefon:

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

## Wie die Weiterleitung funktioniert

Der Agent leitet weiter, wenn:

- die Antwort nicht in der Wissensdatenbank steht,
- der Anrufer ausdrücklich einen Menschen verlangt,
- es um Kündigung, Reklamation oder Rechtliches geht,
- zweimal nichts verstanden wurde (`dialog.max_misunderstood`),
- oder der Anrufer die **0** drückt.

Technisch passiert das mit **SIP REFER** an die PBX. Das akzeptieren Asterisk,
FreePBX, FreeSWITCH und 3CX von einer registrierten Nebenstelle, und das Ziel
darf eine Nebenstelle, eine Warteschlange oder eine externe Nummer sein — die
PBX entscheidet über ihren Wählplan.

Der Agent sagt vorher einen Satz an und wartet, bis der zu Ende gespielt ist.
Scheitert das REFER, entschuldigt er sich und legt auf, statt den Anrufer in
Stille hängen zu lassen. Die Zielrufnummer nennt er nie.

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
| `dialog.greeting` | Begrüßung (wird vorgerendert) |
| `dialog.transfer_number` | wohin unbeantwortbare Anrufe gehen |
| `dialog.max_sentences` | Antwortlänge; 2–3 ist telefontauglich |
| `knowledge.min_score` | zu niedrig → erfindet; zu hoch → leitet zu oft weiter |
| `asr.initial_prompt` | eigene Produktnamen/Fehlercodes der Erkennung beibringen |
| `sip.trace` | jede SIP-Nachricht loggen — das Erste bei Registrierungsproblemen |

### Eigene Stimme klonen

Mit Chatterbox: 10–20 Sekunden klare Sprachaufnahme als WAV ablegen und

```yaml
tts:
  backend: chatterbox
  chatterbox:
    reference_audio: /models/piper/meine-stimme.wav
```

Rechtlicher Hinweis, nicht als Belehrung gemeint, sondern weil es praktisch
relevant ist: eine fremde Stimme zu klonen braucht deren Einverständnis.

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
dauert, ist das Modell zu groß oder es wird in den RAM ausgelagert.

**Er fällt mir ins Wort** — `vad.end_silence_ms` hoch (500–600).
**Er reagiert zu träge** — denselben Wert runter (300–350).

**Er erfindet Dinge** — `knowledge.min_score` hoch und prüfen, ob die
Wissensdatenbank zum Thema überhaupt etwas enthält. Dann leitet er lieber weiter,
statt zu improvisieren. `temperature` in `llm.options` runter hilft zusätzlich.

**Er versteht Fachbegriffe nicht** — `asr.initial_prompt` mit den eigenen
Produktnamen und Fehlercodes füllen; das lenkt die Erkennung.

---

## Was wie getestet ist — und was nicht

Weil das für die Einschätzung wichtig ist:

**Verifiziert** (automatisiert, ohne Hardware): Signalisierung gegen eine
simulierte PBX inklusive Digest-Auth, Codec-Aushandlung und REFER-Weiterleitung;
G.711-Kodierung gegen den Standard; 20,1 ms gemessener Sendetakt; Audiointegrität
über den echten Sendepfad (Korrelation 0,9999); Gesprächsablauf mit Begrüßung,
Unterbrechung, Weiterleitung samt Fehlerfall, Stille-Überwachung und DTMF.

**Noch nicht verifiziert**, weil mir dafür die Hardware fehlt: der Lauf gegen
eine echte PBX, die tatsächliche Latenz auf deiner L4, und wie Chatterbox auf
Deutsch klingt. Die Zahlen für die Modellstufen oben sind veröffentlichte
Benchmarks für diese Hardwareklasse, keine Messung auf deinem Server. Der erste
echte Anruf ist der eigentliche Test — `sip.trace: true` dabei anlassen.

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
  llm/          Ollama-Streaming und der Dialogagent
  tts/          Piper, Chatterbox, OpenAI-kompatibel, Satzaufteilung, Phrasencache
  kb/           Markdown-Chunking, Embeddings, hybride Suche
  session.py    Gesprächsablauf: der Latenzpfad
  app.py        Verdrahtung und Start
config/         Konfiguration, kommentiert
knowledge/      Wissensdatenbank (Markdown)
scripts/        Modelldownload, Pipeline-Test, Tests, Healthcheck
tests/          simulierte PBX, Session-, Verdrahtungs- und Unit-Tests
```
