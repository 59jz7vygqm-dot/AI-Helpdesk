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
| Qwen3-TTS | ~300–700 ms | ~4–8 GB | am besten, Apache-2.0 | `TTS_PROFILE=qwen` |

Warum nicht alles zusammen im Standard-Image steckt: Qwen-TTS pinnt
`transformers==4.57.3`, Chatterbox `transformers==5.2.0`, und Chatterbox zusätzlich
`torch==2.6.0` exakt. Das sind reale Konflikte, keine Vermutungen — deshalb wählt
`TTS_PROFILE` beim Bauen genau eine Stimme, und der Standard ist die, die immer
durchläuft.

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

**Stufe 1: Chatterbox** (natürlich, klonbar, MIT-Lizenz, GPU ~3 GB). Für eine
natürliche Stimme am Telefon ist das die realistische Wahl — Qwen ist dafür zu
langsam, siehe Stufe 2. Image neu bauen und umstellen:

```bash
echo "TTS_PROFILE=quality" >> .env     # sudo löscht sonst die Variable
sudo docker compose build              # ~15 Min, zieht torch
cp config/profiles/quality-chatterbox.yaml config/config.yaml
sudo docker compose up -d --build
sudo docker compose exec helpdesk python3 /app/scripts/try_pipeline.py "Test"
```

Nach dem Start steht im Log, was die Stimme wirklich kann:

```
voice backend: chatterbox
voice speed: rtf <gemessen> (<x>s für <y>s Sprache)
```

Alles unter 0,5 ist für einen Anruf brauchbar, über 1,0 nicht — dann warnt der
Start von selbst. Gemessen wird nach dem Warmlaufen an einem normalen Satz, also
ohne Kernel-Kompilierung: das ist die Geschwindigkeit, die ein Anrufer bekommt.

Eigene Stimme klonen — 10–20 Sekunden klare Aufnahme nach `./voices/` legen:

```yaml
tts:
  backend: chatterbox
  chatterbox:
    reference_audio: /models/piper/meine-stimme.wav
```

**Stufe 2: Qwen3-TTS** — nicht für laufende Anrufe. Gemessen auf einer Tesla L4:

```
qwen3-tts: 37 chars -> 2640 ms audio in 35857 ms (rtf 13.58)
```

35,9 Sekunden für 2,6 Sekunden Sprache beim ersten Satz, warm noch etwa Echtzeit.
Ein Echtzeitfaktor über 1 heißt: der Anrufer wartet jeden Satz ab, und daran
ändert keine Einstellung etwas. Dazu sind die eingebauten Sprecher (`aiden`,
`dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`, `uncle_fu`, `vivian`)
keine deutschen Stimmen — Deutsch kommt mit Akzent heraus.

**Für eine natürliche deutsche Stimme am Telefon ist Chatterbox (Stufe 1) die
Antwort, nicht Qwen.** Qwen bleibt im Projekt für Fälle ohne Zeitdruck
(Ansagen vorrendern, Vergleichsaufnahmen) und weil es dokumentiert, wie ein
GPU-Backend angebunden wird.

Wenn du es trotzdem hören willst:

```bash
echo "TTS_PROFILE=qwen" >> .env
sudo docker compose build       # ~20 Min, zieht torch
cp config/profiles/quality-qwen.yaml config/config.yaml
sudo docker compose up -d --build && sudo docker compose logs -f
```

Beim ersten Start lädt das Modell ~5 GB von Hugging Face. Achte auf diese Zeilen:

```
voice backend: qwen3
language 'German' -> 'german'
using speaker 'aiden' (available: aiden, dylan, eric, ono_anna, ryan, ...)
voice speed: rtf 13.58 -- 35.9s to synthesise 2.6s of speech. Above 1.0 ...
```

Der Sprecher wird gegen die Liste des Modells geprüft: steht in der Konfiguration
nichts, nimmt es den ersten und protokolliert alle verfügbaren. Dann einen davon
in `config/config.yaml` unter `tts.qwen3.speaker` eintragen. Dasselbe gilt für die
Sprachbezeichnung — ein falscher Wert wird korrigiert, nicht quittiert mit einem
Fehler mitten im Anruf.

`voice speed:` wird nach dem Warmlaufen an einem normalen Satz gemessen, also mit
schon kompilierten Kernels. Das ist die Geschwindigkeit, die ein Anrufer bekommt.

**Wichtig zum Paketnamen:** Das richtige PyPI-Paket heißt **`qwen-tts`**. Es gibt
außerdem ein `qwen3-tts`, das ist ein fremdes Kommandozeilen-Werkzeug für Apple
Silicon (hängt an `mlx`) und auf NVIDIA grundsätzlich nicht lauffähig. Ich habe
dieses Projekt zuerst darauf aufgebaut — der Build lief durch, der Container starb
beim Import.

**`qwen` und `quality` schließen sich aus:** Qwen-TTS pinnt `transformers==4.57.3`,
Chatterbox `transformers==5.2.0`. Eins von beiden, nicht beides.

Eigene Stimme klonen — Aufnahme nach `./voices/`:

```yaml
tts:
  backend: qwen3
  qwen3:
    mode: clone
    reference_audio: /models/piper/meine-stimme.wav
    reference_text: "Guten Tag, Sie sprechen mit dem telefonischen Service."
```

Oder die Stimme beschreiben: `mode: design` plus
`instruct: "ruhige, freundliche Frauenstimme, mittleres Tempo"`.

**VRAM:** 1.7B braucht ~8 GB, mit Whisper und Sprachmodell etwa 15 von 23 GB.
Wird es eng: `model_id: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`.

Alternativ gibt es `tts-server/` — dieselbe Stimme als eigener Dienst hinter einer
OpenAI-kompatiblen Schnittstelle, falls du die Abhängigkeiten getrennt halten
willst. Für den Normalfall ist `TTS_PROFILE=qwen` der einfachere Weg.

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

`knowledge/10-it-faq.md` enthält allgemeine Schritte, die überall gelten
(Drucker, Bildschirm, VPN, Outlook) — die kann bleiben. Alles, was nur bei euch
gilt, kommt aus den Vorlagen:

```bash
cp knowledge/vorlagen/00-unternehmen.md.vorlage knowledge/00-unternehmen.md
cp knowledge/vorlagen/20-eigene-geraete.md.vorlage knowledge/20-eigene-geraete.md
```

Platzhalter ersetzen: Firmenname, Erreichbarkeit, Druckermodelle, Fehlercodes,
Portalname, VPN-Client. Solange noch Vorlagentext drinsteht, warnt der Start mit
`knowledge base still contains PLATZHALTER in ...`. **Lieber einen Abschnitt
löschen als raten** — ohne Treffer leitet der Agent weiter, mit einer falschen
Angabe liest er sie überzeugt vor.

Dazu den Firmennamen in `config.yaml` setzen (`dialog.company` und
`dialog.greeting`); standardmäßig steht dort neutral „unserem Unternehmen".

`knowledge/README.md` erklärt, wie man Abschnitte schreibt, damit die Suche sie
findet. Der Index baut sich beim Start automatisch neu, sobald sich der Text
ändert.

### 5. Starten

```bash
docker compose up -d --build
docker compose logs -f
```

Erwartete Ausgabe:

```
voice backend: piper
loading ASR model large-v3-turbo (cuda, int8_float16)
ASR model ready in 4.2s
loading Piper voice /models/piper/de_DE-thorsten-high.onnx (cuda=False)
knowledge sources: 10-it-faq.md(13), 00-unternehmen.md(4)
knowledge base built: 17 chunks from /app/knowledge in 2.1s
warmup complete in 12.4s
registered as 900, refreshing in 225s
helpdesk ready: extension 900 on 192.168.1.10, transfers go to 200
```

Die ersten beiden Zeilen sind die wichtigsten: `voice backend:` sagt, welche
Stimme tatsächlich gewählt wurde, `knowledge sources:` welche Dateien geladen
wurden und mit wie vielen Abschnitten. Beides beantwortet die zwei häufigsten
„warum macht er das nicht"-Fragen aus dem Log.

Dann die Nummer anrufen.

### Nach einem `git pull`: `--build` nicht vergessen

`./config` und `./knowledge` sind als Volume eingehängt und wirken sofort nach
einem Neustart. **`src/` liegt im Image.** Ein `git pull` plus
`docker compose up -d` aktualisiert also die Wissensdatenbank, aber nicht den
Code — der Container läuft dann weiter mit der alten Logik, und zwar ohne sich
zu beschweren.

```bash
git pull
docker compose up -d --build      # die Layer sind gecacht, ~2 Minuten
```

`./scripts/preflight.sh` prüft das inzwischen und meldet
`the image is older than src/`, wenn der Build fehlt.

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

## Zwei Betriebsarten

```yaml
dialog:
  mode: helpdesk      # oder: assistant
```

**`helpdesk`** (Standard): antwortet nur aus der Wissensdatenbank, leitet sonst
weiter. Das ist das Verhalten für den echten Einsatz — es sagt lieber „weiß ich
nicht" als etwas zu erfinden.

**`assistant`**: redet auch frei. Smalltalk („Wie geht es dir?"), allgemeine
Fragen, Scherze. Nutzt die Wissensdatenbank, wenn sie etwas hergibt, antwortet
sonst aus allgemeinem Wissen. Weitergeleitet wird nur noch, wenn der Anrufer
ausdrücklich einen Menschen will. Das ist der Modus für eine Demo, bei der
Kollegen einfach mal anrufen und mit dem Modell reden sollen.

Firmenspezifische Angaben — Preise, Termine, Rufnummern, Zuständigkeiten — bleiben
in beiden Modi an die Wissensdatenbank gebunden.

## Stimme auswählen

Welche Stimme am Telefon gut klingt, lässt sich nicht aus dem Modellnamen ablesen.
Alle verfügbaren laden und vergleichen:

```bash
VOICE="de_DE-thorsten-medium de_DE-thorsten-low de_DE-eva_k-x_low de_DE-kerstin-low de_DE-ramona-low" \
  ./scripts/download_models.sh

docker compose exec helpdesk python3 /app/scripts/compare_voices.py
```

Das schreibt pro Stimme eine WAV in `/tmp/voices` — in 8 kHz, also genau so wie
der Anrufer sie hört — und zeigt den Realtime-Faktor jeder Stimme. Unter 0,15 ist
unkritisch, darüber bremst die Stimme jede Antwort.

Danach den Pfad in `config/config.yaml` unter `tts.piper.model_path` eintragen.
`length_scale` leicht unter 1,0 (etwa 0,95) lässt die Stimme etwas lebendiger
klingen und verkürzt die Audiodauer.

## Wenn das Gespräch sich falsch anfühlt

Die Reihenfolge, in der es sich lohnt zu suchen — aus einem echten ersten Anruf
gelernt:

**1. Steht in der Wissensdatenbank überhaupt eine Antwort?** Das ist mit Abstand
der größte Hebel und wird meist zuletzt geprüft. `knowledge/10-it-faq.md` deckt
nur das Allgemeine ab; zu allem Firmenspezifischen findet der Agent nichts,
solange die Vorlagen aus `knowledge/vorlagen/` nicht ausgefüllt sind.
Die Logzeile `kb hits for '...'` zeigt, was er gefunden hat und wie gut es passt:

```
kb hits for 'Das Display ist schwarz': Der Drucker druckt nicht(0.50), Mein Bildschirm bleibt schwarz(0.49)
```

Zwei mittelmäßige Treffer, keiner beantwortet die Frage — und genau dann neigt
ein Sprachmodell dazu, Schritte zu erfinden. Dagegen hilft nicht am Prompt zu
drehen, sondern der Wissensdatenbank einen Abschnitt zu dieser Frage zu geben.

**2. Ist `min_score` hoch genug?** Lieber kein Kontext und weiterleiten als ein
unpassender Treffer, aus dem improvisiert wird. 0,40 ist der Startwert; wenn der
Agent zu oft weiterleitet, in 0,05er-Schritten senken.

**3. Füllwörter — standardmäßig aus.** `dialog.fillers` kann „einen Moment"
einwerfen, während die Antwort entsteht. Das war sinnvoll, solange Antworten
mehrere Sekunden brauchten; bei Antwortzeiten unter einer Sekunde unterbricht es
den Fluss mehr als es die Pause glättet. Die Liste ist leer — füllen, wenn die
Latenz bei dir doch hoch bleibt.

**4. Wiederholungen.** Wenn der Agent dieselbe Antwort mehrfach gibt, ist das
Gespräch für den Anrufer vorbei. Ähnliche Antworten werden jetzt erkannt: beim
zweiten Mal bekommt das Modell seinen eigenen Satz mit dem Hinweis, ihn nicht zu
wiederholen, beim dritten wird weitergeleitet.

**5. Erst dann an der Latenz drehen.** Und dort zuerst die Sprachausgabe: Im Log
steht pro Antwort, wohin die Zeit ging.

```
turn 1: response 5975 ms (asr 0*, kb 2862, llm_ttft 266, tts 5975)
```

Hier ist alles Synthese — das Sprachmodell braucht 266 ms, die Erkennung dank
Vorausberechnung 0 ms. Die Eingabe zu streamen würde an solchen Zahlen nichts
ändern.

Was tatsächlich wirkt, in der Reihenfolge der Wirkung:

| Maßnahme | Effekt | Kosten |
|---|---|---|
| `tts.piper.use_cuda: true` | Realtime-Faktor ~0,3 → unter 0,05 | ~300 MB VRAM, Image mit `TTS_PROFILE=quality` |
| `tts.piper.threads` auf die Kernzahl | Synthese 2–3× schneller auf CPU | nichts |
| `medium`-Stimme statt `high` | ~3× schnellere Synthese | bei 8 kHz nicht hörbar |
| kurzer Prompt + `top_k: 2` | weniger Prefill → `llm_ttft` runter | weniger Kontext pro Antwort |
| `vad.semantic_endpointing` | antwortet beim erkannten Satzende statt nach Ablauf der Pause | spart die halbe Nachlaufzeit |
| `tts.first_chunk_min_chars: 14` | Sprechbeginn nach Teilsatz statt Satz | minimal andere Betonung |
| `vad.end_silence_ms: 320` | 100 ms weniger Wartezeit pro Turn | schneidet eher mal jemanden ab |

**Zum „Live-Processing":** Die Erkennung läuft bereits, während gesprochen wird —
`asr 0*` im Log heißt genau das: das Transkript lag fertig vor, als der Anrufer
verstummte. Zusätzlich wird jetzt geprüft, ob dieses laufende Transkript schon
einen vollständigen Satz zeigt (Punkt oder Fragezeichen, mindestens drei Wörter).
Wenn ja, wird nicht mehr auf `end_silence_ms` gewartet, sondern sofort geantwortet.
Im Test: 29 ms statt 2.000 ms Nachlaufzeit.

**Zum Prefill, weil es leicht übersehen wird:** Das Modell liest bei *jedem* Turn
den System-Prompt, die Gesprächshistorie und die gefundenen Wissenspassagen neu.
Jede Regel, die man dem Prompt hinzufügt, und jeder zusätzliche `top_k`-Treffer
kostet deshalb Zeit bei jeder einzelnen Antwort. Als in diesem Projekt der
System-Prompt von 3.555 auf 1.403 Zeichen gekürzt und `top_k` von 3 auf 2 gesenkt
wurde, fielen rund 700 Tokens Prefill pro Turn weg — vorher war `llm_ttft` auf
380 ms gestiegen, allein durch zusätzliche Prompt-Regeln.

Zum Nachrechnen: `response_ms` im Log ist die Zeit von „Anrufer verstummt" bis
„erstes Audio raus". Die Sprechpause (`end_silence_ms`) kommt davor noch dazu —
das ist die Pause, die der Anrufer wirklich erlebt.

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

## Eigener Asterisk davor — wenn die PBX fremdverwaltet ist

Der Agent registriert sich als normale Nebenstelle, damit er mit einer verwalteten
PBX auskommt: keine Dialplan-Zeile, kein ARI-Zugang, nur Zugangsdaten. Das ist
der einfachste Weg und kostet einen RTP-Hop weniger.

Mit einem eigenen Asterisk dazwischen bekommst du den Dialplan zurück:

```
Anbieter-PBX  --Registrierung als 1250-->  lokaler Asterisk  --Nebenstelle 7000-->  Agent
```

Der lokale Asterisk registriert sich wie ein Tischtelefon bei der Anbieter-PBX,
mit einer **eigenen** Nummer. Der Agent registriert sich bei ihm.

### Einschalten

Beides liegt in diesem Repo, nicht in einem zweiten: ein `.env`, ein `git pull`,
ein `docker compose up -d`. Bei getrennten Repos müssten Ports und Zugangsdaten
über eine Repo-Grenze hinweg übereinstimmen — genau die Art Aufteilung, die
schon einmal dazu geführt hat, dass die Konfiguration das eine sagte und der
Container das andere tat.

In `.env`:

```bash
COMPOSE_FILE=docker-compose.yml:docker-compose.asterisk.yml

# Die Anbieter-PBX und die NEUE Nummer, die sie für diesen Asterisk ausgegeben hat
PROVIDER_HOST=172.16.0.188
PROVIDER_USER=1250
PROVIDER_PASSWORD=...

# Der Agent, jetzt eine Nebenstelle auf dem lokalen Asterisk
SIP_USERNAME=7000
SIP_PASSWORD=ein-lokales-passwort
SIP_BIND_PORT=5080

# Wohin unbeantwortete und weitergeleitete Anrufe gehen
TRANSFER_NUMBER=1272
```

Dann wie immer:

```bash
sudo docker compose up -d --build
sudo docker compose logs -f
```

`COMPOSE_FILE` in `.env` ist der Trick: `docker compose` liest es von selbst, du
musst dir also kein längeres Kommando merken. Zeile raus, und der Asterisk ist
wieder weg.

`SIP_SERVER_HOST` zeigt der Agent-Container danach selbst auf `127.0.0.1` — das
setzt die Overlay-Datei, damit eine alte Zeile in `.env` ihn nicht
versehentlich direkt zur PBX schickt, wo sich beide um dieselbe Registrierung
streiten würden.

### Was das bringt

* **Rückfall auf einen Menschen.** Antwortet der Agent nicht in
  `AGENT_RING_SECONDS` — Container startet neu, Modell lädt, Image wird
  gebaut —, klingelt der Dialplan bei `TRANSFER_NUMBER`. Der Anrufer landet nie
  auf einer toten Nummer. Das ist der Punkt, den der Agent für sich selbst
  prinzipiell nicht lösen kann.
* **Weiterleitung ohne Tricks.** REFER und die DTMF-Rückfallebene existieren nur,
  weil manche PBX `allow_transfer=no` setzt. Hier ist es ein `Dial()`.
* **Aufzeichnung, Warteschlangen, Zeitsteuerung, mehrere Agenten auf einer
  Nummer** — alles Dialplan, kein Anwendungscode.
* **AudioSocket und ARI** werden möglich, also auch die gepflegten Projekte
  (Agent Voice Response und andere), ohne an der Anbieter-PBX etwas zu ändern.

### Was es kostet

* Ein Dienst mehr, der laufen muss. Fällt dessen Registrierung aus, ist die
  Nummer tot — vorher hing das an einem Dienst, jetzt an zwei.
* Ein zusätzlicher RTP-Hop. Im LAN Einzelstellen von Millisekunden, gegen ~700 ms
  Antwortzeit also nichts — **solange nicht transcodiert wird.** Beide Beine sind
  deshalb auf `alaw` festgenagelt.
* Zwei Audio-Beine heißen zwei Gelegenheiten für einseitigen Ton.

### Die zwei Kollisionen, die das sonst sofort zerlegen

Bei `network_mode: host` gibt es kein Docker-NAT, das sie verdeckt.

1. **SIP-Port 5060.** Asterisk nimmt ihn, der Agent geht auf `SIP_BIND_PORT`.
2. **RTP-Bereich.** Asterisks Standard ist **10000–20000** und enthält den
   Bereich des Agenten (**16000–16200**). Beide Prozesse greifen dann nach
   denselben Ports, der Anruf kommt zustande und ist **stumm**. Die
   Overlay-Datei setzt Asterisk deshalb auf 10000–15999.

### Zugangsdaten: zwei Konten, nicht eins

`PROVIDER_USER` und `SIP_USERNAME` müssen verschieden sein. Ein Satz
Zugangsdaten ist ein Gerät: registrieren sich lokaler Asterisk und Agent beide
als dieselbe Nebenstelle, verwirft die PBX eine der beiden Registrierungen —
und zwar sporadisch, was stundenlanges Suchen bedeutet.

### Konfiguration

Sie wird beim Start aus `asterisk/templates/*.tmpl` gerendert, damit Passwörter
in `.env` bleiben und nicht in einer Image-Schicht. Anpassen heißt: Template
ändern, dann `docker compose restart asterisk`.

Der Entrypoint weigert sich zu starten, wenn etwas leer gerendert hat. Das ist
nicht vorsorglich gemeint — beim Bauen passierte genau das zweimal: `envsubst`
liest die Umgebung, nicht die Shell, also wurde aus einer gesetzten aber nicht
exportierten Variable ein stilles `Dial(PJSIP/,)` und ein `server_uri` ohne
Port. Eine Konfiguration, die lädt, registriert und nichts tut.

Asterisks eigene `${EXTEN}` und `${DIALSTATUS}` überleben das Rendern; es werden
nur die Namen ersetzt, die der Entrypoint selbst kennt.

**Getestet ist das Rendern, nicht der Betrieb** — in dieser Entwicklungsumgebung
gibt es keinen Docker-Daemon und keine PBX. Die pjsip-Struktur folgt der
offiziellen Doku zu
[res_pjsip_outbound_registration](https://docs.asterisk.org/Certified-Asterisk_18.9_Documentation/API_Documentation/Module_Configuration/res_pjsip_outbound_registration),
inklusive der `type=transport`-Sektion, ohne die pjsip überhaupt nicht lauscht.

### Diagnose, in dieser Reihenfolge

```bash
sudo docker compose logs -f asterisk
sudo docker compose exec asterisk asterisk -rx 'pjsip show registrations'
sudo docker compose exec asterisk asterisk -rx 'pjsip show endpoints'
sudo docker compose exec asterisk asterisk -rx 'pjsip set logger on'   # SIP-Mitschnitt
```

`pjsip show registrations` muss `Registered` zeigen. Steht dort
`Rejected`, lehnt die Anbieter-PBX ab: falsches Passwort, oder sie erlaubt
keinen Asterisk mit diesen Daten.

### Vorher mit dem Anbieter klären

* Darf sich ein Asterisk mit diesen Zugangsdaten registrieren? Manche Anbieter
  prüfen den User-Agent oder verbieten Trunking im Vertrag.
* Eine eigene Nummer für den lokalen Asterisk, siehe oben.

### Ob es sich lohnt

Für den Demo-Betrieb: nein, die direkte Registrierung ist weniger beweglich und
läuft schon.

Für den Produktivbetrieb: ja — vor allem wegen des Rückfalls auf einen Menschen.
Ein Agent, der sich selbst überwacht, kann nicht melden, dass er tot ist.

---

## Fehlersuche

**Registrierung schlägt fehl** — `sip.trace: true` setzen und Logs ansehen. Meist
falsches Passwort, oder die PBX erwartet einen separaten Auth-Namen
(`sip.auth_username`). Bei `403 Forbidden` lässt die PBX die IP des Containers
nicht zu.

**Kein Ton, und auf dem Host läuft noch ein Asterisk** — Asterisks
Standard-RTP-Bereich ist 10000-20000 und überlappt den des Agenten
(16000-16200). Bei `network_mode: host` greifen beide nach denselben Ports.
`RTP_PORT_START`/`RTP_PORT_END` oder Asterisks `rtpstart`/`rtpend` verschieben;
siehe den Abschnitt "Eigener Asterisk davor".

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

**Er hört überhaupt nicht zu und entschuldigt sich nur** — im Log wechseln sich
`barge-in on ...` und `nothing recognised` ab, `turns=0` am Ende. Das ist eine
Schleife: der Anrufer redet dazwischen, der Agent antwortet mit „das habe ich
nicht verstanden", redet damit über den Anrufer, dessen nächste Worte wieder
dazwischenfunken. Zwei Dinge halten sie an, beide eingebaut:

* Die Sprache, die den Barge-in *ausgelöst* hat, bleibt Teil der Äußerung. Vorher
  verbrauchte der Detektor seine 200 ms und das Erkannte begann erst danach.
* Unter `asr.min_utterance_ms` (350) wird gar nicht geantwortet, auch nicht
  entschuldigt. Im Log steht dann `ignoring 140 ms of audio: too short`.

Tritt es weiter auf, ist fast immer die Stimme zu langsam: solange der Agent
synthetisiert, geht jede Silbe des Anrufers durch den Barge-in-Detektor statt
durch den Endpointer. `voice speed:` beim Start prüfen.

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

**„nothing recognised" trotz Sprechen** — das Log nennt jetzt den Grund: die Dauer
der Äußerung, den Pegel in dBFS und ob ein Transkript verworfen wurde. Typische
Fälle:

- *Pegel unter etwa −40 dBFS*: Die Leitung ist zu leise, oder die PBX schickt
  kaum Audio. Prüfen, ob der Codec stimmt (im Log `answered … with PCMA`).
- *`discarded transcript`*: Erkannt, aber als unsicher verworfen.
  `asr.min_avg_logprob` auf `-1.4` lockern oder `asr.max_no_speech_prob` auf
  `0.85`.
- *Direkt nach der Begrüßung, ohne dass jemand sprach*: Die Sprachaktivitäts-
  erkennung hat auf Leitungsrauschen angeschlagen. `vad.aggressiveness` auf 3,
  oder `vad.start_frames` auf 5.

**Die Antwort dauert, bis sie kommt** — im Log auf `rtf` achten. Über 0,3 heißt,
die Sprachausgabe ist der Engpass; dann die `medium`- statt der `high`-Stimme
nehmen (bei 8 kHz hörst du keinen Unterschied) und `tts.piper.threads` setzen.

**Er redet zu lang** — `dialog.max_sentences: 2` und `llm.options.num_predict`
runter. Am Telefon sind zwei Sätze plus Rückfrage besser als eine vollständige
Anleitung.

**Er erfindet Rückfragen** („Haben Sie eine Bestellnummer?", obwohl nirgends von
Bestellungen die Rede war) — passiert, wenn die Wissensdatenbank zur Frage nichts
hergibt. Der Prompt unterscheidet jetzt: Gesprächsführung darf er frei
formulieren, Tatsachen über das Unternehmen nur aus dem WISSEN. Wenn es dort
nichts gibt, soll er einmal gezielt nachfragen und dann weiterleiten.

**Er leitet bei „Vielen Dank" weiter** — behoben: eine Verabschiedung beendet das
Gespräch jetzt im Code, ohne Modellaufruf. Abschaltbar über
`dialog.farewell_ends_call: false`.

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
asterisk/       optionaler lokaler Asterisk (Templates + Entrypoint)
scripts/        Preflight, Modelldownload, Pipeline-Test, Tests, Healthcheck
tts-server/     Qwen3-TTS als eigener Dienst (Python 3.13, OpenAI-kompatibel)
tests/          simulierte PBX, Session-, Verdrahtungs-, Unit- und TTS-Tests
```
