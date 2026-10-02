# Wissensdatenbank

Jede `.md`-Datei in diesem Ordner wird beim Start geladen, an den
`##`-Überschriften zerlegt und eingebettet. Pro Anrufer-Frage holt der Agent die
passenden Abschnitte und beantwortet nur daraus. Findet er nichts, sagt er das
und verbindet weiter -- das ist gewollt und besser als eine geratene Antwort.

## Was hier liegt

| Datei | Inhalt |
|---|---|
| `10-it-faq.md` | Allgemeine Schritte, die überall gelten. Kein Gerätemodell, kein Fehlercode, kein Portalname. Kann so bleiben. |
| `vorlagen/*.vorlage` | **Wird nicht geladen.** Vorlagen für alles, was nur bei euch gilt. |

Die Vorlagen in Betrieb nehmen:

```bash
cp knowledge/vorlagen/00-unternehmen.md.vorlage knowledge/00-unternehmen.md
cp knowledge/vorlagen/20-eigene-geraete.md.vorlage knowledge/20-eigene-geraete.md
# Platzhalter ersetzen, dann:
sudo docker compose restart
```

Der Start warnt mit `knowledge base still contains PLATZHALTER in ...`, solange
noch Vorlagentext drinsteht. Jede dieser Zeilen sagt der Agent irgendwann am
Telefon, als wäre sie wahr.

## Wie man gute Abschnitte schreibt

* **Eine Frage pro `##`-Überschrift**, formuliert wie ein Anrufer sie stellt
  ("Der Drucker druckt nicht"), nicht wie ein Handbuch ("Druckerstörungen").
  Die Überschrift ist das, worauf die Suche anschlägt.
* **Zwei bis vier kurze Sätze** als Antwort. Der Agent liest sie am Telefon vor,
  also keine Listen, keine Tabellen, keine Klammern.
* **Zahlen ausschreiben**: "acht bis achtzehn Uhr", nicht "8-18h". Ziffern
  werden zwar umgesetzt, ausgeschrieben klingt es aber sicher richtig.
* **Fehlercodes und Modellnummern genau wie am Gerät.** `E-512` wird als
  `e-512` *und* als `512` indexiert, der Anrufer wird also gefunden, egal wie er
  es ausspricht.
* **Lieber löschen als raten.** Eine fehlende Öffnungszeit führt zur
  Weiterleitung, eine falsche zu einem verärgerten Anrufer.
* **Fälle, die zu einem Menschen gehören, hinschreiben** ("das macht ein
  Mitarbeiter"). Das Modell liest das und leitet weiter, statt zu improvisieren.
* `<!-- Kommentare -->` werden vor dem Einbetten entfernt, Notizen für
  Kolleginnen und Kollegen sind also unschädlich.

## Grenzen

Abschnitte über ~900 Zeichen werden weiter zerlegt; ein Absatz, der
zusammengehört, sollte darunter bleiben. Der Index wird in
`cache/kb.npz` zwischengespeichert und automatisch neu gebaut, sobald sich
Text oder Einbettungsmodell ändern -- Neustart genügt, nichts zu löschen.
