# Wissensdatenbank

Jede `.md`-Datei in diesem Ordner wird beim Start indexiert. Der Index liegt in
`/models/kb-index.npz` und wird automatisch neu gebaut, sobald sich der Text
ändert — es gibt keinen manuellen Schritt.

## Wie man schreibt, damit es am Telefon funktioniert

Der Agent liest Treffer nicht vor, er antwortet daraus. Trotzdem entscheidet die
Struktur, ob er die richtige Stelle findet:

1. **Eine Überschrift pro Thema.** Gesplittet wird an Überschriften, also wird
   aus jeder `##`-Sektion ein eigener Treffer. Eine Sektion sollte eine Frage
   beantworten, nicht fünf.
2. **Die Frage in die Überschrift.** `## Wie setze ich mein Passwort zurück?`
   trifft besser als `## Passwort`, weil Anrufer in Fragen sprechen.
3. **Kurze Sektionen**, 3–8 Sätze. Längere werden an Absätzen getrennt, was
   Zusammenhang kosten kann.
4. **Begriffe der Anrufer benutzen**, nicht die internen. Wer „Drucker geht
   nicht" sagt, findet nichts unter „MFP-Störungsbeseitigung".
5. **Fehlercodes und Modellnummern ausschreiben** (`E-512`, `HP-4500`). Die
   werden sowohl zusammen als auch getrennt indexiert, also findet der Agent sie
   auch, wenn der Anrufer „Fehler fünf zwölf" sagt.
6. **Keine Tabellen, keine Links, keine Bilder.** Das kann niemand vorlesen.
7. **Was der Agent nicht sagen darf, gehört nicht hier rein.** Er antwortet
   ausschließlich aus diesen Dateien — was hier steht, kann am Telefon landen.

## Was absichtlich nicht hier rein gehört

Alles, was eine Entscheidung oder Verantwortung braucht: Preiszusagen,
Kündigungen, Reklamationen, Rechtliches. Dafür ist die Weiterleitung da. Je
klarer die Lücke, desto eher leitet der Agent weiter statt zu improvisieren.
