# Abschlussprojekt: Autonomer Rennwagen

Dieses "Framework" enthält eine grundlegende Struktur für die Umsetzung einer modularen Pipeline für die Gymnasium Car Racing Simulation ([link](https://gymnasium.farama.org/environments/box2d/car_racing/)).

## 0. Requirements

- Python >3.12

## 1. Installation

Wir empfehlen die Pakete in einem virtuellen environment zu installieren. Für Details, siehe die Einführung in Python. 

1. Installieren aller python Pakete: `pip install "gymnasium[box2d]" numpy matplotlib scipy opencv-python`
2. In den Source Ordner wechseln `cd <working-dir>`
3. Testen der Installation: `python test_installation.py`. Danach sollte sich die Simulation öffnen und das Fahrzeug zufällig bewegen.

## 2. Dateistruktur

Das Projekt ist wie folgt zu strukturieren:

```python
working-dir/
| - data/ # Bilder zu getesteten Varianten der Spurerkennung und Pfadplanung
| - docs/ # Aufgabenstellung, Bewertung, Abschlusspräsentation, Quellen
| - .gitignore # Definition von Dateien, die nicht von git getrackt werden sollen
| - car.py # Beschreibt das Fahrzeug und enthält die gesamte Pipeline
| - CHANGES.md # Wichtige Designentscheidungen
| - env_wrapper.py # Wrapper für die Gymnasium Car Racing Simulation
| - FINDINGS.md # Erkenntnisse über Simulation, Fahrzeug und Kamera
| - geometry.py # Umrechnung Pixel -> Fahrzeugkoordinaten, Pfad und Krümmung für die Regler
| - input_controller.py # Manuelle Steuerung des Fahrzeugs
| - lane_detection.py # Modul zur Spurerkennung
| - lateral_control.py # Modul zur Querregelung
| - longitudinal_control.py # Modul zur Längsregelung
| - main.py # Hauptdatei, die die Pipeline über 5 Iterationen ausführt
| - path_planning.py # Modul zur Pfadplanung
| - pyproject.toml # Abhängigkeiten
| - README.md # Diese Datei
| - test_installation.py # Testet die Installation
| - test_lane_detection.py # Testet die Spurerkennung
| - test_lateral_control.py # Testet die Querregelung
| - test_longitudinal_control.py # Testet die Längsregelung
| - test_path_planning.py # Testet die Pfadplanung
| - test_pipeline.py # Testet die gesamte Pipeline
| - TODO.md # Aufgabenliste
```

## 3. Ausführung

Die test Dateien können mit `python <test-file>.py` ausgeführt werden. Die Simulation wird sich öffnen und das Fahrzeug wird sich entsprechend der Pipeline bewegen. Um das Endergebnis zu simulieren, kann die `main.py` Datei ausgeführt werden. Sie führt die Pipeline über 5 Iterationen aus und berechnet den durchschnittlichen Score.

## 4. Hinweise

Die Dateien `main.py`, `env_wrapper.py` und `input_controller.py` dürfen nicht verändert werden, damit die Ergebnisse vergleichbar bleiben. Alle anderen Dateien können beliebig verändert werden.

## 5. Commit Richtlinien

Fange jeden Commit damit an, an welchem Teilpaket gearbeitet wurde. (`LaneDetection`, `PathPlanning`, `LateralControl`, `LongitudinalControl`) oder in Ausnahmefällen `Other`.

Darauf soll eine kurzbeschreibung vorgenommen werden.

Appell: Dokumentiere Design Entscheidungen in `CHANGES.md`

## 6. Gemeinsame Geometrie (`geometry.py`)

Beide Regler rechnen nicht in Pixeln, sondern in Einheiten der Simulation, in einem Fahrzeugkoordinatensystem (vorne, rechts) mit Ursprung im Fahrzeug.

- **Skalierung**: Das 1000x800 Fenster wird auf 96x96 Pixel verkleinert. Quer entspricht eine Einheit 1,56 px, längs 1,94 px. Winkel und Krümmungen direkt in Pixeln wären verzerrt.
- **Zoom am Start**: Die Kamera zoomt in der ersten Sekunde ein. Der Zoom pro Frame ist bekannt (`camera_zoom`) und wird herausgerechnet. Bis zum halben Zoom ist ein Pixel zu grob, solange ignorieren die Regler den Pfad.
- **Nächster Pfadpunkt**: Lotfußpunkt auf dem nächsten Segment statt nächster Stützpunkt. Der Fehler ändert sich dadurch stetig. Lücken im Pfad (z. B. Start/Ziel in der Ground Truth) beenden den Pfad.
- **Krümmung**: Kreis durch drei Punkte (Menger-Krümmung) auf dem gleichmäßig neu abgetasteten Pfad.

## 7. Querregelung (`lateral_control.py`)

Die Querregelung nutzt den **Stanley-Regler** an der Vorderachse:

```
delta = k_psi * psi + arctan(k * e / (k_soft + v)) + k_d * dpsi/dt
k_psi = L / (L + v * tau)
```

| Parameter | Standard | Bedeutung |
| --------- | -------- | --------- |
| `gain` | 1.0 | k, Gewicht der Querabweichung e (1/s) |
| `softening` | 1.0 | k_soft, begrenzt die Verstärkung beim Anfahren (Einheiten/s) |
| `heading_time` | 0.1 | tau, kürzeste Zeit, in der der Kurswinkelfehler psi ausgeregelt wird (s) |
| `damping` | 0.02 | k_d, D-Anteil auf psi (s) |
| `damping_filter` | 0.1 | Zeitkonstante des Tiefpasses auf den D-Anteil (s) |
| `preview` | 12.0 | Sehnenlänge, über die psi gemessen wird (Einheiten) |

Erweiterungen gegenüber dem Standard-Stanley:

- **Kurswinkel über eine Sehne**: Statt der Tangente am Lotfußpunkt wird die Richtung zum Pfadpunkt 12 Einheiten weiter genommen. Das glättet das Pixelrauschen und schneidet Kurven leicht an, dadurch sind höhere Kurvengeschwindigkeiten möglich.
- **Kurswinkel-Verstärkung sinkt mit dem Tempo**: Lenkt das Auto um psi, dreht es sich mit `v * psi / L`. Beim klassischen Stanley (k_psi = 1) wäre der Kurswinkel bei 80 Einheiten/s in 0,04 s ausgeregelt, schneller als die Verzögerung im Regelkreis (ein Frame Kamera, Lenkmotor). Das Auto schwingt dann mit den Frames hin und her. Mit `k_psi = L / (L + v * tau)` dauert das Ausregeln `tau + L / v`, also nie unter 0,1 s. Beim Anfahren bleibt k_psi nahe 1.
- **Dämpfung (D-Anteil)**: Die Änderung des Kurswinkelfehlers entspricht der Differenz aus Gierrate des Pfads und des Autos. Der Term dämpft das Aufschwingen bei hohem Tempo. Ein Tiefpass (0,1 s) filtert das Pixelrauschen heraus, ungefiltert hat der D-Anteil das Rauschen der Spurerkennung verstärkt.
- **Ausgabe in rad**: Die Lenkaktion ist in der Simulation der Soll-Radwinkel in rad (Anschlag ±0,4 rad). Der Regler gibt ihn deshalb direkt aus.
- **Pfadaussetzer**: Ohne Pfad klingt der letzte Lenkwinkel pro Frame mit Faktor 0,9 ab. Kurze Aussetzer werden überbrückt, ohne dass das Auto dauerhaft eingelenkt bleibt.

Als Alternative wurde Pure Pursuit umgesetzt und verglichen (Kreisbogen von der Hinterachse durch den Pfadpunkt im Abstand `l_d = max(0.25 * v, 6)`). Stanley war robuster (Abschnitt 9), Pure Pursuit ist deshalb nicht mehr im Code. Der Stand mit beiden Reglern liegt im Commit `5e750cd`.

## 8. Längsregelung (`longitudinal_control.py`)

**Zielgeschwindigkeit** aus dem sichtbaren Pfad:

1. Kurvengeschwindigkeit an jedem Pfadpunkt: `v = sqrt(a_quer / kappa)`.
2. Bremsweg: Ein Punkt im Abstand s erlaubt `sqrt(v_kurve^2 + 2 * a_brems * s)`. Das Minimum über alle Punkte ist die Zielgeschwindigkeit.
3. Sichtweite: Hinter dem Pfadende kann eine Haarnadelkurve kommen, bis dorthin muss das Auto auf `min_speed` bremsen können. Weniger Sicht bedeutet automatisch langsamer.

Das Krümmungsfenster von ±8 Einheiten lässt kurze Kurven flacher wirken als lange. Das passt zum Fahrverhalten, kurze Kurven schneidet das Auto stärker an. Mit ±4 sinkt der Score auf 924,0 (Haarnadel 913,3, 26/28 Runden).

**PID-Regler**: Der Ausgang ist eine Soll-Beschleunigung. Über gemessene Kennwerte (Vollgas ≈ 44 Einheiten/s², Bremse ≈ 300 · b Einheiten/s²) wird sie in Gas oder Bremse umgerechnet.

- D-Anteil auf die gemessene Beschleunigung statt auf den Fehler, Sprünge der Zielgeschwindigkeit erzeugen so keinen Stoß.
- Anti-Windup: Integriert wird nur, solange der Ausgang nicht in der Begrenzung steckt.
- Kammscher Kreis: Die Querbeschleunigung wird aus Lenkwinkel und Tempo geschätzt (`v² · tan(delta) / L`), nur die restliche Haftung bekommt das Gas. Ohne diese Begrenzung brechen die Hinterräder in Kurven aus.
- Bremse höchstens 0,7, darüber blockieren die Räder nach einigen Schritten.

| Parameter | Standard | Bedeutung |
| --------- | -------- | --------- |
| `kp`, `ki`, `kd` | 8.0, 1.0, 0.05 | PID auf die Soll-Beschleunigung |
| `max_speed`, `min_speed` | 100, 30 | Grenzen der Zielgeschwindigkeit (Einheiten/s) |
| `lateral_accel` | 220 | erlaubte Querbeschleunigung auf der Mittellinie (Einheiten/s²) |
| `brake_accel` | 150 | eingeplante Verzögerung vor Kurven (Einheiten/s²) |
| `curvature_window` | 8 | Punkte links und rechts für die Krümmung |
| `max_brake` | 0.7 | maximaler Bremswert |

`lateral_accel` liegt an der Haftgrenze von 219 Einheiten/s². 240 war auf kurzen Kurven minimal schneller, weil das Auto sie anschneidet. In langen Haarnadeln geht das aber nicht: Dort rutschte das Auto nach außen, verlor die Strecke und drehte sich (U-Turn-Test in Abschnitt 9).

## 9. Performance-Analyse der Regler

Gemessen wurde mit einem eigenen Benchmark-Skript, das nur zum Abstimmen diente und nicht Teil der Abgabe ist. Es fährt ohne Fenster und parallel über feste Seeds und misst Score, Rundenabschluss, Querabweichung zur echten Mittellinie, Offroad-Schritte und die Laufzeit jedes Moduls.

Abgestimmt wurde auf den Seeds 0-23, bewertet auf den Seeds 1000-1049, die beim Abstimmen nicht verwendet wurden. Für den U-Turn-Test wurden aus den Seeds 2000-3199 die 24 schärfsten Kehren nach einer Geraden gesucht (~170° auf 42 Einheiten), dazu kommen die schwierigen Seeds aus `FINDINGS.md`:

`2006, 2730, 2788, 2440, 3171, 2713, 3003, 2022, 2424, 2333, 2892, 2981, 2616, 2342, 2526, 2473, 3020, 2667, 2421, 3159, 2720, 2431, 2846, 2378, 180888, 373468, 693592, 197453`

### Ergebnisse (Seeds 1000-1049, komplette Pipeline)

Spurerkennung und Pfadplanung (Stand 23.10.) sind in allen Zeilen gleich, verändert wurden nur die Regler. Schritte pro Runde nur über abgeschlossene Runden.

| Variante | Ø Score | Std | Min | Runden | Schritte/Runde | Offroad |
| -------- | ------: | --: | --: | -----: | -------------: | ------: |
| **Final**: Stanley (Sehne 12, k_psi mit dem Tempo, gefilterte Dämpfung), Zielgeschwindigkeit aus Krümmung | **926,5** | 7,5 | 904,6 | 50/50 | 735 | 0 |
| Stanley mit k_psi = 1, ungefilterte Dämpfung (Stand 23.10.) | 925,5 | 7,5 | 903,2 | 50/50 | 746 | 0 |
| Stanley mit k_psi = 1, ohne Dämpfung | 925,0 | 7,6 | 902,8 | 50/50 | 750 | 0 |
| Pure Pursuit (`lookahead_gain` 0,25, wieder entfernt) | 926,7 | 7,7 | 902,9 | 50/50 | 733 | 0 |
| `lateral_accel` 240 (alter Standard) | 735,7 | 343,3 | 51,1 | 37/50 | 735 | 6100 |
| Klassischer Stanley (Tangente, ohne Dämpfung) | 380,8 | 256,9 | 40,9 | 1/50 | 768 | 18191 |
| Erste Version (klassischer Stanley, `lateral_accel` 150, Fenster 4) | 907,9 | 27,4 | 781,4 | 45/50 | 849 | 44 |
| Konstante Zielgeschwindigkeit 50 | 860,4 | 51,7 | 684,4 | 18/50 | 955 | 0 |

Offroad zählt Schritte, in denen alle vier Räder neben der Strecke sind.

### U-Turn-Test (28 Seeds mit Kehren, komplette Pipeline)

| Variante | Ø Score | Min | Runden | Offroad |
| -------- | ------: | --: | -----: | ------: |
| **Final** | **916,6** | 881,8 | 27/28 | 0 |
| k_psi = 1, ungefilterte Dämpfung (Stand 23.10.) | 915,1 | 871,4 | 26/28 | 0 |
| Pure Pursuit | 895,7 | 718,2 | 23/28 | 603 |
| `lateral_accel` 240 (alter Standard) | 475,2 | -10,4 | 11/28 | 7927 |

Die fehlende Runde beim finalen Stand ist eine lange Strecke (Seed 2006), die im Zeitlimit von 1000 Schritten zu 98 % gefahren wird.

### Wackeln auf Geraden (Seeds 1000-1049, komplette Pipeline)

Gemessen nur auf Geraden. Hochfrequente Gierrate = Gierrate minus gleitender Mittelwert über 11 Frames, also nur das Zittern ab ~4,5 Hz.

| Variante | Lenkänderung pro Frame | Vorzeichenwechsel der Lenkung | Hochfrequente Gierrate |
| -------- | ---------------------: | ----------------------------: | ---------------------: |
| **Final** | **0,005 rad** | **6,2 /s** | **0,09 rad/s** |
| k_psi = 1, ungefilterte Dämpfung (Stand 23.10.) | 0,026 rad | 17,9 /s | 0,48 rad/s |

Mit fester Lenkung giert das Auto auf der Geraden nicht (0,002 rad/s), das Wackeln entstand also im Regelkreis. Ursachen: Der Pfad der Spurerkennung springt von Frame zu Frame um Bruchteile eines Pixels, der ungefilterte D-Anteil hat das verstärkt, und k_psi = 1 regelt bei hohem Tempo schneller, als das Auto reagieren kann.

Einordnung:

- **Pipeline vs. Ground Truth**: Auf der Ground-Truth-Mittellinie erreichen die Regler auf denselben Seeds 926,9 statt 926,5. Die Regler kommen also mit dem Pixelrauschen der Spurerkennung zurecht.
- **Haarnadeln**: Mit `lateral_accel` 240 war das Auto in langen Kehren zu schnell, rutschte nach außen und drehte sich, sobald die Spurerkennung die Strecke verlor. Aufgefallen ist das erst, seit die Mittellinie in Kehren länger sichtbar ist. Vorher hat die kurze Sichtweite das Auto zufällig abgebremst.
- **Sehne statt Tangente** ist der wichtigste Baustein. Die Tangente aus pixelgenauen Punkten rauscht so stark, dass der klassische Stanley mit derselben Längsregelung fast keine Runde schafft.
- **Pure Pursuit** ist auf normalen Strecken etwa 1 Punkt schneller, weil es Kurven stärker schneidet. In Haarnadeln kommt es aber von der Strecke ab (5 von 28 Runden nicht geschafft). Bei zu kurzer Vorausschau (`lookahead_gain` 0,15) schaukelt es sich auf. Stanley ist robuster und bleibt deshalb als einziger Regler im Code.
- **Dämpfung**: Mit der Ground-Truth-Mittellinie dämpft der D-Anteil das Zittern, mit der Pipeline hat er ungefiltert das Pixelrauschen verstärkt (doppelt so viel Lenkänderung pro Frame). Gefiltert bringt er beides: Ohne D-Anteil fehlen auf den Abstimm-Seeds 0,7 Punkte, in den Haarnadeln 1,3 Punkte (gemessen mit konstantem k_psi = 0,4).
- **Kurswinkel-Verstärkung**: Je kleiner k_psi bei hohem Tempo, desto ruhiger und etwas schneller fährt das Auto, bis es kippt. Mit `tau` 0,1 s liegt k_psi bei 80 Einheiten/s bei 0,29. Mit `tau` ~0,2 s schafft das Auto viele Runden nicht mehr. Ein konstantes k_psi kippt schon bei 0,25, obwohl `tau` 0,13 s bei hohem Tempo etwa denselben Wert hat und noch fährt. Bei niedrigem Tempo braucht das Auto also die volle Verstärkung.
- **Was nicht geholfen hat**: Polynom-Fits über den Pfad ändern am Zittern nichts, der Pfad springt als Ganzes und nicht in einzelnen Punkten. Ein Tiefpass auf die Lenkung kostet Punkte (0,05 s: −3,5 Punkte, Offroad), eine Begrenzung der Lenkrate auf 2 rad/s bringt das Auto in Kurven von der Strecke.
- **Kammscher Kreis**: Ohne die Gasbegrenzung beim Lenken nur Ø 80 Punkte (Abstimm-Seeds), die Hinterräder brechen in fast jeder Kurve aus.
- **Grenzen**: In rund 68 % der Schritte ist die Zielgeschwindigkeit das Höchsttempo, das Auto liegt dort im Mittel aber noch rund 20 Einheiten/s darunter. Der Motor ist der Engpass. Mehr Score bringt vor allem eine Ideallinie aus der Pfadplanung, die Kurvenradien vergrößert. Mit `--domain_randomize` ist die Pipeline noch nicht getestet.

### Laufzeit

Pro Simulationsschritt im Benchmark (8 Prozesse parallel), Budget bei 50 FPS sind 20 ms:

| Modul | p50 | p95 | p99 |
| ----- | --: | --: | --: |
| Spurerkennung | 0,87 ms | 1,33 ms | 1,72 ms |
| Pfadplanung | 0,83 ms | 1,26 ms | 1,47 ms |
| Querregelung | 0,24 ms | 0,33 ms | 0,38 ms |
| Längsregelung | 0,20 ms | 0,29 ms | 0,35 ms |
| Gesamte Pipeline | 2,14 ms | 3,14 ms | 3,65 ms |

Der längste Schritt der gesamten Pipeline dauert 7,8 ms, das p99 liegt bei beiden Reglern unter 0,4 ms. Alle Berechnungen sind mit numpy vektorisiert, Schleifen über Pfadpunkte gibt es nicht.