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
| - .gitignore # Definition von Dateien, die nicht von git getrackt werden sollen
| - benchmark_control.py # Performance-Analyse: Laufzeit und Score über viele Seeds
| - car.py # Beschreibt das Fahrzeug und enthält die gesamte Pipeline
| - env_wrapper.py # Wrapper für die Gymnasium Car Racing Simulation
| - geometry.py # Umrechnung Pixel -> Fahrzeugkoordinaten, Pfad und Krümmung für die Regler
| - input_controller.py # Manuelle Steuerung des Fahrzeugs
| - lane_detection.py # Modul zur Spurerkennung
| - lateral_control.py # Modul zur Querregelung
| - longitudinal_control.py # Modul zur Längsregelung
| - main.py # Hauptdatei, die die Pipeline über 5 Iterationen ausführt
| - path_planning.py # Modul zur Pfadplanung
| - README.md # Diese Datei
| - test_installation.py # Testet die Installation
| - test_lane_detection.py # Testet die Spurerkennung
| - test_lateral_control.py # Testet die Querregelung
| - test_longitudinal_control.py # Testet die Längsregelung
| - test_path_planning.py # Testet die Pfadplanung
| - test_pipeline.py # Testet die gesamte Pipeline
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
delta = psi + arctan(k * e / (k_soft + v)) + k_d * dpsi/dt
```

| Parameter | Standard | Bedeutung |
| --------- | -------- | --------- |
| `gain` | 1.0 | k, Gewicht der Querabweichung e (1/s) |
| `softening` | 1.0 | k_soft, begrenzt die Verstärkung beim Anfahren (Einheiten/s) |
| `damping` | 0.02 | k_d, D-Anteil auf den Kurswinkelfehler psi (s) |
| `preview` | 12.0 | Sehnenlänge, über die psi gemessen wird (Einheiten) |

Erweiterungen gegenüber dem Standard-Stanley:

- **Kurswinkel über eine Sehne**: Statt der Tangente am Lotfußpunkt wird die Richtung zum Pfadpunkt 12 Einheiten weiter genommen. Das glättet das Pixelrauschen und schneidet Kurven leicht an, dadurch sind höhere Kurvengeschwindigkeiten möglich.
- **Dämpfung (D-Anteil)**: Die Änderung des Kurswinkelfehlers entspricht der Differenz aus Gierrate des Pfads und des Autos. Der Term dämpft das Aufschwingen bei hohem Tempo.
- **Ausgabe in rad**: Die Lenkaktion ist in der Simulation der Soll-Radwinkel in rad (Anschlag ±0,4 rad). Der Regler gibt ihn deshalb direkt aus.
- **Pfadaussetzer**: Ohne Pfad klingt der letzte Lenkwinkel pro Frame mit Faktor 0,9 ab. Kurze Aussetzer werden überbrückt, ohne dass das Auto dauerhaft eingelenkt bleibt.

Als Alternative wurde Pure Pursuit umgesetzt und verglichen (Kreisbogen von der Hinterachse durch den Pfadpunkt im Abstand `l_d = max(0.25 * v, 6)`). Stanley war robuster (Abschnitt 9), Pure Pursuit ist deshalb nicht mehr im Code. Der Stand mit beiden Reglern liegt im Commit `5e750cd`.

## 8. Längsregelung (`longitudinal_control.py`)

**Zielgeschwindigkeit** aus dem sichtbaren Pfad:

1. Kurvengeschwindigkeit an jedem Pfadpunkt: `v = sqrt(a_quer / kappa)`.
2. Bremsweg: Ein Punkt im Abstand s erlaubt `sqrt(v_kurve^2 + 2 * a_brems * s)`. Das Minimum über alle Punkte ist die Zielgeschwindigkeit.
3. Sichtweite: Hinter dem Pfadende kann eine Haarnadelkurve kommen, bis dorthin muss das Auto auf `min_speed` bremsen können. Weniger Sicht bedeutet automatisch langsamer.

Das Krümmungsfenster von ±8 Einheiten lässt kurze Kurven flacher wirken als lange. Das passt zum Fahrverhalten, kurze Kurven schneidet das Auto stärker an.

**PID-Regler**: Der Ausgang ist eine Soll-Beschleunigung. Über gemessene Kennwerte (Vollgas ≈ 44 Einheiten/s², Bremse ≈ 300 · b Einheiten/s²) wird sie in Gas oder Bremse umgerechnet.

- D-Anteil auf die gemessene Beschleunigung statt auf den Fehler, Sprünge der Zielgeschwindigkeit erzeugen so keinen Stoß.
- Anti-Windup: Integriert wird nur, solange der Ausgang nicht in der Begrenzung steckt.
- Kammscher Kreis: Die Querbeschleunigung wird aus Lenkwinkel und Tempo geschätzt (`v² · tan(delta) / L`), nur die restliche Haftung bekommt das Gas. Ohne diese Begrenzung brechen die Hinterräder in Kurven aus.
- Bremse höchstens 0,7, darüber blockieren die Räder nach einigen Schritten.

| Parameter | Standard | Bedeutung |
| --------- | -------- | --------- |
| `kp`, `ki`, `kd` | 8.0, 1.0, 0.05 | PID auf die Soll-Beschleunigung |
| `max_speed`, `min_speed` | 100, 30 | Grenzen der Zielgeschwindigkeit (Einheiten/s) |
| `lateral_accel` | 240 | erlaubte Querbeschleunigung auf der Mittellinie (Einheiten/s²) |
| `brake_accel` | 150 | eingeplante Verzögerung vor Kurven (Einheiten/s²) |
| `curvature_window` | 8 | Punkte links und rechts für die Krümmung |
| `max_brake` | 0.7 | maximaler Bremswert |

`lateral_accel` liegt über der Haftgrenze von 219 Einheiten/s², weil sie auf die Krümmung der Mittellinie bezogen ist und das Auto Kurven anschneidet.

## 9. Performance-Analyse der Regler

`benchmark_control.py` fährt ohne Fenster und parallel über feste Seeds und misst Score, Rundenabschluss, Querabweichung zur echten Mittellinie, Offroad-Schritte und die Laufzeit jedes Moduls.

```bash
python benchmark_control.py --mode pipeline --episodes 50   # komplette Pipeline
python benchmark_control.py --mode gt --episodes 50         # Regler auf der Ground-Truth-Mittellinie
```

Abgestimmt wurde auf den Seeds 0-23, bewertet auf den Seeds 1000-1049, die beim Abstimmen nicht verwendet wurden.

### Ergebnisse (Seeds 1000-1049, komplette Pipeline)

Spurerkennung und Pfadplanung (einfache Mittellinie) sind in allen Zeilen gleich, verändert wurden nur die Regler. Schritte pro Runde nur über abgeschlossene Runden.

| Variante | Ø Score | Std | Min | Runden | Schritte/Runde | Offroad |
| -------- | ------: | --: | --: | -----: | -------------: | ------: |
| **Final**: Stanley (Sehne 12, Dämpfung), Zielgeschwindigkeit aus Krümmung | **925,9** | 7,5 | 904,0 | 50/50 | 741 | 0 |
| Stanley ohne Dämpfung | 925,7 | 7,6 | 903,5 | 50/50 | 743 | 0 |
| Pure Pursuit (`lookahead_gain` 0,25, wieder entfernt) | 922,6 | 36,6 | 671,4 | 49/50 | 722 | 489 |
| Klassischer Stanley (Tangente, ohne Dämpfung) | 480,9 | 276,2 | -42,5 | 0/50 | - | 14349 |
| Erste Version (klassischer Stanley, `lateral_accel` 150, Fenster 4) | 902,7 | 28,5 | 743,7 | 43/50 | 882 | 0 |
| Konstante Zielgeschwindigkeit 50 | 865,8 | 49,1 | 697,8 | 18/50 | 949 | 0 |

Offroad zählt Schritte, in denen alle vier Räder neben der Strecke sind.

Einordnung:

- **Pipeline vs. Ground Truth**: Auf den Abstimm-Seeds erreichen die Regler mit der echten Pipeline 925,0 und auf der Ground-Truth-Mittellinie 925,5. Die Regler kommen also mit dem Pixelrauschen der Spurerkennung zurecht.
- **Sehne statt Tangente** ist der wichtigste Baustein. Die Tangente aus pixelgenauen Punkten rauscht so stark, dass der klassische Stanley mit derselben Längsregelung keine Runde schafft.
- **Pure Pursuit** fährt die Runden schneller (722 statt 741 Schritte), schneidet aber stärker und ist einmal von der Strecke abgekommen (Minimum 671 statt 904). Bei zu kurzer Vorausschau (`lookahead_gain` 0,15) schaukelt es sich auf und verlässt die Strecke. Stanley ist robuster und bleibt deshalb als einziger Regler im Code.
- **Dämpfung**: Auf den Abstimm-Seeds +0,6 Punkte und maximale Querabweichung 5,9 statt 7,5 Einheiten, auf den Test-Seeds kein messbarer Unterschied. Sie bleibt drin, weil sie nicht schadet.
- **Kammscher Kreis**: Ohne die Gasbegrenzung beim Lenken nur Ø 80 Punkte (Abstimm-Seeds), die Hinterräder brechen in fast jeder Kurve aus.
- **Grenzen**: In rund 68 % der Schritte ist die Zielgeschwindigkeit das Höchsttempo, das Auto liegt dort im Mittel aber noch rund 20 Einheiten/s darunter. Der Motor ist der Engpass. Mehr Score bringt vor allem eine Ideallinie aus der Pfadplanung, die Kurvenradien vergrößert. Mit `--domain_randomize` ist die Pipeline noch nicht getestet.

### Laufzeit

Pro Simulationsschritt im Benchmark (8 Prozesse parallel), Budget bei 50 FPS sind 20 ms:

| Modul | p50 | p95 | p99 |
| ----- | --: | --: | --: |
| Querregelung | 0,42 ms | 1,16 ms | 2,93 ms |
| Längsregelung | 0,36 ms | 1,01 ms | 2,81 ms |
| Gesamte Pipeline | 2,78 ms | 7,16 ms | 11,47 ms |

Einzeln gemessen ohne Last braucht die Querregelung 0,12 ms und die Längsregelung 0,15 ms. Einzelne Ausreißer bis 40-90 ms treten in allen Modulen auf, auch bei der Längsregelung mit wenigen Rechenschritten. Vermutlich kommen sie vom Scheduling bei paralleler Last, p99 liegt bei allen Reglern unter 3 ms. Alle Berechnungen sind mit numpy vektorisiert, Schleifen über Pfadpunkte gibt es nicht.