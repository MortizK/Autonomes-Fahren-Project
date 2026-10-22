# Changes

Dieses Dokument soll wichtige Design Entscheidungen oder Changes Dokumentieren.

- `LaneDetection` nutzt eine Maske die nach Grün + Puffer ausschaut. Dies ist der Hintergrund. Die Inverse davon ist wohl die Track.
- Die Regler rechnen nicht in Pixeln, sondern in Einheiten der Simulation (`geometry.py`). Das Bild ist quer und längs unterschiedlich skaliert (1,56 / 1,94 px pro Einheit), Winkel und Krümmungen in Pixeln wären verzerrt. Der Zoom in der ersten Sekunde wird über die bekannte Zoomfunktion der Kamera herausgerechnet.
- `LateralControl` gibt den Soll-Radwinkel in rad zurück. Das entspricht der Simulation (Lenkanschlag ±0,4 rad), eine Normierung auf ±1 würde die Lenkung verfälschen.
- `LateralControl` nutzt Stanley an der Vorderachse. Der Kurswinkel wird über eine Sehne von 12 Einheiten gemessen statt über die Tangente: glatter, das Auto schneidet Kurven leicht an und kann schneller fahren. Mit der echten Pipeline 925,9 statt 480,9 Punkte (klassischer Stanley, gleiche Längsregelung).
- Pure Pursuit wurde als Alternative umgesetzt und wieder entfernt. Im Vergleich auf 50 Seeds fuhr es etwas schnellere Runden, kam aber einmal von der Strecke ab (Ø 922,6 statt 925,9). Stanley ist robuster, nur ein Regler hält den Code schlank.
- Ohne Pfad klingt der Lenkwinkel ab, statt gehalten zu werden. Gehalten hat das Auto beim Start (kein Pfad während des Einzoomens) dauerhaft eingelenkt und ist von der Strecke geflogen. Pfade werden erst ab halbem Zoom genutzt.
- `predict_target_speed` bekommt die Trajektorie statt der Krümmung. Die Krümmung berechnet die Längsregelung selbst, die Krümmung aus `PathPlanning` ist damit aktuell ungenutzt.
- Die Zielgeschwindigkeit folgt aus der Physik: Querbeschleunigung in der Kurve, Bremsweg bis zur Kurve und Bremsweg bis zum Ende des sichtbaren Pfads. Weniger Sicht heißt damit automatisch langsamer.
- Der PID-Regler gibt eine Soll-Beschleunigung aus, die über gemessene Kennwerte in Gas oder Bremse umgerechnet wird. Die Reglerparameter haben so eine physikalische Einheit.
- Gas wird beim Lenken über den Kammschen Kreis begrenzt (ohne: Ø 80 statt 924 Punkte). Bremsen bleibt in der Kurve erlaubt, mit Begrenzung flog das Auto häufiger raus.