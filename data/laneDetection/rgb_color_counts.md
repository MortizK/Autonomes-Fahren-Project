# RGB Color Counts

Ich bin mal einen Rundkurs gefahren und habe alle RGB Pixel gezählt und geloggt.

Dadurch können wir einfach mal die Farbgruppen voneinander unterscheiden.

Also Grün ist Hintergrund

Alles Grau ist Straße

Und die Mischdinger sind irgendwas anderes.

## How to use .bas

Verwendung in Excel:

1. CSV-Datei in Excel öffnen.
2. ``Alt + F11`` drücken.
3. Im VBA-Editor: Datei → Datei importieren.
4. [RGB_Fill.bas](./RGB_Fill.bas) auswählen.
5. Zurück zu Excel wechseln.
6. ``Alt + F8`` drücken.
7. ``FillRowsWithRgbColor`` auswählen und ausführen.

## Python


```python
import csv
from pathlib import Path

import numpy as np

class LaneDetection:
    debug_image: np.ndarray

    def __init__(self, pixel_percentages_file: str | Path = "average_pixel_percentages.csv"):
        self.debug_image = np.zeros((98, 98, 3), dtype=np.uint8)
        self.pixel_percentages_file = Path(pixel_percentages_file)
        self.rgb_counts_file = self.pixel_percentages_file.with_name(
            "rgb_color_counts.csv"
        )
        self.frame_count = 0
        self.average_pixel_percentages = np.zeros(256, dtype=np.float64)
        self.rgb_color_counts: dict[tuple[int, int, int], int] = {}

    def _record_rgb_statistics(self, frame: np.ndarray) -> None:
        """Record exact RGB color frequencies for one processed frame."""
        colors, counts = np.unique(
            frame.reshape(-1, 3), axis=0, return_counts=True
        )
        for color, count in zip(colors, counts):
            rgb_color = tuple(int(channel) for channel in color)
            self.rgb_color_counts[rgb_color] = (
                self.rgb_color_counts.get(rgb_color, 0) + int(count)
            )

        self._write_rgb_color_counts()

    def _write_rgb_color_counts(self) -> None:
        """Write the total count for every RGB color found so far."""
        with self.rgb_counts_file.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["red", "green", "blue", "total_count"])
            writer.writerows(
                (*color, count)
                for color, count in sorted(self.rgb_color_counts.items())
            )
```