# OSMI support–safety clustering

Reproduzierbare, explorative Clusteranalyse der **OSMI Mental Health in Tech
Survey 2016**. Analysiert werden ausschließlich zwölf Wahrnehmungen des
aktuellen Arbeitgebers. Diagnosen, Behandlung, Symptome und Demografie sind
keine Clustermerkmale.

Dieses Repository enthält nur den begleitenden Analysecode. Rohdaten,
individuelle Clusterzuordnungen, Prüfungsaufgabe und schriftliche Ausarbeitung
werden nicht veröffentlicht.

## Analytischer Ablauf

1. Öffentlichen Datensatz laden und SHA-256 prüfen.
2. Nicht selbstständige Beschäftigte auswählen (*n* = 1.146).
3. Zwölf kategoriale Arbeitgebermerkmale one-hot-kodieren (40 Dimensionen).
4. Truncated SVD mit 12 Komponenten berechnen.
5. K-Means-Lösungen für *k* = 2 bis 6 vergleichen.
6. Modellwahl über Originalraum-Silhouette, Mindestgröße und Stabilität aus
   20 unabhängigen 80-%-Subsamples.
7. Sensitivität für 8/12/16 SVD-Komponenten und *k* = 2/3 prüfen.
8. Die strukturell abhängige Optionsfrage entfernen und die Zuordnungen mit
   dem Hauptmodell vergleichen.
9. Kategoriale Gower/Hamming-Distanz mit k-Medoids als Robustheitsvergleich
   berechnen.
10. Aggregierte Tabellen, Abbildungen und maschinenlesbare QA-Nachweise erzeugen.

Die Profile beschreiben überlappende Wahrnehmungsmuster. Sie sind weder
Diagnosen noch natürliche Personentypen und dürfen nicht für individuelle
Personalentscheidungen verwendet werden.

## Reproduktion

Vorausgesetzt werden Python 3.11 oder 3.12 und Internetzugang für den einmaligen
Datendownload.

```bash
git clone https://github.com/mavlonfox/osmi-support-safety-clusters.git
cd osmi-support-safety-clusters
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/analyze_osmi2016.py
```

Für einen bereits vorhandenen Datensatz:

```bash
python src/analyze_osmi2016.py --no-download
```

Erwarteter CSV-Pfad:
`data/raw/mental-heath-in-tech-2016_20161114.csv`

Der analysierte Stand ist durch folgenden SHA-256-Fingerprint fixiert:

```text
0bec458b0724cc375a17eb2db0204a9f7a786260441cf702eec210d92bd4ae4d
```

Die Ausführung erzeugt lokal `outputs/`, `data/processed/` und `qa/`. Diese
Ordner sind absichtlich von Git ausgeschlossen.

## Datenquelle

Open Sourcing Mental Illness Ltd. (2017, November 8). *OSMI mental health in
tech survey* [Data set]. figshare.
https://doi.org/10.6084/m9.figshare.5579458.v1

Das Skript verwendet den in der Aufgabenquelle genannten Kaggle-Spiegel und
prüft anschließend den Datei-Fingerprint. Die Rohdaten enthalten sensible
Gesundheitsangaben und werden trotz öffentlicher Verfügbarkeit nicht in diesem
Repository gespiegelt.

## Lizenz

Der Code steht unter der MIT-Lizenz. Für den Datensatz gelten ausschließlich
die Bedingungen der Datenquelle; er ist nicht Bestandteil dieser Lizenz.
