
# F1 Pitstop Strategy Simulator

This project analyzes and simulates Formula 1 race strategies using [FastF1](https://theoehrly.github.io/Fast-F1/) telemetry data. It uses pre-trained machine learning models to recommend pit stop strategies and visualize them in comparison with actual race data.

## Features

- Predicts optimal pit stop strategy (1 or 2 stops).
- Predicts lap numbers for each pit stop.
- Compares actual vs. predicted strategy lap times.
- Visualizes race telemetry on track maps.
- Animates actual vs. predicted driver paths.

## Project Structure

```

F1-Pitstop/
├── cache/                     # FastF1 cache directory
├── models/
│   ├── pitstop\_classifier.pkl
│   ├── pitlap1\_regressor.pkl
│   └── pitlap2\_regressor.pkl
├── main.py                    # Main script (the one you provided)
├── requirements.txt
└── README.md

````

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
````

Example `requirements.txt`:

```text
fastf1
pandas
matplotlib
numpy
joblib
```

## How to Use

1. Place your trained models (`pitstop_classifier.pkl`, `pitlap1_regressor.pkl`, `pitlap2_regressor.pkl`) in a `models/` directory.
2. Run the script:

```bash
python main.py
```

3. The output includes:

   * Recommended number of stops and pit laps.
   * Lap time comparison charts.
   * Track map visualization with pit stops.
   * Animated movement of the car.

## Notes

* Make sure to **enable the FastF1 cache** to speed up data loading and avoid hitting API limits.
* This version uses Charles Leclerc (`LEC`) at Silverstone 2023 as an example.

## Troubleshooting

### GitHub Push Issues

If you're pushing this project to GitHub:

* Avoid committing large files (like `.ff1pkl`, `.sqlite`, `.pkl`) directly to Git.
* Use `.gitignore` to exclude cache and model files:

```text
cache/
*.ff1pkl
*.sqlite
*.pkl
```


This repository implements the F1-SIM pipeline from the paper:
**Data-Driven Pit-Stop Strategy Optimization in Formula 1: Simulation, Prediction, and Policy Evaluation**  
Authors: Srimadhav Seebu Kumar, Seyan Singampalli, Dr. Dinakaran M.

## Quick links
- Paper: included in the project repository (see `/paper/` or upload final PDF)
- Code: this repo
- Dataset summary: `data_summary.json`
- Models and metadata: `/models` and `/models/metadata`

## Reproducibility statement
All experiments in the paper are reproducible with the code here. For exact dataset counts, preprocessing, splits, seed, and hyperparameters see `data_summary.json` and `/models/metadata/*_meta.json`.

## Requirements
Install required packages (pinned versions provided):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

---

## License

MIT License. Use and modify freely.

```

