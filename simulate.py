import argparse
import joblib
import numpy as np
import pandas as pd
import os
import json
import random

def sample_safety_cars(num_scenarios, race_length, seed=None):
    """Return list of scenarios; each scenario is a list of SC insertion laps."""
    np.random.seed(seed)
    # Fit Poisson on historical SC counts is done offline; use lambda=expected_sc per race type (example lambda=0.5)
    lam = 0.5
    scenarios = []
    for i in range(num_scenarios):
        k = np.random.poisson(lam)
        sc_laps = []
        if k > 0:
            # uniformly sample insertion laps across race length
            sc_laps = list(np.random.randint(1, race_length+1, size=k))
            sc_laps.sort()
        scenarios.append(sc_laps)
    return scenarios

def evaluate_strategy_on_scenario(strategy, baseline_strategy, scenario, lap_time_predictor, pit_delta):
    # strategy: list of pit laps e.g., [25], baseline_strategy similar
    # This function should simulate lap by lap applying pit delta on pit laps and adding SC time effects
    # For brevity assume lap_time_predictor.predict(lap_features) returns lap_time
    # Return total_time_model, total_time_actual
    total_model = 0.0
    total_actual = 0.0
    race_length = 52  # should be derived per race
    for lap in range(1, race_length+1):
        # determine features for lap for predictor (user to implement)
        features = construct_features_for_lap(lap, scenario)
        lap_t = lap_time_predictor.predict([features])[0]
        total_model += lap_t
        total_actual += lap_t
        if lap in strategy:
            total_model += pit_delta
        if lap in baseline_strategy:
            total_actual += pit_delta
    # apply safety car lap time adjustments (coarse approach)
    # If any SC in scenario, add small adjustments or re-run per-lap predictor with SC flag
    return total_model, total_actual

def run_evaluation(models_dir, races_df, scenarios_per_race=200, out_csv="results/deltaT.csv", seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # load models
    pitstop_clf = joblib.load(os.path.join(models_dir, "pitstop_classifier.pkl"))
    pitlap1 = joblib.load(os.path.join(models_dir, "pitlap1_regressor.pkl"))
    pitlap2 = joblib.load(os.path.join(models_dir, "pitlap2_regressor.pkl"))

    results = []
    for idx, race in races_df.iterrows():
        race_length = int(race.total_laps)
        scenarios = sample_safety_cars(scenarios_per_race, race_length, seed + idx)
        # baseline_strategy: derive from race actual pit stops
        baseline_strategy = race.actual_pit_laps  # expects list
        # model_strategy: use classifier to predict number of stops then regressors for lap
        pred_count = pitstop_clf.predict([race.feature_vector])[0]
        if pred_count == 1:
            pred_lap = int(pitlap1.predict([race.feature_vector])[0])
            model_strategy = [max(1, min(pred_lap, race_length))]
        else:
            l1 = int(pitlap1.predict([race.feature_vector])[0])
            l2 = int(pitlap2.predict([race.feature_vector])[0])
            model_strategy = [max(1, min(l1, race_length)), max(1, min(l2, race_length))]

        # simulate over scenarios
        delta_list = []
        for s in scenarios:
            total_model, total_actual = evaluate_strategy_on_scenario(model_strategy, baseline_strategy, s, lap_time_predictor=DummyLapPredictor(), pit_delta=race.pit_delta)
            delta_list.append(total_actual - total_model)
        results.append({
            "race_id": race.race_id,
            "pred_count": int(pred_count),
            "actual_count": len(baseline_strategy),
            "mean_deltaT": float(np.mean(delta_list)),
            "std_deltaT": float(np.std(delta_list)),
            "median_deltaT": float(np.median(delta_list))
        })
    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print("Wrote deltaT results to", out_csv)
