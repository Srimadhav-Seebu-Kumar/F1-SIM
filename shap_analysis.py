import argparse
import joblib
import shap
import os
import pandas as pd
import json
import matplotlib.pyplot as plt

def main(model_path, X_sample_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = joblib.load(model_path)

    X = pd.read_csv(X_sample_path).reset_index(drop=True)

    explainer = shap.TreeExplainer(model) if hasattr(shap, "TreeExplainer") else shap.Explainer(model, X)
    shap_values = explainer(X)

    fig = shap.summary_plot(shap_values, X, show=False)
    plt.savefig(os.path.join(out_dir, "shap_summary.png"), bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    feat_imp.to_csv(os.path.join(out_dir, "feature_importance_shap.csv"), index=False)

    print("Saved SHAP artifacts to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--X", required=True, help="CSV of feature matrix sample")
    parser.add_argument("--out_dir", default="analysis/shap")
    args = parser.parse_args()
    main(args.model, args.X, args.out_dir)
