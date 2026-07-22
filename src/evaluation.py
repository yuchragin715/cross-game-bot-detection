import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def calibrate_threshold(human_scores, target_fpr=0.05):
    human_scores = np.asarray(human_scores)
    return float(np.quantile(human_scores, 1.0 - target_fpr))

def _rate_above(scores, thr):
    return float((np.asarray(scores) > thr).mean())

def train_bot_detector(
    human_df,
    bot_df,
    feature_cols,
    random_state=42,
    name=None,
    show_feature_importance=True,
):
    dataset = pd.concat(
        [human_df.assign(is_bot=0), bot_df.assign(is_bot=1)],
        ignore_index=True,
    )
    input_data = dataset[list(feature_cols)]
    output_data = dataset["is_bot"]

    input_train, input_test, output_train, output_test = train_test_split(
        input_data, output_data, test_size=0.2, random_state=random_state, stratify=output_data
    )

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(input_train, output_train)
    output_pred = model.predict(input_test)
    acc = accuracy_score(output_test, output_pred)

    tn, fp, fn, tp = confusion_matrix(output_test, output_pred, labels=[0, 1]).ravel()
    n_test = len(output_test)
    detect = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    tag = f" [{name}]" if name else ""
    print(f"=== In-domain Human vs Bot{tag} ===")
    print(f"Test accuracy: {acc:.2%}  ({(output_pred == output_test).sum()}/{n_test})")
    print(f"Bot detect rate (recall): {detect:.1%}, Human FP rate: {fpr:.1%}")
    print()
    print("Confusion counts (positive = bot):")
    print(f"  TN (human→human): {tn:4d}    FP (human→bot): {fp:4d}")
    print(f"  FN (bot→human):   {fn:4d}    TP (bot→bot):   {tp:4d}")
    print()
    if show_feature_importance:
        imp = pd.Series(model.feature_importances_, index=list(feature_cols)).sort_values(
            ascending=False
        )
        print("Feature importance:")
        print(imp.to_string(float_format=lambda x: f"{x:.4f}"))
        print()

    return model, acc

def threshold_sweep_table(ph, pb, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)
    return pd.DataFrame({
        "threshold": thresholds,
        "detect": [_rate_above(pb, t) for t in thresholds],
        "FP": [_rate_above(ph, t) for t in thresholds],
    })

def diagnose_cross_game(
    tag,
    model,
    human_df,
    bot_df,
    feature_cols,
    target_fpr=0.05,
    thresholds=None,
    plot=True,
    plot_proba=True,
    title_suffix="",
):
    cols = list(feature_cols)
    ph = model.predict_proba(human_df[cols])[:, 1]
    pb = model.predict_proba(bot_df[cols])[:, 1]
    auc = roc_auc_score(np.r_[np.zeros(len(ph)), np.ones(len(pb))], np.r_[ph, pb])
    thr = calibrate_threshold(ph, target_fpr=target_fpr)
    detect_cal = _rate_above(pb, thr)
    fp_cal = _rate_above(ph, thr)
    detect_05 = float((pb >= 0.5).mean())
    fp_05 = float((ph >= 0.5).mean())

    label = f"{tag}" + (f" | {title_suffix}" if title_suffix else "")
    print(f"=== [{label}] cross-game diagnose ===")
    print(f"AUC = {auc:.3f}")
    print(
        f"P(bot) human: min={ph.min():.3f}  median={np.median(ph):.3f}  max={ph.max():.3f}"
    )
    print(
        f"P(bot) bot:   min={pb.min():.3f}  median={np.median(pb):.3f}  max={pb.max():.3f}"
    )
    print(f"@0.5: detect={detect_05:.1%}, FP={fp_05:.1%}")
    print(
        f"@human-calibrated thr={thr:.3f} (target FP≈{target_fpr:.0%}): "
        f"detect={detect_cal:.1%}, FP={fp_cal:.1%}"
    )

    sweep = threshold_sweep_table(ph, pb, thresholds=thresholds)
    print()
    print(f"--- threshold sweep [{label}] ---")
    print(sweep.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if plot or plot_proba:
        n_plots = int(plot) + int(plot_proba)
        fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 3.5))
        if n_plots == 1:
            axes = [axes]
        ax_i = 0
        if plot_proba:
            ax = axes[ax_i]
            ax.hist(ph, bins=15, alpha=0.7, label="human", density=True)
            ax.hist(pb, bins=15, alpha=0.7, label="bot", density=True)
            ax.axvline(0.5, color="gray", ls="--", lw=0.8)
            ax.axvline(thr, color="black", ls=":", lw=0.8, label=f"cal thr={thr:.2f}")
            ax.set_title(f"{label}: P(bot) distribution")
            ax.set_xlabel("P(bot)")
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
            ax_i += 1
        if plot:
            ax = axes[ax_i]
            ax.plot(sweep["threshold"], sweep["detect"], label="detect (bots)")
            ax.plot(sweep["threshold"], sweep["FP"], label="FP (humans)")
            ax.axvline(0.5, color="gray", ls="--", lw=0.8, label="thr=0.5")
            ax.axvline(thr, color="black", ls=":", lw=0.8, label=f"cal thr={thr:.2f}")
            ax.set_title(f"{label}: threshold sweep")
            ax.set_xlabel("threshold")
            ax.set_ylabel("rate")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)
        fig.tight_layout()
        plt.show()

    return {
        "auc": auc,
        "ph": ph,
        "pb": pb,
        "thr": thr,
        "detect_05": detect_05,
        "fp_05": fp_05,
        "detect_cal": detect_cal,
        "fp_cal": fp_cal,
        "sweep": sweep,
    }
