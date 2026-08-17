import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from src.bots import generate_bot_mouse_games
from src.config import RNG_SEED, WINDOW_MIN_EVENTS, WINDOW_MS
from src.features import build_window_feature_table

def calibrate_threshold(human_scores, target_fpr=0.05):
    human_scores = np.asarray(human_scores)
    if human_scores.size == 0:
        return 0.5
    return float(np.quantile(human_scores, 1.0 - target_fpr))

def _rate_above(scores, thr):
    return float((np.asarray(scores) > thr).mean())


def _metrics_from_scores(y_true, y_score, target_fpr=0.05):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    human = y_score[y_true == 0]
    bot = y_score[y_true == 1]
    thr = calibrate_threshold(human, target_fpr=target_fpr)

    y_pred_05 = (y_score >= 0.5).astype(int)
    y_pred_cal = (y_score > thr).astype(int)

    def _pack(y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        detect = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return {
            "acc": float(accuracy_score(y_true, y_pred)),
            "detect": float(detect),
            "fp": float(fpr),
        }

    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")

    m05 = _pack(y_pred_05)
    mcal = _pack(y_pred_cal)
    return {
        "auc": auc,
        "acc": mcal["acc"],
        "detect": mcal["detect"],
        "fp": mcal["fp"],
        "thr": float(thr),
        "target_fpr": float(target_fpr),
        "acc_05": m05["acc"],
        "detect_05": m05["detect"],
        "fp_05": m05["fp"],
        "n_test": int(len(y_true)),
        "n_human": int(human.size),
        "n_bot": int(bot.size),
    }


def _format_fold_metrics(metrics):
    return (
        f"auc={metrics['auc']:.3f}  |  "
        f"cal@thr={metrics['thr']:.3f} acc={metrics['acc']:.2%} detect={metrics['detect']:.1%} "
        f"fp={metrics['fp']:.1%}  |  "
        f"@0.5 acc={metrics['acc_05']:.2%} detect={metrics['detect_05']:.1%} fp={metrics['fp_05']:.1%}"
    )


def _print_bot_detector_report(
    model, output_test, output_pred, feature_cols, name, show_feature_importance
):
    acc = accuracy_score(output_test, output_pred)
    tn, fp, fn, tp = confusion_matrix(
        output_test, output_pred, labels=[0, 1]
    ).ravel()
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
    return acc

# train bot detector for cross-game
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
    acc = _print_bot_detector_report(
        model, output_test, output_pred, feature_cols, name, show_feature_importance
    )
    return model, acc

# train bot detector for in domain (k fold)
def _fit_presplit(
    human_train,
    human_test,
    bot_train,
    bot_test,
    feature_cols,
    random_state,
    target_fpr=0.05,
):
    train_df = pd.concat(
        [human_train.assign(is_bot=0), bot_train.assign(is_bot=1)],
        ignore_index=True,
    )
    test_df = pd.concat(
        [human_test.assign(is_bot=0), bot_test.assign(is_bot=1)],
        ignore_index=True,
    )
    cols = list(feature_cols)
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(train_df[cols], train_df["is_bot"])
    y_true = test_df["is_bot"].to_numpy()
    y_score = model.predict_proba(test_df[cols])[:, 1]
    return model, _metrics_from_scores(y_true, y_score, target_fpr=target_fpr)


def _list_session_names(feat_df, prefix="session"):
    if "gameId" in feat_df.columns:
        return feat_df["gameId"].astype(str).tolist()
    if "source_file" in feat_df.columns:
        return feat_df["source_file"].astype(str).tolist()
    return [f"{prefix}_{i}" for i in range(len(feat_df))]

# calculate the mean and standard deviation of the metrics and print the results
def _mean_std_summary(metrics_by_bot, bot_types, name, n_splits_eff, prefix=""):
    summary_rows = []
    label = f"{name}{prefix}"
    print(f"\n=== {label}: mean ± std over {n_splits_eff} folds ===")
    print(
        "  (primary: AUC + human-calibrated thr≈95th pct of test-human scores; "
        "fixed-0.5 acc can collapse to ~50% on small folds — see auc / @_05)"
    )
    for bt in bot_types:
        aucs = np.array([m["auc"] for m in metrics_by_bot[bt]], dtype=float)
        accs = np.array([m["acc"] for m in metrics_by_bot[bt]], dtype=float)
        detects = np.array([m["detect"] for m in metrics_by_bot[bt]], dtype=float)
        fps = np.array([m["fp"] for m in metrics_by_bot[bt]], dtype=float)
        accs_05 = np.array(
            [m.get("acc_05", np.nan) for m in metrics_by_bot[bt]], dtype=float
        )
        detects_05 = np.array(
            [m.get("detect_05", np.nan) for m in metrics_by_bot[bt]], dtype=float
        )
        fps_05 = np.array(
            [m.get("fp_05", np.nan) for m in metrics_by_bot[bt]], dtype=float
        )
        summary_rows.append({
            "bot": bt,
            "auc_mean": float(np.nanmean(aucs)),
            "auc_std": float(np.nanstd(aucs, ddof=0)),
            "acc_mean": float(np.nanmean(accs)),
            "acc_std": float(np.nanstd(accs, ddof=0)),
            "detect_mean": float(np.nanmean(detects)),
            "detect_std": float(np.nanstd(detects, ddof=0)),
            "fp_mean": float(np.nanmean(fps)),
            "fp_std": float(np.nanstd(fps, ddof=0)),
            "acc_05_mean": float(np.nanmean(accs_05)),
            "acc_05_std": float(np.nanstd(accs_05, ddof=0)),
            "detect_05_mean": float(np.nanmean(detects_05)),
            "detect_05_std": float(np.nanstd(detects_05, ddof=0)),
            "fp_05_mean": float(np.nanmean(fps_05)),
            "fp_05_std": float(np.nanstd(fps_05, ddof=0)),
        })
        print(
            f"  {bt}: auc {np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}  |  "
            f"cal acc {np.nanmean(accs):.2%} ± {np.nanstd(accs):.2%}  "
            f"detect {np.nanmean(detects):.1%} ± {np.nanstd(detects):.1%}  "
            f"fp {np.nanmean(fps):.1%} ± {np.nanstd(fps):.1%}  |  "
            f"@0.5 acc {np.nanmean(accs_05):.2%} ± {np.nanstd(accs_05):.2%}  "
            f"detect {np.nanmean(detects_05):.1%} ± {np.nanstd(detects_05):.1%}  "
            f"fp {np.nanmean(fps_05):.1%} ± {np.nanstd(fps_05):.1%}"
        )
    return pd.DataFrame(summary_rows)

# aggregate the session scores
def _aggregate_session_scores(win_df, scores, feature_cols=None):
    tmp = win_df[["session_id", "is_bot"]].copy()
    tmp["score"] = np.asarray(scores, dtype=float)
    agg = tmp.groupby("session_id", sort=False).agg(
        is_bot=("is_bot", "first"),
        score=("score", "mean"),
        n_windows=("score", "size"),
    )
    return agg

# train the model and evaluate the metrics for the window feature table
def _fit_presplit_windows(
    human_train,
    human_test,
    bot_tr,
    bot_te,
    feature_cols,
    random_state,
    report_session_agg=True,
):
    model, win_metrics = _fit_presplit(
        human_train, human_test, bot_tr, bot_te, feature_cols, random_state
    )
    result = {"window": win_metrics, "session": None}
    if not report_session_agg:
        return model, result

    cols = list(feature_cols)
    test_df = pd.concat(
        [human_test.assign(is_bot=0), bot_te.assign(is_bot=1)],
        ignore_index=True,
    )
    scores = model.predict_proba(test_df[cols])[:, 1]
    agg = _aggregate_session_scores(test_df, scores)
    y_true = agg["is_bot"].to_numpy()
    y_score = agg["score"].to_numpy()
    result["session"] = _metrics_from_scores(y_true, y_score, target_fpr=0.05)
    return model, result


_FOLD_METRIC_KEYS = (
    "auc",
    "thr",
    "acc",
    "detect",
    "fp",
    "acc_05",
    "detect_05",
    "fp_05",
)

def _store_fold_metrics(fold_logs, bot_type, metrics, prefix=""):
    for key in _FOLD_METRIC_KEYS:
        fold_logs[f"{bot_type}_{prefix}{key}"] = metrics[key]

def evaluate_group_kfold_windows(
    human_df,
    groups,
    traces_for_df,
    feature_cols,
    n_splits=5, #
    rng_seed_base=RNG_SEED,
    round_deltas=True,
    name="in-domain",
    show_fold_detail=True, #
    vae_bundle=None,
):
    human_df = human_df.reset_index(drop=True)
    groups = np.asarray(groups)
    bot_types = ("stitch", "smooth", "bezier", "vae")
    if len(human_df) != len(groups):
        raise ValueError(
            "evaluate_group_kfold_windows: len(groups) != len(human_df)"
        )
    if vae_bundle is None:
        raise ValueError("evaluate_group_kfold_windows: vae_bundle is None")

    n_groups = int(len(np.unique(groups)))
    n_splits_eff = int(min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits_eff)

    fold_rows = []
    metrics_by_bot = {bt: [] for bt in bot_types}
    session_metrics_by_bot = {bt: [] for bt in bot_types}

    print(
        f"=== {name}: GroupKFold split players"
        f"(window_ms={WINDOW_MS}, min_events={WINDOW_MIN_EVENTS}, "
        f"fold={n_splits_eff}, groups={n_groups} ==="
    )

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(np.zeros(len(human_df)), groups=groups)
    ):
        human_train = human_df.iloc[train_idx].reset_index(drop=True)
        human_test = human_df.iloc[test_idx].reset_index(drop=True)
        test_groups = np.unique(groups[test_idx])

        traces_train = traces_for_df(human_train)
        traces_test = traces_for_df(human_test)
        session_id_train = _list_session_names(human_train, prefix=f"{name}_f{fold}_htr")
        session_id_test = _list_session_names(human_test, prefix=f"{name}_f{fold}_hte")

        # build window feature table for human training and testing
        human_win_train = build_window_feature_table(
            traces_train,
            groups=groups[train_idx],
            session_ids=session_id_train,
            is_bot=0,
            bot_type="human",
        )
        human_win_test = build_window_feature_table(
            traces_test,
            groups=groups[test_idx],
            session_ids=session_id_test,
            is_bot=0,
            bot_type="human",
        )

        # generate the bot traces for each bot type
        bots_train = generate_bot_mouse_games(
            traces_train,
            human_train,
            rng_seed=rng_seed_base + fold,
            round_deltas=round_deltas,
            id_prefix=f"{name}_f{fold}_tr",
            vae_bundle=vae_bundle,
        )
        bots_test = generate_bot_mouse_games(
            traces_test,
            human_test,
            rng_seed=rng_seed_base + 100 + fold,
            round_deltas=round_deltas,
            id_prefix=f"{name}_f{fold}_te",
            vae_bundle=vae_bundle,
        )

        # record the data for this fold
        fold_logs = {
            "fold": fold,
            "n_train_sessions": len(human_train),
            "n_test_sessions": len(human_test),
            "n_train_windows_human": len(human_win_train),
            "n_test_windows_human": len(human_win_test),
            "n_test_groups": len(test_groups),
            "test_groups": ",".join(map(str, test_groups)),
        }
        if show_fold_detail:
            print(
                f"\n--- fold {fold}: sessions train/test="
                f"{fold_logs['n_train_sessions']}/{fold_logs['n_test_sessions']}  "
                f"human windows train/test="
                f"{fold_logs['n_train_windows_human']}/{fold_logs['n_test_windows_human']}  "
                f"test_groups={fold_logs['n_test_groups']} ({fold_logs['test_groups']}) ---"
            )

        # extract the window feature table for each bot type
        for bot_type in bot_types:
            bot_win_train = build_window_feature_table(
                bots_train[bot_type],
                groups=np.full(len(bots_train[bot_type]), f"bot_{bot_type}_tr"),
                session_ids=bots_train[f"{bot_type}_ids"],
                is_bot=1,
                bot_type=bot_type,
            )
            bot_win_test = build_window_feature_table(
                bots_test[bot_type],
                groups=np.full(len(bots_test[bot_type]), f"bot_{bot_type}_te"),
                session_ids=bots_test[f"{bot_type}_ids"],
                is_bot=1,
                bot_type=bot_type,
            )
            fold_logs[f"n_train_windows_{bot_type}"] = len(bot_win_train)
            fold_logs[f"n_test_windows_{bot_type}"] = len(bot_win_test)

            # train the model and evaluate the metrics
            _, evaluate_result = _fit_presplit_windows(
                human_win_train,
                human_win_test,
                bot_win_train,
                bot_win_test,
                feature_cols,
                random_state=RNG_SEED,
                report_session_agg=True,
            )
            window_metrics = evaluate_result["window"]
            session_metrics = evaluate_result["session"]
            metrics_by_bot[bot_type].append(window_metrics)
            _store_fold_metrics(fold_logs, bot_type, window_metrics)
            if show_fold_detail:
                print(
                    f"  {bot_type} [window]: {_format_fold_metrics(window_metrics)}  "
                    f"(windows te human/bot="
                    f"{fold_logs['n_test_windows_human']}/{fold_logs[f"n_test_windows_{bot_type}"]})"
                )
            if session_metrics is not None:
                session_metrics_by_bot[bot_type].append(session_metrics)
                _store_fold_metrics(fold_logs, bot_type, session_metrics, prefix="sess_")
                if show_fold_detail:
                    print(
                        f"  {bot_type} [session mean]: {_format_fold_metrics(session_metrics)}  "
                        f"(n_sessions={session_metrics['n_test']})"
                    )
        fold_rows.append(fold_logs)

    fold_df = pd.DataFrame(fold_rows)
    summary_df = _mean_std_summary(
        metrics_by_bot, bot_types, name, n_splits_eff, prefix=" [window]"
    )
    session_summary_df = None
    if any(session_metrics_by_bot[bot_type] for bot_type in bot_types):
        session_summary_df = _mean_std_summary(
            session_metrics_by_bot,
            bot_types,
            name,
            n_splits_eff,
            prefix=" [session mean of windows]",
        )

    return {
        "n_splits": n_splits_eff,
        "n_groups": n_groups,
        "window_ms": WINDOW_MS,
        "min_events": WINDOW_MIN_EVENTS,
        "fold_df": fold_df,
        "summary_df": summary_df,
        "session_summary_df": session_summary_df,
        "metrics_by_bot": metrics_by_bot,
        "session_metrics_by_bot": session_metrics_by_bot,
    }


def threshold_sweep_table(p_bot_on_humans, p_bot_on_bots, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)
    return pd.DataFrame({
        "threshold": thresholds,
        "detect": [_rate_above(p_bot_on_bots, t) for t in thresholds],
        "FP": [_rate_above(p_bot_on_humans, t) for t in thresholds],
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
    p_bot_on_humans = model.predict_proba(human_df[cols])[:, 1]
    p_bot_on_bots = model.predict_proba(bot_df[cols])[:, 1]
    auc = roc_auc_score(np.r_[np.zeros(len(p_bot_on_humans)), np.ones(len(p_bot_on_bots))], np.r_[p_bot_on_humans, p_bot_on_bots])
    human_cal_thr = calibrate_threshold(p_bot_on_humans, target_fpr=target_fpr)
    detect_cal = _rate_above(p_bot_on_bots, human_cal_thr)
    fp_cal = _rate_above(p_bot_on_humans, human_cal_thr)
    detect_05 = float((p_bot_on_bots >= 0.5).mean())
    fp_05 = float((p_bot_on_humans >= 0.5).mean())

    label = f"{tag}" + (f" | {title_suffix}" if title_suffix else "")
    print(f"=== [{label}] cross-game diagnose ===")
    print(f"AUC = {auc:.3f}")
    print(
        f"P(bot) human: min={p_bot_on_humans.min():.3f}  median={np.median(p_bot_on_humans):.3f}  max={p_bot_on_humans.max():.3f}"
    )
    print(
        f"P(bot) bot:   min={p_bot_on_bots.min():.3f}  median={np.median(p_bot_on_bots):.3f}  max={p_bot_on_bots.max():.3f}"
    )
    print(f"@0.5: detect={detect_05:.1%}, FP={fp_05:.1%}")
    print(
        f"@human-calibrated thr={human_cal_thr:.3f} (target FP≈{target_fpr:.0%}): "
        f"detect={detect_cal:.1%}, FP={fp_cal:.1%}"
    )

    sweep = threshold_sweep_table(p_bot_on_humans, p_bot_on_bots, thresholds=thresholds)
    # draw the threshold sweep plot and the probability distribution plot
    if plot or plot_proba:
        n_plots = int(plot) + int(plot_proba)
        fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 3.5))
        if n_plots == 1:
            axes = [axes]
        ax_i = 0
        if plot_proba:
            ax = axes[ax_i]
            ax.hist(p_bot_on_humans, bins=15, alpha=0.7, label="human", density=True)
            ax.hist(p_bot_on_bots, bins=15, alpha=0.7, label="bot", density=True)
            ax.axvline(0.5, color="gray", ls="--", lw=0.8)
            ax.axvline(human_cal_thr, color="black", ls=":", lw=0.8, label=f"cal thr={human_cal_thr:.2f}")
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
            ax.axvline(human_cal_thr, color="black", ls=":", lw=0.8, label=f"cal thr={human_cal_thr:.2f}")
            ax.set_title(f"{label}: threshold sweep")
            ax.set_xlabel("threshold")
            ax.set_ylabel("rate")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)
        fig.tight_layout()
        plt.show()

    return {
        "auc": auc,
        "ph": p_bot_on_humans,
        "pb": p_bot_on_bots,
        "thr": human_cal_thr,
        "detect_05": detect_05,
        "fp_05": fp_05,
        "detect_cal": detect_cal,
        "fp_cal": fp_cal,
        "sweep": sweep,
    }
