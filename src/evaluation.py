import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def calibrate_threshold(human_scores, target_fpr=0.05):
    human_scores = np.asarray(human_scores)
    return float(np.quantile(human_scores, 1.0 - target_fpr))

def _rate_above(scores, thr):
    return float((np.asarray(scores) > thr).mean())


def group_train_test_indices(groups, test_size=0.2, random_state=42):
    groups = np.asarray(groups)
    n = len(groups)
    if n < 2:
        raise ValueError("group_train_test_indices: need at least 2 rows")
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        # Degenerate: every row same group — fall back to row shuffle.
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n)
        n_te = max(1, int(round(n * test_size)))
        return perm[n_te:], perm[:n_te]

    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(gss.split(np.arange(n), groups=groups))
    return train_idx, test_idx


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

def _fit_presplit(human_train, human_test, bot_train, bot_test, feature_cols, random_state):
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
    y_pred = model.predict(test_df[cols])
    y_score = model.predict_proba(test_df[cols])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    detect = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    return model, {
        "acc": float(acc),
        "auc": auc,
        "detect": float(detect),
        "fp": float(fpr),
        "n_test": int(len(y_true)),
    }


def train_bot_detector_presplit(
    human_train,
    human_test,
    bot_train,
    bot_test,
    feature_cols,
    random_state=42,
    name=None,
    show_feature_importance=True,
):
    model, metrics = _fit_presplit(
        human_train, human_test, bot_train, bot_test, feature_cols, random_state
    )
    test_df = pd.concat(
        [human_test.assign(is_bot=0), bot_test.assign(is_bot=1)],
        ignore_index=True,
    )
    cols = list(feature_cols)
    _print_bot_detector_report(
        model,
        test_df["is_bot"],
        model.predict(test_df[cols]),
        cols,
        name,
        show_feature_importance,
    )
    return model, metrics["acc"]


def _session_ids_from_feat_df(feat_df, prefix="session"):
    if "gameId" in feat_df.columns:
        return feat_df["gameId"].astype(str).tolist()
    if "source_file" in feat_df.columns:
        return feat_df["source_file"].astype(str).tolist()
    return [f"{prefix}_{i}" for i in range(len(feat_df))]


def _mean_std_summary(metrics_by_bot, bot_types, name, n_splits_eff, prefix=""):
    summary_rows = []
    label = f"{name}{prefix}"
    print(f"\n=== {label}: mean ± std over {n_splits_eff} folds ===")
    for bt in bot_types:
        accs = np.array([m["acc"] for m in metrics_by_bot[bt]], dtype=float)
        aucs = np.array([m["auc"] for m in metrics_by_bot[bt]], dtype=float)
        detects = np.array([m["detect"] for m in metrics_by_bot[bt]], dtype=float)
        fps = np.array([m["fp"] for m in metrics_by_bot[bt]], dtype=float)
        summary_rows.append({
            "bot": bt,
            "acc_mean": float(np.nanmean(accs)),
            "acc_std": float(np.nanstd(accs, ddof=0)),
            "auc_mean": float(np.nanmean(aucs)),
            "auc_std": float(np.nanstd(aucs, ddof=0)),
            "detect_mean": float(np.nanmean(detects)),
            "detect_std": float(np.nanstd(detects, ddof=0)),
            "fp_mean": float(np.nanmean(fps)),
            "fp_std": float(np.nanstd(fps, ddof=0)),
        })
        print(
            f"  {bt}: acc {np.nanmean(accs):.2%} ± {np.nanstd(accs):.2%}  |  "
            f"auc {np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}  |  "
            f"detect {np.nanmean(detects):.1%} ± {np.nanstd(detects):.1%}  |  "
            f"fp {np.nanmean(fps):.1%} ± {np.nanstd(fps):.1%}"
        )
    return pd.DataFrame(summary_rows)


def _aggregate_session_scores(win_df, scores, feature_cols=None):
    tmp = win_df[["session_id", "is_bot"]].copy()
    tmp["score"] = np.asarray(scores, dtype=float)
    agg = tmp.groupby("session_id", sort=False).agg(
        is_bot=("is_bot", "first"),
        score=("score", "mean"),
        n_windows=("score", "size"),
    )
    return agg


def _fit_presplit_windows(
    human_tr,
    human_te,
    bot_tr,
    bot_te,
    feature_cols,
    random_state,
    report_session_agg=True,
):
    model, win_metrics = _fit_presplit(
        human_tr, human_te, bot_tr, bot_te, feature_cols, random_state
    )
    out = {"window": win_metrics, "session": None}
    if not report_session_agg:
        return model, out

    cols = list(feature_cols)
    test_df = pd.concat(
        [human_te.assign(is_bot=0), bot_te.assign(is_bot=1)],
        ignore_index=True,
    )
    scores = model.predict_proba(test_df[cols])[:, 1]
    agg = _aggregate_session_scores(test_df, scores)
    y_true = agg["is_bot"].to_numpy()
    y_score = agg["score"].to_numpy()
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    detect = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    out["session"] = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "auc": auc,
        "detect": float(detect),
        "fp": float(fpr),
        "n_test": int(len(y_true)),
    }
    return model, out


def evaluate_split_first_group_kfold(
    human_df,
    groups,
    mice_for_df,
    feature_cols,
    n_splits=5,
    random_state=42,
    rng_seed_base=42,
    round_deltas=True,
    bot_types=("stitch", "smooth", "bezier"),
    name="in-domain",
    show_fold_detail=True,
):
    from src.bots import generate_bot_feature_tables

    human_df = human_df.reset_index(drop=True)
    groups = np.asarray(groups)
    if len(human_df) != len(groups):
        raise ValueError("evaluate_split_first_group_kfold: len(groups) != len(human_df)")

    n_groups = int(len(np.unique(groups)))
    if n_groups < 2:
        raise ValueError("evaluate_split_first_group_kfold: need at least 2 groups")
    n_splits_eff = int(min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits_eff)

    fold_rows = []
    metrics_by_bot = {bt: [] for bt in bot_types}

    print(
        f"=== {name}: GroupKFold split-first "
        f"(requested={n_splits}, used={n_splits_eff}, groups={n_groups}) ==="
    )

    for fold, (tr_idx, te_idx) in enumerate(
        gkf.split(np.zeros(len(human_df)), groups=groups)
    ):
        human_tr = human_df.iloc[tr_idx].reset_index(drop=True)
        human_te = human_df.iloc[te_idx].reset_index(drop=True)
        te_groups = np.unique(groups[te_idx])

        bots_tr = generate_bot_feature_tables(
            mice_for_df(human_tr),
            human_tr,
            rng_seed=rng_seed_base + fold,
            round_deltas=round_deltas,
            id_prefix=f"{name}_f{fold}_tr",
        )
        bots_te = generate_bot_feature_tables(
            mice_for_df(human_te),
            human_te,
            rng_seed=rng_seed_base + 100 + fold,
            round_deltas=round_deltas,
            id_prefix=f"{name}_f{fold}_te",
        )

        row = {
            "fold": fold,
            "n_train": len(human_tr),
            "n_test": len(human_te),
            "n_test_groups": len(te_groups),
            "test_groups": ",".join(map(str, te_groups)),
        }
        if show_fold_detail:
            print(
                f"\n--- fold {fold}: train={len(human_tr)} test={len(human_te)} "
                f"test_groups={len(te_groups)} ({row['test_groups']}) ---"
            )

        for bt in bot_types:
            _, m = _fit_presplit(
                human_tr,
                human_te,
                bots_tr[bt],
                bots_te[bt],
                feature_cols,
                random_state=random_state,
            )
            metrics_by_bot[bt].append(m)
            row[f"{bt}_acc"] = m["acc"]
            row[f"{bt}_auc"] = m["auc"]
            row[f"{bt}_detect"] = m["detect"]
            row[f"{bt}_fp"] = m["fp"]
            if show_fold_detail:
                print(
                    f"  {bt}: acc={m['acc']:.2%}  auc={m['auc']:.3f}  "
                    f"detect={m['detect']:.1%}  fp={m['fp']:.1%}"
                )
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    summary_df = _mean_std_summary(metrics_by_bot, bot_types, name, n_splits_eff)
    return {
        "n_splits": n_splits_eff,
        "n_groups": n_groups,
        "fold_df": fold_df,
        "summary_df": summary_df,
        "metrics_by_bot": metrics_by_bot,
    }


def evaluate_split_first_group_kfold_windows(
    human_df,
    groups,
    mice_for_df,
    feature_cols,
    n_splits=5,
    random_state=42,
    rng_seed_base=42,
    round_deltas=True,
    bot_types=("stitch", "smooth", "bezier"),
    name="in-domain",
    show_fold_detail=True,
    window_ms=None,
    min_events=None,
    report_session_agg=True,
):
    from src.bots import generate_bot_mouse_games
    from src.config import WINDOW_MIN_EVENTS, WINDOW_MS
    from src.features import build_window_feature_table

    if window_ms is None:
        window_ms = WINDOW_MS
    if min_events is None:
        min_events = WINDOW_MIN_EVENTS

    human_df = human_df.reset_index(drop=True)
    groups = np.asarray(groups)
    if len(human_df) != len(groups):
        raise ValueError(
            "evaluate_split_first_group_kfold_windows: len(groups) != len(human_df)"
        )

    n_groups = int(len(np.unique(groups)))
    if n_groups < 2:
        raise ValueError(
            "evaluate_split_first_group_kfold_windows: need at least 2 groups"
        )
    n_splits_eff = int(min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits_eff)

    fold_rows = []
    metrics_by_bot = {bt: [] for bt in bot_types}
    session_metrics_by_bot = {bt: [] for bt in bot_types}

    print(
        f"=== {name}: GroupKFold split-first WINDOWS "
        f"(window_ms={window_ms}, min_events={min_events}, "
        f"requested={n_splits}, used={n_splits_eff}, groups={n_groups}) ==="
    )

    for fold, (tr_idx, te_idx) in enumerate(
        gkf.split(np.zeros(len(human_df)), groups=groups)
    ):
        human_tr = human_df.iloc[tr_idx].reset_index(drop=True)
        human_te = human_df.iloc[te_idx].reset_index(drop=True)
        te_groups = np.unique(groups[te_idx])

        mice_tr = mice_for_df(human_tr)
        mice_te = mice_for_df(human_te)
        sid_tr = _session_ids_from_feat_df(human_tr, prefix=f"{name}_f{fold}_htr")
        sid_te = _session_ids_from_feat_df(human_te, prefix=f"{name}_f{fold}_hte")

        human_win_tr = build_window_feature_table(
            mice_tr,
            groups=groups[tr_idx],
            session_ids=sid_tr,
            is_bot=0,
            window_ms=window_ms,
            min_events=min_events,
            bot_type="human",
        )
        human_win_te = build_window_feature_table(
            mice_te,
            groups=groups[te_idx],
            session_ids=sid_te,
            is_bot=0,
            window_ms=window_ms,
            min_events=min_events,
            bot_type="human",
        )

        bots_tr = generate_bot_mouse_games(
            mice_tr,
            human_tr,
            rng_seed=rng_seed_base + fold,
            round_deltas=round_deltas,
            id_prefix=f"{name}_f{fold}_tr",
        )
        bots_te = generate_bot_mouse_games(
            mice_te,
            human_te,
            rng_seed=rng_seed_base + 100 + fold,
            round_deltas=round_deltas,
            id_prefix=f"{name}_f{fold}_te",
        )

        row = {
            "fold": fold,
            "n_train_sessions": len(human_tr),
            "n_test_sessions": len(human_te),
            "n_train_windows_human": len(human_win_tr),
            "n_test_windows_human": len(human_win_te),
            "n_test_groups": len(te_groups),
            "test_groups": ",".join(map(str, te_groups)),
        }
        if show_fold_detail:
            print(
                f"\n--- fold {fold}: sessions train/test="
                f"{len(human_tr)}/{len(human_te)}  "
                f"human windows train/test="
                f"{len(human_win_tr)}/{len(human_win_te)}  "
                f"test_groups={len(te_groups)} ({row['test_groups']}) ---"
            )

        for bt in bot_types:
            bot_win_tr = build_window_feature_table(
                bots_tr[bt],
                groups=np.full(len(bots_tr[bt]), f"bot_{bt}_tr"),
                session_ids=bots_tr[f"{bt}_ids"],
                is_bot=1,
                window_ms=window_ms,
                min_events=min_events,
                bot_type=bt,
            )
            bot_win_te = build_window_feature_table(
                bots_te[bt],
                groups=np.full(len(bots_te[bt]), f"bot_{bt}_te"),
                session_ids=bots_te[f"{bt}_ids"],
                is_bot=1,
                window_ms=window_ms,
                min_events=min_events,
                bot_type=bt,
            )
            row[f"n_train_windows_{bt}"] = len(bot_win_tr)
            row[f"n_test_windows_{bt}"] = len(bot_win_te)

            _, packed = _fit_presplit_windows(
                human_win_tr,
                human_win_te,
                bot_win_tr,
                bot_win_te,
                feature_cols,
                random_state=random_state,
                report_session_agg=report_session_agg,
            )
            m = packed["window"]
            metrics_by_bot[bt].append(m)
            row[f"{bt}_acc"] = m["acc"]
            row[f"{bt}_auc"] = m["auc"]
            row[f"{bt}_detect"] = m["detect"]
            row[f"{bt}_fp"] = m["fp"]
            if show_fold_detail:
                print(
                    f"  {bt} [window]: acc={m['acc']:.2%}  auc={m['auc']:.3f}  "
                    f"detect={m['detect']:.1%}  fp={m['fp']:.1%}  "
                    f"(windows te human/bot={len(human_win_te)}/{len(bot_win_te)})"
                )
            if packed["session"] is not None:
                sm = packed["session"]
                session_metrics_by_bot[bt].append(sm)
                row[f"{bt}_sess_acc"] = sm["acc"]
                row[f"{bt}_sess_auc"] = sm["auc"]
                row[f"{bt}_sess_detect"] = sm["detect"]
                row[f"{bt}_sess_fp"] = sm["fp"]
                if show_fold_detail:
                    print(
                        f"  {bt} [session mean]: acc={sm['acc']:.2%}  "
                        f"auc={sm['auc']:.3f}  detect={sm['detect']:.1%}  "
                        f"fp={sm['fp']:.1%}  (n_sessions={sm['n_test']})"
                    )
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    summary_df = _mean_std_summary(
        metrics_by_bot, bot_types, name, n_splits_eff, prefix=" [window]"
    )
    session_summary_df = None
    if report_session_agg and any(session_metrics_by_bot[bt] for bt in bot_types):
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
        "window_ms": window_ms,
        "min_events": min_events,
        "fold_df": fold_df,
        "summary_df": summary_df,
        "session_summary_df": session_summary_df,
        "metrics_by_bot": metrics_by_bot,
        "session_metrics_by_bot": session_metrics_by_bot,
    }


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
