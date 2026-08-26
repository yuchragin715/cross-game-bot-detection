# TODO — Zero-Day Bot Detection via Mouse Dynamics

> This file is the single source of truth for project tasks. After completing and verifying a task, update the **Changelog / Progress** section below.
>
> Project background: see [`CONTEXT.md`](CONTEXT.md). Technical details (features, classifiers, synthesis strategies, etc.) are adjusted on a rolling basis and finalised when the corresponding Phase is reached.

---

## Current Status

| Item | Status |
|------|--------|
| **Current Phase** | **Project complete** (experiments finalised, thesis finished, repo cleaned up) |
| **Next task** | None |
| **Learning mode** | One step at a time; technical terms explained in plain language on the spot |
| **Project week** | Finished (12 weeks total) |
| **Last updated** | 2026-08-24 (final wrap-up; see Changelog for D.8 notebook split and F.5 cleanup) |

### Notebook layout (experiment entry points; after the 2026-08-21 split)

| File | Role |
|------|------|
| `notebooks/red_eclipse.ipynb` | RE pipeline: cache + RE classifiers |
| `notebooks/lol.ipynb` | LoL pipeline: cache |
| `notebooks/csgo.ipynb` | CSGO pipeline: cache + CSGO classifiers |
| `notebooks/cross_game.ipynb` | Cross-game zero-shot (reads `artifacts/` only, no training) |

(The old `main_segment` / `main_split_first` / `trajectory_qa` / `main` were removed after the split and consolidation on 2026-08-21.)

**VAE weights (Restart only loads; see `artifacts/README.md`):**
- Current: `vae_re_v2.pt`, `vae_lol_v4.pt`, `vae_csgo_v2.pt` (`axis_std_v1`, per-axis)
- Legacy: `vae_re_v1` / `vae_lol_v3` / `vae_csgo_v1` (scalar norm; do not treat as official)

**LoL human data:**
- Matches aligned to game clock **minutes 10–13**; `load_lol_match_windows` (derived cache)
- Cache: `data/lol2/derived/mouse_windows_10_13_v1/`; timestamps `ms_rjust_v1`
- Approx. **59** windows / 7 users / 16 matches

**Evaluation API:**
- Session level: `evaluate_split_first_group_kfold`
- Windowed: `evaluate_group_kfold_windows` (reports window + session-mean)
- Main metrics: **AUC + human-calibrated threshold** (~95th percentile of test humans, target FP≈5%); `@0.5` as a footnote only (`*_05`)

**Decision (updated 2026-07-27):** Official main results = **`main_segment` (window-level, split-by-player)**; session level reported in the same notebook via **session-mean-of-windows** as a footnote.
Four synthetic bot tiers: **stitch / smooth / bezier / VAE** (VAE = learned trajectory-generation proxy, not a full game cheat).
Two feature tracks: **SI-min (4) main table** / **SI-ext (10) ablation**; `xy_corr` **kept** (importance≈0, hypothesis not confirmed, but code untouched).

**Numbers status (2026-07-27):**
- `main_segment`: rerun with new features + calibration; SI-ext rescues VAE cross-game calibrated detection; −`xy_corr` / −`vh_ratio` ablation (RE→LoL) shows VAE does **not** depend on these two
- Narrative: write **cross-game transferable**, not domain-invariant (existing ~99% fingerprint + SI-ext human differences between games)

**Performance (2026-07-27):** split-first still regenerates bots per fold (leakage prevention); VAE changed to **decoding a shared pool once per call** (`vae_n_pool_segments=256` kept) + bundle model cache. CV still spends time on stitch / smooth / bezier, but is no longer dragged down by "re-running the VAE decode for every bot session".

---

## Phase 0: Project Foundations

- [x] **0.1** Document the planned directory structure (create on demand, no empty directories in advance)
  - Write the expected structure and purpose of each directory in `docs/project_structure.md`
  - Actual directories are created **the first time a file needs to go there** (e.g. create `notebooks/` when doing EDA, `src/data/` when writing loaders)
  - Reference structure:
    ```
    src/data/, src/trajectory/, src/features/, src/models/,
    src/synthetic/, src/evaluation/
    notebooks/, configs/, experiments/, docs/
    ```
- [~] **0.2** ~~Define experiment logging conventions~~ (dropped: replaced in the end by notebook + artifacts naming conventions; no separate `experiments/` was created)
  - Agree on an `experiments/` naming scheme (e.g. `{phase}_{date}_{description}/`)
  - Set up a simple run-log format (settings, metrics, output paths)
- [x] **0.3** Red Eclipse exploratory EDA (notebook)
  - Player counts, session distribution, event-type statistics
  - MouseEvent trajectory lengths, time spans, basic distributions
  - Produce an EDA summary (input for Phase A decisions)

**Phase 0 completion criteria**: directory-structure plan written, EDA summary done, first version of the data dictionary done (directories appear on demand as tasks need them).

---

## Phase A: Human Authentication Pipeline (Baseline)

> **Goal**: using real human mouse data, train a classifier that can guess which player is which.
> **Approach**: in a single notebook, complete four steps in order: load data → extract features → classifier → evaluation.
> **Principle**: get it running first, refactor later; one step at a time.

- [x] **A.1 Load data**
  - Load a small number of game JSONs in `notebooks/main.ipynb` (1 game first, then scale to 20–50)
  - From each game extract `MouseEvent` (`dx`, `dy`, `time`) and the player label (`userId`)
  - Check: loads successfully and prints basic info (plots not required for completion)

- [x] **A.2 Extract features**
  - Turn each game's mouse trajectory into **a set of numbers** (a feature vector)
  - Start with 2–3 simple features (e.g. average movement speed, total movement)
  - Check: each game maps to one row of numbers + one player ID

- [x] **A.3 Classifier**
  - Split the data into training and test sets
  - Train a classifier with sklearn to learn "features → which player"
  - Check: the model can predict on test data

- [x] **A.4 Evaluation**
  - See how often the model is right (accuracy)
  - Check: result beats random guessing, and the whole pipeline is rerunnable

- [x] **A.5 Wrap-up** (before Phase B)
  - Run the 37-player subset (≥8 games) and compare accuracy with the 45-player version
  - Plot a confusion matrix
  - Record experiment settings and results in the changelog

**Phase A completion criteria**: A.1–A.5 done. ✅

---

## Phase B: Synthetic Data and Bot Detection

> Goal: generate synthetic AI trajectories and train a True Human vs. Synthetic AI classifier.

- [x] **B.1** Synthesis strategy v0: segment stitching (block bootstrap)
- [x] **B.2** Implement the bot generators (`notebooks/main.ipynb` Phase B section)
- [x] **B.3** Build the Human vs Synthetic dataset (`is_bot` 0/1)
- [x] **B.4** Train and evaluate the Human vs Synthetic classifier
  - (Rerun after the angle-wrap fix, 2026-07-17) stitch in-domain **~88%**; smooth **~99.5%**
  - Old numbers (before the fix): stitch 85.34% / smooth ~100%
- [~] **B.5** ~~Comparison against Phase A metrics~~ (skipped: low contribution to the core thesis; different class counts make direct comparison impossible)
- [x] **B.7** Ablation: stitch vs. smooth bot detection difficulty
  - After the fix: stitch is harder, smooth still separates almost perfectly (same conclusion as before)
- [x] **B.8** Fix `turn_angle_mean` angle wrap (±π) and rerun everything
  - Human `turn_angle_mean` median ~0.69 → ~0.33 (magnitude now sensible)
  - Phase A player identification ~11–12% (still far above chance)
- [x] **B.9** Synthetic-bot realism / anti-fingerprint revision (2026-07-22; official generation settings)
  - **Timing**: stitch gaps and smooth intervals both sampled from that game's empirical human `dt` distribution (`round`); removed the fixed 20–80ms and `normal(mean,2)+int`
  - **Duration**: stitch `target_duration` = median human window duration
  - **Motion**: `collect_human_motion_samples` — angles ← `atan2`; step lengths ← `hypot` (movement events + **p25–p90 truncation**, so a single huge flick cannot lock a whole segment and blow up the trajectory)
  - **Scale**: `step_median = median(hypot)`, `jitter = 0.1 * step_median` (no longer using the unit-error-prone avg_speed×dt as the generation step length; legacy comparison still printed)
  - **Segment length**: `SEGMENT_MS_RANGE=(1500,2500)` randomised, reducing the fixed 2s periodicity
  - After rerun: in-domain / SF smooth cross-game main conclusions recovered, flat or slightly better (see Changelog)
- [x] **B.10** Third synthesis type: **Bézier** curve chaining (2026-07-23; pure numpy)
  - Controlled comparison against smooth: same motion pools, same `stroke_points_range=(20,60)`, `distortion=0.1×step_median`, zero pause between strokes; differs only in cubic curvature + switchable ease-out
  - With `round_deltas=True`, points that round to `(0,0)` are skipped (event-driven semantics); CSGO stays `False`
  - Notebook: RE / LoL / CSGO generation + previews + in-domain + raw / SF diagnose all wired up; plus an RE→LoL inversion-diagnostic cell
  - Rerun highlights: in-domain still 100% (easy to catch); RE↔CSGO SF as strong as smooth (AUC≈1); **RE→LoL inversion** (raw AUC≈0.12) — `speed_std` sign-flip + `avg_speed` overshoot (see Changelog)
- [x] **B.6** Fourth bot tier: **VAE** (learned trajectory-generation proxy; not a full game cheat)
  - Weights: `vae_re_v2` / `vae_lol_v4` / `vae_csgo_v2` (`axis_std_v1`); wired into the official split-first / segment notebooks
  - Difficulty narrative: cross-game roughly smooth / bezier ≫ VAE ≫ stitch; SI-ext can rescue VAE calibrated detection (see Changelog)
  - No GAN; VAE = generative comparison point, not a real cheat client

**Phase B completion criteria**: stable, reproducible Human vs. Synthetic metrics on Red Eclipse; at least one synthesis method usable. ✅
(B.6 VAE done; four tiers: stitch / smooth / bezier / VAE.)

---

## Phase C: Second Dataset, AMuCS / CSGO Integration

> Goal: FPS vs. FPS cross-dataset experiments (as a comparison axis against the cross-genre RE→LoL results).
> Dataset: Affective Multimodal Counter-Strike (AMuCS); already downloaded to `data/CSGO/`.

- [x] **C.1** Obtain and initially explore the AMuCS / CSGO data
  - Author's caveat: real bot sessions <4 (added by mistake), **cannot be used as a genuine bot label set**
  - `mouseposition.csv` is **unusable** (the engine locks the cursor)
  - Usable: `eyeVectorX/Y/Z` in `gameFlt.csv` (view-direction unit vector) to derive crosshair dynamics
  - Questionnaires / docs **do not record** per-player mouse sensitivity; resolution and other settings may differ
- [~] **C.2** Write the CSGO / AMuCS data dictionary (eye_vector semantics, Y-up evidence, limitations)
  - **2026-08-04**: content largely merged into thesis **§3.1.2** (F.2); a standalone `docs/` dictionary is optional and does not block writing
- [x] **C.3** Implement the conversion: `eye_vector → (dx, dy, time)` (crosshair angular-velocity proxy)
  - Code: `src/data_csgo.py`; path `CSGO_DATA_ROOT`; 237 `gameFlt.csv` files
  - Y-up empirically: `asin(Y)` caps at the engine's ±89°; away-from-poles round-trip PASS (threshold 1e-5)
  - `d_yaw` wraps at ±180°; teleport frames \|Δ\|>30° dropped (median ~0.5%)
  - **Windowing (official)**: `window_mouse_round_alive` — Round 2 start → +`CSGO_WINDOW_MIN=3` min, and `gameInt.health≥1`; if only the Round 2 end is available, backtrack end−660s to infer the start (AMuCS fixed block); 237/237 succeed; alive share of the window median ~95%
  - This is **not** OS mouse pixels (no sensitivity); the up-down wobble in the cumsum preview = accumulated pitch, expected (recoil and hand control are mixed in the crosshair and cannot be cleanly separated)
  - smooth bot: `round_deltas=False`; `jitter = base * 0.1` (0.5 floor removed, see Changelog)
- [x] **C.4** Cross-dataset experiment design
  - CSGO in-domain; **RE → CSGO** and **CSGO → RE** (raw + scale-invariant)
  - Comparison axes: RE→LoL (cross-genre) vs RE↔CSGO (within-genre FPS)
- [x] **C.5** Run the FPS vs. FPS cross-dataset experiments (**using the latest synthesis settings + Round2+alive**; see Changelog)
  - CSGO in-domain (9 feat): stitch **~78%**, smooth **100%** (old first-3-min-of-recording ~99% and Round2-only ~81% no longer used; timing fingerprints keep getting weaker)
  - **RE → CSGO scale-invariant smooth**: AUC **1.0, @0.5 100%/FP 0%** (recovered after the step-length truncation fixed the exploding trajectories)
  - **CSGO → RE scale-invariant smooth**: AUC **1.0, @0.5 ~100%/FP ~0%**
  - Raw cross-game: stitch still weak; some smooth raw cells cleaner than before (see Changelog)
- [x] **C.6** Quantify within-genre vs cross-genre + write into D.7 / thesis (written into the thesis results chapter)
  - Scale table: RE vs CSGO still differ hugely (avg_speed ~1.0 vs ~0.05; idle_ratio 0 vs ~0.82) — windowing only slightly lowers idle; **units / per-frame logging** still dominate
  - Narrative: same genre does not guarantee bidirectional zero-shot; scale-invariant + smooth works both ways; stitch is hard in both directions
  - Make clear: after switching the generators to empirical dt / motion sampling, in-domain relies more on behavioural features; the old fixed gap is kept only as a historical comparison
  - Clarify: the gap is mainly due to "angle proxy vs mouse grid" units / feature semantics, not a failure to convert trajectories

**Phase C completion criteria**: CSGO convertible to the unified trajectory format and at least one RE→CSGO experiment with analysis. ✅
**Status**: all done. C.2 covered by thesis §3.1; C.6 written into the thesis.

---

## Phase D: Cross-Game Generalisation (Core Research Question)

> Goal: train a Human vs. AI model on Game A (FPS), deploy it on Game B (LoL / MOBA) without retraining, and measure the performance drop.

- [x] **D.1** Obtain and explore the LoL data (`data/lol2` — Tilburg / Dutch College League)
  - Raw mouse: 41 `*-keylogger-new.txt` files (Moved + X,Y + timestamp)
  - Conversion: X,Y diff → dx,dy; `LOL_WINDOW_MIN=3` to match RE session length
  - `data/lol` is an aggregated CSV, not raw, deprecated
  - `FAST_TEST` switch (small sample / full run) added to the notebook
- [~] **D.2** Write the LoL data dictionary (`docs/lol2_data_dictionary.md`)
  - **2026-08-04**: content largely merged into thesis **§3.1.3** (F.2: keylogger / matchid / shift / minutes 10–13 / 59 trajectories); a standalone dictionary is optional
- [x] **D.3** Refactor the main pipeline into `src/` (2026-07-17; CSGO + bot realism revision added 2026-07-22)
  - Present: `config` / `features` / `bots` / `data` (RE+LoL) / `data_csgo` / `evaluation` / `plotting`
  - Main diagnostic flow: `diagnose_cross_game` (raw + scale-invariant; calibration uses `>`)
  - Known fixes: angle wrap; stitch concat; CSGO round-trip / Round2+alive; jitter floor; SF drops idle; **empirical dt / angle / step + randomised segment length** (B.9)
- [x] **D.4** Cross-game feature alignment
  - Cross-game uses `cross_game_feature_cols` (7 features, excluding `total_movement`, `n_events`)
  - Phases A/B still use the full 9 features; when writing the triangle table, in-domain should also use the **7-feature** alignment
- [x] **D.5** Zero-shot deployment experiment (full data; both bots self-calibrated per game)
  - accuracy@0.5: stitch / smooth both **0%** (still the case after the angle fix)
  - **Do not report just the 0%**: see D.5b
- [x] **D.5b** Diagnostics (rerun after the angle fix, checked 2026-07-17; notebook finalised 2026-07-21)
  - Same RE model catching RE stitch: **98.1%** (467/476) → training is not broken
  - **AUC (raw features)**: stitch **0.242**; smooth **1.000** (human P≈0.13, bot P≈0.39)
  - Human-calibrated threshold (human scores only): rescues smooth; stitch still weak
  - **Threshold sweep** (raw): table + detect/FP curves (diagnostic only; do not pick the main-result threshold using bot labels)
  - Scale-invariant: stitch AUC ~0.81; smooth AUC **1.000**, detect@0.5 = **100%**
  - Source-only StandardScaler + RF: no-op (experiment cell deleted; one sentence in the thesis is enough)
  - Per-domain: deleted (optional ablation; the main line goes raw → scale-invariant)
  - **Writing note**: thresholds must be human-calibrated or report AUC only; never max(detect−FP) using bot labels
- [x] **D.6** Control experiment: LoL in-domain
  - stitch **94.12%**, smooth **100%** (test set is tiny, ~17 rows)
- [x] **D.7** Generalisation analysis and writing (written into the thesis results chapter)
  - Fill the tables: RE / LoL / CSGO in-domain + RE→LoL + RE↔CSGO (raw / SI-min / SI-ext / human-calibrated / @0.5) — **use the `main_segment` numbers**
  - Main-result narrative: SI-ext (10) cross-game; SI-min (4) as the compact comparison; raw 6 to demonstrate scale failure
  - Make clear: CSGO is a view-angle / crosshair proxy (with recoil mixed in); the up-down wobble in plots ≠ conversion failure; `idle_ratio` stays out of the SI headline claims
  - Limitations: stitch is hard; LoL / CSGO small n; no sensitivity info; not domain-invariant; synthetic bots = attack proxies
  - Completed and written into the thesis

**Phase D completion criteria**: clear numbers and analysis answering the core research question, sufficient to support the thesis's main experimental chapter. ✅
**Status**: all done. D.2 covered by thesis §3.1; D.7 written into the thesis; D.8 executed on 2026-08-21.

### Cross-Game Diagnostic Checklist

When bots cannot be detected, work through these in order; rerun the same set for each new target game (LoL / CSGO).

| # | Step | Purpose | Where in the notebook | Output format | Status |
|---|------|---------|-----------------------|---------------|--------|
| 1 | Source-domain in-domain | Rule out "model / features broken" | `## Human vs Bot classification`; cross-game training cell (7-feat live numbers) | **auto-printed by `train_bot_detector`**: accuracy, TP/TN/FP/FN, feature importance (no plots) | [x] |
| 2 | Target-domain in-domain | Rule out "target data / bots undetectable" | LoL / **CSGO** in-domain sanity check | Same as above (same function) | [x] |
| 3 | `predict_proba` distribution | See whether human vs bot scores separate | `diagnose_cross_game`: `## Zero-shot diagnose (raw…)` + `## Scale-invariant features…` | **Numeric summary + histogram**; no per-player grids / heatmaps | [x] |
| 4 | AUC | Ranking ability (independent of any single threshold) | Same as above (same function) | **Number** | [x] |
| 5 | Threshold sweep | See whether tuning the threshold can rescue it | Same as above (same function; merged into diagnose, no longer a separate cell) | **Table + curves**; **no** confusion matrix per threshold | [x] |
| 6 | Feature scale comparison | See whether it is scale / domain shift | RE vs LoL; **RE vs CSGO**; **CSGO vs RE** | **Table** (medians); no extra plots needed | [x] |

Shared implementation: `src/evaluation.diagnose_cross_game`; scale-invariant: `to_scale_free` / `SCALE_FREE_COLS` (**no** `idle_ratio`; relative quantities: speed_cv / speed_peak / dist_cv / turn_angle).
Human calibration / sweep: use **`score > thr`** on the RF's discrete probabilities (avoids `>=` sweeping a whole mass point into false positives).

Supplementary (not part of the main diagnostic chain, but already in the notebook):

| Item | Where | Output format | Notes |
|------|-------|---------------|-------|
| @0.5 detect / FP + human-calibrated threshold | `diagnose_cross_game` (raw + scale-invariant) | **Numbers** | Main results use these; thresholds must **not** be tuned with bot labels |
| Scale-invariant feature experiments | `## Scale-invariant features (RE → LoL)` | Training via `train_bot_detector` + the same diagnose | Main line: raw → scale-invariant |
| Feature importance | Auto-printed in-domain; separate scale-invariant cell | **Table** | **≠ the scale table** (see below) |
| Trajectory previews | Per-bot / LoL preview cells | **Plots** | Method illustration; not required for diagnostics |
| Source-scaler / per-domain | (deleted) | — | One sentence in the thesis: RF + source-fit scaler = no-op; per-domain optional ablation |

> **Scale comparison ≠ feature importance**
> - Scale table: whether the two games' feature **values are of similar magnitude** (domain shift)
> - Importance: which features the model **relies on most** (interpretability)

#### Figures / Tables Conclusions (2026-07-21)

| Plot or not | Item | Recommendation |
|-------------|------|----------------|
| Table / numbers suffice | in-domain, AUC, scale comparison, importance, human-calibrated threshold | A clear notebook listing is enough |
| Plot (diagnose already outputs) | threshold sweep + proba distribution | Curves + histograms; one round each for raw / scale-invariant |
| Optional for the thesis | in-domain 2×2 confusion matrix | At most 1 per model; no per-threshold matrices in the sweep |
| Keep as illustration | trajectory plots | For the Method chapter |
| No | per-threshold confusion matrices, per-player heatmaps | Nobody reads them; curves / distributions cover it |

Thesis figure budget (about 2–3 core figures): proba distribution, sweep curves, (optional) confusion matrix; everything else as tables.

#### Ideal Execution Order

1. Train the source-domain model → print step 1 (importance optional)
2. Target-domain human / bot features ready → print steps 2 + 6
3. **Raw** zero-shot → `diagnose_cross_game(...)` (numbers + proba plot + AUC + sweep)
4. **Scale-invariant** training → call the **same** `diagnose_cross_game(...)` again

Extracted into `src/evaluation.diagnose_cross_game` on 2026-07-21; reused directly once the CSGO target domain was ready.

### Later Notebook Split (agreed; **executed on 2026-08-21**, actual file names in "Notebook layout" above)

> Decision (2026-07-21): keep RE↔LoL in `notebooks/main.ipynb` for now. Split once **CSGO joins** (or main becomes clearly hard to maintain).
> Do **not** build "one all-purpose notebook per game that handles every cross-game combination" — cross-game is an experiment over "a pair of games", and files do not share memory, so models / features must be **persisted to disk** to be shared.

**Target structure:**

| File | Responsibility |
|------|----------------|
| `prepare_re.ipynb` (or `red_eclipse.ipynb`) | load → features → synthetic bots → in-domain (step 1) → save artifacts (feature tables + raw / scale-invariant models) |
| `prepare_lol.ipynb` | Same as above; includes target-domain in-domain (step 2) |
| `prepare_csgo.ipynb` | Same as above (after the C.3 conversion is done) |
| `transfer.ipynb` (or `re_to_lol.ipynb` etc.) | Load source models + target features → scale table (step 6) → raw `diagnose_cross_game` → scale-invariant training / diagnose |

**Principles:**
- Shared logic stays in `src/` (`diagnose_cross_game`, `to_scale_free`, bots / loaders)
- Each game notebook only "prepares its own"; cross-game experiments live in a separate transfer notebook
- Suggested artifacts paths (create when implementing): e.g. `artifacts/{game}/features_*.parquet`, `models_*.joblib`
- Do **not** split main heavily at this stage; when CSGO joins, use this scheme first to avoid triplicating the diagnostic code

- [x] **D.8** Split the notebooks per the table above + artifacts I/O (2026-08-21, commit `refactor: split experiment notebook by game`)
  - Actually split into four notebooks: `red_eclipse` / `lol` / `csgo` / `cross_game`; shared logic stays in `src/`
  - Artifacts I/O: `artifacts/data/` (features + traces CSV), `artifacts/classifiers/` (joblib), `artifacts/vae_weights/`; managed centrally by `src/artifacts_store.py` (2026-08-17 `refactor: consolidate segment pipeline`)
  - `cross_game.ipynb` reproduces all cross-game diagnostics from files alone (runs features-only too; traces need not be pushed to git)

---

## Phase E: Extensions (time permitting)

- [~] **E.1** ~~Player skill ranking / profiling exploration~~ (dropped: no time, outside the core research question)
- [~] **E.2** ~~Merge with B.6: VAE / other generator comparisons, adversarial samples~~ (dropped: VAE already included in the B.6 four tiers; adversarial samples not pursued)
- [x] **E.3** Feature importance and interpretability analysis
  - Notebook already has RF `feature_importances_` (including SI features); the thesis narrates from these, no separate experiment
- [~] **E.4** ~~(Optional) feature ablation (removing `dist_std` / `mean_dt` etc.)~~ (dropped: the SI-ext ablation already covers the main question)

---

## Thesis and Wrap-Up (can run in parallel with experiments; do not leave until the final two weeks)

- [x] **F.1** Thesis structure draft (Introduction → Related Work → Method → Experiments → Conclusion)
- [x] **F.2** Method and experiments chapters (thesis finished, incl. RE / LoL / CSGO tables; VAE / SI-ext)
- [x] **F.3** Figures and tables (thesis finished, figures finalised)
- [x] **F.4** Limitations and Future Work (thesis finished)
- [x] **F.5** Repo cleanup (2026-08-17 ~ 2026-08-21, see git log)
  - 2026-08-17 `refactor: consolidate segment pipeline and simplify src modules`
  - 2026-08-21 `refactor: split experiment notebook by game`, `refactor: integrate src/data`
  - 2026-08-21 `docs: add readme.md`, `docs: add [TOC] in readme` (project description, environment setup, reproduction steps)
- [x] **F.6** Final self-check: all experiments reproducible, paths and documentation consistent (2026-08-24)

---

## Rolling Decision List

- **Evaluation metrics**: cross-game always reports **AUC** in addition to accuracy@0.5; threshold sweep when needed; human calibration uses **`>`** (discrete probabilities)
- **Scaling**: main results use true zero-shot (source-domain or scale-invariant); per-domain only as ablation; **source-fit scaler + RF = no-op**
- **Scale-invariant feature list**: excludes `idle_ratio` (`distance < 1` is unit-bound + CSGO's per-frame logging; not comparable across games)
- **Smooth / Stitch / Bézier timing (official)**: each bot samples from a **single human session's** `dt` distribution (`dt_by_session`); no longer the globally event-weighted mixed pool (fixes LoL `mean_dt` being dominated by high-frequency sessions)
- **Smooth / Stitch motion**: direction ← `atan2`; step length ← `hypot` (p25–p90); segment length randomised 1.5–2.5s; `jitter = 0.1×step_median`
- **Bézier**: chord length = one step × point count per stroke; `bend_scale=0.15`; ease-out switchable (for ablation); tremor magnitude matched to smooth jitter
- **CSGO trajectories**: `eye_vector` crosshair proxy; windowing = Round2+alive; from-scratch bots use `round_deltas=False`
- **CSGO stitch in-domain**: headline ~76–78% (behavioural level); the old ~99% only as a timing-artifact comparison
- **Synthetic bots**: stitch / smooth / Bézier / **VAE**, four official tiers (no GAN); difficulty narrative driven by the cross-game gap
- **Cross-domain main metrics**: AUC + human-calibrated (`>`); @0.5 as a footnote only (operating point drifts easily)
- **Refactoring**: main-line modules complete; diagnostic AUC / scale-invariant code may stay in notebooks
- **Notebook organisation** (from 2026-08-21): official = the four notebooks `red_eclipse` / `lol` / `csgo` / `cross_game`; the old `main_segment` etc. removed (historical numbers carry over from their results)
- **Feature freeze**: SI-min main, SI-ext ablation; no more features; `xy_corr` kept as a column but not claimed as a contribution

---

## Changelog / Progress

> After completing and verifying a task, add an entry here (newest at the top).

### 2026-08-24 — Project wrap-up: all tasks closed

- **Thesis (F.1–F.4, C.6, D.7)**: finished (written over an extended period, no single completion date); C.2 / D.2 data dictionaries covered by thesis §3.1
- **Repo refactoring and cleanup (D.8, F.5)**, dated by git commits:
  - 2026-08-17 `refactor: consolidate segment pipeline and simplify src modules` (module consolidation; `artifacts_store` unifies the data / classifier caches)
  - 2026-08-18 `fix: use_cols to usecols`
  - 2026-08-21 `refactor: split experiment notebook by game` (split into `red_eclipse` / `lol` / `csgo` / `cross_game`)
  - 2026-08-21 `refactor: integrate src/data` (`data_csgo` merged into `data`)
  - 2026-08-21 `docs: add readme.md`, `docs: add [TOC] in readme`, `doc: add meeting logs`
- **Dropped items**: 0.2 (experiment logging conventions), E.1 (skill ranking), E.2 (adversarial samples), E.4 (extra ablation); struck through with reasons noted
- **Final state**: all four notebooks rerunnable; `cross_game.ipynb` reproduces the cross-game results from `artifacts/` alone; README includes reproduction steps

### 2026-08-04 — Thesis Methodology: first drafts of §3.1 Datasets, §3.2 Feature Extraction

- **Progress (F.2)**: started writing the Method chapter
  - **§3.1**: RE / CS:GO (AMuCS) / LoL → unified `(dx, dy, t)` (incl. eye_vector, Round2+alive, LoL 10–13 + shift)
  - **§3.2**: in-domain 14, cross-game raw 6, SI-min 4, SI-ext 10 (incl. `efficiency_sub`, the dual-track SI design)
- **Mapping to existing items**: C.2 / D.2 content covered mainly by thesis §3.1 (`[~]`); D.7 / C.6 results narrative not yet started
- **Next**: fact-check §3.1–3.2 against the code and finalise → remaining Method subsections → D.7 / C.6

### 2026-07-27 — Feature validation wrap-up: freeze `main_segment` as the official entry point

- **Done**
  - SI-min / SI-ext rerun in `main_segment` (in-domain + RE↔LoL / CSGO)
  - SI-ext greatly improves VAE cross-game **calibrated detection** (vs SI-min threshold collapse)
  - Ablation RE→LoL: EXT−`xy_corr`, EXT−{`xy_corr`,`vh_ratio`} → VAE calibrated detection still **100%** (not carried by fingerprint features alone)
  - Hypothesis revision: `xy_corr` does not catch VAE; `step_autocorr` does not catch stitch seams; `vh_ratio` carries game-fingerprint risk but is not the VAE's sole dependency
- **Decisions**
  - **Official main results** switched to `main_segment`; `main_split_first` **no longer needs rerunning**
  - **`xy_corr` not removed** (stays in cols; thesis says "tried, importance≈0")
  - Game fingerprints: **no new large experiment**; use the existing ~99% + SI-ext inter-game human differences to write the limitation
- **Next**: freeze the tables → write D.7 / C.6 / Scope; save the time for the thesis
- **Suggested branch / commit**: see that day's conversation; do **not** commit the accidentally modified `.gitignore` (ignoring `docs/` / `TODO.md` / itself)

### 2026-07-27 — Feature finalisation implemented (in-domain 14 / raw 6 / SI-min 4 / SI-ext 10)

- **`src/features.py`**
  - New in-domain: `efficiency_sub` (mean over 40-event subsegments), `turn_angle_std`, `vh_ratio`, `dir_change_rate`, `xy_corr`
  - Raw cross-game: removed `mean_dt` (kept `idle_ratio` as a failure demonstration)
  - SI main set: `speed_cv` / `speed_peak` / `turn_angle_mean` / **`accel_cv` (replacing dist_cv)**; `dist_cv` still computed by `to_scale_invariant` for comparison
  - SI ablation: `SCALE_INVARIANT_EXT_COLS` (+ turn_angle_std, efficiency_sub, vh_ratio, dir_change_rate, xy_corr, step_autocorr)
- **Verified premises (before implementing)**: speed_cv↔dist_cv collinear; accel_cv ablation improves bezier→LoL; VAE needs a monitored rerun; fingerprint check → write "transferable", not domain-invariant
- **Follow-up**: closed the same day by the "feature validation wrap-up" entry (segment finalised; split_first no longer scheduled)

### 2026-07-27 — fix: VAE generation speed-up (shared pool + model cache)

- **Problem**: `evaluate_split_first_*` calls `generate_bot_feature_tables` per fold; the old implementation ran `sample_vae_segments(256)` and rebuilt the model for **every** VAE bot session → RE in-domain extremely slow
- **Fix**
  - `generate_vae_bot_games`: decode once → stitch multiple sessions
  - `generate_bot_mouse_games` / notebook preview cells switched to the batch API
  - `vae_from_bundle` caches `_model_cache`; `save_vae_bundle` strips it when saving
- **Unchanged**: `vae_n_pool_segments=256`; the split-first "regenerate per fold" protocol stays
- **Suggested commit**: `fix: share VAE segment pool across fold bot generation`

### 2026-07-27 — VAE merged into the official notebooks + duration / axis revisions + calibrated metrics + QA split-out

- **Merge state (`dev/vae-bot`)**
  - Both `main_split_first` / `main_segment` sources include: VAE, `load_lol_match_windows`, `target_duration_ms`, `vae_bundle` fold generation
  - `src/evaluation.py`: headline **AUC + human-calibrated**; `@0.5` kept as `*_05`
  - `src/bots.py`: smooth supports `target_duration_ms` (aligned to ~180s with bezier / stitch / VAE)
  - `src/vae_bot.py`: `axis_std_v1` per-axis; default weights **v2 / v4 / v2**
- **Trajectory QA**: moved **out of** `main_segment` → `notebooks/trajectory_qa.ipynb` (standalone load / generate / sanity)
- **Verified (segment rerun)**: durations aligned; RE VAE dx/dy ratio improved; in-domain strong for non-stitch; cross-FPS headline is SI smooth / bezier; VAE weak cross-game; RE→LoL bezier raw prone to inversion
- **Still pending**: rerun `main_split_first` with the new yardstick → freeze the session-level main table; fill D.7 / C.6; thesis Scope (synthetic bots = attack proxies)
- **Commit note**: clear notebook output before pushing; do **not** commit the accidentally added `.gitignore` (ignoring `docs/` / `scripts/` / itself)

### 2026-07-25 — LoL match 10–13 + timestamp fix + derived cache + VAE v3

- **Windowing**: `load_lol_match_windows` (`matchid` + `shift.py`); fixed game clock **10–13**; `min_events=100` → **59** windows
- **Bugfix**: keylogger fractional milliseconds need `rjust` (`.43` = 43ms); the old `ljust` caused time to run backwards within a second, scrambled ordering and fake `speed_max~2000`
- **Post-fix human medians (notebook)**: `avg_speed≈2.56`, `speed_max≈158`, `mean_dt≈2.82` (vs bugged ~32 / ~2058)
- **Cache**: `data/lol2/derived/mouse_windows_10_13_v1/`; daily load ~0.3s; `build_lol_match_windows_cache()` to re-extract
- **VAE**: `vae_lol_v1.pt` (retrained after the parse fix); RE / CSGO still v1
- **`main.ipynb` rerun (RE→LoL raw)**: stitch AUC ~0.64; smooth ~0.98; bezier ~0.23 (still inverted); vae ~1.0 (@0.5 FP still high; after calibration 100%/0%)
- **Next**: merge VAE + the new LoL loader into `main_post_split`; D.7 / C.6

### 2026-07-25 — Full windowing comparison done; main results stay session-level post-split (split players first, then synthesise bots)

- **Decision**: official in-domain = `main_post_split` (session level); `main_segment` = comparison / early check; no return to the old `main` random split
- **Full windowing (10s) highlights**
  - RE: window-level stitch ~65%; session aggregation ~84% ≈ session-level post-split (split players first, then synthesise bots)
  - LoL (7 players): uglier / unstable; windowing cannot fix "too few players"
  - CSGO (4 players, many sessions): window / aggregated numbers inflated (~85% / ~99% vs session-level ~70%) → do not use directly as the main table
- **Player-count clarification**: LoL 41 files = 7 players; CSGO's many sessions = P1–P4, only 4 players

### 2026-07-25 — Segment-level (windowed) pipeline wired into `main_segment`

- **Settings**: non-overlapping fixed-length windows, default `WINDOW_MS=10000`, `WINDOW_MIN_EVENTS=30` (overridable in the notebook)
- **src**: `segment_trace` / `build_window_feature_table`; `generate_bot_mouse_games`; `evaluate_post_split_group_kfold_windows`
- **Notebook**: only `main_segment` changed; `main` / `main_post_split` untouched

### 2026-07-23 — Bézier bot + per-session dt (all three games rerun)

- **Implementation (`src/bots.py`; full pipeline wired in `notebooks/main.ipynb`)**
  - **B.10 Bézier**: cubic + ease-out + normal-direction tremor; controlled comparison with smooth (same pools / stroke lengths / tremor scale / no inter-stroke pauses)
  - Event-driven: `round_deltas=True` discards post-rounding `(0,0)` (avoids fake `idle_ratio`)
  - **dt protocol upgrade**: `dt_by_session`; stitch / smooth / bezier each bind one human rhythm template per session
  - Diagnostic cell: RE→LoL Bézier inversion (four-group medians + importance + `sign_flip`)
- **Calibration**: LoL bot `mean_dt` ~7.4 → **~11–12** (human ~10.5); parameters `mean_interval_ms` LoL 1.0 → **10.5**, RE 17 → **30.9**
- **After rerun (vs pre-Bézier; main conclusions)**
  | Experiment | stitch | smooth | bezier |
  |------------|--------|--------|--------|
  | RE / LoL / CSGO in-domain | ~76–88% | **100%** | **100%** |
  | RE→LoL raw / SF | weak / SF moderate | **AUC 1.0** | **inverted** AUC ~0.12 / 0.36 |
  | RE↔CSGO SF | weak | **1.0 / cal ~100%** | **1.0 / cal ~100%** |
- **Interpretation**: Bézier ≠ a middle-difficulty tier (still easy to catch in-domain); can invert cross-genre (LoL); within-genre FPS (CSGO) SF as strong as smooth. The `avg_speed` overshoot remains (filtered step pool + independent dt), not a dt-pool problem.
- **Next**: use this table for D.7 / C.6; fourth tier pending supervisor; optional ease ablation

### 2026-07-22 — Synthetic-bot realism revision (empirical motion sampling; official generation settings)

- **Motivation**: fixed gaps / fixed dt noise, isotropic angles, short-tailed speeds, fixed 2s segment length → generator fingerprints; validity risk at the viva
- **Implementation (`src/bots.py` / `config.py`; RE / LoL / CSGO generation cells wired up)**
  - `collect_human_motion_samples`: `dt` + truncated `step` (hypot p25–p90) + `angle` (atan2, movement events)
  - stitch: gaps ← `dt`; target duration ← median human duration; `build_segments` segment length randomised **1500–2500ms**
  - smooth: intervals ← `dt`; per-segment direction / step ← empirical distributions; `jitter = 0.1 * step_median`
  - **Pitfall**: once sampled from the untruncated full hypot → a giant flick locked whole segments → exploding trajectories, SF smooth cross-game collapse (AUC ~0.02); recovered after switching to p25–p90
- **After rerun (vs the "empirical dt only" version: main conclusions flat or slightly better)**
  | Experiment | stitch | smooth |
  |------------|--------|--------|
  | RE in-domain | ~89.5% | **100%** |
  | LoL in-domain | 94% | **100%** |
  | CSGO in-domain | **~78%** | **100%** |
  | RE→LoL raw / SF (smooth) | — | AUC **1.0** (SF @0.5 best paired with human calibration) |
  | RE→CSGO SF smooth | weak | **1.0 / 100%·FP 0%** |
  | CSGO→RE SF smooth | weak | **1.0 / ~100%·FP ~0%** |
- **Next**: ~~B.10 Bézier~~ → done 2026-07-23; use the latest table for D.7 / C.6

### 2026-07-22 — Code-review revisions + Round2+alive rerun (numbers partially superseded by the "synthetic realism" entry)

- **Revisions (by severity)**
  1. **`estimate_smooth_params` jitter**: removed `max(0.5, …)` → `base * 0.1` (CSGO relative jitter back to ~10%; RE / LoL unchanged)
  2. **CSGO stitch ~99%**: attributed to the fixed 64Hz + 20–80ms stitch-gap timing fingerprint; for Discussion / dual reporting, not a headline "behavioural detection success" number
  3. **`SCALE_FREE_COLS`**: removed `idle_ratio` (unit threshold + logging-mechanism trap)
  4. **Human calibration / sweep**: `score > thr` (fixes the RF discrete-mass-point `>=` artefact of calibrated FP=100%)
  5. **Windowing**: `window_mouse_round_alive` (Round 2 start + alive); replaces first-3-min-of-recording; end−660s when only the end exists; 237/237 OK
- **Key post-rerun numbers (official)**
  | Experiment | stitch | smooth |
  |------------|--------|--------|
  | CSGO in-domain (9 feat) | **~81%** | **100%** |
  | CSGO in-domain (SF) | **~81%** | **100%** |
  | RE→CSGO raw AUC | ~0.46 | ~0.95 (@0.5 still unusable) |
  | RE→CSGO SF AUC / @0.5 | ~0.58 / undetected | **1.0 / 100%·FP 0%** |
  | CSGO→RE SF AUC / @0.5 | ~0.41 / weak | **1.0 / 100%·FP 0%** |
- **Interpretation**: windowing makes stitch in-domain honest (99%→81%); SF smooth still strong both ways; the idle / scale gulf remains; raw cross-game still fails
- **Next**: write D.7 / C.6 from the table above + C.2 dictionary + F.1; no need to add pitch filtering for the "wobbly trajectory" issue

### 2026-07-22 — CSGO conversion + bidirectional cross-game (Phase C main experiment; **numbers superseded by the same-day revision entry**)

- **`src/data_csgo.py`**: eye_vector → angle differences `(dx,dy,time_ms)`; axis check + `validate_real_roundtrip`; all 237 traces
- **CSGO pipeline (main.ipynb)**: features → stitch/smooth (`round_deltas=False`) → in-domain → RE→CSGO / CSGO→RE (raw + scale-invariant) → scale table
- **At the time (first 3 min of recording)**: in-domain stitch ~99% / smooth 100%; RE→CSGO raw fails; CSGO→RE SF smooth AUC 1.0 (later confirmed bidirectional)
- **Scale table**: avg_speed 1.0 vs 0.05; idle_ratio 0 vs ~0.84 → primary evidence of domain shift
- **Superseded**: windowing changed to Round2+alive; stitch headline number changed to ~81%; see the previous Changelog entry

### 2026-07-21 — Diagnostics finalised; later notebook split agreed (not split yet)

- RE→LoL: `diagnose_cross_game` unifies raw / scale-invariant (proba + AUC + human-calibrated threshold + sweep); scale-invariant train / diagnose in separate cells; labels avoid `/` so the IDE does not mistake them for paths
- Scale-comparison table: for argumentation, not pipeline-critical; kept as step 6 for now
- **Notebook**: keep `main.ipynb`; **later** split into prepare_re / prepare_lol / prepare_csgo + transfer (artifacts persisted); see D.8 and "Later Notebook Split"
- **Next**: CSGO C.3 → D.7 / F.1; split when CSGO lands or main becomes hard to maintain

### 2026-07-21 — Diagnostic notebook finalised (threshold sweep + headings / live numbers)

- Added **threshold sweep** (raw RE→LoL): 21 points over 0–1, table + detect/FP curves; noted that bot labels must not be used to pick the main-result threshold
- Fixed md headings: the human-calibrated cell no longer mislabelled as "AUC only"
- Cross-game per-tier references switched to live numbers `re_acc_stitch` / `re_acc_smooth` (7-feature in-domain)
- Scale-invariant / importance headings anglicised; deleted the source-scaler and per-domain experiment cells (main line raw → scale-invariant)
- TODO: the 6-step diagnostic table gained **where / output format**, figure conclusions and the ideal execution order; next **CSGO C.3** + **D.7**
- **Later the same day**: extracted `diagnose_cross_game`; shared by raw / scale-invariant (incl. proba histogram + sweep); `to_scale_free` moved into `src/features.py`

### 2026-07-17 — Main-line refactor ready + full rerun after the angle wrap fix (output verified)

- **Refactor conclusion**: `src/` = `config` / `features` / `bots` / `data` / `evaluation` / `plotting`; diagnostic scripts stay in the notebook
- **Bugfix**: `turn_angle_mean` wrap; `stitch_bot_game` → `pd.concat`
- **Verified rerun numbers**:
  | Experiment | stitch | smooth |
  |------------|--------|--------|
  | RE in-domain (9 feat) | 87.96% | 99.48% |
  | RE in-domain (7 feat, cross-game training) | 87.43% | 99.48% |
  | LoL in-domain | 94.12% | 100% |
  | RE→LoL detect@0.5 | 0% | 0% |
  | RE→LoL AUC | 0.242 | **1.000** |
  | scale-invariant detect@0.5 | ~0% | **100%** (FP 2.4%) |
  - Phase A (37 players): **11.36%** (chance 2.70%)
  - `turn_angle_mean` median: RE human ~**0.33** (pre-fix ~0.69)
- **Next**: write D.7 → CSGO (C.3) → thesis outline (F.1)

### 2026-07-16 — Diagnosed the LoL "0%", obtained CSGO, planned next steps

- **Diagnosis (pipeline not entirely broken)**:
  - The same model still catches RE stitch bots at **97.3%**
  - Raw-feature AUC: stitch **0.475**, smooth **1.000** → smooth has ranking ability; the 0% is a threshold problem
  - Feature scale differences: `dist_std`, `mean_dt` etc.
  - Scale-invariant features: smooth zero-shot reaches detect@0.5 **100%** (FP 7.3%); stitch still weak
- **CSGO / AMuCS**: data in `data/CSGO/`; needs `eye_vector → dx,dy`; no sensitivity records; virtually no real bot sessions
- **Refactor**: `src/features.py`, `src/bots.py`, `src/config.py`; later extended to include data/evaluation/plotting (see 2026-07-17)
- **Next (priority order)**: write the diagnostics into D.7 → CSGO unit conversion (C.3) → thesis outline (F.1) → then B.6 / VAE

### 2026-07-11 — D.6 LoL in-domain control done; the triangle of experiments complete

- **LoL in-domain (train+test both LoL, shared feature functions)**: stitch **94.12%**, smooth **100.00%**
- **Triangle complete**:
  | Experiment | stitch | smooth |
  |------------|--------|--------|
  | RE in-domain | ~85% | ~96% |
  | LoL in-domain | 94% | 100% |
  | RE→LoL (cross-game) | 0% | 0% |
- **Core conclusion**: both games are detectable in-domain, cross-game is all 0% → the failure comes from cross-game transfer (domain shift), not from the data or method
- **Notebook**: added two Phase D (D.6) cells (mirroring RE Phase B's `eval_human_vs_bot`)
- **Files**: md docs moved to `docs/` (including this TODO)
- **Next**: refactor code (notebook → `src/`, starting with features.py); then write D.7 + book the supervisor

### 2026-07-11 — smooth bot switched to per-game self-calibration (fixing the inflated 100%)

- **Problem**: the smooth bot previously used hard-coded parameters (identical across games); the LoL smooth 100% was an artefact
- **Fix**: added `estimate_smooth_params()`, deriving the smooth bot's scale from each game's human `mean_dt`/`avg_speed`; RE and LoL calibrated separately
- **Parameters**: RE dt_mean~33, base 16–49; LoL dt_mean~10.5, base 10–30
- **New results (zero-shot RE→LoL)**:
  - stitch: **0%**, smooth: **0%** (the old 100% gone)
  - in-domain still high: RE smooth ~96%, stitch ~85%
- **Post-fix conclusion (more honest)**: both bot types detectable in-domain; both fail cross-game → handcrafted features carry "game-specific scale", thresholds break across games (domain shift, not a bot-type problem)
- **Next lever**: feature normalisation / scale-independent features (the only plausible way to lift cross-game off 0%; echoes T-Detector's universal representations)

### 2026-07-11 — Phase D adds both LoL bots (stitch + smooth) as a comparison

- **Motivation**: stitch still 0% after the timestamp fix; add a second bot type to create contrast
- **Results (zero-shot, trained on RE → tested on LoL)**:
  - stitch: detection **0%**, FP 0%
  - smooth: detection **100%**, FP 7.3%
- **Key insight**:
  - The smooth bot is "game-independent mechanical generation" (fixed parameters); RE/LoL features nearly identical → the universal signal transfers
  - The stitch bot carries the "game-specific human distribution" → domain shift, does not transfer
- **Thesis caveat**: the smooth 100% arises because both games share the same generation procedure, not strictly cross-game; must be stated explicitly
- **Notebook**: LoL stitch/smooth each with generation + trajectory plots + zero-shot; final cell prints the tier comparison

### 2026-07-11 — Fixed the LoL timestamp parsing bug (important)

- **Bug**: the keylogger timestamp's decimal part is "integer milliseconds (0–999, 1–3 digits)", not a decimal fraction; the old `%f` read `.48` as 480ms and `.97` as 970ms, and the subsequent `sort_values` scrambled event order and caused displacement spikes
- **Verification**: scan of 300k rows; under the integer-millisecond interpretation, 297,875 rows monotonically increasing, 0 scrambled; max frac = 999
- **Fix**: parse datetime to the second + `int(frac)` milliseconds
- **Effect (first 3 min, single file)**: avg_speed 19.3→1.53, speed_std 97.6→1.72, speed_max 1215→34.4, now in the same ballpark as RE (1.0 / 1.7 / 21)
- **Impact**: the earlier Phase D zero-shot 0% was built on corrupted data and **must be rerun**; the column mapping (dx,dy,time) and `extract_features` themselves were correct, only the parsing was wrong
- **Residual differences (not bugs)**: LoL mean_dt ~11ms (faster sampling), smaller dist_std (screen-pixel diffs vs FPS relative aiming displacements) — genuine game differences

### 2026-07-11 — Phase D v1 done: zero-shot cross-game experiment running

- **Done**: lol2 loader, cross_game features, full RE→LoL zero-shot (41 LoL sessions)
- **Results**:
  - RE in-domain bot detection: **84.82%**
  - LoL human false positives: **0%** (0/41)
  - LoL bot detection (zero-shot): **0%** (0/41)
- **Interpretation**: severe FPS→MOBA domain shift; 7-dim features + RandomForest do not transfer across games; **the negative result can go in the thesis**
- **Technical decisions**: `cross_game_feature_cols` excludes cumulative quantities; `FAST_TEST` switch; LoL takes the first 3 min
- **Next**: D.6 LoL in-domain sanity check; D.7 feature comparison table + thesis paragraphs

### 2026-07-11 — Phase D kick-off: lol2 confirmed usable

- **Dataset**: `data/lol2` = Tilburg University Dutch College League
- **Usable raw data**: 41 keylogger files; `data/lol` deprecated
- **Notebook**: Phase D loader + trajectory previews + zero-shot cells

### 2026-07-10 — Phase B wrap-up: B.5 skipped, marked complete

- **B.5 skip rationale**: Phase A (37-class player identification) vs Phase B (2-class bot detection) are different tasks with different baselines; comparing accuracies directly contributes little to the core thesis questions (cross-game generalisation, the bot spectrum)
- **Phase B complete**: stitch + smooth bots, Human vs Bot classifier, bot tier comparison
- **Next**: **Phase D** (the core) — train on Game A → zero-shot deploy on Game B; Phase C in parallel if the AMuCS DUA comes through

### 2026-07-10 — B.7 bot tier comparison results

- **stitch bot**: 85.34% accuracy (human/bot F1 ~0.85)
- **smooth bot**: **100.00%** accuracy (complete separation)
- **Interpretation**: smooth looks smoother visually but is statistically more mechanical (tiny dt variance, regular turning angles), which makes it easier to catch; stitch uses real human segments, is statistically closer to human, and is harder to catch
- **Thesis angle**: confirms the bot spectrum — the more mechanical, the easier to detect; "looks smooth" ≠ "statistically human"

### 2026-07-10 — Phase B main line running (B.1–B.4)

- **Done**: segment-stitching bot + Human vs Bot classifier running on all 476 games
- **Bot generation**: 35,693 chunks → 476 synthetic bot games
- **Results**: accuracy **85.34%** (test 191 rows), human/bot F1 both ~0.85; beats the 50% baseline
- **Interpretation**: the segment-stitching bot still leaves detectable traces (stitch boundaries, temporal discontinuities) that the model can learn
- **Next**: B.5 comparing Phase A (player identification ~12–15%) vs bot detection (85%) difficulty

### 2026-07-10 — Phase B main line (segment stitching) added to the notebook

- **Done**: `notebooks/main.ipynb` gained block-bootstrap bot generation + Human vs Bot classification
- **Method**: cut human trajectories into 2s chunks → randomly stitch into synthetic bots → same 9 features → RandomForest (0/1)
- **Next**: run Feature extraction → Phase B → Human vs Bot in order, confirm accuracy and update the changelog

### 2026-07-09 — Phase A complete (incl. wrap-up)

- **Evaluation setup**: 80/20 split by game (`stratify=y`); 9 mouse features; RandomForest
- **45 players (476 games)**: accuracy **14.58%**, chance baseline 2.22%
- **37 players (436 games, ≥8 games each)**: accuracy **12.50%**, chance baseline 2.70%
- **Excluded players**: userId 9, 17, 18, 24, 34, 40, 43, 44
- **Confusion matrix**: only 2–5 test games per player, so large per-player fluctuation is normal; overall far above chance
- **Deferred**: sliding windows (a later optimisation, not a Phase B prerequisite)
- **Next**: Phase B — generate simple synthetic bot trajectories

### 2026-07-07 — Simplified the Phase A plan

- **Adjustment**: no five-layer `src/` module structure; switched to four notebook steps (load → features → classify → evaluate)
- **Learning style**: one step at a time, technical terms in plain language; advanced options (sliding windows, 37 vs 45 players etc.) left off the list for now
- **Next**: A.1 load data

### 2026-07-07 — Phase 0 complete

- **Done**:
  - 0.1 `docs/project_structure.md` (directories on demand, small-sample testing convention)
  - 0.3 `notebooks/eda_red_eclipse.ipynb` (with `MAX_GAMES` small-sample mode)
  - 0.4 `docs/red_eclipse_data_dictionary.md` (fields, filtering rules, known issues)
  - Updated `CONTEXT.md` Phase A to multi-player identification
- **Decisions confirmed**: 37/45 players as a filtering parameter; whole game = 1 sample; mouse only; 0.2 deferred; evaluation protocol left to Phase A
- **Next**: A.1 load data (`notebooks/phase_a_baseline.ipynb`)

<!-- Example format:
### 2026-07-07 — 0.1 Set up the directory structure
- **Done**: created the `src/` module skeleton and the `experiments/` directory
- **Issues**: none
- **Next**: 0.2 define experiment logging conventions
-->
