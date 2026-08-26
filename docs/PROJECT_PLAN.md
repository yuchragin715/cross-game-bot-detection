Project Background
This is the final project of a one-year MSc in the UK, completed independently over roughly 12 weeks in the final term, with a supervisor, regular meetings, code pushed periodically to GitLab, and a report plus the project submitted at the end.

* Initial starting point: the original idea was to study continuous authentication across different contexts (e.g. office vs. gaming), but that difference is too obvious and lacks academic challenge.
* Narrowing down the pain point: anti-cheat AI models in the games industry are mostly trained for a single game. When a new game launches, the publisher simply does not have enough cheat data to train a defence model. This is known as the "Zero-Day Bot Detection" problem.
* Core research question: can we train a general-purpose model on Game A using real player data plus "AI-synthesised fake trajectories", then deploy that model directly on Game B and successfully catch the cheats in Game B?

Final topic direction: the core is the cross-context generalisation of behavioural biometrics in game settings. Concretely:

* Human vs. AI detection (bot detection): use real human data + generated synthetic (AI) input data to train a "human vs. AI" classifier, and test whether that model generalises across games — that is, whether a detector trained on one game still works when moved to another game.
* The real-world problem to solve: games / online systems worry about AI bots impersonating humans (cheating, account theft). Behavioural biometrics can detect human vs. bot, but can an anti-cheat model generalise across games, or does every game need retraining? This is an open question with industrial relevance.

Data Sources and Characteristics (The Datasets)
To ensure academic rigour, we discard databases without published backing and focus on the following public datasets with high academic value:

* Game A (FPS genre): AMuCS (CS:GO, Nature journal) or Red Eclipse (IEEE journal).
   * Data characteristics: records event-triggered timestamps and the mouse's "relative displacement (dx, dy)", or trajectories can be recovered through view-angle conversion. Needs converting to absolute screen coordinates.
* Game B (different mechanics, or control group): Smerdov eSports (LoL, MOBA genre).
* Cross-dataset technical challenge: the datasets record the mouse differently (Red Eclipse is dx/dy displacement, AMuCS is view-angle rotation, LoL is XY position per second), so features must be abstracted into a "comparable form" (e.g. converting everything into speed / direction distributions rather than comparing raw values). Red Eclipse and AMuCS are both "relative displacement", so they should in theory align more easily.


Overall Execution Plan
Phase A: player identification pipeline. Using the single Red Eclipse dataset, run the full "raw data → feature extraction → train classifier → evaluate accuracy" pipeline; classifier to be discussed. No generated data yet. Using only real player data, build a model that can distinguish "player A from player B".
Phase B: synthetic data generation + human vs. bot detection. Generate synthetic "bot input" data for the game data (generation methods from simple to hard: simple statistical / scripted bots → advanced generative models), and train a "human vs. bot" classifier.
Phase C: bring in the second dataset (AMuCS). Once the DUA is approved, process the second game dataset and run FPS vs. FPS cross-dataset experiments.
Phase D: cross-game generalisation test (the core research question). Test whether "a human-vs-bot detector trained on one game generalises to another game", and quantify the degradation in generalisation.
Final weeks: write the report.

Resources and Tools

* Development environment: Mac (M4) + Cursor + a Python virtual environment.
* Main packages: pandas, numpy, matplotlib, scikit-learn, jupyterlab.
* Version control: GitLab, with regular commits.

Working with Cursor:

* `PROJECT_PLAN.md`: the core background document for Cursor (reference with `@PROJECT_PLAN.md`).
* `TODO.md`: tracks task progress in check-boxes. Top section: ## To-Do; bottom section: ## Changelog / Progress.