# AI Usage Details and Methods

## Division of Work

| | Human | Claude | Cursor |
|---|---|---|---|
| Role | Decide the research question and scope, design experiments, weigh every technical option, verify all AI output | Method discussion, literature organisation, concept explanation, structuring and polishing the report and slides | Writing code, refactoring, debugging, cross-file consistency checks |
| Strengths | Deep understanding of the actual content and its meaning, and making the final decisions | Good at in-depth discussion of a single question | Can read the whole project folder and edit code files directly |

For the same question, the answers from the two AIs can be cross-compared, checking each other for logical gaps, incorrect approaches, or better implementations. The final decision is always mine.

## Understanding and Supervising AI

- The AI needs to be given specific instructions in the correct direction. Otherwise the output tends to drift, or contains hallucinations and errors.
- AI does not truly understand the task. It predicts likely answers from patterns in large amounts of data, without truly understanding the context and meaning behind them. So I must have a solid grasp of the task at hand, and the instructions must be specific, or the output stays at the surface level.
- The amount of content in a prompt also matters. When the input is too long, the AI pays less attention to the middle part, and anything beyond the context limit is cut off. In these cases the error rate rises clearly and hallucinations appear.
- The AI also tends to take shortcuts to save computation, so its output must be checked for completeness.
- AI tends to extend its answer from content that looks similar on the surface but actually follows a different logic. These errors read smoothly, so the logic must be verified section by section.
- AI output is often too detailed and flat. It can easily produce a long piece of text that is fragmented and lacks focus, so the key points and their hierarchy then need to be re-selected and re-arranged.
- Code written by AI is often overly defensive and redundant, so what to keep must be decided and the code cleaned up. Naming and coding style may also be over-simplified or inconsistent, so suitable names and styles are decided after discussion with the AI.

## How AI Was Used in This Project

1. The scope and steps of the project were planned with Claude.
2. After the plan was confirmed through back-and-forth discussion, the conclusions were written into TODO.md, which Cursor then maintained as the main record of project progress.
3. Cursor is an IDE (similar to VS Code), so code can be written directly in it. I wrote concrete pseudo code in Cursor, and Cursor turned it into actual code. The generated code was then verified, modified, tested and integrated by me.
4. When there were doubts or when necessary, the code or the content in question was brought back to Claude to check for problems, errors or gaps. Or in the other direction, Claude's suggestions were given to Cursor to evaluate.

### Files Related to AI Collaboration

`PROJECT_PLAN.md`:
- Fully hand-written by me to keep the content correct and faithful to the original intent. AI only reads it and never edits it.
- Contains the project background and scope, ensuring the AI understands them correctly and that the premises do not drift as the conversations progress.

`TODO.md`:
- Maintained by Cursor. Only the AI edits it. I do not touch it.
- The upper part holds the project steps, to-do items and their checked status. The lower part is a timeline, updated at the end of each stage.

## Usage Examples

### Claude

- Initial planning of the concrete steps of the project.
- Discussing possible approaches and method choices, for example which features to extract, which methods to use for generating synthetic bots, and the reasons behind them.
- Presenting my own ideas and asking Claude to assess their feasibility and weaknesses. For example, the original idea was to let the scripted bot follow game events when generating trajectories. After discussion it was confirmed that randomly generating game events is a separate large topic, so the idea was dropped.
- Clarifying the reasons behind certain decisions. For example, why the mouse trajectories during in-game events are kept in data cleaning rather than skipped or processed uniformly. It was later decided that keeping natural gameplay behaviour is what makes the comparison truly cross-game.
- Discussing the choice of the final evaluation methods, the meaning of the results, and which diagnostics are needed so that the numbers are complete and can be clearly presented in the report.
- Asking Claude to explain complex concepts, knowledge and terminology, using plain, concrete, everyday examples so that they could be understood.
- Asking Claude to produce one summary file for each of the specified key papers, as quick-reference notes, with the source pages and sections cited for verification.
- Checking the content of the report I had written had any gaps or contradictions when checked against the project as a whole. After writing, checking the text, correcting grammar and wording, and adjusting the LaTeX layout.

### Cursor

- At the start of the project, Cursor read PROJECT_PLAN.md, followed the planned steps and structure, and after confirmation created TODO.md, breaking the workflow into individual to-do items before the project started.
- Before any fix or refactor, Cursor was first asked to review the existing content thoroughly and list the suggested changes as bullet points, without editing anything directly. Changes were only made after I confirmed them.
- Refactoring and project-wide renaming. For example, the original main notebook was split into per-game data processing and cross-game evaluation and diagnostics, and the data was turned into a cache so the project could run on remote GitLab.
- Explaining how an existing function works and why it is written this way rather than another (why and why not). Also discussing why a function or variable has its current name and what better names could be used.
- At the end of each project stage, reviewing the overall project structure and providing suggestions for improvement.
- Consulting Cursor when I hesitated over project execution, coding or refactoring. For example, whether `vae_bot.py` should be merged into `bots.py`: Cursor pointed out that merging would force torch to load every time, so they were kept separate. `data_csgo.py` was originally separate from `data.py`: Cursor suggested keeping them separate, but I judged that keeping everything together was tidier, and the suggestion was not adopted.
- When writing the report, asking Cursor to list the implementation steps of a specific part of the project, as the basis for turning code into written description, so that no detail was missed.

### Using Both AIs for Cross-Comparison and Mutual Checking

One AI's answer was given to the other for evaluation, asking it to raise questions and different opinions and to look for gaps. I passed the messages between them, acted as the intermediary, and decided the final approach. The result may partly follow Cursor, partly follow Claude, or neither. I chose according to the needs of the project, or designed a different approach myself.

Examples:
- Deciding the content of the feature sets. The candidate feature table compiled from the literature was given to both AIs separately, asking which features should be included or excluded and why. Their answers were then cross-examined. Each was asked why the other had included a feature it had not. Meanwhile I compared the practices in the related literature as verification and to weigh the two answers. The final feature table was decided by me. The screening and selection of synthetic bots followed a similar process.
- For specific functions, Claude was asked to write the code and Cursor was asked to write the same function in parallel. I cross-compared the two versions for differences in approach and content, where they differed and why. The AIs were asked to explain the differences, and I decided the final version.
- Both AIs were regularly asked to step back from the details and review the whole project, checking for gaps, logical errors or wrong directions, from the logic of the research itself, feature extraction, dataset field alignment and synthetic data methods to the use of the classifier, and whether everything matches the real-world workflow of game anti-cheat and deploying an old model to a new game.

### Questions and Challenges to the AI

- Representativeness of synthetic bots.
Synthetic trajectories are not produced by actually playing the game, so how can a model trained on them be expected to handle real cheats in other games? What real-world situation do these synthetic bots correspond to, and what is their practical value?
Real cheat trajectories cannot be obtained, and the literature generally uses synthetic data. The research is therefore positioned accordingly. This study measures detection and transfer against synthetic attack proxies, and this scope is stated clearly in the report.

- The baseline for comparison when in-domain performance is poor.
Stitch performs poorly even in-domain, so how can a model that cannot measure its own game accurately be tested on other games? Should a more accurate method be found first, or an aligned baseline be established?
This question led to later revisions of the evaluation protocol, including using AUC as the main metric and calibrating the threshold from the human scores of the target game.

- Reviewing the necessity and positioning of the research.
After a new game launches, cheats also take time to develop, so is zero-day detection really needed? Has this topic already been studied?
Through discussion and literature checks it was confirmed that cheat tools are largely portable across games, so the early period after launch is a defensive gap. Detection within a single game is well studied, but cross-game zero-shot transfer has no systematic research, only one early side experiment.

- Rejecting an AI proposal.
For example, Claude originally proposed z-score normalisation of the features, to pull the value distributions of the two games onto the same scale. I rejected it for three reasons. Random Forest only looks at the ordering of values, so z-scoring within a single game is a no-op. Across games, computing z-scores requires the mean and standard deviation of the target game, and if bot data are included, it amounts to peeking at the target game's bots. Finally, z-scores can only align shift and scale, not differences in distribution shape.