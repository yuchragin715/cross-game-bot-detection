# Project Meeting Notes

## 18 February 2026

### Discussion

Presented the initial proposals — two directions, four ideas in total:

* Direction A: Social Media & Disinformation
    1. Visualizing Coordinated Inauthentic Behavior (CIB): Using graph theory to visualize suspicious network structures (e.g., automated spread by bots) on social media platforms such as X (Twitter), Threads, etc.
    2. The Disinformation Network Map: Developing a tool to map out disinformation networks by tracking connections between content farms. For example, sites sharing the same Google AdSense IDs or domain registrants.

* Direction B: Browser Fingerprinting & Privacy
    1. Browser Fingerprinting & Privacy Dashboard: A web interface that shows users exactly what data (Hardware, Canvas, Fonts) might reveal their identity, and analyzes what kind of features (such as font list vs. screen resolution) are most unique and more likely to lead to identity leaks.
    2. Behavioral Biometrics Analysis: Analyzing mouse and keyboard dynamics to distinguish between human users and automated bots, including generating "human-like" mouse paths to test whether they can bypass detection.

---

## 13 March 2026

### Discussion

Completed the literature review and identified seven research gaps (supporting papers listed under each):

- **Lack of Multi-Modal Integration & Cross-Device Validation** — current studies often focus on a single modality (e.g., only mouse or only keystroke) and lack research on trackpads, touch dynamics, or device orientation.
  - Mouse Dynamics Behavioral Biometrics: A Survey
  - Deep Learning-Driven User Legitimacy Prediction Using Keystroke and Mouse Behavioural Dynamics
  - Hiding in the Crowd: an Analysis of the Effectiveness of Browser Fingerprinting at Large Scale
  - Multi-Modal Adversarial Activity Detection Using Keyboard and Mouse Dynamics
  - Mouse2Vec: Learning Reusable Semantic Representations of Mouse Behaviour

- **Hardware Discrepancies & Environmental Variables** — external variables such as mouse sensitivity, screen refresh rate, network latency, and different hardware significantly affect the analysis results and data quality.
  - Mouse Dynamics Behavioral Biometrics: A Survey
  - Multifractal Mice: Operationalising Dimensions of Readiness-to-hand via a Feature of Hand Movement
  - Characterizing and Quantifying Expert Input Behavior in League of Legends

- **Impact of Time, Emotions, and Task Scenes on User Habits** — users' typing and clicking habits change significantly over time; psychological and emotional states also affect their trajectories, and existing systems often ignore how different task scenes (e.g., working vs. gaming) impact behavior.
  - Advancing Multi-Modal Behavioral Biometric Authentication: A Deep Learning Approach With Synthetic Data Generation
  - Keystroke Dynamics: Concepts, Techniques, and Applications
  - User Authentication Method Based on Keystroke Dynamics and Mouse Dynamics Using Scene-Irrelated and User-Related Features
  - Multi-Modal Adversarial Activity Detection Using Keyboard and Mouse Dynamics
  - Mouse2Vec: Learning Reusable Semantic Representations of Mouse Behaviour
  - Mouse Dynamics Behavioral Biometrics: A Survey

- **Demographic, Cultural, and Geographic Biases** — most datasets suffer from severe demographic and geographical biases, lacking diversity in culture and language.
  - Advancing Multi-Modal Behavioral Biometric Authentication
  - AgeGen Bio Track: Continuous Mouse Behavioral Biometrics-Based Age and Gender Profiling
  - Deep Learning-Driven User Legitimacy Prediction Using Keystroke and Mouse Behavioural Dynamics
  - Hiding in the Crowd: an Analysis of the Effectiveness of Browser Fingerprinting at Large Scale
  - How Unique is Whose Web Browser? The Role of Demographics in Browser Fingerprinting among US Users

- **High Computational Cost & Real-Time Deployment Challenges** — advanced AI models face a severe trade-off between accuracy and response time, making real-time deployment difficult and resource-intensive.
  - Advancing Multi-Modal Behavioral Biometric Authentication: A Deep Learning Approach With Synthetic Data Generation
  - Deep Learning-Driven User Legitimacy Prediction Using Keystroke and Mouse Behavioural Dynamics
  - Multi-Modal Adversarial Activity Detection Using Keyboard and Mouse Dynamics

- **Vulnerability to Synthetic Attacks & Lack of Practical Defenses** — some studies identify privacy risks but fail to explore practical solutions; since mouse trajectories can be easily synthesized, there is a lack of testing against advanced spoofing attacks.
  - Mouse Dynamics Behavioral Biometrics: A Survey
  - Advancing Multi-Modal Behavioral Biometric Authentication
  - Deep Learning-Driven User Legitimacy Prediction Using Keystroke and Mouse Behavioural Dynamics
  - Keystroke Dynamics: Concepts, Techniques, and Applications
  - Multi-Modal Adversarial Activity Detection Using Keyboard and Mouse Dynamics
  - How Unique is Whose Web Browser? The Role of Demographics in Browser Fingerprinting among US Users

- **Discrepancy Between Lab Environments and Real-World Scenarios** — data collected in laboratory settings is often too artificial and does not fully reflect real-world situations.
  - Evaluation in Human-Computer Interaction – Beyond Lab Studies
  - Investigating the Tradeoffs of Everyday Text-Entry Collection Methods

Based on these gaps, proposed three candidate project plans:

#### Plan A: Cross-Dataset Context Analysis
- **Concept:** Load two drastically different public datasets (e.g., Balabit Office dataset vs. Minecraft Gaming dataset) into the dashboard.
- **Feature:** Allow users (security researchers) to toggle between scenes to visually compare how task contexts (working vs. gaming) fundamentally alter biometric trajectories and features.

#### Plan B: Behavioral Noise Modifiers
- **Concept:** Use a clean baseline dataset and implement mathematical parameter sliders in the frontend to simulate human or hardware variations.
- **Feature:** Include sliders for "Anxiety/Jerkiness" (adding Gaussian noise to coordinates) or "Hardware Degradation" (reducing sampling rates). This visualizes how emotional or hardware changes can easily break existing authentication models.

#### Plan C: Multi-Modal Time-Sync Visualizer
- **Concept:** Utilize the open-source `BB-MAS` dataset, which contains simultaneous keystroke and mouse records.
- **Feature:** Create a synchronized dual-timeline UI. Researchers can select a specific timeframe to instantly analyze the correlation between "Keystroke Flight Time" and "Mouse Acceleration," addressing the lack of multi-modal visualization tools in current research.

Also raised one open question with the supervisor: most public datasets lack detailed labels for attributes such as emotion, region, or gender. Asked whether this data would need to be collected first-hand, or whether there is a better way to work around the limitation.

---

## 25 March 2026

### Discussion

Following up on the three plans from 13 March, compiled a list of candidate public datasets:

| Dataset Name | Devices | Users | Environment / Scene | Activities and Data | Demographics | Link |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SU-AIS BB-MAS** | Desktop, 2 Phones (hand and pocket), Tablet | 117 | Lab | Typing, Gait and Swipes from the SAME USERS on all devices<br> Desktop: Typing, Browsing; Keystroke and Mouse<br>Tablet Typing: Keystroke, Swipe; Accelerometer, Gyroscope | User ID, Age, Gender, Height, Ethnicity, Languages Spoken, Typing Languages, Handedness, Desktop Hours, Smartphone Hours, Tablet Hours, Typing Style, Major/Minor | [SU-AIS BB-MAS](https://ieee-dataport.org/open-access/su-ais-bb-mas-syracuse-university-and-assured-information-security-behavioral) |
| **Balabit Mouse Challenge** | Mouse | 10 | Remote Desktop / Administrative tasks | record timestamp, client timestamp, mouse button, mouse state (Move, Drag, Pressed, Released), X, Y | | [Balabit Mouse Challenge](https://github.com/balabit/Mouse-Dynamics-Challenge) |
| **Minecraft Mouse Dynamics** | Mouse | 10 (Original) / 40 (Extended) | Game | Timestamp, X, Y, Button Press, Subject ID | | [Minecraft Mouse Dynamics](https://github.com/NyleSiddiqui/Minecraft-Mouse-Dynamics-Dataset) |
| **CMU Keystroke Dynamics** | Keystroke | 51 | Lab (typing password) | Typing a fixed password (.tie5Roanl) 400 times across 8 sessions; Subject ID, Session Index, Repetition, Hold time (H), Keydown-Keydown time (DD), Keyup-Keydown time (UD)| | [CMU Keystroke Dynamics](https://www.cs.cmu.edu/~keystroke/) |
| **AmIUnique** | Browser Fingerprint | > 2 Million records (fingerprints) | | Website visiting. Collects browser properties: HTTP headers (User-Agent, Accept), JS attributes (Canvas/WebGL hashes, Fonts, Screen resolution, Plugins, Timezone) | | [AmIUnique](https://amiunique.org/) |
| TWOS (The Wolf of SUTD) Dataset | keystrokes, mouse | 24 | Office | keystrokes, mouse, host monitor, network traffic, SMTP logs, and logon | | [TWOS](https://github.com/ivan-homoliak-sutd/twos) |
| Bogazici Mouse Dynamics Dataset | Mouse | 24 | Daily computer use / Office | Action Type, Timestamp, X, Y, Button, State, Window Name | | [Bogazici Mouse Dynamics Dataset](https://data.mendeley.com/datasets/w6cxr8yc7p/2) |
| AMuCS | keyboard, mouse | 245 | Game | Mouse/keyboard button presses, Game data, Physiological data, Eyetracker data | | [AMuCS](https://yareta.unige.ch/archives/8563a0e8-0242-4c67-98fe-8b929b5a403a)<br>[Intro](https://www.nature.com/articles/s41597-025-05596-3) |
| KeyRecs: Keystroke Dynamics Dataset | keyboard | 100 | fixed text, free text | | age, gender, handedness, and nationality | [KeyRecs](https://zenodo.org/records/7886743) |
| Observations on Typing from 136 Million Keystrokes | keyboard | ~168,000 | Web-based typing test (Non-gaming) | | Age, Gender, Handedness, Native language, English proficiency | [Aalto](https://userinterfaces.aalto.fi/136Mkeystrokes/) |
| UB (Buffalo) Dataset | keyboard | 148 | Office | Transcribing text, navigating websites, and answering questions. Records mouse coordinates, clicks, and keystroke timings. | | [Buffalo](https://www.buffalo.edu/cubs/research/datasets.html)<br>Available upon request |
| Clarkson II Dataset | keyboard, mouse | 103 | Daily computer use (over 2.5 years) | keystroke timestamps, mouse events, active programs | | [Clarkson](https://citer.clarkson.edu/clarkson-university-keystroke-dataset-ii/) |

[Other open video game datasets with physiological or affective modalities.](https://www.nature.com/articles/s41597-025-05596-3/tables/1)
![Overview of open video game datasets](https://hackmd.io/_uploads/Bk8-8G09bg.png)

---

## 8 May 2026

### Discussion

Proposed two problem-driven ideas — one continuing the earlier behavioural biometrics theme, the other extending the supervisor's paper:

#### Idea 1: Cross-Context False Rejection in Continuous Authentication
- **Research question:** Can a continuous authentication model trained on an office-scene population generalise to a gaming-scene population?
- **Real-world problem:** Continuous authentication (Zero Trust security) keeps checking identity from mouse behaviour, but a model trained on one scene may fail when deployed elsewhere, leading to false rejections.
- **Approach:** Compare keyboard or mouse dynamics between two different environments using open datasets covering different task scenes, showing that security models must account for the scene to avoid high false rejection rates.

#### Idea 2: Coordinated Amplification of High-Risk Financial Content
- **Research question:** Do UK finfluencers promoting high-risk topics (crypto / foreign exchange) show coordinated amplification in the follower network, and which influencers are the key nodes?
- **Real-world problem:** Social media users are easily exposed to high-risk financial advice (e.g., crypto, foreign exchange), and it is hard for regulators to track how these trends spread and who the key promoters are.
- **Approach:** Build an interactive network map to test whether these influencers form closed circles, and identify the key connectors spreading the risky information — helping regulators focus on the main targets instead of monitoring everyone.
- **Analysis angles:** Network density comparison (high-risk vs. low-risk sub-networks); coordination signals such as posting-time similarity, hashtag overlap, and mutual follows.

---

## 26 June 2026

### Discussion

Proposed two concrete comparison plans:

#### Cross-Game Generalization in Mouse Dynamics (Using BEACON & AMuCS)
- **Datasets:** BEACON (Valorant) and AMuCS (CS:GO).
- **Concept:** Both are competitive FPS (first-person shooter) games requiring high precision, but with slightly different game mechanics.
- **Research question:** Can a continuous authentication model trained on CS:GO mouse dynamics successfully identify players (or detect bots) in Valorant?
- **Value:** Tests whether behavioural security models are "game-dependent"; starting with two similar games makes it possible to observe whether the data distribution shifts.

#### Cultural Bias in Keystroke Dynamics (Using Aalto Database)
- **Dataset:** Aalto University Keystroke Database (136 million keystrokes, 160k+ users globally).
- **Concept:** All participants type English text, but their native languages differ.
- **Research question:** Do native English speakers and non-native speakers (e.g., from Asia or Europe) show different keystroke biometrics, such as hold time or flight time?
- **Value:** If yes, current security models may carry cultural bias; if no, it suggests keystroke models are robust across cultures.

---

## 2 July 2026

### Discussion

Shortlisted candidate game datasets for the cross-game behavioural biometrics comparison.

Main candidates:

| Dataset | Game / Genre | Source |
| :--- | :--- | :--- |
| AMuCS | CS:GO / FPS | AMuCS: Affective multimodal Counter-Strike video game dataset (Nature Scientific Data) |
| Red Eclipse | Red Eclipse / FPS | Rapid Skill Capture in a First-Person Shooter (IEEE T-CIAIG) |
| Smerdov eSports Sensors | League of Legends / MOBA | Collection and Validation of Psychophysiological Data from Professional and Amateur Players: a Multimodal eSports Dataset (IEEE Transactions on Games) |

Backup options (from weaker sources, only if needed):

| Dataset | Game / Genre | Source |
| :--- | :--- | :--- |
| Minecraft Mouse | Minecraft / sandbox | Continuous User Authentication Using Mouse Dynamics, Machine Learning, and Minecraft (ICECET) |
| BEACON | Valorant / FPS | BEACON: A Multimodal Dataset for Learning Behavioral Fingerprints from Gameplay Data (arXiv (Thapar)) |

AMuCS, Red Eclipse, and BEACON are all first-person shooters, so their controls are the closest; Smerdov (LoL) is a MOBA and one step removed, while Minecraft is the most different.

---

## 16 July 2026

### Discussion

The project is roughly at the halfway point, with the foundational pipeline in place. Progress reported:

- **Foundational pipeline on the first game — Red Eclipse (FPS):**
  - Implemented feature extraction and a Random Forest–based human-vs-bot detection model.
  - Features follow those listed in the dataset's original paper, keeping the ones suitable for this setup.
  - Built two synthetic bot generators: block-bootstrap resampling of real human segments, and a mechanical/scripted-style generator. Both work well, though the generation methods are not yet grounded in existing literature — flagged as the next area to strengthen.
- **Initial cross-game comparison with the second game — LoL (MOBA):**
  - Early finding: the model trained on Red Eclipse does not transfer to LoL at all — bot detection drops to 0%. This is an early-stage test without deeper optimisation.
  - The two games have very different mouse behaviour, so the current features do not appear to transfer across genres.
  - Planned next steps: explore more general features to improve cross-game detection, improve the synthetic data generation to better reflect realistic bot-cheating behaviour, then re-test cross-game detection from several angles.
- **AMuCS (CS:GO) dataset:** using it for a cross-dataset comparison between two FPS games requires a signed Data Use Agreement; the DUA document was passed to the supervisor. Alternatively, if the Red Eclipse (FPS) vs. LoL (MOBA) framework provides sufficient depth, the project scope can be adjusted.

---

## 24 July 2026

### Discussion

Meeting agenda:

- Full pipeline walkthrough
- Follow-up on the LoL 0% result from 16 July
- Review of all current results
- New synthetic Bézier bot
- Fourth bot tier: VAE or GAN (pick one)
- Next steps

#### Follow-up on the LoL "0%" result

- The 0% figure came from using a fixed 0.5 threshold; many scores sat below 0.5, making the transfer look like a total failure — especially for the stitch bot.
- The same models still work in-domain on Red Eclipse (~88–100%).
- Fix: report AUC, and set the threshold from the target game's human scores only (95th percentile ≈ 5% false alarms — no bot labels needed, so still zero-day compatible).
- Result: the smooth bot is detected at 100% in all three transfer directions.
- The stitch bot is genuinely different — its AUC really is low, so that one is a real transfer failure.

---

## 7 August 2026

### Discussion

Progress reported:

- Added the fourth synthetic bot tier (VAE-based); all four tiers (scripted, Bézier, resampled, VAE) now run across all three games.
- Enhanced the evaluation pipeline: bots are now generated within each fold using a subject-disjoint (GroupKFold) split to remove leakage, and a window-level segmentation analysis was added to compare with session-level results.
- Surveyed the mouse-dynamics feature literature and finalised a revised feature set.

Report and demonstration status:

- The demonstration slides are essentially complete; content and details are being refined. Current draft: https://canva.link/m6m2uauxli2byk5
- Report writing has started with the Methodology chapter: the datasets, feature-extraction, and bot-generation sections are drafted, and the evaluation chapter is in progress. The Background and Related Work chapters are being expanded in parallel, as they share much of the same material.

#### Methodology improvements

- **Leakage-free split.** Players are split into train/test first (GroupKFold); synthetic bots are generated within each side only.
- **Windowed evaluation.** Non-overlapping 10-s windows, aggregated per session by mean score. Metrics: AUC + human-calibrated threshold at FP ≈ 5%.
- **Window alignment.** All three games now use ~3 minutes of natural mid-game play.
  - RE: full match, 3 min.
  - CS:GO: from Round 2, alive frames only.
  - LoL: match clock 10:00–13:00 via match metadata.

---

## 19 August 2026

### Discussion

Wrap-up meeting: presented the viva slides, walked through the presentation content, and demonstrated the project code.
