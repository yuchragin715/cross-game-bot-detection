# Project Meeting Notes

## 18 February 2026

### Discussion

Presented the initial proposals — two directions, four ideas in total:

* Direction A: Social Media & Disinformation
    1. Visualizing Coordinated Inauthentic Behavior (CIB): Using graph theory to visualize suspicious network structures (e.g., automated spread by bots) on social media platforms such as X (Twitter), Threads, etc.
    2. The Disinformation Network Map: Developing a tool to map out disinformation networks by tracking connections between content farms. For example, sites sharing the same Google AdSense IDs or domain registrants.

* Direction B: Browser Fingerprinting & Privacy
    1. Browser Fingerprinting & Privacy Dashboard: A web interface that shows users exactly what data (Hardware, Canvas, Fonts) might reveal their identity, and analyzes what kind of features (such as font list vs. screen resolution) are most unique and more likely to lead to identity leaks.
    2. Behavioral Biometrics Analysis: Analyzing mouse and keyboard dynamics to distinguish between human users and automated bots. I also want to try generating "human-like" mouse paths to test if they bypass detection.

### Supervisor's feedback

- Do background research in the literature from the last 3–4 years of top-tier Security and Privacy conferences, and identify gaps in current practice.
- B1 (the fingerprinting dashboard) is also interesting, and would become more so if potential users of the web app/plugin were engaged after development to gather opinions and impressions.
- Suggested venues for the literature review: SOUPS, ACM CHI, TheWebConference, etc.

---

## 13 March 2026

### Discussion

Completed the literature review and identified seven research gaps.
Each gap keeps its supporting papers (nothing dropped — only the HackMD `<details>` markup was removed for plain Markdown / GitLab compatibility).

1. Lack of Multi-Modal Integration & Cross-Device Validation  
   Current studies often focus on a single modality (e.g., only mouse or only keystroke) and lack research on trackpads, touch dynamics, or device orientation.  
   - Mouse Dynamics Behavioral Biometrics: A Survey  
   - Deep Learning-Driven User Legitimacy Prediction Using Keystroke and Mouse Behavioural Dynamics  
   - Hiding in the Crowd: an Analysis of the Effectiveness of Browser Fingerprinting at Large Scale  
   - Multi-Modal Adversarial Activity Detection Using Keyboard and Mouse Dynamics  
   - Mouse2Vec: Learning Reusable Semantic Representations of Mouse Behaviour  

2. Hardware Discrepancies & Environmental Variables  
   External variables such as mouse sensitivity, screen refresh rate, network latency, and different hardware significantly affect the analysis results and data quality.  
   - Mouse Dynamics Behavioral Biometrics: A Survey  
   - Multifractal Mice: Operationalising Dimensions of Readiness-to-hand via a Feature of Hand Movement  
   - Characterizing and Quantifying Expert Input Behavior in League of Legends  

3. Impact of Time, Emotions, and Task Scenes on User Habits  
   Users' typing and clicking habits change significantly over time. Psychological and emotional states also affect their trajectories. Existing systems often ignore how different task scenes (e.g., working vs. gaming) impact behavior.  
   - Advancing Multi-Modal Behavioral Biometric Authentication: A Deep Learning Approach With Synthetic Data Generation  
   - Keystroke Dynamics: Concepts, Techniques, and Applications  
   - User Authentication Method Based on Keystroke Dynamics and Mouse Dynamics Using Scene-Irrelated and User-Related Features  
   - Multi-Modal Adversarial Activity Detection Using Keyboard and Mouse Dynamics  
   - Mouse2Vec: Learning Reusable Semantic Representations of Mouse Behaviour  
   - Mouse Dynamics Behavioral Biometrics: A Survey  

4. Demographic, Cultural, and Geographic Biases  
   Most datasets suffer from severe demographic and geographical biases, lacking diversity in culture and language.  
   - Advancing Multi-Modal Behavioral Biometric Authentication  
   - AgeGen Bio Track: Continuous Mouse Behavioral Biometrics-Based Age and Gender Profiling  
   - Deep Learning-Driven User Legitimacy Prediction Using Keystroke and Mouse Behavioural Dynamics  
   - Hiding in the Crowd: an Analysis of the Effectiveness of Browser Fingerprinting at Large Scale  
   - How Unique is Whose Web Browser? The Role of Demographics in Browser Fingerprinting among US Users  

5. High Computational Cost & Real-Time Deployment Challenges  
   Advanced AI models face a severe trade-off between accuracy and response time, making real-time deployment difficult and resource-intensive.  
   - Advancing Multi-Modal Behavioral Biometric Authentication: A Deep Learning Approach With Synthetic Data Generation  
   - Deep Learning-Driven User Legitimacy Prediction Using Keystroke and Mouse Behavioural Dynamics  
   - Multi-Modal Adversarial Activity Detection Using Keyboard and Mouse Dynamics  

6. Vulnerability to Synthetic Attacks & Lack of Practical Defenses  
   Some studies identify privacy risks but fail to explore practical solutions. Since mouse trajectories can be easily synthesized, there is a lack of testing against advanced spoofing attacks.  
   - Mouse Dynamics Behavioral Biometrics: A Survey  
   - Advancing Multi-Modal Behavioral Biometric Authentication  
   - Deep Learning-Driven User Legitimacy Prediction Using Keystroke and Mouse Behavioural Dynamics  
   - Keystroke Dynamics: Concepts, Techniques, and Applications  
   - Multi-Modal Adversarial Activity Detection Using Keyboard and Mouse Dynamics  
   - How Unique is Whose Web Browser? The Role of Demographics in Browser Fingerprinting among US Users  

7. Discrepancy Between Lab Environments and Real-World Scenarios  
   Data collected in laboratory settings is often too artificial and does not fully reflect real-world situations.  
   - Evaluation in Human-Computer Interaction – Beyond Lab Studies  
   - Investigating the Tradeoffs of Everyday Text-Entry Collection Methods  

Based on these gaps, proposed three candidate project plans:

#### Plan A: Cross-Dataset Context Analysis
* **Concept:** Load two drastically different public datasets (e.g., Balabit Office dataset vs. Minecraft Gaming dataset) into the dashboard.
* **Feature:** Allow users (security researchers) to toggle between scenes to visually compare how task contexts (working vs. gaming) fundamentally alter biometric trajectories and features.

#### Plan B: Behavioral Noise Modifiers
* **Concept:** Use a clean baseline dataset and implement mathematical parameter sliders in the frontend to simulate human or hardware variations.
* **Feature:** Include sliders for "Anxiety/Jerkiness" (adding Gaussian noise to coordinates) or "Hardware Degradation" (reducing sampling rates). This visualizes how emotional or hardware changes can easily break existing authentication models.

#### Plan C: Multi-Modal Time-Sync Visualizer
* **Concept:** Utilize the open-source `BB-MAS` dataset, which contains simultaneous keystroke and mouse records.
* **Feature:** Create a synchronized dual-timeline UI. Researchers can select a specific timeframe to instantly analyze the correlation between "Keystroke Flight Time" and "Mouse Acceleration," addressing the lack of multi-modal visualization tools in current research.

Also raised one open question with the supervisor: most public datasets lack detailed labels for attributes such as emotion, region, or gender. Asked whether this data would need to be collected first-hand, or whether there is a better way to work around the limitation.

### Supervisor's feedback

- The supervisor took the HackMD write-up away to read, with a follow-up discussion arranged a few days later.
- Reminder: for digital-footprint-style topics that require collecting specific data first-hand, ethics approval is required, which takes longer and adds procedural overhead — this must be factored into planning.
- Decision: use public datasets instead, avoiding the ethics-approval process that first-hand data collection would require.

---

## 25 March 2026

### Discussion

Following up on the three plans from 13 March, compiled a list of candidate public datasets:

| Dataset Name | Devices | Users | Environment / Scene | Activities and Data | Demographics | Link |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SU-AIS BB-MAS** | Desktop, 2 Phones (hand and pocket), Tablet  | 117 | Lab | Typing, Gait and Swipes from the SAME USERS on all devices<br> Desktop: Typing, Browsing; Keystroke and Mouse<br>Tablet Typing: Keystroke, Swipe; Accelerometer, Gyroscope | User ID, Age, Gender, Height, Ethnicity, Languages Spoken, Typing Languages, Handedness, Desktop Hours, Smartphone Hours, Tablet Hours, Typing Style, Major/Minor | [SU-AIS BB-MAS](https://ieee-dataport.org/open-access/su-ais-bb-mas-syracuse-university-and-assured-information-security-behavioral) |
| **Balabit Mouse Challenge** | Mouse | 10 | Remote Desktop  / Administrative tasks | record timestamp, client timestamp, mouse button, mouse state (Move, Drag, Pressed, Released), X, Y | | [Balabit Mouse Challenge](https://github.com/balabit/Mouse-Dynamics-Challenge) |
| **Minecraft Mouse Dynamics** | Mouse | 10(Original) / 40(Extended) | Game | Timestamp, X, Y, Button Press, Subject ID | | [Minecraft Mouse Dynamics](https://github.com/NyleSiddiqui/Minecraft-Mouse-Dynamics-Dataset) |
| **CMU Keystroke Dynamics** | Keystroke | 51 | Lab (typing password) | Typing a fixed password (.tie5Roanl) 400 times across 8 sessions; Subject ID, Session Index, Repetition, Hold time (H), Keydown-Keydown time (DD), Keyup-Keydown time (UD)| | [CMU Keystroke Dynamics](https://www.cs.cmu.edu/~keystroke/) |
| **AmIUnique** | Browser Fingerprint | > 2 Million records (fingerprints) | | Website visiting. Collects browser properties: HTTP headers (User-Agent, Accept), JS attributes (Canvas/WebGL hashes, Fonts, Screen resolution, Plugins, Timezone) | | [AmIUnique](https://amiunique.org/) |
| TWOS (The Wolf of SUTD) Dataset | keystrokes, mouse | 24 | Office | keystrokes, mouse, host monitor, network traffic, SMTP logs, and logon | | [TWOS](https://github.com/ivan-homoliak-sutd/twos) |
| Bogazici Mouse Dynamics Dataset | Mouse | 24 | Daily computer useOffice | Action Type, Timestamp, X, Y, Button, State, Window Name | | [Bogazici Mouse Dynamics Dataset](https://data.mendeley.com/datasets/w6cxr8yc7p/2) |
| AMuCS | keyboard, mouse | 245 | Game | Mouse/keyboard button presses, Game data, Physiological data, Eyetracker data | | [AMuCS](https://yareta.unige.ch/archives/8563a0e8-0242-4c67-98fe-8b929b5a403a)<br>[Intro](https://www.nature.com/articles/s41597-025-05596-3) |
| KeyRecs: Keystroke Dynamics Dataset | keyboard | 100 | fixed text, free text | | age, gender, handedness, and nationality | [KeyRecs](https://zenodo.org/records/7886743) |
| Observations on Typing from 136 Million Keystrokes | keyboard | ~168,000 | Web-based typing test (Non-gaming) | | Age, Gender, Handedness, Native language, English proficiency | [Aalto](https://userinterfaces.aalto.fi/136Mkeystrokes/) |
| UB (Buffalo) Dataset | keyboard | 148 | Office | Transcribing text, navigating websites, and answering questions. Records mouse coordinates, clicks, and keystroke timings. | | [Buffalo](https://www.buffalo.edu/cubs/research/datasets.html)<br>Available upon request |
| Clarkson II Dataset | keyboard, mouse | 103 | Daily computer use(over 2.5 years) | keystroke timestamps, mouse events, active programs | | [Clarkson](https://citer.clarkson.edu/clarkson-university-keystroke-dataset-ii/) |

[Other open video game datasets with physiological or affective modalities.](https://www.nature.com/articles/s41597-025-05596-3/tables/1)
![截圖 2026-03-23 01.23.04](https://hackmd.io/_uploads/Bk8-8G09bg.png)

### Supervisor's feedback

- The supervisor pointed out that the current proposals essentially amount to building a tool; the project needs a clearer answer to what practical (or academic) problem it solves.
- The supervisor shared one of their papers as a reference — either for inspiration, or as a basis for extending the project using the data collected in it: U.K. Finfluencers: Exploring Content, Reach, Responsibility and Public Stance.

---

## 8 May 2026

### Discussion

Proposed two problem-driven ideas — one continuing the earlier behavioural biometrics theme, the other extending the supervisor's paper:

#### Idea 1: Cross-Context False Rejection in Continuous Authentication

Can a continuous authentication model trained on an office-scene population generalise to a gaming-scene population.

Real-world problem:
Continuous Authentication (or Zero Trust security) keeps checking identity from mouse behaviour.
But a model trained on one scene may fail when deployed elsewhere. This can lead to false rejections.

- Compare keyboard or mouse dynamics between two different environments using open datasets covering different task scenes.
- Show security models must consider the scene to avoid high false rejection rates.

#### Idea 2: Coordinated Amplification of High-Risk Financial Content

Do UK finfluencers promoting high-risk topics (crypto / foreign exchange) show coordinated amplification in the follower network? 
Which influencer are the key node?

Problem:
Social media users are easily exposed to high-risk financial advice (e.g., Crypto, foreign exchange). It is hard for regulators to track how these high-risk trends spread and who the key promoters are.

- Build an interactive network map to see if these influencers form closed circles.
- Identify the key connectors who help spread this risky information.
- Helps regulators find the main target instead of monitoring everyone.
- Compare network density: high-risk vs. low-risk sub-networks
- Coordination signals: posting time similarity, hashtag overlap, mutual follows

### Supervisor's feedback

- Conclusion of the discussion: shift towards comparing different mouse/keyboard dynamics datasets, starting by surveying which datasets can be meaningfully compared.
- The supervisor felt the gaming-vs-office contrast is too obvious — one context is calm, the other fast-paced — so the comparison would offer limited insight.
- Examples of better comparison settings given by the supervisor: AI-assisted coding vs. manual coding, or writing emails on a phone vs. on a computer — i.e. contexts that are similar yet subtly different.
- The supervisor suggested comparing different game datasets; I noted the comparison need not be limited to games — different regions, demographics, or hardware categories could also be candidates.

---

## 26 June 2026

### Discussion

Proposed two concrete comparison plans:

#### Cross-Game Generalization in Mouse Dynamics (Using BEACON & AMuCS)
Datasets: BEACON (Valorant) and AMuCS (CS:GO).
Concept: Both are competitive FPS (First-Person Shooter) games requiring high precision, but with slightly different game mechanics.
Research Question: Can a continuous authentication model trained on CS:GO mouse dynamics successfully identify players (or detect bots) in Valorant?
Value: Tests if behavioral security models are "game-dependent." We can start with these two similar games to see if the data distribution shifts.

#### Cultural Bias in Keystroke Dynamics (Using Aalto Database)
Dataset: Aalto University Keystroke Database (136 million keystrokes, 160k+ users globally).
Concept: Everyone is typing English text, but their native languages are different.
Research Question: Do people from English-speaking countries and non-native speakers (e.g., Asia, Europe) have different keystroke biometrics (like hold time or flight time)?
Value: If yes, current security models might have cultural bias. If no, it proves keystroke models are universally robust.

### Supervisor's feedback

- The supervisor selected the cross-game comparison (Cross-Game Generalization) direction.
- Since the proposal cited the BEACON paper, the supervisor asked for a broader literature search, prioritising datasets published in reputable venues; less credible datasets should only be used as a fallback.
- The supervisor had reservations about the credibility of BEACON's publication source, and advised reviewing more papers before settling on the final direction.
- Action: continue collecting relevant papers and candidate datasets.

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

AMuCS, Red Eclipse, and BEACON are all First-Person Shooter games, so the controls are closest. Smerdov (LoL) is a MOBA, so it's a bit further away, and Minecraft is the most different.

### Supervisor's feedback

- The supervisor again pressed on what real problem the project is meant to solve.
- After revisiting the earlier discussions during the meeting, a new idea emerged: generate synthetic data for one game, train a model to distinguish human from synthetic input, then transfer the model to a second game to test whether it can still tell them apart. (Open question: would synthetic data then need to be generated for both games?)
- This connects back to the real-world problem framed on 8 May: a continuous authentication (Zero Trust) model trained in one context may fail when deployed in another, leading to false rejections.
- Direction agreed in the meeting: generate synthetic mouse trajectories on game A, train a detection model, then transfer it to game B to test whether it can still detect synthetic bots. The supervisor approved this direction.
- Also informed the supervisor that access to the AMuCS (CS:GO) dataset requires a signed Data Use Agreement (DUA).

---

## 16 July 2026

### Discussion

The project is roughly at the halfway point, with the foundational pipeline in place. (Self-note: the synthetic-data generation methods are not yet grounded in existing literature — flagged as a priority to address next.)

Progress summary:

1. Foundational pipeline on the first game: Red Eclipse (FPS game)
I have implemented feature extraction and a human-vs-bot detection model based on Random Forest.
The features are based on those listed in the dataset's original paper, selecting the ones suitable to my setup. For the synthetic bots, I built generators in two different ways, one using block-bootstrap resampling of real human segments, and one mechanical/scripted-style generator.
Both methods work well, though I haven't yet grounded the generation methods in existing literature — this is one area I plan to strengthen next.

2. Initial cross-game comparison with the second game: LoL (MOBA game)
The initial finding is that the model trained on Red Eclipse (FPS) does not transfer to LoL at all, bot detection drops to 0%. This is an early stage test without deeper optimization yet.
The two games have very different mouse behaviour, so the current features do not seem to transfer across genres.
Next I plan to explore more general features to improve detection ability across different games, and improve the synthetic data generation to better reflect realistic bot cheating behaviour, then re-test cross-game detection from several angles.

There is also one request regarding the AMuCS (CS:GO) dataset.
If we still intend to use it for a cross-dataset analysis between two FPS games, I will need the signed Data Use Agreement to get access. I have attached the document here.
Alternatively, if the current Red Eclipse (FPS) vs LoL (MOBA) framework provides sufficient depth, I can adjust the project scope.

### Supervisor's feedback

- The 0% cross-game detection result is not acceptable as-is; the cause needs to be investigated as the next step.
- The supervisor suggested exploring AI / deep-learning-based approaches (e.g., generative models or LLM-based methods) for producing the synthetic data. Follow-up action: began evaluating generative models such as GANs and VAEs.
- The AMuCS (CS:GO) DUA has been signed; the dataset files were received on 17 July.