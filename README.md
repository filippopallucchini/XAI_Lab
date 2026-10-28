# Explainable AI in practice: LIME & SHAP

Four hands-on notebooks for the course **Big Data Analytics** (University of Milano-Bicocca, Degree Programme in Economics, Data Analysis and Management). Language: English. Platform: Google Colab (CPU only, no setup).

| # | Notebook | Data | Method | What students do | Run time on Colab |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | [`01_XAI_LIME_Tabular.ipynb`](01_XAI_LIME_Tabular.ipynb) | 🏦 mortgage applications | LIME | train the mortgage model, rebuild LIME **by hand**, use the library, audit the model for racial bias, fix it | ≈ 1 minute |
| 2 | [`02_XAI_LIME_NLP.ipynb`](02_XAI_LIME_NLP.ipynb) | 🏥 medical documents | LIME | train the text model, rebuild LIME **by hand**, use the library, find a hidden shortcut, fix it | ≈ 1 minute |
| 3 | [`03_XAI_SHAP_Tabular.ipynb`](03_XAI_SHAP_Tabular.ipynb) | 🏦 mortgage applications | SHAP | **repeat the training of Notebook 1**, compute Shapley values **by hand**, use the library (local, global, fairness views), fix the model | ≈ 2 minutes |
| 4 | [`04_XAI_SHAP_NLP.ipynb`](04_XAI_SHAP_NLP.ipynb) | 🏥 medical documents | SHAP | **repeat the training of Notebook 2**, compute Shapley values of words **by hand**, use the library, build a global word ranking, fix the model | ≈ 2 minutes |

**The core idea:** each model is trained in its LIME notebook and **trained again, with identical code**, in its SHAP notebook: Notebooks **1 and 3** explain the same mortgage model, Notebooks **2 and 4** the same text model. Students compare **explainers**, not trainings. Every notebook is self-contained and can also be used on its own.

**Suggested order:** 1 → 2 → 3 → 4 (one method at a time), or 1 → 3 → 2 → 4 (one data type at a time).

---

## 1. The two case studies

Both case studies come from **sensitive domains**, where an unexplained decision is not acceptable.

### 🏦 Mortgage approvals (tabular data) - Notebooks 1 and 3

| | |
| :--- | :--- |
| **Question** | Why did the model deny this mortgage? |
| **Data** | Boston HMDA: 2,380 real mortgage applications collected by the Federal Reserve Bank of Boston (Munnell et al., 1996; also used in Stock & Watson, *Introduction to Econometrics*) |
| **Download** | automatic, CSV from the Rdatasets archive: `https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/AER/HMDA.csv` |
| **Features** | 9: `pirat` (debt payments / income), `hirat` (housing expenses / income), `lvrat` (loan-to-value), `chist` (consumer credit history, 1 = best … 6 = worst), `mhist` (mortgage credit history, 1 … 4), `phist` (public bad credit record), `selfemp`, `single`, `afam` (African-American applicant: **protected attribute**) |
| **Target** | `deny` (12% of applications were denied) |
| **Black box** | `RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42)`, stratified 75/25 split |
| **Test performance** | ROC-AUC 0.806 |

**What the explanations reveal:** the model uses **race**. Applicant #112 is denied (P(deny) = 0.52); an identical applicant with `afam = 0` would be approved (0.46). LIME finds `afam=yes` pushing toward denial for practically all the 85 African-American applicants of the test set (rarely the main reason, but always there). SHAP measures it in probability points and splits the gap in predicted risk between the two groups into feature contributions: race explains about 3 of the 11 points of gap.

> The variable `insurance` (= a separate insurer refused mortgage insurance) is left out on purpose: it is almost a copy of the outcome and would dominate every explanation.

### 🏥 Routing medical documents (text data) - Notebooks 2 and 4

| | |
| :--- | :--- |
| **Question** | Should this document go to Oncology or to Cardiology? |
| **Data** | Medical Abstracts corpus (Schopf et al., 2023; licence CC BY-SA 3.0). We keep *neoplasms* (→ Oncology) and *cardiovascular diseases* (→ Cardiology): 4,971 training and 1,243 test documents, first 30 words of each abstract |
| **Download** | automatic, CSV from GitHub: `https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/` |
| **Black box** | TF-IDF (`min_df=2`, English stop words) + `LogisticRegression(max_iter=1000)` in one scikit-learn pipeline |
| **Test performance** | accuracy 94.9% |

**⚠️ A simulated confounder (important for the instructor).** Each document is prefixed with its "source": `[Source: Milan hospital]` or `[Source: Rome hospital]`. In training and test, 90% of Cardiology documents come from Milan and 90% of Oncology documents from Rome. The notebooks state clearly that this tag is **added by us** to simulate **site confounding**, a well-documented real problem (e.g. Zech et al., 2018: pneumonia detectors trained on X-rays from several hospitals learned to recognise the hospital and generalised worse to new ones).

**What the explanations reveal:** the model learned **geography, not medicine**. Document #532 (*"Pseudoaneurysm of the abdominal aorta…"*, clearly cardiology) is sent to Oncology because of the word *Rome*; a document containing only the word *Rome* gets P(oncology) = 1.00. LIME finds a city as the most important word in 33 of 50 documents; the SHAP global ranking puts *milan* and *rome* at the top. In a simulated "deployment" (both sites now treat both conditions, 50/50), accuracy drops from 94.9% to 86.1%.

---

## 2. How "the same identical model" is guaranteed

- The **training part** of each SHAP notebook (data description, data loading, model training) is **identical, cell by cell,** to the training part of the matching LIME notebook: same data, same split, same random seeds, same code. Nothing is loaded from the LIME notebook.
- The training cell prints a **fingerprint** of the model, which is the same in both notebooks of a pair:

| Model | Notebooks | Fingerprint |
| :--- | :---: | :--- |
| 🏦 Mortgage | 1 and 3 | AUC = **0.8062**, P(deny) of applicant #112 = **0.523** |
| 🏥 Text | 2 and 4 | test accuracy = **0.9485**, P(oncology) of document #532 = **0.554** |

- Why retrain instead of saving and loading the model? Every Colab notebook runs on its own machine: retraining takes about 2 seconds and needs no Google Drive permissions.
- All notebooks were executed end-to-end on two software stacks and produced **identical numbers**:
  - Colab-like: Python 3.12, NumPy 2.0.2, pandas 2.2.2, scikit-learn 1.6.1, matplotlib 3.10.0, shap 0.48.0, lime 0.2.0.1, IPython 7.34;
  - recent: Python 3.13, NumPy 2.5, pandas 3.0, scikit-learn 1.9, shap 0.54, IPython 9.

---

## 3. Structure of the notebooks

All four notebooks follow the same structure:

1. **Part 1 — Train the black box** (data, model, evaluation, fingerprint).
2. **Part 2 — How the method works**: intuition, formula, recipe, then the method **rebuilt by hand in a few lines of code** and compared with the library. The part ends with a table *"from the mechanism to the practice"*: because the method works in a certain way, it *can* do X, *but* it cannot do Y.
3. **Part 3 — Using the explanations** to audit or debug the model, with one hands-on exercise.
4. **Part 4 — The limits** of the method, shown with code where possible.
5. **Summary**, discussion questions (in capital letters, to start a class discussion), final exercises, references.

| Notebook | By hand | With the library | What it is used for |
| :--- | :--- | :--- | :--- |
| 1. LIME · tabular | applicant #112: quartile ranges, 5,000 copies drawn from real applicants, Gaussian kernel, weighted Ridge regression; a chart of "what LIME sees" (the LIME weight ≈ the gap in P(deny) between copies that keep and copies that leave the applicant's range) | `LimeTabularExplainer` (HTML view + bar chart) | LIME on all 85 African-American applicants; "identical twin" test; stability; unrealistic copies (`hirat > pirat`) |
| 2. LIME · text | document #532: 2,000 copies with random words removed, kernel, weighted Ridge; scatter plot of the copies | `LimeTextExplainer` (HTML view + bar chart) | most important word of 50 documents; deployment test; stability |
| 3. SHAP · tabular | applicant #112: value of a team $v(S)$ with a background of 100 applicants, two orders → two stories, average over 1,000 orders = Shapley values | exact explainer: waterfall, beeswarm, mean \|SHAP\| bar, dependence plots | LIME vs SHAP on the same applicant; fairness audit and decomposition of the gap between groups |
| 4. SHAP · text | document #532: words as players, the empty document as base value, two orders, 1,000 orders; efficiency and dummy (ignored words get exactly 0) | text explainer: interactive text plot | LIME vs SHAP on the same document; global word ranking over 100 documents; deployment test |

**By hand vs library**, as students will see:
- Notebooks 2 and 3: the hand-made version and the library give practically the same numbers.
- Notebook 1: same top reasons and signs, but different numbers, because the library draws the values inside each range from a bell curve while the hand-made version reuses real values. The notebook uses this as a lesson: **LIME depends on how the neighbourhood is built**.
- Notebook 4: close values, because the library uses a faster hierarchical approximation (teams of neighbouring words).

---

## 4. Hands-on exercises

Each notebook has one exercise in which students **edit one line** and re-run two cells; the notebook then shows a "before vs after" chart. With the line left empty the notebook still runs (and says that nothing changed), so *Run all* never breaks. Solutions are in collapsible cells (**✅ Solution**, click to open).

| Notebook | Exercise | Line to edit | Expected result |
| :--- | :--- | :--- | :--- |
| 1 | A fairer model? | `features_to_drop = ['afam']` | AUC 0.806 → 0.795; applicant #112 approved (0.523 → 0.398); average predicted risk: African-American 20.5% → 17.6%, others 9.9% → 10.2% |
| 2 | Fix the text model | `extra_stop_words = ['Milan', 'Rome']` | deployment accuracy 86.1% → 92.9%; test 94.9% → 92.9% (the original model was "cheating"); document #532 goes to Cardiology |
| 3 | Where does the gap go? | `features_to_drop = ['afam']` | the race bar disappears, total gap 0.109 → 0.074, the other features barely change |
| 4 | Fix the model using only the global view | `extra_stop_words = ['Milan', 'Rome']` | ranking led by *cardiac*, *cancer*, *hypertension*; deployment accuracy 86.1% → 92.9% |

Each notebook also ends with 4 open **final exercises** (e.g. what-if on a feature, kernel width, another neighbourhood or background, your own document, fewer orders, `TreeExplainer`).

---

## 5. How to run

**Google Colab (recommended)**
1. Open <https://colab.research.google.com> → *File* → *Upload notebook* (or the *GitHub* tab, if the notebooks are in a repository; any notebook on GitHub opens at `https://colab.research.google.com/github/<user>/<repository>/blob/main/<notebook>.ipynb`).
2. *Runtime* → *Run all*. The default CPU runtime is enough.
3. The first cell installs what is missing (`lime` in Notebooks 1-2, `shap` and `lime` in Notebooks 3-4); the data are downloaded automatically.

**Locally**
```bash
pip install numpy pandas scikit-learn matplotlib lime shap jupyter
jupyter notebook 01_XAI_LIME_Tabular.ipynb
```

---

## 6. Notes for the instructor

- **Choices made for clarity:** two classes and the first 30 words for the text data, 9 features for the mortgage data, one showcase instance per model (applicant #112, document #532) used in both notebooks of the pair.
- **Sensitive content:** race is used as a feature **on purpose**, to show how explanations reveal discrimination learned from historical decisions. The notebooks frame it as an audit and leave the open questions (legitimate risk factors or proxies?) to the class discussion.
- **Interactive outputs:** the LIME HTML views and the SHAP text plot need a live notebook (Colab, Jupyter). A static preview (e.g. on GitHub) shows them empty; all the other charts are static images.
- **Timing:** the slowest cells are the SHAP explanations of 300 applicants (≈ 20-40 s, set by `N_EXPLAIN`) and of 100 documents (≈ 15-30 s).
- **Charts:** our own charts use a colorblind-safe palette; red/blue always mean "toward / away from" the explained class, as in the SHAP plots.

## 7. Troubleshooting

| Problem | Solution |
| :--- | :--- |
| A data download fails | Your network blocks `raw.githubusercontent.com`: download `HMDA.csv`, `medical_tc_train.csv` and `medical_tc_test.csv` from the URLs above, upload them to Colab (*Files* panel) and set `URL_HMDA = "HMDA.csv"` and `URL_MED = ""` |
| The LIME view is empty in Jupyter | Re-run the cell; in classic Jupyter use *File → Trust notebook* (JavaScript is needed) |
| `ModuleNotFoundError: lime` / `shap` | Re-run the first cell (installation) |
| Many progress-bar lines below the SHAP cells | Normal: they are `shap`'s progress bars (a single bar on Colab) |

## 8. References

- Ribeiro, Singh, Guestrin (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD.
- Lundberg, Lee (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Shapley (1953). *A Value for n-Person Games*. Contributions to the Theory of Games II.
- Lundberg et al. (2020). *From local explanations to global understanding with explainable AI for trees*. Nature Machine Intelligence.
- Molnar. *Interpretable Machine Learning* (online book).
- Munnell, Tootell, Browne, McEneaney (1996). *Mortgage Lending in Boston: Interpreting HMDA Data*. American Economic Review 86(1).
- Schopf, Braun, Matthes (2023). *Evaluating Unsupervised Text Classification: Zero-Shot and Similarity-Based Approaches*. NLPIR 2022 (Medical Abstracts dataset).
- Zech et al. (2018). *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study*. PLOS Medicine 15(11).
