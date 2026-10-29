# Explainable AI in practice: LIME & SHAP

Four hands-on notebooks for the course **Big Data Analytics** (University of Milano-Bicocca, Degree Programme in Economics, Data Analysis and Management). Language: English. Platform: Google Colab (the free CPU is enough).

| # | Notebook | Data | Method | What students do | Run time on Colab (CPU) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | [`01_XAI_LIME_Tabular.ipynb`](01_XAI_LIME_Tabular.ipynb) | 🏦 mortgage applications | LIME | train the mortgage model, rebuild LIME **by hand**, use the library, measure the **fidelity** of the explanation, audit the model for racial bias, fix it | ≈ 1 minute |
| 2 | [`02_XAI_LIME_NLP.ipynb`](02_XAI_LIME_NLP.ipynb) | 🛒 customer reviews (phones, restaurants) | LIME | train a sentiment model (sentence embeddings + logistic regression), rebuild LIME **by hand**, use the library, measure **fidelity**, find a hidden bug, **make the model more accurate** | ≈ 2 minutes |
| 3 | [`03_XAI_SHAP_Tabular.ipynb`](03_XAI_SHAP_Tabular.ipynb) | 🏦 mortgage applications | SHAP | **repeat the training of Notebook 1**, compute Shapley values **by hand**, use the library (local, global, fairness views), compare the **fidelity** of SHAP and LIME (deletion test), fix the model | ≈ 2 minutes |
| 4 | [`04_XAI_SHAP_NLP.ipynb`](04_XAI_SHAP_NLP.ipynb) | 🛒 customer reviews (phones, restaurants) | SHAP | **repeat the training of Notebook 2**, compute Shapley values of words **by hand**, use the library, compare the **fidelity** of SHAP and LIME, find the bug with a global view, **make the model more accurate** | ≈ 3 minutes |

**The core idea:** each model is trained in its LIME notebook and **trained again, with identical code**, in its SHAP notebook: Notebooks **1 and 3** explain the same mortgage model, Notebooks **2 and 4** the same sentiment model. Students compare **explainers**, not trainings. Every notebook is self-contained and can also be used on its own.

**Three uses of XAI, one per case study:**
- 🛒 **Debugging → a more accurate model** (Notebooks 2 and 4): the explanations reveal that the model cannot see *not*, *no*, *never*...; fixing the preprocessing raises the test accuracy from about 86% to about 96% (on the sentences with a negation from about 63% to about 96%).
- 🏦 **Auditing → a fairer model** (Notebooks 1 and 3): the explanations reveal that race pushes applications toward denial; removing it keeps the same ranking quality (AUC −0.01).
- ✅ **Checking the explanation itself (fidelity)** (all notebooks): an explanation is judged by **fidelity** (does it describe what the model does?), never by **accuracy** (is the model right?).

**Suggested order:** 1 → 2 → 3 → 4 (one method at a time), or 1 → 3 → 2 → 4 (one data type at a time).

---

## 1. The two case studies

### 🏦 Mortgage approvals (tabular data) - Notebooks 1 and 3

| | |
| :--- | :--- |
| **Question** | Why did the model deny this mortgage? |
| **Data** | Boston HMDA: 2,380 real mortgage applications collected by the Federal Reserve Bank of Boston (Munnell et al., 1996; also used in Stock & Watson, *Introduction to Econometrics*) |
| **Download** | automatic, CSV from the Rdatasets archive: `https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/AER/HMDA.csv` |
| **Features** | 9, renamed in the notebooks for readability (see the table below) |
| **Target** | `deny` (12% of applications were denied) |
| **Black box** | `RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42)`, stratified 75/25 split |
| **Test performance** | ROC-AUC 0.806 |

| Name in the notebooks | Meaning | Original name |
| :--- | :--- | :---: |
| `debt_to_income` | total debt payments / income | `pirat` |
| `housing_to_income` | housing expenses / income | `hirat` |
| `loan_to_value` | loan / value of the property | `lvrat` |
| `credit_problems` | consumer credit history: 1 = no late payments, 2 = one or two, 3 = more than two, 4 = not enough history, 5 = 60 days overdue, 6 = 90 days overdue (higher = worse) | `chist` |
| `mortgage_problems` | mortgage payment history: 1 = no late payments, 2 = no mortgage history, 3 = one or two late payments, 4 = more than two (higher = worse) | `mhist` |
| `public_bad_record` | public record of credit problems (e.g. bankruptcy) | `phist` |
| `self_employed` | self-employed | `selfemp` |
| `single` | single | `single` |
| `african_american` | African-American applicant: **protected attribute** | `afam` |

The levels of the two credit histories are those of the documentation of the Boston HMDA data (a low value is a good score). The renaming is done in the data cell (`hmda.rename(...)`), identical in Notebooks 1 and 3.

**What the explanations reveal:** the model uses **race**. Applicant #112 is denied (P(deny) = 0.52); an identical applicant with `african_american = 0` would be approved (0.46). LIME finds `african_american=yes` pushing toward denial for practically all the 85 African-American applicants of the test set (rarely the main reason, but always there); SHAP measures it in probability points for every applicant.

> The variable `insurance` (= a separate insurer refused mortgage insurance) is left out on purpose: it is almost a copy of the outcome and would dominate every explanation.

### 🛒 Sentiment of customer reviews (text data) - Notebooks 2 and 4

| | |
| :--- | :--- |
| **Question** | Is this review sentence positive or negative? (e.g. to send the complaints to customer service automatically) |
| **Data** | *Sentiment Labelled Sentences* (Kotzias et al., 2015; UCI Machine Learning Repository, CC BY 4.0): 1,000 sentences from **Amazon** reviews of mobile phones and accessories and 1,000 from **Yelp** restaurant reviews, each labelled positive or negative by hand (50/50). The 14 sentences that appear twice are kept once (1,986 sentences); stratified 75/25 split (1,489 train, 497 test) |
| **Download** | automatic, a zip file of ≈ 80 KB from the UCI repository: `https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip` (the IMDb movie reviews of the same zip are not used) |
| **Black box** | a scikit-learn pipeline: **1. clean** (lower case, drop punctuation, remove scikit-learn's 318 English stop words) → **2. embed** with the sentence-embedding model `BAAI/bge-small-en-v1.5` (33M parameters, 384 numbers per sentence, downloaded from Hugging Face, ≈ 130 MB) → **3. classify** with `LogisticRegression(max_iter=1000)` |
| **Test performance** | accuracy ≈ 86% (as trained in Part 1); ≈ 96% after the fix |

**What the explanations reveal:** the model ignores *not*, *no*, *never*... because the **cleaning step** removes them: they are on the stop-word list. *"I would not recommend this item to anyone."* (test sentence #370, negative) reaches the network as *"recommend item"* and is classified as positive with P = 0.81.
- **LIME** (Notebook 2): removing *not* from the copies of the sentence changes nothing → weight ≈ 0, while *recommend* gets all the weight; the same happens in four other misclassified sentences (*"Never had anything to complain about here."* → *"complain"*).
- **SHAP** (Notebook 4): by the **dummy** property, *not* gets **exactly 0** in sentence #370 (also visible in the lattice figure), and a global view over 50 sentences lists the words that **never** change any prediction.
- **The fix** (exercise): keeping the negations gives ≈ 86% → ≈ 92%; removing the cleaning altogether gives **≈ 96%** (on the sentences with a negation ≈ 63% → ≈ 91% → ≈ 96%).

**Why a sentence-embedding model and why `bge-small-en-v1.5`.** A modern small transformer is a real black box (nobody can read its 384 numbers), which is exactly the situation where model-agnostic explanations are needed. We tested alternatives on the same split, without the cleaning step: TF-IDF + logistic regression reaches about 82%, the popular `all-MiniLM-L6-v2` about 89% (it is trained for semantic similarity, not sentiment), `bge-small-en-v1.5`, of similar size, about 96%.

---

## 2. How "the same identical model" is guaranteed

- The **training part** of each SHAP notebook (data description, data loading, model training) is **identical, cell by cell,** to the training part of the matching LIME notebook: same data, same split, same random seeds, same code. Nothing is loaded from the LIME notebook.
- The training cell prints a **fingerprint** of the model, which is the same in both notebooks of a pair:

| Model | Notebooks | Fingerprint |
| :--- | :---: | :--- |
| 🏦 Mortgage | 1 and 3 | AUC = **0.8062**, P(deny) of applicant #112 = **0.523** |
| 🛒 Sentiment | 2 and 4 | test accuracy = **0.8592**, P(positive) of test sentence #370 = **0.810** |

- Why retrain instead of saving and loading the model? Every Colab notebook runs on its own machine: retraining needs no Google Drive permissions (about 2 seconds for the forest, well under a minute for the sentiment model on a CPU).
- The sentiment fingerprint is computed by a neural network: on a different machine (e.g. GPU vs CPU) the last digit of the probability may differ, but within the same machine the two notebooks always print the same numbers.

---

## 3. Structure of the notebooks

All four notebooks follow the same structure:

1. **Part 1 — Train the black box** (data, model, evaluation, fingerprint). Every concept is introduced in plain words (decision tree and Random Forest, ROC-AUC, stop words, sentence embeddings, logistic regression, ...).
2. **Part 2 — How the method works**: intuition (with a figure), formula, recipe, how the perturbed copies are built and what you can set, then the method **rebuilt by hand in a few lines of code**, only to show every detail of the mechanism. **One** short table then checks that the library gives similar numbers, and from there on the notebook uses the library. The LIME notebooks end Part 2 with **fidelity** (Section 2.6) and a table *"from the mechanism to the practice"*.
3. **Part 3 — Using the explanations** to audit or debug the model (in the SHAP notebooks also LIME vs SHAP and the **fidelity** comparison), with one hands-on exercise.
4. **Part 4 — The limits** of the method, shown with code where possible.
5. **Summary**, discussion questions (in capital letters, to start a class discussion), final exercises, references.

Every code cell is commented line by line.

**A figure opens the explanation of each method** (start of Part 2; the images are embedded in the notebooks, no external files needed):
- **LIME notebooks**: the original LIME intuition figure (the black box's decision function, the instance, the weighted perturbed copies, the local linear explanation), followed by a table that explains every element. Figure from the LIME repository, © 2016 Marco Tulio Ribeiro, BSD licence.
- **SHAP notebooks**: a lattice of the 8 teams that can be formed with 3 players (`public_bad_record`, `credit_problems`, `loan_to_value` for applicant #112; *recommend*, *item*, *not* for sentence #370), with the **real values** of our model. An order of the players is a path through the lattice; the orange arrows show when one player joins, with the weight and the marginal contribution of each arrow, and the text computes its Shapley value step by step (+0.217 for `public_bad_record`, +0.241 for *recommend*). In the text lattice *not* is a **dummy** player: adding it never changes the value.

**How the copies are built, and what you can set** (each notebook answers these questions explicitly, with code):

| | How a copy of the instance is built | How many features/words change in a copy? | What you can set |
| :--- | :--- | :--- | :--- |
| LIME · tabular | **every** feature is drawn again from the training data, each one from a **different random applicant** (the notebook shows copy #1 feature by feature, with the applicant each value comes from) | **random**, not set: a feature keeps the applicant's range with probability = share of training applicants in that range; for applicant #112, 6 of 9 features change on average (chart) | `num_samples`, `kernel_width`, `discretizer`, `num_features` |
| LIME · text | LIME draws **how many** words to remove (uniformly between 1 and the number of distinct words $d$), then **which** ones; no words are taken from other sentences | **random**: from 1 to $d$, about half on average | `num_samples`, `kernel_width`, `bow`, `num_features` |
| SHAP · tabular | for a team $S$: the features in $S$ from the applicant, **all** the others from **one** background applicant (the whole row), for each of the 100 background applicants (the notebook shows 3 copies) | **not random**: $9 - \|S\|$, and all team sizes are used (exact: all 512 teams) | the background (which and how many applicants), the algorithm (exact, permutation, tree) |
| SHAP · text | for a team $S$: the sentence with only the words of $S$, all the others removed (no background) | **not random**: $d - \|S\|$; teams are sampled (1,000 random orders by hand, `max_evals` in the library) | what a player is (the masker), the number of orders / `max_evals` |

**Fidelity: how we check that an explanation is right**

| | Measure | Where | What students see |
| :--- | :--- | :--- | :--- |
| LIME | **local $R^2$**: weighted $R^2$ of LIME's linear model on the perturbed copies (the numerator is LIME's own loss $\mathcal{L}$); also $g(x)$ vs $f(x)$ | Notebooks 1 and 2, Section 2.6 (`explanation.score`, `explanation.local_pred`) | a chart of fidelity vs number of features shown (the $\mathcal{L}$ vs $\Omega$ trade-off); applicant #112: low (0.32), sentence #370: high (0.95); in Notebook 1 the fidelity of all 85 explanations (it varies a lot) |
| SHAP | **local accuracy** (efficiency: base value + sum = $f(x)$ exactly) + **deletion test**: remove the top-$k$ features/words according to the explanation and watch the prediction fall | Notebooks 3 and 4, Sections 3.2-3.3 | SHAP vs LIME vs random on the 30 riskiest applicants / 50 sentences: both explanations beat random removal by far (mortgages: P(deny) 0.45 → 0.10 with SHAP, 0.11 with LIME, 0.32 at random after removing 3 reasons; reviews: P(predicted class) 0.84 → 0.44 with SHAP, 0.44 with LIME, 0.70 at random after removing 3 words) |

The message repeated in every notebook: **accuracy** compares the model with the truth; **fidelity** compares the explanation with the model. A faithful explanation of a wrong prediction is exactly what we need to find the model's mistakes.

| Notebook | By hand (to show the details) | With the library (used from then on) | What it is used for |
| :--- | :--- | :--- | :--- |
| 1. LIME · tabular | applicant #112: quartile ranges, 5,000 copies drawn from real applicants, how many features change per copy, Gaussian kernel, weighted Ridge regression, "what LIME sees" | `LimeTabularExplainer` (HTML view) | fidelity (local $R^2$, fewer features); LIME on all 85 African-American applicants; "identical twin" test; fix (drop race); stability; unrealistic copies (`housing_to_income > debt_to_income`); fidelity applicant by applicant |
| 2. LIME · text | sentence #370: 1,000 copies with random words removed, kernel, weighted Ridge; what happens to the copies when one word is removed | `LimeTextExplainer` (HTML view) | fidelity (local $R^2$, fewer words); LIME on four misclassified sentences; the cleaning bug; fix (keep the negations / no cleaning); stability |
| 3. SHAP · tabular | applicant #112: value of a team $v(S)$ with a background of 100 applicants, two orders → two stories, average over 1,000 orders = Shapley values, efficiency | exact explainer: waterfall, beeswarm, mean \|SHAP\| bar, dependence plots | LIME vs SHAP; deletion test (fidelity); fairness audit; fix (drop race) with global importance before/after |
| 4. SHAP · text | sentence #370: words as players, the empty sentence as base value, two orders, 1,000 orders; efficiency and dummy (*not* gets exactly 0) | text explainer: interactive text plot | LIME vs SHAP; deletion test (fidelity); words that never matter over 50 sentences; the cleaning bug; fix |

**By hand vs library:** the hand-made versions exist only to make the mechanism visible. Each notebook compares them with the library **once**, in one short table: same main reasons, same signs, similar sizes (small differences because LIME's copies are random, the LIME library draws tabular values from a bell curve inside each range, and the SHAP text explainer uses a faster approximation). The SHAP tabular explainer is exact, so there the numbers practically coincide.

---

## 4. Hands-on exercises

Each notebook has one exercise in which students **edit one or two lines** and re-run two cells; the notebook then shows a "before vs after" chart. With the lines left unchanged the notebook still runs (and says that nothing changed), so *Run all* never breaks. Solutions are in collapsible cells (**✅ Solution**, click to open).

| Notebook | Exercise | Lines to edit | Expected result |
| :--- | :--- | :--- | :--- |
| 1 | A fairer model | `features_to_drop = ['african_american']` | AUC 0.806 → 0.795 (practically unchanged); applicant #112 approved (0.523 → 0.398); average predicted risk: African-American 20.5% → 17.6%, others 9.9% → 10.2% |
| 2 | Fix the cleaning | Step 1: `words_to_keep = ['not', 'no', 'never', ...]`; Step 2: `CLEAN_TEXT = False` | Step 1: test accuracy ≈ 86% → ≈ 92% (with a negation ≈ 63% → ≈ 91%); Step 2: ≈ 96% (≈ 96% also with a negation); sentence #370 becomes negative (P(positive) 0.81 → 0.02) and LIME shows *not* as its strongest word |
| 3 | Remove the protected attribute | `features_to_drop = ['african_american']` | as in Notebook 1, plus the global importance (mean \|SHAP\|) of every feature before/after: race drops to zero, the other features barely change |
| 4 | Fix the cleaning (found with the global view) | as in Notebook 2 | as in Notebook 2; in sentence #370 *not* gets the largest SHAP value, toward negative |

Each notebook also ends with 4 open **final exercises** (e.g. what-if on a feature, kernel width and fidelity, another neighbourhood or background, your own reviews, fewer orders, a harder deletion test).

---

## 5. How to run

**Google Colab (recommended)**
1. Open <https://colab.research.google.com> → *File* → *Upload notebook* (or the *GitHub* tab, if the notebooks are in a repository; any notebook on GitHub opens at `https://colab.research.google.com/github/<user>/<repository>/blob/main/<notebook>.ipynb`).
2. *Runtime* → *Run all* (the free CPU runtime is enough).
3. The first cell installs what is missing (`lime`, `shap`, `sentence-transformers`); the data are downloaded automatically, and the embedding model is downloaded from Hugging Face the first time (≈ 130 MB).

**Locally**
```bash
pip install numpy pandas scikit-learn matplotlib lime shap sentence-transformers jupyter
jupyter notebook 01_XAI_LIME_Tabular.ipynb
```

---

## 6. Notes for the instructor

- **Choices made for clarity:** 9 features for the mortgage data; short review sentences for the text data; one showcase instance per model (applicant #112, sentence #370) used in both notebooks of the pair.
- **The text bug is realistic:** removing stop words is a standard step of bag-of-words pipelines and is often copied into pipelines that use embedding models. scikit-learn's own documentation warns that its English list has known issues. Here it deletes *not*, *no*, *never*, *nothing*, *without*, *too*, *very*, *but*, *well*, *less*, ... The notebooks never say this in advance: students discover it through the explanations (LIME weight ≈ 0, SHAP exactly 0).
- **Sensitive content:** race is used as a feature **on purpose**, to show how explanations reveal discrimination learned from historical decisions. The notebooks frame it as an audit and leave the open questions (legitimate risk factors or proxies?) to the class discussion. A few review sentences contain mild slang (e.g. *"crap"*), as real reviews do.
- **Interactive outputs:** the LIME HTML views and the SHAP text plots need a live notebook (Colab, Jupyter). A static preview (e.g. on GitHub) shows them empty; all the other charts are static images.
- **Timing (CPU):** every notebook runs in a few minutes. The slowest cell is the SHAP explanation of 300 applicants in Notebook 3 (≈ 30 seconds); in the text notebooks the sentiment model is retrained in the exercise (less than a minute), and a cache in the pipeline makes sure that each distinct sentence is encoded only once.
- **Charts:** our own charts use a colorblind-safe palette; red/blue always mean "toward / away from" the explained class, as in the SHAP plots.

## 7. Troubleshooting

| Problem | Solution |
| :--- | :--- |
| A data download fails | The mortgage CSV comes from `raw.githubusercontent.com`, the review sentences from `archive.ics.uci.edu`. If your network blocks one of them, download the file from the URL above, upload it to Colab (*Files* panel) and set `URL_HMDA = "HMDA.csv"` or `URL_REVIEWS = "file:///content/sentiment+labelled+sentences.zip"` |
| The embedding model does not download | It comes from Hugging Face (`BAAI/bge-small-en-v1.5`). A warning about a missing `HF_TOKEN` is harmless; if the download fails, re-run the cell |
| The LIME view is empty in Jupyter | Re-run the cell; in classic Jupyter use *File → Trust notebook* (JavaScript is needed) |
| `ModuleNotFoundError: lime` / `shap` / `sentence_transformers` | Re-run the first cell (installation) |
| Many progress-bar lines below the SHAP cells | Normal: they are `shap`'s progress bars (a single bar on Colab) |

## 8. References

- Ribeiro, Singh, Guestrin (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD.
- Lundberg, Lee (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Shapley (1953). *A Value for n-Person Games*. Contributions to the Theory of Games II.
- Lundberg et al. (2020). *From local explanations to global understanding with explainable AI for trees*. Nature Machine Intelligence.
- DeYoung et al. (2020). *ERASER: A Benchmark to Evaluate Rationalized NLP Models*. ACL (deletion-based fidelity).
- Molnar. *Interpretable Machine Learning* (online book).
- Munnell, Tootell, Browne, McEneaney (1996). *Mortgage Lending in Boston: Interpreting HMDA Data*. American Economic Review 86(1).
- Kotzias, Denil, de Freitas, Smyth (2015). *From Group to Individual Labels Using Deep Features*. KDD. Dataset: *Sentiment Labelled Sentences*, UCI Machine Learning Repository, <https://doi.org/10.24432/C57604>.
- Xiao et al. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding* (the BGE embedding models). arXiv:2309.07597.
- Reimers, Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.
