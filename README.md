# Cost-sensitive credit default prediction

ES 335 final project by Tagalpallewar Abhishek Prashant (24110364) and Siddhantkumar Vinay Thute (24110373).

**Project website:** https://siddhant171.github.io/credit-default-prediction/ (hosted from the team member's fork). The static site source is in [`docs/`](docs/).  
**Repository:** https://github.com/TagalpallewarAbhishek/credit-default-prediction

## Question and data

Can account and payment history predict next-month default, and which probability threshold minimizes loss under different error costs? One row is one client account. The target is next-month default payment, not whether a new loan should be approved.

The [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) has 30,000 Taiwan accounts, 23 predictors, and one binary target. It contains credit limits, demographics, six months of repayment statuses, bills, and payments. Roughly 22.12% of rows are defaults.

## Setup and run

Use Python 3.11 or newer. Run commands from the repository root.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/01_inspect_data.py
python src/02_split_data.py
python src/train_baselines.py
python src/analyze_baselines.py
python src/calibrate_and_optimize.py
python src/train_ensembles.py
python src/evaluate_all_models.py
python src/cost_optimizer.py
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000/`. `01_inspect_data.py` downloads the UCI `.xls` file and saves `data/raw_credit_default.csv`. If the URL is unavailable, download the file from UCI and save a CSV copy with that path and the original column names. The split script then exports `data/train.csv`, `data/val.csv`, and `data/test.csv`.

The fixed stratified split is 70/15/15 with `random_state=42`. The script drops `ID`, renames the target to `TARGET` and `PAY_0` to `PAY_1`, groups undocumented education and marriage codes, and retains negative bill balances. Learned scaling and encoding fit inside model pipelines on training folds only.

## Results and verification

`results/metrics_summary.csv` contains all six validation model comparisons. `results/ensemble_val_metrics.csv` has ensemble hyperparameters and cross-validation scores. The `.npz` files retain validation probabilities. `results/test_metrics.json` records the one-time held-out test report. `results/test_predictions.json` holds true labels and probabilities for the interactive simulator; `docs/assets/test_predictions.json` is its website copy. Figures are in `docs/assets/`.

The selected HistGradientBoosting model had validation PR-AUC 0.5526, ROC-AUC 0.7800, and loss $4,187,500 at the theoretical 0.1304 threshold. A validation sweep selected 0.135. On 4,500 test accounts, PR-AUC was 0.5596, ROC-AUC 0.7798, and Brier score 0.1348. At 0.135, 171 defaults were missed and 1,686 non-defaults were flagged. Test-sample assumed loss was $4,239,000 versus $6,677,000 at 0.50. These are scenario costs using $10,000 per missed default and $1,500 per false alarm, not observed savings.

## Pending handoff items

- Representative false-negative and false-positive borrower-profile analysis awaits review. The website shows final aggregate counts and marks profile examples pending.
- The planned notebooks are not yet present. The scripts above are the executable workflow and are the source for current saved results.
- To host the canonical URL from the original repository, its owner must enable GitHub Pages using GitHub Actions. The same site is already enabled on the team member's fork. Verify the public URL in a private browser window.

## Publish with GitHub Actions

The workflow in [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml) uploads `docs/` and deploys it on pushes to `main` that change the site. The repository owner must enable Pages once:

1. Open the original repository's **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Open **Actions → Deploy website to GitHub Pages → Run workflow**, select `main`, and run it.
4. Wait for the `deploy` job to finish. Open the URL shown in **Settings → Pages** in a private browser window.

The expected canonical URL is `https://tagalpallewarabhishek.github.io/credit-default-prediction/`. The owner needs admin or maintainer permission to change the Pages source. If the workflow is blocked, check **Settings → Actions → General** for Actions permissions and **Settings → Environments → github-pages** for deployment restrictions.

## Limitations

The data describe Taiwan accounts in 2005. The dataset does not include actual loss amounts, recovery, intervention outcomes, or current populations. The costs are assumptions. Sex and marital status raise fairness questions requiring separate checks before any operational use.

## AI-use disclosure

AI tools assisted with code, analysis workflow, website wording/layout, and the original halftone hero illustration in `docs/assets/hero-ledger.webp`. The team remains responsible for checking computations, source citations, and claims against the saved data and code. **Pending team review:** add specific tools and the division of AI-assisted work before submission.

The website's visual direction was inspired by the [GoShuin — Every stamp, a memory](https://www.awwwards.com/inspiration/desktop-goshuin-every-stamp-a-memory) example. Its artwork and interface assets were not reused.

## Citations

- Yeh, I.-C. (2009). [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). UCI Machine Learning Repository. DOI: [10.24432/C55S3H](https://doi.org/10.24432/C55S3H).
- Pedregosa, F. et al. (2011). [Scikit-learn: Machine Learning in Python](https://jmlr.org/papers/v12/pedregosa11a.html). *Journal of Machine Learning Research*, 12, 2825–2830.
