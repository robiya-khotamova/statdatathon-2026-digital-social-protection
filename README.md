# Digital Social Protection in Kazakhstan: Who Is at Risk of Being Left Out?

**Stat.Datathon 2026 · Finalist** (organised by the Bureau of National Statistics of Kazakhstan and CIS-STAT, 108 applicant teams)
Team **Datum Duo**: Robiyabonu Khotamova & Kamilla Irgasheva

**Question:** Kazakhstan is moving social support to proactive digital channels. Which households are most likely to need help *and* be hardest to reach digitally, and which channel should reach them?

**Answer:** Digital access is a separate dimension of vulnerability. A digital-only model would miss most of the 4.7% of households that are both debt-stressed and digitally excluded: only **7%** of them can be served purely digitally, **42%** are reachable by SMS / zero-rated mobile, and **51%** need offline or assisted service. We recommend a **hybrid delivery model** (digital + SMS/zero-rating + assisted offline).

## Framework
Digital-divide theory (Van Dijk, 2005), social and digital exclusion (Helsper, 2012) and targeted social assistance (World Bank, 2022), set against Kazakhstan's proactive "Digital Family Card" model (since 2022).

## Data & method
- **47,826 household-years**, 2021–2024, 17 regions: synthetic national household-survey microdata provided by the organisers (budget, roster, dwelling and quality-of-life forms).
- **Pipeline** (Python: pandas, statsmodels, scikit-learn, matplotlib):
  1. `src/01_discover.py`: profile every survey table and code book.
  2. `src/02_build_core.py`: build one row per household per year from 4 survey forms; harmonise questionnaire and regional-boundary changes across waves.
  3. `src/03_digital_index.py`: add quality-of-life variables and build a **composite digital-barrier index (0–1)**.
  4. `src/04_analysis.py`: test 6 hypotheses: quintile comparisons, log-log elasticity, logistic regressions with income, household-size, region and year controls, χ² independence test, k-means clustering, channel sizing.
- The raw microdata belongs to the organisers and is **not included**; aggregated results are in `results/`.

## Results by hypothesis

| # | Hypothesis | Verdict | Key evidence |
|---|---|---|---|
| H1 | Poorest households spend less on communication | Partially supported | Q5 spends 2.1× more per person than Q1, but income elasticity is only 0.01: connectivity is a near-fixed cost, so Q1 spends **4.1% of income** on it vs 2.3% in Q5 |
| H2 | Rural households face a higher digital barrier | Supported | Adjusted odds of no home internet, rural vs urban = **1.28** [1.23; 1.34], controlling for income and region |
| H3 | Benefit recipients borrow more for everyday needs | Not supported | Adjusted OR = 0.99 [0.88; 1.11], p = 0.85 |
| H4 | Debt service erodes social transfers | Partially supported | 45% of recipients service debt; upper-bound offset = **33.7%** of transfer money |
| H5 | Debt stress and digital exclusion compound each other | Not supported as an overlap | 4.7% are double-vulnerable, but the factors are independent (co-occurrence ratio 0.99, χ² p = 0.80) |
| H6 | SMS / zero-rating can reach the most vulnerable | Supported (scenario) | 79% of double-vulnerable households have mobile coverage |

*`results/tables/scorecard.csv` is the automatic first-pass output; the verdicts for H1 and H4 above were refined after manual review, as presented to the jury.*

## Other findings
- **The take-up gap is not a digital problem.** 37% of low-income households receive no social transfer, but a high digital barrier does not make this worse (OR 0.91, p = 0.10). The issue is eligibility criteria and awareness.
- **Geography matters more than income.** The highest barrier is in Shymkent (0.36), Turkestan (0.34) and West Kazakhstan (0.33). In Atyrau, 37.5% of households have no mobile signal at the dwelling, which needs an infrastructure fix, not a tariff.
- **The double-vulnerable group splits into 4 profiles** (k-means). The priority is large low-income families with **no** benefits (22% of the group): the highest debt load and no support coverage.

## Recommendations
1. **SMS-first + zero-rated portal** where mobile coverage exists but home internet doesn't.
2. **Assisted offline service** in Turkestan, Shymkent and Kyzylorda.
3. **Proactive eligibility matching** for large low-income families without benefits.
4. **A social connectivity tariff:** the poorest quintile spends almost twice the income share on communication.
5. **A mobile-coverage audit** in Atyrau.

## Limitations
Synthetic data keeps roughly half of the real-world associations, so effect sizes are lower bounds. The data are repeated cross-sections, not a true panel. 2021 is only partly comparable with later waves.

## Repository
```
src/        4-step Python pipeline
results/    findings.txt, key_numbers.json, tables/*.csv (aggregated outputs)
figures/    charts used in the defence (labels in Russian, the competition language)
slides/     final presentation as defended (PDF, Russian)
```

## Run it
Download the Stat.Datathon 2026 microdata from the organisers, then:
```bash
pip install -r requirements.txt
python src/02_build_core.py "path/to/synthetic_microdata_2021-2024"
python src/03_digital_index.py "path/to/synthetic_microdata_2021-2024"
python src/04_analysis.py work results
```
