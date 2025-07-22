# PRISM: Price Recommendation via Iterative Subgroup Modeling

This repository contains the implementation of **PRISM**, a modular framework for modeling price elasticity and generating interpretable pricing recommendations using **Exceptional Model Mining (EMM)**. The methodology was developed as part of a Master’s thesis in the Data & AI research group at the Eindhoven University of Technology and applied in collaboration with industry.

---

## Project Scope

**PRISM** addresses the challenge of data-driven pricing in complex, B2B industrial settings, where:

- Product-level sales data is sparse, irregular, and heterogeneous
- Price elasticity varies across product groups and time periods
- Interpretability is essential for decision support

The framework integrates the following components:
- Time series condensation and inflation adjustment
- Subgroup discovery via tailored EMM targets, quality functions, and evaluation metrics
- Forecasting with ARIMA and hierarchical MinT reconciliation
- Elasticity-informed price recommendation logic

---

##  Repository Structure

```bash
PRISM/
├── data specifications/
│   └── data_requirements.md     # Input data specifications (no real data included)
├── notebooks/
│   ├── Step 2 Aggregate Price Elasticity.ipynb
│   ├── Step 3 Subgroup Discovery.ipynb
│   ├── Step 4 Predict Price Elasticity ARIMA.ipynb
|   ├── Step 5 Compute Price Recommendation.ipynb
│   └── functions_definitions.ipynb                     # Utility functions 
├── src/
│   └──  my_pysubgroup_extensions/  # Custom EMM target & quality functions
├── README.md
└── requirements.txt
```

---

##  Input Data

This repository does **not** include any real or synthetic data due to confidentiality. Instead, input schema requirements are fully documented in [`data specifications/data_requirements.md`](data/data_requirements.md).

The framework assumes the data has already been preprocessed. Specifically, it expects:

1. A dataset of order-level data, enriched with stationary product-level data for each order
2. A time series of inflation indices (e.g., from ECB or national sources)

---

##  Methodology Overview

1. **Feature Engineering**  
   Aggregates transactional data to product level, computing average prices, discounts, order metrics, and cost-related features. Computes a timeseries of price elasticities for each product.

2. **Subgroup Discovery**  
   Applies EMM using the numerical array of historical elasticity values as a target

3. **Forecasting**  
   Uses ARIMA with MinT reconciliation to predict future elasticity per product while maintaining coherence across product groups.

4. **Price Recommendation**  
   Suggests pricing actions based on predicted elasticity, inflation, and revenue projections. Built-in safeguards prevent harmful pricing decisions.


### Target Compatibility Disclaimer

PRISM is built and evaluated around the numerical array target. While the categorical and correlation-based targets are supported, they have limitations:

- **Categorical Target** : Subgroups are discovered per class (e.g., ‘positive’, ‘negative’), leading to overlapping group membership. This violates disjoint segmentation assumptions and makes it incompatible with forecasting out-of-the-box.
- **Array Target** : Fully supported. Used in all forecasting and price recommendation components.
- **Correlation Target** : Supported and disjoint, but with lower subgroup quality and narrower coverage in experiments.

Refer to the full thesis for detailed methodology and evaluation results, as well as target choice trade-offs.

##  Academic Reference

If you use this code in academic work, please cite the following:

> Rogozan, M. (2025). *PRISM: A Framework for Price Elasticity Modeling via Exceptional Model Mining*. Master’s thesis, Eindhoven University of Technology, Department of Mathematics and Computer Science.

---

##  Setup Instructions

This project uses Python 3.13.1+ and the following core libraries:

- `pandas`, `numpy`, `scipy`, `matplotlib`, `statsmodels`
- `pysubgroup` (extended with custom targets and quality functions)
- `pmdarima` for time series forecasting


Install requirements using:

```bash
pip install -r requirements.txt
```

### Dependencies and Extensions
This project extends the [pysubgroup](https://github.com/flemme/pysubgroup) library.

The original library is NOT included in this repo. After installing the requirements, the library extensions in [`src`](src/) should be manually patched into the user's local [pysubgroup](https://github.com/flemme/pysubgroup) fork.

---

##  Disclaimer

This is academic research code applied to a real-world industrial dataset. It is shared for transparency and reproducibility of the PRISM methodology. No part of the original dataset is included. Users are responsible for adapting the code to their own data context.

---

##  Contact

For questions about the methodology or implementation, feel free to  open an issue on this repository.

---