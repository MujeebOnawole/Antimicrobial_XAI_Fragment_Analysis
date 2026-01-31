# Decoding Antimicrobial Selectivity with Explainable AI Fragments

**Fragment Analysis Repository**

## Overview

This repository contains the data analysis pipeline and extracted fragment datasets supporting the manuscript *"Decoding Antimicrobial Selectivity with Explainable AI Fragments"*. The analysis derives pathogen-specific antimicrobial design rules from XAI (Explainable AI) attributed molecular fragments. The analysis identifies molecular substructures (scaffolds and substituents) associated with activity against three major pathogen classes:

- **S. aureus** (Gram-positive bacteria)
- **E. coli** (Gram-negative bacteria)
- **C. albicans** (Fungi)

The resulting **SELECT Rules** provide actionable guidance for medicinal chemistry optimization of antimicrobial compounds.

> **Note:** This is a supporting repository for fragment analysis. The main repository containing model development and training is available at: [https://github.com/MujeebOnawole/Antimicrobial_XAI](https://github.com/MujeebOnawole/Antimicrobial_XAI)

## Key Findings

### Fragment Extraction Summary

| Stage | S. aureus | E. coli | C. albicans | Total |
|-------|-----------|---------|-------------|-------|
| Training Compounds | 54,277 | 44,920 | 28,476 | 68,736 unique |
| Reliable Compounds | 47,282 (87.1%) | 39,233 (87.3%) | 24,741 (86.9%) | 63,169 unique |
| **Positive Fragments** | 2,332 | 537 | 1,234 | **12,993** |
| **Negative Fragments** | 398 | 161 | 227 | **5,114** |

### Fragment Breakdown by Testing Category

| Category | Positive | Negative |
|----------|----------|----------|
| Single-pathogen (SA, EC, CA) | 4,103 | 786 |
| Dual-pathogen (SA+EC, SA+CA, EC+CA) | 6,087 | 1,575 |
| Pan-pathogen (all three) | 2,803 | 2,753 |
| **Total** | **12,993** | **5,114** |

### SELECT Design Rules

| Pathogen | LogP | TPSA (Å²) | HBD | Design Principle |
|----------|------|-----------|-----|------------------|
| S. aureus (G+) | 1.9–4.1 | 17–43 | 0–1 | Lipophilic, membrane-penetrating |
| E. coli (G-) | 1.3–3.4 | 25–50+ | 3–5 | Polar, porin-transported |
| C. albicans | 1.7–3.7 | 34–38 | 0–2 | Balanced, N-heterocycles |

### Statistically Significant Property Differences (|Cohen's d| ≥ 0.3)

| Property | Comparison | Cohen's d | Interpretation |
|----------|------------|-----------|----------------|
| LogP | SA vs EC | 0.42 | SA more lipophilic |
| TPSA | SA vs EC | -0.32 | SA lower polarity |
| HBD | SA vs CA | 0.34 | SA more H-bond donors |
| HBD | EC vs CA | 0.67 | EC more H-bond donors |

## Repository Structure

```
fragments_analysis/
├── README.md
├── requirements.txt
├── .gitignore
│
├── raw_data/                    # Original XAI prediction data (zipped)
│   ├── S_aureus_pred_class_murcko.csv.zip
│   ├── E_coli_pred_class_murcko.csv.zip
│   ├── C_albicans_pred_class_murcko.csv.zip
│   └── source_data/             # Original activity data with MIC values (zipped)
│
├── data/                        # Extracted fragment datasets
│   ├── single_pathogen/
│   │   ├── positive/            # SA-only, EC-only, CA-only positive
│   │   └── negative/            # SA-only, EC-only, CA-only negative
│   ├── dual_pathogen/
│   │   ├── positive/            # SA+EC, SA+CA, EC+CA positive
│   │   └── negative/
│   ├── pan_pathogen/
│   │   ├── positive/            # Triple-active (all three)
│   │   └── negative/
│   └── manuscript/              # Curated datasets for publication
│
├── sql/                         # SQL extraction scripts
│   ├── single_pathogen/
│   ├── dual_pathogen/
│   └── pan_pathogen/
│
├── scripts/                     # Python analysis scripts
│   ├── analysis/                # Statistical analysis
│   ├── visualization/           # Plot generation
│   └── extraction/              # Data processing utilities
│
├── results/                     # Analysis outputs
│   ├── reports/                 # Text reports and summaries
│   └── statistics/              # Statistical results (CSV)
│
└── figures/                     # Publication figures
    ├── radar_plots/             # Property profile visualizations
    ├── molecular_structures/    # Exemplar compound structures
    ├── distributions/           # Property distribution plots
    └── exemplar_fragments/      # Fragment visualization images
```

## Raw Data Schema

The raw data files (`*_pred_class_murcko.csv`) contain XAI-attributed predictions:

| Column | Description |
|--------|-------------|
| `COMPOUND_ID` | ChEMBL compound identifier |
| `SMILES` | Molecular structure |
| `ensemble_prediction` | Predicted activity probability (0–1) |
| `prediction` | Binary class (0=inactive, 1=active) |
| `prediction_std` | Prediction uncertainty |
| `murcko_substructure_X_attribution` | XAI attribution score for scaffold X |
| `murcko_substructure_X_smiles` | SMILES of scaffold X |
| `murcko_substituent_X_Y_smiles` | SMILES of substituent Y on scaffold X |
| `murcko_substituent_X_Y_attribution` | XAI attribution score |

## Fragment Extraction Criteria

Fragments are classified as positive or negative based on:

- **Positive fragments**: Mean attribution ≥ 0.1 across reliable compounds
- **Negative fragments**: Mean attribution ≤ -0.1 across reliable compounds
- **Reliable compounds**: `has_mismatch = 0` AND `reliability_level = 'RELIABLE'`
- **Minimum support**: ≥5 true positive cases (single-pathogen) or ≥3 per pathogen (dual/pan)

### Pathogen Specificity Logic

- **Single-pathogen (e.g., SA-only)**: Present in SA, NOT in EC or CA
- **Dual-pathogen (e.g., SA+EC)**: Present in both SA and EC, NOT in CA
- **Pan-pathogen**: Present in all three pathogens with positive attribution

## Exemplar Compounds

### Single-Pathogen Exemplars (0 SELECT Violations)

| Pathogen | ChEMBL ID | MIC (µg/mL) | Activity (µM) |
|----------|-----------|-------------|---------------|
| S. aureus | CHEMBL4536843 | 8.0 | 29.3 |
| E. coli | CHEMBL369493 | 2.5 | 8.8 |
| C. albicans | CHEMBL4277673 | 4.0 | 14.2 |

### Dual-Active Exemplars

| Combination | ChEMBL ID | MIC (µg/mL) |
|-------------|-----------|-------------|
| SA + EC | CHEMBL2178320 | SA: 0.25, EC: 4.0 |
| SA + CA | CHEMBL5207371 | SA: 31.2, CA: 31.2 |
| EC + CA | CHEMBL5409101 | EC: 0.1, CA: 0.1 |

### Broad-Spectrum (Triple-Active) Exemplar

| ChEMBL ID | SA MIC | EC MIC | CA MIC |
|-----------|--------|--------|--------|
| CHEMBL2297203 | 30.0 µg/mL | 40.0 µg/mL | 30.0 µg/mL |

## Usage

### Statistical Analysis

```python
# Calculate physicochemical properties and run statistical tests
python scripts/analysis/pathogen_analyzer.py
python scripts/analysis/triple_active_statistics.py
```

### Visualization

```python
# Generate radar plots for exemplar compounds
python scripts/visualization/exemplar_compounds_radar_final.py
python scripts/visualization/dual_triple_exemplar_radar_plots.py

# Generate positive vs negative comparison
python scripts/visualization/positive_negative_comparison_radar.py
```

## Requirements

```
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
matplotlib>=3.5.0
seaborn>=0.12.0
rdkit>=2022.09.1
statsmodels>=0.13.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Related Repository

**Main Repository (Model Development & Training):**
[https://github.com/MujeebOnawole/Antimicrobial_XAI](https://github.com/MujeebOnawole/Antimicrobial_XAI)

The main repository contains:
- XAI model architecture and training code
- Ensemble prediction methodology
- Attribution calculation methods
- Original training datasets

## Citation

If you use this data or methodology, please cite:

> **Decoding Antimicrobial Selectivity with Explainable AI Fragments**
>
> Abdulmujeeb T. Onawole<sup>1</sup>, Mark A.T. Blaskovich<sup>1,2</sup> and Johannes Zuegg<sup>1*</sup>
>
> <sup>1</sup>Centre for Superbug Solutions, Institute for Molecular Bioscience, The University of Queensland, Brisbane, Queensland, Australia.
>
> <sup>2</sup>ARC Centre for Agricultural and Environmental Solutions to Antimicrobial Resistance, Institute for Molecular Bioscience, The University of Queensland, Brisbane, Queensland, Australia.

## License

MIT License

## Contact

- **Abdulmujeeb T. Onawole** - Centre for Superbug Solutions, The University of Queensland
- **Johannes Zuegg*** (Corresponding Author) - Centre for Superbug Solutions, The University of Queensland
