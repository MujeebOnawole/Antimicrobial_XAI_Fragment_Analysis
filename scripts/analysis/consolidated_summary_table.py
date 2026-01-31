"""
Consolidated Single-Pathogen Summary Table
==========================================
Creates ONE comprehensive table with all properties, means, and significance
across all three pathogens for publication use.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

BASE_DIR = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results" / "statistics"

PROPERTIES = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Rings', 'Fsp3']


def calculate_properties(smiles):
    if not RDKIT_AVAILABLE or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': rdMolDescriptors.CalcTPSA(mol),
            'HBD': rdMolDescriptors.CalcNumHBD(mol),
            'HBA': rdMolDescriptors.CalcNumHBA(mol),
            'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'Rings': rdMolDescriptors.CalcNumRings(mol),
            'Fsp3': rdMolDescriptors.CalcFractionCSP3(mol)
        }
    except:
        return None


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def sig_symbol(d, p):
    """Return significance symbol based on effect size and p-value."""
    if p > 0.05:
        return ""
    d_abs = abs(d)
    if d_abs >= 0.8:
        return "***"
    elif d_abs >= 0.5:
        return "**"
    elif d_abs >= 0.2:
        return "*"
    return ""


def load_data(filepath):
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath)
    smiles_col = None
    for col in ['fragment_smiles', 'smiles', 'SMILES']:
        if col in df.columns:
            smiles_col = col
            break
    if smiles_col is None:
        return None

    props_list = []
    for _, row in df.iterrows():
        props = calculate_properties(row[smiles_col])
        if props:
            props_list.append(props)
    return pd.DataFrame(props_list)


def main():
    print("=" * 70)
    print("CONSOLIDATED SINGLE-PATHOGEN SUMMARY TABLE")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data_sources = {
        'SA': DATA_DIR / 'single_pathogen' / 'positive' / 'SA_specific_positive_scaffolds.csv',
        'EC': DATA_DIR / 'single_pathogen' / 'positive' / 'EC_specific_positive_scaffolds.csv',
        'CA': DATA_DIR / 'single_pathogen' / 'positive' / 'CA_specific_positive_scaffolds.csv',
    }

    all_data = {}
    for pathogen, filepath in data_sources.items():
        df = load_data(filepath)
        if df is not None:
            all_data[pathogen] = df
            print(f"Loaded {pathogen}: n={len(df)}")

    # Build consolidated table
    rows = []
    for prop in PROPERTIES:
        row = {'Property': prop}

        # Mean +/- SD for each pathogen
        for p in ['SA', 'EC', 'CA']:
            if p in all_data:
                vals = all_data[p][prop].dropna()
                row[f'{p}_Mean_SD'] = f"{vals.mean():.2f} +/- {vals.std():.2f}"
                row[f'{p}_n'] = len(vals)
            else:
                row[f'{p}_Mean_SD'] = "N/A"
                row[f'{p}_n'] = 0

        # Cohen's d for each pairwise comparison
        comparisons = [('SA', 'EC'), ('SA', 'CA'), ('EC', 'CA')]
        for p1, p2 in comparisons:
            if p1 in all_data and p2 in all_data:
                vals1 = all_data[p1][prop].dropna()
                vals2 = all_data[p2][prop].dropna()
                d = cohens_d(vals1, vals2)
                _, p_val = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
                sig = sig_symbol(d, p_val)
                row[f'd_{p1}v{p2}'] = f"{d:.2f}{sig}"
            else:
                row[f'd_{p1}v{p2}'] = "N/A"

        rows.append(row)

    # Create DataFrame
    df_summary = pd.DataFrame(rows)

    # Reorder columns for clarity
    col_order = [
        'Property',
        'SA_Mean_SD', 'EC_Mean_SD', 'CA_Mean_SD',
        'd_SAvEC', 'd_SAvCA', 'd_ECvCA'
    ]
    df_summary = df_summary[col_order]

    # Print table
    print("\n" + "=" * 120)
    print("CONSOLIDATED SUMMARY TABLE")
    print("=" * 120)
    print(f"\n{'Property':<10} {'S. aureus (G+)':<18} {'E. coli (G-)':<18} {'C. albicans':<18} {'d SA-EC':<10} {'d SA-CA':<10} {'d EC-CA':<10}")
    print("-" * 110)

    for _, row in df_summary.iterrows():
        print(f"{row['Property']:<10} {row['SA_Mean_SD']:<18} {row['EC_Mean_SD']:<18} {row['CA_Mean_SD']:<18} {row['d_SAvEC']:<10} {row['d_SAvCA']:<10} {row['d_ECvCA']:<10}")

    # Add sample sizes row
    print("-" * 110)
    print(f"{'n':<10} {all_data['SA'].shape[0]:<18} {all_data['EC'].shape[0]:<18} {all_data['CA'].shape[0]:<18}")

    # Save
    output_file = OUTPUT_DIR / 'single_pathogen_consolidated_summary.csv'
    df_summary.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # Also save a nicely formatted version with headers
    df_formatted = df_summary.copy()
    df_formatted.columns = [
        'Property',
        'S. aureus (G+) Mean +/- SD',
        'E. coli (G-) Mean +/- SD',
        'C. albicans Mean +/- SD',
        "Cohen's d (SA vs EC)",
        "Cohen's d (SA vs CA)",
        "Cohen's d (EC vs CA)"
    ]

    formatted_file = OUTPUT_DIR / 'single_pathogen_consolidated_summary_formatted.csv'
    df_formatted.to_csv(formatted_file, index=False)
    print(f"Saved: {formatted_file}")

    print("\n" + "=" * 70)
    print("Legend: * |d|>=0.2 (small), ** |d|>=0.5 (medium), *** |d|>=0.8 (large)")
    print("=" * 70)


if __name__ == '__main__':
    main()
