"""
Dual-Active Fragment Summary Statistics
========================================
Calculates mean +/- SD for all 8 physicochemical properties
for each dual-active fragment combination (SA+EC, SA+CA, EC+CA).

Output format matches single_pathogen_consolidated_summary.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis")
DATA_DIR = BASE_DIR / "data" / "dual_pathogen" / "positive"
OUTPUT_DIR = BASE_DIR / "results" / "statistics"


def calc_properties(smiles):
    """Calculate 8 physicochemical properties from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MW': round(Descriptors.MolWt(mol), 2),
        'LogP': round(Descriptors.MolLogP(mol), 2),
        'TPSA': round(rdMolDescriptors.CalcTPSA(mol), 2),
        'HBD': rdMolDescriptors.CalcNumHBD(mol),
        'HBA': rdMolDescriptors.CalcNumHBA(mol),
        'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'Rings': rdMolDescriptors.CalcNumRings(mol),
        'Fsp3': round(rdMolDescriptors.CalcFractionCSP3(mol), 2)
    }


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (group1.mean() - group2.mean()) / pooled_std


def load_dual_fragments():
    """Load all dual-active fragment files and calculate properties."""

    combinations = {
        'SA+EC': ['dual_SA_EC_positive_scaffolds.csv', 'dual_SA_EC_positive_substitutents.csv'],
        'SA+CA': ['dual_SA_CA_positive_scaffolds.csv', 'dual_SA_CA_positive_substitutents.csv'],
        'EC+CA': ['dual_CA_EC_positive_scaffolds.csv', 'dual_CA_EC_positive_substitutents.csv']
    }

    all_data = {}

    for combo, files in combinations.items():
        fragments = []
        for file in files:
            filepath = DATA_DIR / file
            if filepath.exists():
                df = pd.read_csv(filepath)
                if 'fragment_smiles' in df.columns:
                    fragments.extend(df['fragment_smiles'].dropna().unique().tolist())

        # Calculate properties for unique fragments
        props_list = []
        for smiles in set(fragments):
            props = calc_properties(smiles)
            if props:
                props['SMILES'] = smiles
                props_list.append(props)

        all_data[combo] = pd.DataFrame(props_list)
        print(f"{combo}: {len(all_data[combo])} unique fragments with valid properties")

    return all_data


def calculate_summary_statistics(data_dict):
    """Calculate mean +/- SD for each combination and property."""

    properties = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Rings', 'Fsp3']
    combos = ['SA+EC', 'SA+CA', 'EC+CA']

    # Summary table
    summary_rows = []
    for prop in properties:
        row = {'Property': prop}
        for combo in combos:
            df = data_dict[combo]
            mean = df[prop].mean()
            sd = df[prop].std()
            n = len(df)
            row[f'{combo}_Mean'] = round(mean, 2)
            row[f'{combo}_SD'] = round(sd, 2)
            row[f'{combo}_N'] = n
            row[f'{combo}_Mean_SD'] = f"{mean:.2f} +/- {sd:.2f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def calculate_pairwise_comparisons(data_dict):
    """Calculate Cohen's d for all pairwise comparisons."""

    properties = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Rings', 'Fsp3']
    combos = ['SA+EC', 'SA+CA', 'EC+CA']
    comparisons = [('SA+EC', 'SA+CA'), ('SA+EC', 'EC+CA'), ('SA+CA', 'EC+CA')]

    results = []
    for prop in properties:
        row = {'Property': prop}
        for c1, c2 in comparisons:
            d = cohens_d(data_dict[c1][prop], data_dict[c2][prop])
            # Add asterisk for practical significance
            sig = '*' if abs(d) >= 0.3 else ''
            if abs(d) >= 0.5:
                sig = '**'
            if abs(d) >= 0.8:
                sig = '***'
            row[f'd_{c1}_v_{c2}'] = f"{d:.2f}{sig}"
        results.append(row)

    return pd.DataFrame(results)


def create_consolidated_table(data_dict):
    """Create a consolidated table matching single-pathogen format."""

    properties = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Rings', 'Fsp3']
    combos = ['SA+EC', 'SA+CA', 'EC+CA']
    comparisons = [('SA+EC', 'SA+CA'), ('SA+EC', 'EC+CA'), ('SA+CA', 'EC+CA')]

    rows = []
    for prop in properties:
        row = {'Property': prop}

        # Mean +/- SD for each combination
        for combo in combos:
            df = data_dict[combo]
            mean = df[prop].mean()
            sd = df[prop].std()
            row[f'{combo}_Mean_SD'] = f"{mean:.2f} +/- {sd:.2f}"

        # Cohen's d for pairwise comparisons
        for c1, c2 in comparisons:
            d = cohens_d(data_dict[c1][prop], data_dict[c2][prop])
            sig = '*' if abs(d) >= 0.3 else ''
            if abs(d) >= 0.5:
                sig = '**'
            row[f'd_{c1}v{c2}'] = f"{d:.2f}{sig}"

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("DUAL-ACTIVE FRAGMENT SUMMARY STATISTICS")
    print("Calculating mean +/- SD for all 8 physicochemical properties")
    print("=" * 70)

    # Load data
    print("\nLoading dual-active fragments...")
    data_dict = load_dual_fragments()

    # Sample sizes
    print("\n" + "=" * 50)
    print("SAMPLE SIZES")
    print("=" * 50)
    for combo, df in data_dict.items():
        print(f"  {combo}: N = {len(df)}")

    # Summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS (Mean +/- SD)")
    print("=" * 50)
    summary_df = calculate_summary_statistics(data_dict)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_file = OUTPUT_DIR / 'dual_active_summary_statistics_8props.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved: {summary_file}")

    # Pairwise comparisons
    print("\n" + "=" * 50)
    print("PAIRWISE COMPARISONS (Cohen's d)")
    print("* = |d| >= 0.3 (small), ** = |d| >= 0.5 (medium), *** = |d| >= 0.8 (large)")
    print("=" * 50)
    pairwise_df = calculate_pairwise_comparisons(data_dict)
    print(pairwise_df.to_string(index=False))

    # Save pairwise
    pairwise_file = OUTPUT_DIR / 'dual_active_pairwise_comparisons_8props.csv'
    pairwise_df.to_csv(pairwise_file, index=False)
    print(f"\nSaved: {pairwise_file}")

    # Consolidated table
    print("\n" + "=" * 50)
    print("CONSOLIDATED TABLE (like single-pathogen format)")
    print("=" * 50)
    consolidated_df = create_consolidated_table(data_dict)
    print(consolidated_df.to_string(index=False))

    # Save consolidated
    consolidated_file = OUTPUT_DIR / 'dual_active_consolidated_summary.csv'
    consolidated_df.to_csv(consolidated_file, index=False)
    print(f"\nSaved: {consolidated_file}")

    # Print table for manuscript
    print("\n" + "=" * 70)
    print("TABLE FOR MANUSCRIPT")
    print("=" * 70)

    combos = ['SA+EC', 'SA+CA', 'EC+CA']
    print(f"\n{'Property':<12} ", end="")
    for combo in combos:
        print(f"{combo} (N={len(data_dict[combo])}){' '*8}", end="")
    print()
    print("-" * 70)

    for prop in ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Rings', 'Fsp3']:
        print(f"{prop:<12} ", end="")
        for combo in combos:
            df = data_dict[combo]
            mean = df[prop].mean()
            sd = df[prop].std()
            print(f"{mean:.2f} +/- {sd:.2f}{' '*6}", end="")
        print()

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
