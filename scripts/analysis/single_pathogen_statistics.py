"""
Single-Pathogen Fragment Statistical Analysis
==============================================
Generates comprehensive statistics for single-pathogen fragments
including all 8 physicochemical properties with Cohen's d effect sizes.

Compares:
- S. aureus (SA) vs E. coli (EC)
- S. aureus (SA) vs C. albicans (CA)
- E. coli (EC) vs C. albicans (CA)
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
    print("WARNING: RDKit not available")

# Paths
BASE_DIR = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results" / "statistics"

# All 8 properties matching radar plots
PROPERTIES = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Rings', 'Fsp3']


def calculate_properties(smiles):
    """Calculate all 8 physicochemical properties from SMILES."""
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
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def effect_size_category(d):
    """Categorize effect size based on Cohen's conventions."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return 'negligible'
    elif d_abs < 0.5:
        return 'small'
    elif d_abs < 0.8:
        return 'medium'
    else:
        return 'large'


def benjamini_hochberg(p_values):
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Calculate adjusted p-values
    adjusted = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        adjusted[idx] = sorted_p[i] * n / (i + 1)

    # Ensure monotonicity
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    return adjusted


def load_fragment_data(filepath, pathogen_name):
    """Load fragment data and calculate properties."""
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} fragments from {filepath.name}")

    # Find SMILES column
    smiles_col = None
    for col in ['fragment_smiles', 'smiles', 'SMILES']:
        if col in df.columns:
            smiles_col = col
            break

    if smiles_col is None:
        print(f"    No SMILES column found")
        return None

    # Calculate properties
    props_list = []
    for idx, row in df.iterrows():
        props = calculate_properties(row[smiles_col])
        if props:
            props['pathogen'] = pathogen_name
            props_list.append(props)

    print(f"    Calculated properties for {len(props_list)} fragments")
    return pd.DataFrame(props_list)


def main():
    print("=" * 70)
    print("SINGLE-PATHOGEN FRAGMENT STATISTICAL ANALYSIS")
    print("All 8 Physicochemical Properties with Cohen's d")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define data sources for single-pathogen specific fragments
    data_sources = {
        'SA': DATA_DIR / 'single_pathogen' / 'positive' / 'SA_specific_positive_scaffolds.csv',
        'EC': DATA_DIR / 'single_pathogen' / 'positive' / 'EC_specific_positive_scaffolds.csv',
        'CA': DATA_DIR / 'single_pathogen' / 'positive' / 'CA_specific_positive_scaffolds.csv',
    }

    # Load all data
    print("\nLoading fragment data...")
    all_data = {}
    for pathogen, filepath in data_sources.items():
        print(f"\n{pathogen}:")
        df = load_fragment_data(filepath, pathogen)
        if df is not None and len(df) > 0:
            all_data[pathogen] = df

    if len(all_data) < 2:
        print("\nERROR: Need at least 2 pathogen datasets for comparisons")
        return

    # =========================================================================
    # 1. Summary Statistics for Each Pathogen
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS BY PATHOGEN")
    print("=" * 60)

    summary_rows = []
    for pathogen, df in all_data.items():
        print(f"\n{pathogen} (n={len(df)}):")
        for prop in PROPERTIES:
            values = df[prop].dropna()
            summary_rows.append({
                'Pathogen': pathogen,
                'Property': prop,
                'N': len(values),
                'Mean': values.mean(),
                'SD': values.std(),
                'Median': values.median(),
                'Min': values.min(),
                'Max': values.max(),
                'Mean_SD': f"{values.mean():.2f} +/- {values.std():.2f}"
            })
            print(f"  {prop}: {values.mean():.2f} +/- {values.std():.2f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_file = OUTPUT_DIR / 'single_pathogen_summary_statistics_8props.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved: {summary_file.name}")

    # =========================================================================
    # 2. Pairwise Comparisons with Cohen's d
    # =========================================================================
    print("\n" + "=" * 60)
    print("PAIRWISE COMPARISONS (Cohen's d)")
    print("=" * 60)

    comparison_pairs = [
        ('SA', 'EC'),
        ('SA', 'CA'),
        ('EC', 'CA'),
    ]

    comparison_rows = []

    for p1, p2 in comparison_pairs:
        if p1 not in all_data or p2 not in all_data:
            continue

        df1 = all_data[p1]
        df2 = all_data[p2]
        comparison_name = f"{p1} vs {p2}"
        print(f"\n{comparison_name}:")

        for prop in PROPERTIES:
            vals1 = df1[prop].dropna()
            vals2 = df2[prop].dropna()

            if len(vals1) < 2 or len(vals2) < 2:
                continue

            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')

            # Cohen's d
            d = cohens_d(vals1, vals2)

            # Means
            mean1 = vals1.mean()
            mean2 = vals2.mean()

            # Fold change
            fold_change = mean1 / mean2 if mean2 != 0 else np.inf

            # Direction
            direction = "higher" if d > 0 else "lower"

            comparison_rows.append({
                'Comparison': comparison_name,
                'Property': prop,
                f'{p1}_Mean': mean1,
                f'{p1}_SD': vals1.std(),
                f'{p1}_N': len(vals1),
                f'{p2}_Mean': mean2,
                f'{p2}_SD': vals2.std(),
                f'{p2}_N': len(vals2),
                'Cohens_d': d,
                'Effect_Size': effect_size_category(d),
                'Direction': f"{p1} {direction}",
                'Fold_Change': fold_change,
                'U_Statistic': statistic,
                'p_value': p_value,
            })

            sig_marker = "***" if abs(d) >= 0.5 else ("**" if abs(d) >= 0.3 else ("*" if abs(d) >= 0.2 else ""))
            print(f"  {prop}: d={d:.3f} ({effect_size_category(d)}) p={p_value:.2e} {sig_marker}")

    # Apply FDR correction
    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df['p_corrected'] = benjamini_hochberg(comparison_df['p_value'].values)
        comparison_df['Significant'] = comparison_df['p_corrected'] < 0.05
        comparison_df['Practically_Significant'] = comparison_df['Cohens_d'].abs() >= 0.2

        # Save comparisons
        comparison_file = OUTPUT_DIR / 'single_pathogen_pairwise_comparisons_8props.csv'
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\nSaved: {comparison_file.name}")

        # =====================================================================
        # 3. Significant Patterns Summary
        # =====================================================================
        print("\n" + "=" * 60)
        print("PRACTICALLY SIGNIFICANT PATTERNS (|d| >= 0.2)")
        print("=" * 60)

        sig_patterns = comparison_df[comparison_df['Practically_Significant']].copy()
        sig_patterns = sig_patterns.sort_values('Cohens_d', key=abs, ascending=False)

        if len(sig_patterns) > 0:
            for _, row in sig_patterns.iterrows():
                print(f"  {row['Property']}: {row['Comparison']} | d={row['Cohens_d']:.3f} ({row['Effect_Size']}) | {row['Direction']}")

            sig_file = OUTPUT_DIR / 'single_pathogen_significant_patterns_8props.csv'
            sig_patterns.to_csv(sig_file, index=False)
            print(f"\nSaved: {sig_file.name}")
        else:
            print("  No practically significant patterns found")

    # =========================================================================
    # 4. Formatted Table for Publication
    # =========================================================================
    print("\n" + "=" * 60)
    print("FORMATTED TABLE (Mean +/- SD)")
    print("=" * 60)

    # Create pivot table
    pivot_data = []
    for prop in PROPERTIES:
        row = {'Property': prop}
        for pathogen in ['SA', 'EC', 'CA']:
            if pathogen in all_data:
                values = all_data[pathogen][prop].dropna()
                row[pathogen] = f"{values.mean():.2f} +/- {values.std():.2f}"
            else:
                row[pathogen] = "N/A"
        pivot_data.append(row)

    pivot_df = pd.DataFrame(pivot_data)
    print(f"\n{'Property':<12} {'S. aureus (G+)':<20} {'E. coli (G-)':<20} {'C. albicans':<20}")
    print("-" * 72)
    for _, row in pivot_df.iterrows():
        print(f"{row['Property']:<12} {row.get('SA', 'N/A'):<20} {row.get('EC', 'N/A'):<20} {row.get('CA', 'N/A'):<20}")

    pivot_file = OUTPUT_DIR / 'single_pathogen_formatted_table_8props.csv'
    pivot_df.to_csv(pivot_file, index=False)
    print(f"\nSaved: {pivot_file.name}")

    # =========================================================================
    # 5. Effect Size Summary Matrix
    # =========================================================================
    print("\n" + "=" * 60)
    print("COHEN'S d EFFECT SIZE MATRIX")
    print("=" * 60)

    effect_matrix = []
    for prop in PROPERTIES:
        row = {'Property': prop}
        for p1, p2 in comparison_pairs:
            comp_name = f"{p1} vs {p2}"
            match = comparison_df[(comparison_df['Comparison'] == comp_name) &
                                  (comparison_df['Property'] == prop)]
            if len(match) > 0:
                d = match.iloc[0]['Cohens_d']
                effect = match.iloc[0]['Effect_Size']
                row[f'{p1}_vs_{p2}'] = f"{d:.3f} ({effect[0]})"
            else:
                row[f'{p1}_vs_{p2}'] = "N/A"
        effect_matrix.append(row)

    effect_df = pd.DataFrame(effect_matrix)
    print(f"\n{'Property':<12} {'SA vs EC':<18} {'SA vs CA':<18} {'EC vs CA':<18}")
    print("-" * 66)
    for _, row in effect_df.iterrows():
        print(f"{row['Property']:<12} {row.get('SA_vs_EC', 'N/A'):<18} {row.get('SA_vs_CA', 'N/A'):<18} {row.get('EC_vs_CA', 'N/A'):<18}")

    effect_file = OUTPUT_DIR / 'single_pathogen_cohens_d_matrix_8props.csv'
    effect_df.to_csv(effect_file, index=False)
    print(f"\nSaved: {effect_file.name}")

    print("\n" + "=" * 70)
    print("SINGLE-PATHOGEN STATISTICAL ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files in: {OUTPUT_DIR}")
    print("\nFiles generated:")
    print("  - single_pathogen_summary_statistics_8props.csv")
    print("  - single_pathogen_pairwise_comparisons_8props.csv")
    print("  - single_pathogen_significant_patterns_8props.csv")
    print("  - single_pathogen_formatted_table_8props.csv")
    print("  - single_pathogen_cohens_d_matrix_8props.csv")


if __name__ == '__main__':
    main()
