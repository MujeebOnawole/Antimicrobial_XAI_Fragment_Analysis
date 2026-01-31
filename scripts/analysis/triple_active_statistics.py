"""
Triple-Active Fragment Statistical Analysis
============================================
Generates comprehensive statistics for triple-active (broad-spectrum) fragments
including all 8 physicochemical properties with Cohen's d effect sizes.

Compares triple-active fragments against:
1. Single-pathogen fragments (SA-only, EC-only, CA-only)
2. Dual-active fragments (SA+EC, SA+CA, EC+CA)
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

# Property names
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


def load_fragment_data(filepath, category_name):
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
            props['category'] = category_name
            props_list.append(props)

    print(f"    Calculated properties for {len(props_list)} fragments")
    return pd.DataFrame(props_list)


def main():
    print("=" * 70)
    print("TRIPLE-ACTIVE FRAGMENT STATISTICAL ANALYSIS")
    print("All 8 Physicochemical Properties with Cohen's d")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define data sources - BOTH scaffolds AND substituents for consistency
    data_sources = {
        # Single-pathogen positive (scaffolds + substituents)
        'SA_only': [
            DATA_DIR / 'single_pathogen' / 'positive' / 'SA_specific_positive_scaffolds.csv',
            DATA_DIR / 'single_pathogen' / 'positive' / 'SA_specific_positive_substitutents.csv',
        ],
        'EC_only': [
            DATA_DIR / 'single_pathogen' / 'positive' / 'EC_specific_positive_scaffolds.csv',
            DATA_DIR / 'single_pathogen' / 'positive' / 'EC_specific_positive_substitutents.csv',
        ],
        'CA_only': [
            DATA_DIR / 'single_pathogen' / 'positive' / 'CA_specific_positive_scaffolds.csv',
            DATA_DIR / 'single_pathogen' / 'positive' / 'CA_specific_positive_substitutents.csv',
        ],
        # Dual-pathogen positive (scaffolds + substituents)
        'SA_EC': [
            DATA_DIR / 'dual_pathogen' / 'positive' / 'dual_SA_EC_positive_scaffolds.csv',
            DATA_DIR / 'dual_pathogen' / 'positive' / 'dual_SA_EC_positive_substitutents.csv',
        ],
        'SA_CA': [
            DATA_DIR / 'dual_pathogen' / 'positive' / 'dual_SA_CA_positive_scaffolds.csv',
            DATA_DIR / 'dual_pathogen' / 'positive' / 'dual_SA_CA_positive_substitutents.csv',
        ],
        'EC_CA': [
            DATA_DIR / 'dual_pathogen' / 'positive' / 'dual_CA_EC_positive_scaffolds.csv',
            DATA_DIR / 'dual_pathogen' / 'positive' / 'dual_CA_EC_positive_substitutents.csv',
        ],
        # Triple-active (pan-pathogen) - scaffolds + substituents
        'TRIPLE': [
            DATA_DIR / 'pan_pathogen' / 'positive' / 'Multi_positive_scaffolds.csv',
            DATA_DIR / 'pan_pathogen' / 'positive' / 'Multi_positive_substituents.csv',
        ],
    }

    # Load all data (combining scaffolds + substituents for each category)
    print("\nLoading fragment data (scaffolds + substituents)...")
    all_data = {}
    for category, filepaths in data_sources.items():
        print(f"\n{category}:")
        combined_dfs = []
        for filepath in filepaths:
            df = load_fragment_data(filepath, category)
            if df is not None and len(df) > 0:
                combined_dfs.append(df)

        if combined_dfs:
            combined_df = pd.concat(combined_dfs, ignore_index=True)
            all_data[category] = combined_df
            print(f"  Combined total: {len(combined_df)} fragments")

    if 'TRIPLE' not in all_data:
        print("\nERROR: Triple-active data not loaded")
        return

    triple_df = all_data['TRIPLE']
    print(f"\nTriple-active fragments: {len(triple_df)}")

    # =========================================================================
    # 1. Summary Statistics for Triple-Active
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRIPLE-ACTIVE SUMMARY STATISTICS")
    print("=" * 60)

    summary_rows = []
    for prop in PROPERTIES:
        values = triple_df[prop].dropna()
        summary_rows.append({
            'Property': prop,
            'N': len(values),
            'Mean': values.mean(),
            'SD': values.std(),
            'Median': values.median(),
            'Min': values.min(),
            'Max': values.max(),
            'Mean_SD': f"{values.mean():.2f} ± {values.std():.2f}"
        })
        print(f"  {prop}: {values.mean():.2f} ± {values.std():.2f} (n={len(values)})")

    summary_df = pd.DataFrame(summary_rows)
    summary_file = OUTPUT_DIR / 'triple_active_summary_statistics.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved: {summary_file.name}")

    # =========================================================================
    # 2. Pairwise Comparisons: Triple vs Single-Pathogen
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRIPLE vs SINGLE-PATHOGEN COMPARISONS")
    print("=" * 60)

    comparison_rows = []

    for single_cat in ['SA_only', 'EC_only', 'CA_only']:
        if single_cat not in all_data:
            continue

        single_df = all_data[single_cat]
        comparison_name = f"TRIPLE vs {single_cat}"
        print(f"\n{comparison_name}:")

        for prop in PROPERTIES:
            triple_vals = triple_df[prop].dropna()
            single_vals = single_df[prop].dropna()

            if len(triple_vals) < 2 or len(single_vals) < 2:
                continue

            # Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(triple_vals, single_vals, alternative='two-sided')

            # Cohen's d
            d = cohens_d(triple_vals, single_vals)

            # Fold change
            mean_triple = triple_vals.mean()
            mean_single = single_vals.mean()
            fold_change = mean_triple / mean_single if mean_single != 0 else np.inf

            comparison_rows.append({
                'Comparison': comparison_name,
                'Property': prop,
                'Triple_Mean': mean_triple,
                'Triple_SD': triple_vals.std(),
                'Triple_N': len(triple_vals),
                'Single_Mean': mean_single,
                'Single_SD': single_vals.std(),
                'Single_N': len(single_vals),
                'Cohens_d': d,
                'Effect_Size': effect_size_category(d),
                'Fold_Change': fold_change,
                'U_Statistic': statistic,
                'p_value': p_value,
            })

            sig = "***" if abs(d) >= 0.3 else ""
            print(f"  {prop}: d={d:.3f} ({effect_size_category(d)}) p={p_value:.2e} {sig}")

    # =========================================================================
    # 3. Pairwise Comparisons: Triple vs Dual-Active
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRIPLE vs DUAL-ACTIVE COMPARISONS")
    print("=" * 60)

    for dual_cat in ['SA_EC', 'SA_CA', 'EC_CA']:
        if dual_cat not in all_data:
            continue

        dual_df = all_data[dual_cat]
        comparison_name = f"TRIPLE vs {dual_cat}"
        print(f"\n{comparison_name}:")

        for prop in PROPERTIES:
            triple_vals = triple_df[prop].dropna()
            dual_vals = dual_df[prop].dropna()

            if len(triple_vals) < 2 or len(dual_vals) < 2:
                continue

            statistic, p_value = stats.mannwhitneyu(triple_vals, dual_vals, alternative='two-sided')
            d = cohens_d(triple_vals, dual_vals)

            mean_triple = triple_vals.mean()
            mean_dual = dual_vals.mean()
            fold_change = mean_triple / mean_dual if mean_dual != 0 else np.inf

            comparison_rows.append({
                'Comparison': comparison_name,
                'Property': prop,
                'Triple_Mean': mean_triple,
                'Triple_SD': triple_vals.std(),
                'Triple_N': len(triple_vals),
                'Single_Mean': mean_dual,
                'Single_SD': dual_vals.std(),
                'Single_N': len(dual_vals),
                'Cohens_d': d,
                'Effect_Size': effect_size_category(d),
                'Fold_Change': fold_change,
                'U_Statistic': statistic,
                'p_value': p_value,
            })

            sig = "***" if abs(d) >= 0.3 else ""
            print(f"  {prop}: d={d:.3f} ({effect_size_category(d)}) p={p_value:.2e} {sig}")

    # Apply FDR correction
    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df['p_corrected'] = benjamini_hochberg(comparison_df['p_value'].values)
        comparison_df['Significant'] = comparison_df['p_corrected'] < 0.05
        comparison_df['Practically_Significant'] = comparison_df['Cohens_d'].abs() >= 0.3

        # Save comparisons
        comparison_file = OUTPUT_DIR / 'triple_active_pairwise_comparisons.csv'
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\nSaved: {comparison_file.name}")

        # =====================================================================
        # 4. Significant Patterns Summary
        # =====================================================================
        print("\n" + "=" * 60)
        print("PRACTICALLY SIGNIFICANT PATTERNS (|d| >= 0.3)")
        print("=" * 60)

        sig_patterns = comparison_df[comparison_df['Practically_Significant']].copy()
        sig_patterns = sig_patterns.sort_values('Cohens_d', key=abs, ascending=False)

        if len(sig_patterns) > 0:
            for _, row in sig_patterns.iterrows():
                direction = "higher" if row['Cohens_d'] > 0 else "lower"
                print(f"  {row['Property']}: {row['Comparison']} | d={row['Cohens_d']:.3f} ({row['Effect_Size']}) | Triple {direction}")

            sig_file = OUTPUT_DIR / 'triple_active_significant_patterns.csv'
            sig_patterns.to_csv(sig_file, index=False)
            print(f"\nSaved: {sig_file.name}")
        else:
            print("  No practically significant patterns found")

    # =========================================================================
    # 5. All Categories Summary Table
    # =========================================================================
    print("\n" + "=" * 60)
    print("ALL CATEGORIES SUMMARY TABLE")
    print("=" * 60)

    all_summary_rows = []
    for cat_name, cat_df in all_data.items():
        for prop in PROPERTIES:
            values = cat_df[prop].dropna()
            all_summary_rows.append({
                'Category': cat_name,
                'Property': prop,
                'N': len(values),
                'Mean': values.mean(),
                'SD': values.std(),
                'Mean_SD': f"{values.mean():.2f} ± {values.std():.2f}"
            })

    all_summary_df = pd.DataFrame(all_summary_rows)
    all_summary_file = OUTPUT_DIR / 'triple_active_all_categories_summary.csv'
    all_summary_df.to_csv(all_summary_file, index=False)
    print(f"Saved: {all_summary_file.name}")

    # Print summary table
    print("\nMean ± SD by Category and Property:")
    pivot = all_summary_df.pivot(index='Property', columns='Category', values='Mean_SD')
    print(pivot.to_string())

    print("\n" + "=" * 70)
    print("TRIPLE-ACTIVE STATISTICAL ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
