"""
Extract Comprehensive Physicochemical Property Statistics
For manuscript supporting tables - Results section

This script extracts summary statistics and pairwise comparisons for
positive fragments across S. aureus, E. coli, and C. albicans.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Set paths
BASE_DIR = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis")
DATA_DIR = BASE_DIR / "data" / "single_pathogen" / "positive"
RESULTS_DIR = BASE_DIR / "results" / "statistics"
OUTPUT_DIR = BASE_DIR / "results" / "statistics"

# Property mapping (statistical_summary.csv uses different names)
PROPERTY_MAP = {
    'MW': 'MW',
    'LogP': 'LogP',
    'TPSA': 'TPSA',
    'HBD': 'HBD',
    'HBA': 'HBA',
    'NumRotatableBonds': 'Rotatable Bonds',
    'NumAromaticRings': 'Aromatic Rings',
    'BertzCT': 'Bertz Complexity'
}

# Read the existing statistical summary
print("=" * 80)
print("COMPREHENSIVE PHYSICOCHEMICAL PROPERTY STATISTICS")
print("Single-Pathogen Positive Fragments Analysis")
print("=" * 80)

# Load the statistical summary file
stat_summary = pd.read_csv(RESULTS_DIR / "statistical_summary.csv")
print(f"\nLoaded statistical summary with {len(stat_summary)} comparisons")

# Filter to the 8 key properties
key_properties = ['MW', 'LogP', 'TPSA', 'HBA', 'HBD', 'NumRotatableBonds', 'NumAromaticRings', 'BertzCT']
stat_filtered = stat_summary[stat_summary['property'].isin(key_properties)].copy()

# =============================================================================
# SECTION 1: SUMMARY STATISTICS TABLE
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 1: SUMMARY STATISTICS BY PATHOGEN")
print("=" * 80)

# Extract unique values for each pathogen from pairwise comparisons
summary_stats = []

for prop in key_properties:
    prop_data = stat_filtered[stat_filtered['property'] == prop]

    # Extract SA data (from SA vs EC comparison)
    sa_row = prop_data[(prop_data['pathogen1'] == 'SA') & (prop_data['pathogen2'] == 'EC')].iloc[0]
    sa_mean = sa_row['mean1']
    sa_std = sa_row['std1']
    sa_n = int(sa_row['n1'])

    # Extract EC data (from SA vs EC comparison)
    ec_mean = sa_row['mean2']
    ec_std = sa_row['std2']
    ec_n = int(sa_row['n2'])

    # Extract CA data (from SA vs CA comparison)
    ca_row = prop_data[(prop_data['pathogen1'] == 'SA') & (prop_data['pathogen2'] == 'CA')].iloc[0]
    ca_mean = ca_row['mean2']
    ca_std = ca_row['std2']
    ca_n = int(ca_row['n2'])

    # Add to summary list
    for pathogen, mean, std, n in [('SA', sa_mean, sa_std, sa_n),
                                    ('EC', ec_mean, ec_std, ec_n),
                                    ('CA', ca_mean, ca_std, ca_n)]:
        summary_stats.append({
            'Property': PROPERTY_MAP.get(prop, prop),
            'Pathogen': pathogen,
            'Mean': mean,
            'SD': std,
            'N': n,
            'Mean_SD': f"{mean:.2f} ± {std:.2f}"
        })

summary_df = pd.DataFrame(summary_stats)

# Print summary table
print("\nSummary Statistics (Mean ± SD) by Pathogen:")
print("-" * 80)

# Pivot for display
pivot_display = summary_df.pivot(index='Property', columns='Pathogen', values='Mean_SD')
pivot_display = pivot_display[['SA', 'EC', 'CA']]  # Reorder columns

# Also create numeric pivot for calculations
pivot_mean = summary_df.pivot(index='Property', columns='Pathogen', values='Mean')
pivot_mean = pivot_mean[['SA', 'EC', 'CA']]

pivot_sd = summary_df.pivot(index='Property', columns='Pathogen', values='SD')
pivot_sd = pivot_sd[['SA', 'EC', 'CA']]

# Get sample sizes
sample_sizes = summary_df.groupby('Pathogen')['N'].first()
print(f"\nSample sizes: SA={sample_sizes['SA']}, EC={sample_sizes['EC']}, CA={sample_sizes['CA']}")

print("\n" + pivot_display.to_string())

# =============================================================================
# SECTION 2: PAIRWISE COMPARISON STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 2: PAIRWISE COMPARISON STATISTICS")
print("=" * 80)

pairwise_results = []

for prop in key_properties:
    prop_data = stat_filtered[stat_filtered['property'] == prop]

    for _, row in prop_data.iterrows():
        comparison = f"{row['pathogen1']} vs {row['pathogen2']}"

        # Determine practical significance
        cohens_d = row['cohens_d']
        practically_sig = "Yes" if abs(cohens_d) >= 0.3 else "No"

        # Calculate fold difference
        if row['mean2'] != 0:
            fold_diff = row['mean1'] / row['mean2']
        else:
            fold_diff = np.nan

        mean_diff = row['mean1'] - row['mean2']

        pairwise_results.append({
            'Property': PROPERTY_MAP.get(prop, prop),
            'Comparison': comparison,
            "Cohen's d": cohens_d,
            'p-value': row['p_value'],
            'FDR p-value': row['p_corrected'],
            'Practical Sig': practically_sig,
            'Mean Diff': mean_diff,
            'Fold Diff': fold_diff,
            'Mean_1': row['mean1'],
            'Mean_2': row['mean2'],
            'Stat Sig': row['significant'],
            'Both Sig': (row['significant'] and abs(cohens_d) >= 0.3)
        })

pairwise_df = pd.DataFrame(pairwise_results)

print("\nPairwise Comparison Results:")
print("-" * 120)
print(f"{'Property':<20} {'Comparison':<12} {'Cohen d':>10} {'p-value':>12} {'FDR p':>12} {'Pract Sig':>10} {'Mean Diff':>12} {'Fold':>8}")
print("-" * 120)

for _, row in pairwise_df.iterrows():
    p_str = f"{row['p-value']:.2e}" if row['p-value'] < 0.001 else f"{row['p-value']:.3f}"
    fdr_str = f"{row['FDR p-value']:.2e}" if row['FDR p-value'] < 0.001 else f"{row['FDR p-value']:.3f}"
    cohens_d = row["Cohen's d"]
    prop = row['Property']
    comp = row['Comparison']
    prac_sig = row['Practical Sig']
    mean_diff = row['Mean Diff']
    fold_diff = row['Fold Diff']
    print(f"{prop:<20} {comp:<12} {cohens_d:>10.3f} {p_str:>12} {fdr_str:>12} {prac_sig:>10} {mean_diff:>12.2f} {fold_diff:>8.2f}x")

# =============================================================================
# SECTION 3: PRACTICALLY SIGNIFICANT COMPARISONS (|d| >= 0.3)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 3: PRACTICALLY SIGNIFICANT COMPARISONS (|Cohen's d| >= 0.3)")
print("=" * 80)

sig_comparisons = pairwise_df[pairwise_df['Both Sig'] == True].copy()
sig_comparisons = sig_comparisons.sort_values("Cohen's d", key=abs, ascending=False)

print(f"\n{len(sig_comparisons)} comparisons met both statistical (FDR p < 0.05) and practical (|d| >= 0.3) significance:\n")

print(f"{'Property':<20} {'Comparison':<12} {'Cohen d':>10} {'Mean1':>10} {'Mean2':>10} {'Interpretation'}")
print("-" * 100)

for _, row in sig_comparisons.iterrows():
    d = row["Cohen's d"]
    comp_parts = row['Comparison'].split(' vs ')
    if d > 0:
        interp = f"{comp_parts[0]} higher"
    else:
        interp = f"{comp_parts[1]} higher"
    prop = row['Property']
    comp = row['Comparison']
    mean1 = row['Mean_1']
    mean2 = row['Mean_2']
    print(f"{prop:<20} {comp:<12} {d:>10.3f} {mean1:>10.2f} {mean2:>10.2f} {interp}")

# =============================================================================
# SECTION 4: SPECIFIC VALUES FOR MANUSCRIPT TEXT
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 4: SPECIFIC VALUES FOR MANUSCRIPT TEXT")
print("=" * 80)

# Extract specific values
sa_logp_row = stat_filtered[(stat_filtered['property'] == 'LogP') & (stat_filtered['pathogen1'] == 'SA') & (stat_filtered['pathogen2'] == 'EC')].iloc[0]
sa_tpsa_row = stat_filtered[(stat_filtered['property'] == 'TPSA') & (stat_filtered['pathogen1'] == 'SA') & (stat_filtered['pathogen2'] == 'EC')].iloc[0]
sa_mw_row = stat_filtered[(stat_filtered['property'] == 'MW') & (stat_filtered['pathogen1'] == 'SA') & (stat_filtered['pathogen2'] == 'EC')].iloc[0]
sa_hbd_row = stat_filtered[(stat_filtered['property'] == 'HBD') & (stat_filtered['pathogen1'] == 'SA') & (stat_filtered['pathogen2'] == 'EC')].iloc[0]

hbd_sa_ca = stat_filtered[(stat_filtered['property'] == 'HBD') & (stat_filtered['pathogen1'] == 'SA') & (stat_filtered['pathogen2'] == 'CA')].iloc[0]
hbd_ec_ca = stat_filtered[(stat_filtered['property'] == 'HBD') & (stat_filtered['pathogen1'] == 'EC') & (stat_filtered['pathogen2'] == 'CA')].iloc[0]

logp_sa_ec = pairwise_df[(pairwise_df['Property'] == 'LogP') & (pairwise_df['Comparison'] == 'SA vs EC')].iloc[0]
tpsa_sa_ec = pairwise_df[(pairwise_df['Property'] == 'TPSA') & (pairwise_df['Comparison'] == 'SA vs EC')].iloc[0]

print("\n--- S. aureus Fragments ---")
print(f"LogP: Mean = {sa_logp_row['mean1']:.2f} ± {sa_logp_row['std1']:.2f}")
print(f"TPSA: Mean = {sa_tpsa_row['mean1']:.2f} ± {sa_tpsa_row['std1']:.2f} A^2")
print(f"MW: Mean = {sa_mw_row['mean1']:.1f} ± {sa_mw_row['std1']:.1f} Da")
print(f"HBD: Mean = {sa_hbd_row['mean1']:.2f} ± {sa_hbd_row['std1']:.2f}")
print(f"N = {int(sa_logp_row['n1'])} fragments")

print("\n--- E. coli Fragments ---")
print(f"LogP: Mean = {sa_logp_row['mean2']:.2f} ± {sa_logp_row['std2']:.2f}")
print(f"TPSA: Mean = {sa_tpsa_row['mean2']:.2f} ± {sa_tpsa_row['std2']:.2f} A^2")
print(f"HBD: Mean = {sa_hbd_row['mean2']:.2f} ± {sa_hbd_row['std2']:.2f}")
print(f"N = {int(sa_logp_row['n2'])} fragments")

ca_logp_row = stat_filtered[(stat_filtered['property'] == 'LogP') & (stat_filtered['pathogen1'] == 'SA') & (stat_filtered['pathogen2'] == 'CA')].iloc[0]
ca_tpsa_row = stat_filtered[(stat_filtered['property'] == 'TPSA') & (stat_filtered['pathogen1'] == 'SA') & (stat_filtered['pathogen2'] == 'CA')].iloc[0]
ca_hbd_row = stat_filtered[(stat_filtered['property'] == 'HBD') & (stat_filtered['pathogen1'] == 'SA') & (stat_filtered['pathogen2'] == 'CA')].iloc[0]

print("\n--- C. albicans Fragments ---")
print(f"LogP: Mean = {ca_logp_row['mean2']:.2f} ± {ca_logp_row['std2']:.2f}")
print(f"TPSA: Mean = {ca_tpsa_row['mean2']:.2f} ± {ca_tpsa_row['std2']:.2f} A^2")
print(f"HBD: Mean = {ca_hbd_row['mean2']:.2f} ± {ca_hbd_row['std2']:.2f}")
print(f"N = {int(ca_logp_row['n2'])} fragments")

print("\n--- Key Pairwise Comparisons ---")
logp_d = logp_sa_ec["Cohen's d"]
logp_fold = logp_sa_ec['Fold Diff']
tpsa_d = tpsa_sa_ec["Cohen's d"]
tpsa_fold = tpsa_sa_ec['Fold Diff']
hbd_sa_ca_d = hbd_sa_ca['cohens_d']
hbd_ec_ca_d = hbd_ec_ca['cohens_d']
print(f"LogP (SA vs EC): d = {logp_d:.3f}, Fold diff = {logp_fold:.2f}x")
print(f"TPSA (SA vs EC): d = {tpsa_d:.3f}, SA/EC ratio = {tpsa_fold:.2f}")
print(f"HBD (SA vs CA): d = {hbd_sa_ca_d:.3f}")
print(f"HBD (EC vs CA): d = {hbd_ec_ca_d:.3f}")

# =============================================================================
# SECTION 5: VERIFICATION AGAINST EXPECTED VALUES
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 5: VERIFICATION AGAINST EXPECTED VALUES")
print("=" * 80)

expected_values = {
    'LogP (SA vs EC)': 0.42,
    'TPSA (SA vs EC)': 0.32,
    'HBD (SA vs CA)': 0.34,
    'HBD (EC vs CA)': 0.66
}

calculated_values = {
    'LogP (SA vs EC)': logp_d,
    'TPSA (SA vs EC)': abs(tpsa_d),  # TPSA is lower for SA, so negative d
    'HBD (SA vs CA)': hbd_sa_ca_d,
    'HBD (EC vs CA)': hbd_ec_ca_d
}

print("\nComparison of Expected vs Calculated Cohen's d values:")
print("-" * 70)
print(f"{'Comparison':<25} {'Expected':>12} {'Calculated':>12} {'Difference':>12} {'Status'}")
print("-" * 70)

all_match = True
for comp, expected in expected_values.items():
    calculated = calculated_values[comp]
    diff = abs(calculated - expected)
    status = "[OK] MATCH" if diff < 0.05 else "[!] DIFFERS"
    if diff >= 0.05:
        all_match = False
    print(f"{comp:<25} {expected:>12.2f} {calculated:>12.3f} {diff:>12.3f} {status}")

if all_match:
    print("\n[OK] All calculated values match expected values within tolerance (±0.05)")
else:
    print("\n[!] Some values differ from expected - please verify")

# =============================================================================
# SAVE OUTPUT FILES
# =============================================================================
print("\n" + "=" * 80)
print("SAVING OUTPUT FILES")
print("=" * 80)

# 1. Summary statistics CSV
summary_output = summary_df[['Property', 'Pathogen', 'Mean', 'SD', 'N']].copy()
summary_output['Mean ± SD'] = summary_output.apply(lambda x: f"{x['Mean']:.2f} ± {x['SD']:.2f}", axis=1)
summary_output.to_csv(OUTPUT_DIR / "summary_statistics_table.csv", index=False)
print(f"[OK] Saved: summary_statistics_table.csv")

# 2. Pairwise comparisons CSV
pairwise_output = pairwise_df[['Property', 'Comparison', "Cohen's d", 'p-value', 'FDR p-value',
                               'Practical Sig', 'Mean Diff', 'Fold Diff', 'Mean_1', 'Mean_2']].copy()
pairwise_output.to_csv(OUTPUT_DIR / "pairwise_comparisons_table.csv", index=False)
print(f"[OK] Saved: pairwise_comparisons_table.csv")

# 3. Specific values text file
with open(OUTPUT_DIR / "specific_values_for_text.txt", 'w') as f:
    f.write("SPECIFIC VALUES FOR MANUSCRIPT TEXT\n")
    f.write("=" * 60 + "\n\n")

    f.write("S. aureus Positive Fragments (N = 2,332):\n")
    f.write(f"  LogP: {sa_logp_row['mean1']:.2f} ± {sa_logp_row['std1']:.2f}\n")
    f.write(f"  TPSA: {sa_tpsa_row['mean1']:.1f} ± {sa_tpsa_row['std1']:.1f} A^2\n")
    f.write(f"  MW: {sa_mw_row['mean1']:.1f} ± {sa_mw_row['std1']:.1f} Da\n")
    f.write(f"  HBD: {sa_hbd_row['mean1']:.2f} ± {sa_hbd_row['std1']:.2f}\n\n")

    f.write("E. coli Positive Fragments (N = 537):\n")
    f.write(f"  LogP: {sa_logp_row['mean2']:.2f} ± {sa_logp_row['std2']:.2f}\n")
    f.write(f"  TPSA: {sa_tpsa_row['mean2']:.1f} ± {sa_tpsa_row['std2']:.1f} A^2\n")
    f.write(f"  HBD: {sa_hbd_row['mean2']:.2f} ± {sa_hbd_row['std2']:.2f}\n\n")

    f.write("C. albicans Positive Fragments (N = 1,234):\n")
    f.write(f"  LogP: {ca_logp_row['mean2']:.2f} ± {ca_logp_row['std2']:.2f}\n")
    f.write(f"  TPSA: {ca_tpsa_row['mean2']:.1f} ± {ca_tpsa_row['std2']:.1f} A^2\n")
    f.write(f"  HBD: {ca_hbd_row['mean2']:.2f} ± {ca_hbd_row['std2']:.2f}\n\n")

    f.write("KEY EFFECT SIZES (Cohen's d):\n")
    f.write(f"  LogP (SA vs EC): d = {logp_d:.3f}\n")
    f.write(f"  TPSA (SA vs EC): d = {tpsa_d:.3f}\n")
    f.write(f"  HBD (SA vs CA): d = {hbd_sa_ca_d:.3f}\n")
    f.write(f"  HBD (EC vs CA): d = {hbd_ec_ca_d:.3f}\n\n")

    f.write("FOLD DIFFERENCES:\n")
    f.write(f"  LogP (SA/EC): {logp_fold:.2f}x\n")
    f.write(f"  TPSA (SA/EC): {tpsa_fold:.2f}x\n")

print(f"[OK] Saved: specific_values_for_text.txt")

# =============================================================================
# MARKDOWN OUTPUT
# =============================================================================
print("\n" + "=" * 80)
print("MARKDOWN TABLES FOR MANUSCRIPT")
print("=" * 80)

print("\n### Table 1: Summary Statistics by Pathogen\n")
print("| Property | S. aureus (N=2,332) | E. coli (N=537) | C. albicans (N=1,234) |")
print("|----------|---------------------|-----------------|----------------------|")

for prop in ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'NumRotatableBonds', 'NumAromaticRings', 'BertzCT']:
    prop_name = PROPERTY_MAP.get(prop, prop)
    sa_val = summary_df[(summary_df['Property'] == prop_name) & (summary_df['Pathogen'] == 'SA')]['Mean_SD'].values[0]
    ec_val = summary_df[(summary_df['Property'] == prop_name) & (summary_df['Pathogen'] == 'EC')]['Mean_SD'].values[0]
    ca_val = summary_df[(summary_df['Property'] == prop_name) & (summary_df['Pathogen'] == 'CA')]['Mean_SD'].values[0]
    print(f"| {prop_name} | {sa_val} | {ec_val} | {ca_val} |")

print("\n### Table 2: Pairwise Comparisons with Practical Significance\n")
print("| Property | Comparison | Cohen's d | p-value (FDR) | Practical Sig |")
print("|----------|------------|-----------|---------------|---------------|")

for _, row in sig_comparisons.iterrows():
    p_str = f"{row['FDR p-value']:.2e}" if row['FDR p-value'] < 0.001 else f"{row['FDR p-value']:.3f}"
    d_val = row["Cohen's d"]
    print(f"| {row['Property']} | {row['Comparison']} | {d_val:.3f} | {p_str} | Yes |")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
