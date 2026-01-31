"""
Comprehensive Fragment Statistics - All Categories
===================================================
Creates ONE compact table with mean ± SD for:
- Single-pathogen (SA, EC, CA) - from summary_statistics_table.csv
- Dual-active (SA+EC, SA+CA, EC+CA)
- Triple-active (SA+EC+CA)

Note: Single-pathogen uses "Aromatic Rings" not total "Rings", and lacks Fsp3.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis")
STATS_DIR = BASE_DIR / "results" / "statistics"
OUTPUT_DIR = STATS_DIR


def load_all_data():
    """Load all statistics from their respective files."""

    # Single-pathogen (from summary_statistics_table.csv - the manuscript source)
    single_file = STATS_DIR / "summary_statistics_table.csv"
    single_df = pd.read_csv(single_file)

    # Dual-active
    dual_file = STATS_DIR / "dual_active_summary_statistics_8props.csv"
    dual_df = pd.read_csv(dual_file)

    # Triple-active
    triple_file = STATS_DIR / "triple_active_summary_statistics.csv"
    triple_df = pd.read_csv(triple_file)

    return single_df, dual_df, triple_df


def create_comprehensive_table():
    """Create the comprehensive table."""

    single_df, dual_df, triple_df = load_all_data()

    # Get N values
    n_single = {'SA': 2332, 'EC': 537, 'CA': 1234}
    n_dual = {
        'SA+EC': int(dual_df['SA+EC_N'].iloc[0]),
        'SA+CA': int(dual_df['SA+CA_N'].iloc[0]),
        'EC+CA': int(dual_df['EC+CA_N'].iloc[0])
    }
    n_triple = int(triple_df['N'].iloc[0])

    # Build rows for common properties
    properties_map = {
        'MW': 'MW',
        'LogP': 'LogP',
        'TPSA': 'TPSA',
        'HBD': 'HBD',
        'HBA': 'HBA',
        'Rotatable Bonds': 'RotBonds',
        'Aromatic Rings': 'Rings',
    }

    rows = []

    # Process each property
    for single_prop, std_prop in properties_map.items():
        row = {'Property': std_prop}

        # Single-pathogen
        for pathogen in ['SA', 'EC', 'CA']:
            mask = (single_df['Property'] == single_prop) & (single_df['Pathogen'] == pathogen)
            if mask.any():
                row[pathogen] = single_df.loc[mask, 'Mean ± SD'].values[0]
            else:
                row[pathogen] = 'N/A'

        # Dual-active
        dual_row = dual_df[dual_df['Property'] == std_prop]
        if not dual_row.empty:
            row['SA+EC'] = dual_row['SA+EC_Mean_SD'].values[0]
            row['SA+CA'] = dual_row['SA+CA_Mean_SD'].values[0]
            row['EC+CA'] = dual_row['EC+CA_Mean_SD'].values[0]
        else:
            row['SA+EC'] = row['SA+CA'] = row['EC+CA'] = 'N/A'

        # Triple-active
        triple_row = triple_df[triple_df['Property'] == std_prop]
        if not triple_row.empty:
            row['Triple'] = triple_row['Mean_SD'].values[0]
        else:
            row['Triple'] = 'N/A'

        rows.append(row)

    # Add Fsp3 (only in dual and triple)
    fsp3_row = {'Property': 'Fsp3', 'SA': 'N/A*', 'EC': 'N/A*', 'CA': 'N/A*'}
    dual_fsp3 = dual_df[dual_df['Property'] == 'Fsp3']
    if not dual_fsp3.empty:
        fsp3_row['SA+EC'] = dual_fsp3['SA+EC_Mean_SD'].values[0]
        fsp3_row['SA+CA'] = dual_fsp3['SA+CA_Mean_SD'].values[0]
        fsp3_row['EC+CA'] = dual_fsp3['EC+CA_Mean_SD'].values[0]
    triple_fsp3 = triple_df[triple_df['Property'] == 'Fsp3']
    if not triple_fsp3.empty:
        fsp3_row['Triple'] = triple_fsp3['Mean_SD'].values[0]
    rows.append(fsp3_row)

    # Create DataFrame
    cols = ['Property', 'SA', 'EC', 'CA', 'SA+EC', 'SA+CA', 'EC+CA', 'Triple']
    df = pd.DataFrame(rows)[cols]

    return df, n_single, n_dual, n_triple


def main():
    print("=" * 120)
    print("COMPREHENSIVE FRAGMENT STATISTICS - ALL CATEGORIES")
    print("=" * 120)

    df, n_single, n_dual, n_triple = create_comprehensive_table()

    # Print header with N values
    print("\n" + "-" * 120)
    print("Property   | --- SINGLE-PATHOGEN ---                        | --- DUAL-ACTIVE ---                               | TRIPLE")
    sa_hdr = f"SA (N={n_single['SA']})"
    ec_hdr = f"EC (N={n_single['EC']})"
    ca_hdr = f"CA (N={n_single['CA']})"
    saec_hdr = f"SA+EC (N={n_dual['SA+EC']})"
    saca_hdr = f"SA+CA (N={n_dual['SA+CA']})"
    ecca_hdr = f"EC+CA (N={n_dual['EC+CA']})"
    triple_hdr = f"(N={n_triple})"
    print(f"           | {sa_hdr:<14} {ec_hdr:<14} {ca_hdr:<14} | {saec_hdr:<16} {saca_hdr:<15} {ecca_hdr:<15} | {triple_hdr}")
    print("-" * 120)

    # Print data
    for _, row in df.iterrows():
        print(f"{row['Property']:<10} | {str(row['SA']):<14} {str(row['EC']):<14} {str(row['CA']):<14} | {str(row['SA+EC']):<16} {str(row['SA+CA']):<15} {str(row['EC+CA']):<15} | {row['Triple']}")

    print("-" * 120)
    print("*N/A: Fsp3 not available in single-pathogen summary (uses Aromatic Rings instead of total Rings)")

    # Save to CSV
    output_file = OUTPUT_DIR / 'comprehensive_all_categories_statistics.csv'

    # Add N values to column names for clarity
    df_save = df.copy()
    df_save.columns = [
        'Property',
        f'SA (N={n_single["SA"]})',
        f'EC (N={n_single["EC"]})',
        f'CA (N={n_single["CA"]})',
        f'SA+EC (N={n_dual["SA+EC"]})',
        f'SA+CA (N={n_dual["SA+CA"]})',
        f'EC+CA (N={n_dual["EC+CA"]})',
        f'Triple (N={n_triple})'
    ]
    df_save.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # Also create markdown table for easy copy-paste
    print("\n" + "=" * 120)
    print("MARKDOWN TABLE FOR MANUSCRIPT")
    print("=" * 120)

    print(f"\n| Property | SA (N={n_single['SA']}) | EC (N={n_single['EC']}) | CA (N={n_single['CA']}) | SA+EC (N={n_dual['SA+EC']}) | SA+CA (N={n_dual['SA+CA']}) | EC+CA (N={n_dual['EC+CA']}) | Triple (N={n_triple}) |")
    print("|----------|" + "---------------|" * 7)
    for _, row in df.iterrows():
        print(f"| {row['Property']} | {row['SA']} | {row['EC']} | {row['CA']} | {row['SA+EC']} | {row['SA+CA']} | {row['EC+CA']} | {row['Triple']} |")

    print("\n" + "=" * 120)
    print("COMPLETE!")
    print("=" * 120)


if __name__ == '__main__':
    main()
