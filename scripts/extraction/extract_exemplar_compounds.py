"""
Extract Exemplar Compound Details
=================================
Extracts compound-level information for dual-active exemplar scaffolds including:
- Compound IDs (ChEMBL IDs)
- Per-pathogen predicted activity
- Fragment type (scaffold/substituent)
- Cross-combination occurrence counts (TP counts)
"""

import pandas as pd
import numpy as np
import re
import os
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis\DUAL_ACTIVE_POSITIVE"

# All scaffold and substituent files
FILES = {
    'SA_CA_scaffolds': os.path.join(BASE_DIR, 'dual_SA_CA_positive_scaffolds.csv'),
    'SA_CA_substituents': os.path.join(BASE_DIR, 'dual_SA_CA_positive_substitutents.csv'),
    'SA_EC_scaffolds': os.path.join(BASE_DIR, 'dual_SA_EC_positive_scaffolds.csv'),
    'SA_EC_substituents': os.path.join(BASE_DIR, 'dual_SA_EC_positive_substitutents.csv'),
    'CA_EC_scaffolds': os.path.join(BASE_DIR, 'dual_CA_EC_positive_scaffolds.csv'),
    'CA_EC_substituents': os.path.join(BASE_DIR, 'dual_CA_EC_positive_substitutents.csv'),
}

OUTPUT_DIR = os.path.join(BASE_DIR, 'radar_plots')


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_pathogen_examples(example_str):
    """
    Parse the pathogen_examples column to extract compound details.

    Example format:
    "c_albicans Example: CHEMBL5411312 | SMILES: CCN... | Target: 1 | Attribution: 0.762 | Prediction: 0.913 | MW: 379.3"
    """
    if pd.isna(example_str) or not example_str:
        return []

    compounds = []

    # Split by pathogen sections - look for patterns like "pathogen Example:"
    # Pattern: pathogen_name Example: COMPOUND_ID | SMILES: ... | Target: ... | Attribution: ... | Prediction: ... | MW: ...
    pattern = r'(\w+)\s+Example:\s+(\w+)\s*\|\s*SMILES:\s*([^\|]+)\s*\|\s*Target:\s*(\d+)\s*\|\s*Attribution:\s*([\d\.\-]+)\s*\|\s*Prediction:\s*([\d\.]+)\s*\|\s*MW:\s*([\d\.]+)'

    matches = re.findall(pattern, example_str)

    seen = set()  # Avoid duplicates
    for match in matches:
        pathogen, compound_id, smiles, target, attribution, prediction, mw = match
        key = (pathogen, compound_id)
        if key not in seen:
            seen.add(key)
            compounds.append({
                'pathogen': pathogen.lower().replace('_', ' ').title(),
                'compound_id': compound_id,
                'smiles': smiles.strip(),
                'target': int(target),
                'attribution': float(attribution),
                'prediction': float(prediction),
                'mw': float(mw)
            })

    return compounds


def parse_pathogen_breakdown(breakdown_str):
    """
    Parse pathogen_breakdown to get per-pathogen statistics.

    Example format:
    "e_coli: 270 compounds (257 TP, 13 TN) | Avg Attr: 0.658 | Activity: 95.19% || s_aureus: 352 compounds..."
    """
    if pd.isna(breakdown_str) or not breakdown_str:
        return {}

    result = {}

    # Pattern for each pathogen section
    pattern = r'(\w+):\s*(\d+)\s*compounds\s*\((\d+)\s*TP,\s*(\d+)\s*TN\)\s*\|\s*Avg Attr:\s*([\d\.\-]+)\s*\|\s*Activity:\s*([\d\.]+)%'

    matches = re.findall(pattern, breakdown_str)

    for match in matches:
        pathogen, total, tp, tn, avg_attr, activity = match
        pathogen_clean = pathogen.lower().replace('_', ' ').title()
        if pathogen_clean not in result:
            result[pathogen_clean] = {
                'total_compounds': int(total),
                'tp_count': int(tp),
                'tn_count': int(tn),
                'avg_attribution': float(avg_attr),
                'activity_rate': float(activity)
            }

    return result


def get_combination_pathogens(combination_name):
    """Get the two target pathogens from combination name."""
    if 'S. aureus' in combination_name and 'C. albicans' in combination_name:
        return ['S Aureus', 'C Albicans']
    elif 'S. aureus' in combination_name and 'E. coli' in combination_name:
        return ['S Aureus', 'E Coli']
    elif 'C. albicans' in combination_name and 'E. coli' in combination_name:
        return ['C Albicans', 'E Coli']
    return []


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def load_all_fragment_data():
    """Load all scaffold and substituent data with fragment occurrence tracking."""

    all_data = []
    fragment_counts = defaultdict(lambda: defaultdict(int))  # fragment_id -> combination -> count
    fragment_smiles = {}  # fragment_id -> smiles

    for file_key, filepath in FILES.items():
        if not os.path.exists(filepath):
            print(f"Skipping missing file: {filepath}")
            continue

        # Determine fragment type and combination
        if 'scaffold' in file_key.lower():
            fragment_type = 'scaffold'
        else:
            fragment_type = 'substituent'

        if 'SA_CA' in file_key:
            combination = 'SA_CA'
            combination_label = 'S. aureus + C. albicans'
        elif 'SA_EC' in file_key:
            combination = 'SA_EC'
            combination_label = 'S. aureus + E. coli'
        elif 'CA_EC' in file_key:
            combination = 'CA_EC'
            combination_label = 'C. albicans + E. coli'
        else:
            continue

        print(f"Loading: {file_key}")
        df = pd.read_csv(filepath)

        for idx, row in df.iterrows():
            frag_id = row['fragment_id']
            frag_smiles = row['fragment_smiles']

            # Track fragment occurrences
            tp_count = row.get('total_tp_count', row.get('total_compounds_both_pathogens', 0))
            fragment_counts[frag_id][combination] += tp_count
            fragment_smiles[frag_id] = frag_smiles

            # Store row data
            all_data.append({
                'fragment_id': frag_id,
                'fragment_smiles': frag_smiles,
                'fragment_type': fragment_type,
                'combination': combination,
                'combination_label': combination_label,
                'rank': row.get('rank', idx + 1),
                'total_compounds': row.get('total_compounds_both_pathogens', 0),
                'total_tp_count': tp_count,
                'activity_rate': row.get('avg_activity_rate_percent', 0),
                'avg_attribution': row.get('overall_avg_attribution', 0),
                'pathogen_breakdown': row.get('pathogen_breakdown', ''),
                'pathogen_examples': row.get('pathogen_examples', ''),
            })

    return pd.DataFrame(all_data), fragment_counts, fragment_smiles


def extract_exemplar_details(df, fragment_counts, top_n=3):
    """
    Extract detailed compound information for top exemplars.
    """

    results = []

    # Process each combination
    for combination in ['SA_CA', 'SA_EC', 'CA_EC']:
        combination_labels = {
            'SA_CA': 'S. aureus + C. albicans',
            'SA_EC': 'S. aureus + E. coli',
            'CA_EC': 'C. albicans + E. coli',
        }

        pathogen_map = {
            'SA_CA': ('S Aureus', 'C Albicans'),
            'SA_EC': ('S Aureus', 'E Coli'),
            'CA_EC': ('C Albicans', 'E Coli'),
        }

        # Filter to this combination's scaffolds (prefer scaffolds over substituents)
        df_combo = df[
            (df['combination'] == combination) &
            (df['fragment_type'] == 'scaffold') &
            (df['activity_rate'] >= 90) &
            (df['total_compounds'] >= 10)
        ].sort_values(
            by=['total_compounds', 'activity_rate', 'avg_attribution'],
            ascending=[False, False, False]
        ).head(top_n)

        if df_combo.empty:
            # Fall back to substituents if no scaffolds meet criteria
            df_combo = df[
                (df['combination'] == combination) &
                (df['activity_rate'] >= 90)
            ].sort_values(
                by=['total_compounds', 'activity_rate'],
                ascending=[False, False]
            ).head(top_n)

        pathogen1, pathogen2 = pathogen_map[combination]

        for rank, (idx, row) in enumerate(df_combo.iterrows(), 1):
            frag_id = row['fragment_id']

            # Parse pathogen breakdown for per-pathogen stats
            breakdown = parse_pathogen_breakdown(row['pathogen_breakdown'])

            # Parse compound examples
            examples = parse_pathogen_examples(row['pathogen_examples'])

            # Get unique compound examples for each pathogen
            pathogen1_examples = [e for e in examples if pathogen1.lower().replace(' ', '_') in e['pathogen'].lower().replace(' ', '_') or
                                  pathogen1.lower().replace(' ', '') in e['pathogen'].lower().replace(' ', '')]
            pathogen2_examples = [e for e in examples if pathogen2.lower().replace(' ', '_') in e['pathogen'].lower().replace(' ', '_') or
                                  pathogen2.lower().replace(' ', '') in e['pathogen'].lower().replace(' ', '')]

            # More flexible matching
            for e in examples:
                p = e['pathogen'].lower()
                if 's aureus' in p or 's_aureus' in p or 'saureus' in p:
                    if pathogen1 == 'S Aureus' and e not in pathogen1_examples:
                        pathogen1_examples.append(e)
                    elif pathogen2 == 'S Aureus' and e not in pathogen2_examples:
                        pathogen2_examples.append(e)
                elif 'e coli' in p or 'e_coli' in p or 'ecoli' in p:
                    if pathogen1 == 'E Coli' and e not in pathogen1_examples:
                        pathogen1_examples.append(e)
                    elif pathogen2 == 'E Coli' and e not in pathogen2_examples:
                        pathogen2_examples.append(e)
                elif 'c albicans' in p or 'c_albicans' in p or 'calbicans' in p:
                    if pathogen1 == 'C Albicans' and e not in pathogen1_examples:
                        pathogen1_examples.append(e)
                    elif pathogen2 == 'C Albicans' and e not in pathogen2_examples:
                        pathogen2_examples.append(e)

            # Get cross-combination counts
            cross_counts = fragment_counts[frag_id]
            other_combinations = [c for c in ['SA_CA', 'SA_EC', 'CA_EC'] if c != combination]

            # Get per-pathogen stats from breakdown
            p1_stats = None
            p2_stats = None
            for p_name, stats in breakdown.items():
                p_lower = p_name.lower().replace(' ', '')
                if 'aureus' in p_lower or 'saureus' in p_lower:
                    if 'S Aureus' in [pathogen1, pathogen2]:
                        if pathogen1 == 'S Aureus':
                            p1_stats = stats
                        else:
                            p2_stats = stats
                elif 'coli' in p_lower or 'ecoli' in p_lower:
                    if 'E Coli' in [pathogen1, pathogen2]:
                        if pathogen1 == 'E Coli':
                            p1_stats = stats
                        else:
                            p2_stats = stats
                elif 'albicans' in p_lower or 'calbicans' in p_lower:
                    if 'C Albicans' in [pathogen1, pathogen2]:
                        if pathogen1 == 'C Albicans':
                            p1_stats = stats
                        else:
                            p2_stats = stats

            # Create result entries for each compound example
            # First, try to get unique compounds from both pathogens
            all_compounds = {}

            for e in pathogen1_examples[:5]:  # Limit to 5 per pathogen
                cid = e['compound_id']
                if cid not in all_compounds:
                    all_compounds[cid] = {
                        'compound_id': cid,
                        'compound_smiles': e['smiles'],
                        'mw': e['mw'],
                        f'{pathogen1}_prediction': e['prediction'],
                        f'{pathogen1}_attribution': e['attribution'],
                        f'{pathogen2}_prediction': None,
                        f'{pathogen2}_attribution': None,
                    }
                else:
                    all_compounds[cid][f'{pathogen1}_prediction'] = e['prediction']
                    all_compounds[cid][f'{pathogen1}_attribution'] = e['attribution']

            for e in pathogen2_examples[:5]:
                cid = e['compound_id']
                if cid not in all_compounds:
                    all_compounds[cid] = {
                        'compound_id': cid,
                        'compound_smiles': e['smiles'],
                        'mw': e['mw'],
                        f'{pathogen1}_prediction': None,
                        f'{pathogen1}_attribution': None,
                        f'{pathogen2}_prediction': e['prediction'],
                        f'{pathogen2}_attribution': e['attribution'],
                    }
                else:
                    all_compounds[cid][f'{pathogen2}_prediction'] = e['prediction']
                    all_compounds[cid][f'{pathogen2}_attribution'] = e['attribution']

            # If we have compound examples, create entries
            if all_compounds:
                for cid, comp_data in all_compounds.items():
                    result_entry = {
                        'combination': combination_labels[combination],
                        'exemplar_rank': rank,
                        'fragment_id': frag_id,
                        'fragment_smiles': row['fragment_smiles'],
                        'fragment_type': row['fragment_type'],
                        'compound_id': comp_data['compound_id'],
                        'compound_smiles': comp_data['compound_smiles'],
                        'compound_mw': comp_data['mw'],
                        f'{pathogen1.replace(" ", "_")}_prediction': comp_data.get(f'{pathogen1}_prediction'),
                        f'{pathogen1.replace(" ", "_")}_attribution': comp_data.get(f'{pathogen1}_attribution'),
                        f'{pathogen2.replace(" ", "_")}_prediction': comp_data.get(f'{pathogen2}_prediction'),
                        f'{pathogen2.replace(" ", "_")}_attribution': comp_data.get(f'{pathogen2}_attribution'),
                        f'{pathogen1.replace(" ", "_")}_activity_rate': p1_stats['activity_rate'] if p1_stats else None,
                        f'{pathogen1.replace(" ", "_")}_tp_count': p1_stats['tp_count'] if p1_stats else None,
                        f'{pathogen2.replace(" ", "_")}_activity_rate': p2_stats['activity_rate'] if p2_stats else None,
                        f'{pathogen2.replace(" ", "_")}_tp_count': p2_stats['tp_count'] if p2_stats else None,
                        'fragment_total_compounds': row['total_compounds'],
                        'fragment_total_tp': row['total_tp_count'],
                        'fragment_avg_attribution': row['avg_attribution'],
                        'fragment_activity_rate': row['activity_rate'],
                        'appears_in_SA_CA': cross_counts.get('SA_CA', 0),
                        'appears_in_SA_EC': cross_counts.get('SA_EC', 0),
                        'appears_in_CA_EC': cross_counts.get('CA_EC', 0),
                        'total_cross_combination_tp': sum(cross_counts.values()),
                    }
                    results.append(result_entry)
            else:
                # No compound examples parsed, create summary entry
                result_entry = {
                    'combination': combination_labels[combination],
                    'exemplar_rank': rank,
                    'fragment_id': frag_id,
                    'fragment_smiles': row['fragment_smiles'],
                    'fragment_type': row['fragment_type'],
                    'compound_id': 'N/A (see source data)',
                    'compound_smiles': row['fragment_smiles'],
                    'compound_mw': None,
                    f'{pathogen1.replace(" ", "_")}_prediction': None,
                    f'{pathogen1.replace(" ", "_")}_attribution': None,
                    f'{pathogen2.replace(" ", "_")}_prediction': None,
                    f'{pathogen2.replace(" ", "_")}_attribution': None,
                    f'{pathogen1.replace(" ", "_")}_activity_rate': p1_stats['activity_rate'] if p1_stats else row['activity_rate'],
                    f'{pathogen1.replace(" ", "_")}_tp_count': p1_stats['tp_count'] if p1_stats else None,
                    f'{pathogen2.replace(" ", "_")}_activity_rate': p2_stats['activity_rate'] if p2_stats else row['activity_rate'],
                    f'{pathogen2.replace(" ", "_")}_tp_count': p2_stats['tp_count'] if p2_stats else None,
                    'fragment_total_compounds': row['total_compounds'],
                    'fragment_total_tp': row['total_tp_count'],
                    'fragment_avg_attribution': row['avg_attribution'],
                    'fragment_activity_rate': row['activity_rate'],
                    'appears_in_SA_CA': cross_counts.get('SA_CA', 0),
                    'appears_in_SA_EC': cross_counts.get('SA_EC', 0),
                    'appears_in_CA_EC': cross_counts.get('CA_EC', 0),
                    'total_cross_combination_tp': sum(cross_counts.values()),
                }
                results.append(result_entry)

    return pd.DataFrame(results)


def create_summary_table(df_details):
    """Create a summary table of exemplar scaffolds across combinations."""

    summary = df_details.groupby(['combination', 'fragment_id', 'fragment_smiles', 'fragment_type']).agg({
        'exemplar_rank': 'first',
        'fragment_total_compounds': 'first',
        'fragment_total_tp': 'first',
        'fragment_activity_rate': 'first',
        'fragment_avg_attribution': 'first',
        'appears_in_SA_CA': 'first',
        'appears_in_SA_EC': 'first',
        'appears_in_CA_EC': 'first',
        'total_cross_combination_tp': 'first',
        'compound_id': lambda x: list(x.unique())[:3],  # First 3 unique compound IDs
    }).reset_index()

    summary['example_compound_ids'] = summary['compound_id'].apply(lambda x: '; '.join(str(c) for c in x if c != 'N/A (see source data)'))
    summary = summary.drop('compound_id', axis=1)

    return summary


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("EXTRACTING EXEMPLAR COMPOUND DETAILS")
    print("=" * 70)

    # Load all data
    print("\n1. Loading all fragment data...")
    df_all, fragment_counts, fragment_smiles = load_all_fragment_data()
    print(f"   Loaded {len(df_all)} total fragment records")
    print(f"   Unique fragments tracked: {len(fragment_counts)}")

    # Extract exemplar details
    print("\n2. Extracting exemplar compound details...")
    df_details = extract_exemplar_details(df_all, fragment_counts, top_n=3)
    print(f"   Extracted {len(df_details)} compound records")

    # Create summary table
    print("\n3. Creating summary table...")
    df_summary = create_summary_table(df_details)

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Detailed compound-level CSV
    detail_path = os.path.join(OUTPUT_DIR, 'exemplar_compounds_detailed.csv')
    df_details.to_csv(detail_path, index=False)
    print(f"\n   Saved: {detail_path}")

    # Summary CSV
    summary_path = os.path.join(OUTPUT_DIR, 'exemplar_scaffolds_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"   Saved: {summary_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXEMPLAR SUMMARY")
    print("=" * 70)

    for combo in df_summary['combination'].unique():
        print(f"\n{combo}:")
        combo_data = df_summary[df_summary['combination'] == combo]
        for _, row in combo_data.iterrows():
            print(f"  Rank {int(row['exemplar_rank'])}: Fragment {row['fragment_id']}")
            print(f"    Type: {row['fragment_type']}")
            print(f"    SMILES: {row['fragment_smiles'][:50]}...")
            print(f"    Activity: {row['fragment_activity_rate']:.1f}%")
            print(f"    Total compounds: {row['fragment_total_compounds']}")
            print(f"    Cross-combination TP: SA_CA={row['appears_in_SA_CA']}, SA_EC={row['appears_in_SA_EC']}, CA_EC={row['appears_in_CA_EC']}")
            if row['example_compound_ids']:
                print(f"    Example compounds: {row['example_compound_ids'][:80]}")

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
