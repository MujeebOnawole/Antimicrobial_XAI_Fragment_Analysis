#!/usr/bin/env python3
"""
Extract Best Exemplar Compounds from Fragment Analysis CSVs
"""

import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

# Quality thresholds for filtering high-quality fragments
AVG_ATTRIBUTION_THRESHOLD = 0.30
TP_COUNT_THRESHOLD = 20
POSITIVE_CONSISTENCY_THRESHOLD = 80.0
ACTIVITY_RATE_THRESHOLD = 70.0

# SELECT physicochemical criteria
SELECT_CRITERIA = {
    's_aureus': {'logp_min': 1.9, 'logp_max': 4.1, 'mw_min': 206, 'mw_max': 325},
    'e_coli': {'logp_min': 1.3, 'logp_max': 3.4, 'mw_min': 196, 'mw_max': 317},
    'e_coli_relaxed': {'logp_min': 0.0, 'logp_max': 4.5, 'mw_min': 150, 'mw_max': 400},
    'c_albicans': {'logp_min': 1.7, 'logp_max': 3.7, 'mw_min': 207, 'mw_max': 303}
}


def parse_highest_attribution_example(example_str):
    """
    Parse the highest_attribution_example column to extract compound information.

    Format: "HIGHEST ATTR (ACTIVE): COMPOUND_ID | SMILES: ... | Attribution: X.XXX | Prediction: X.XXX | MW: XXX.X | LogP: X.XX"

    Returns dict with: compound_id, smiles, attribution, prediction, mw, logp
    """
    if pd.isna(example_str) or not example_str:
        return None

    try:
        # Extract compound ID
        compound_match = re.search(r'HIGHEST ATTR \(ACTIVE\):\s*(\S+)', example_str)
        if not compound_match:
            return None
        compound_id = compound_match.group(1)

        # Extract SMILES
        smiles_match = re.search(r'SMILES:\s*([^\|]+)', example_str)
        smiles = smiles_match.group(1).strip() if smiles_match else None

        # Extract attribution score
        attr_match = re.search(r'Attribution:\s*([\d\.\-]+)', example_str)
        attribution = float(attr_match.group(1)) if attr_match else None

        # Extract prediction
        pred_match = re.search(r'Prediction:\s*([\d\.\-]+)', example_str)
        prediction = float(pred_match.group(1)) if pred_match else None

        # Extract MW
        mw_match = re.search(r'MW:\s*([\d\.\-]+)', example_str)
        mw = float(mw_match.group(1)) if mw_match else None

        # Extract LogP
        logp_match = re.search(r'LogP:\s*([\d\.\-]+)', example_str)
        logp = float(logp_match.group(1)) if logp_match else None

        return {
            'compound_id': compound_id,
            'smiles': smiles,
            'attribution': attribution,
            'prediction': prediction,
            'mw': mw,
            'logp': logp
        }
    except Exception as e:
        print(f"Warning: Failed to parse example string: {example_str[:100]}... Error: {e}")
        return None


def filter_high_quality_fragments(df, fragment_type, relax_level=0):
    """
    Filter fragments that meet all quality thresholds.
    relax_level: 0 (strict), 1 (relax tp_count), 2 (relax attribution), 3 (relax consistency)
    """
    # Adjust thresholds based on relax_level
    tp_threshold = TP_COUNT_THRESHOLD if relax_level < 1 else 10
    attr_threshold = AVG_ATTRIBUTION_THRESHOLD if relax_level < 2 else 0.20
    consistency_threshold = POSITIVE_CONSISTENCY_THRESHOLD if relax_level < 3 else 70.0

    filtered = df[
        (df['avg_attribution'] >= attr_threshold) &
        (df['tp_count'] >= tp_threshold) &
        (df['positive_consistency_percent'] >= consistency_threshold) &
        (df['activity_rate_percent'] >= ACTIVITY_RATE_THRESHOLD)
    ].copy()

    threshold_msg = f"(tp>={tp_threshold}, attr>={attr_threshold}, cons>={consistency_threshold}%)"
    print(f"  {fragment_type}: {len(filtered)} fragments pass quality filters {threshold_msg} (from {len(df)} total)")
    return filtered


def extract_compound_data(scaffolds_df, substituents_df, pathogen_name):
    """
    Extract compound information from high-quality fragments.
    Returns a dict mapping compound_id to compound info and fragment lists.
    """
    compounds = defaultdict(lambda: {
        'smiles': None,
        'mw': None,
        'logp': None,
        'prediction': None,
        'scaffolds': [],
        'substituents': [],
        'scaffold_count': 0,
        'substituent_count': 0,
        'compound_score': 0
    })

    # Process scaffolds
    for _, row in scaffolds_df.iterrows():
        compound_info = parse_highest_attribution_example(row['highest_attribution_example'])
        if compound_info:
            cid = compound_info['compound_id']
            compounds[cid]['smiles'] = compound_info['smiles']
            compounds[cid]['mw'] = compound_info['mw']
            compounds[cid]['logp'] = compound_info['logp']
            compounds[cid]['prediction'] = compound_info['prediction']
            compounds[cid]['scaffolds'].append({
                'fragment_id': row['fragment_id'],
                'fragment_smiles': row['fragment_smiles'],
                'avg_attribution': row['avg_attribution'],
                'tp_count': row['tp_count'],
                'activity_rate_percent': row['activity_rate_percent']
            })

    # Process substituents
    for _, row in substituents_df.iterrows():
        compound_info = parse_highest_attribution_example(row['highest_attribution_example'])
        if compound_info:
            cid = compound_info['compound_id']
            # Update compound info if not already set
            if not compounds[cid]['smiles']:
                compounds[cid]['smiles'] = compound_info['smiles']
                compounds[cid]['mw'] = compound_info['mw']
                compounds[cid]['logp'] = compound_info['logp']
                compounds[cid]['prediction'] = compound_info['prediction']

            compounds[cid]['substituents'].append({
                'fragment_id': row['fragment_id'],
                'fragment_smiles': row['fragment_smiles'],
                'avg_attribution': row['avg_attribution'],
                'tp_count': row['tp_count'],
                'activity_rate_percent': row['activity_rate_percent']
            })

    # Calculate counts and scores
    for cid, data in compounds.items():
        data['scaffold_count'] = len(data['scaffolds'])
        data['substituent_count'] = len(data['substituents'])

        # Scoring: scaffolds × 2.0 + substituents × 1.0 + prediction × 0.5
        prediction_score = data['prediction'] if data['prediction'] else 0
        data['compound_score'] = (
            data['scaffold_count'] * 2.0 +
            data['substituent_count'] * 1.0 +
            prediction_score * 0.5
        )

    print(f"\nExtracted {len(compounds)} unique compounds for {pathogen_name}")
    return dict(compounds)


def apply_select_filters(compounds, pathogen_name):
    """
    Apply SELECT physicochemical filters to compounds.
    Returns list of (compound_id, compound_data, compliance_status) tuples.
    Only include compounds with ≥2 total fragments (scaffolds + substituents).
    """
    criteria = SELECT_CRITERIA[pathogen_name]

    results = []
    strict_pass = 0
    multi_fragment_count = 0

    for cid, data in compounds.items():
        mw = data['mw']
        logp = data['logp']

        if mw is None or logp is None:
            continue

        # CRITICAL: Only include compounds with ≥2 fragments (not molecule-specific)
        total_fragments = data['scaffold_count'] + data['substituent_count']
        if total_fragments < 2:
            continue

        multi_fragment_count += 1

        # Check strict criteria
        if (criteria['logp_min'] <= logp <= criteria['logp_max'] and
            criteria['mw_min'] <= mw <= criteria['mw_max']):
            results.append((cid, data, '✓'))
            strict_pass += 1
        elif pathogen_name == 'e_coli':
            # Try relaxed criteria for E. coli
            relaxed = SELECT_CRITERIA['e_coli_relaxed']
            if (relaxed['logp_min'] <= logp <= relaxed['logp_max'] and
                relaxed['mw_min'] <= mw <= relaxed['mw_max']):
                results.append((cid, data, '⚠️ Divergent from SELECT-G⁻'))

    print(f"  {pathogen_name}: {multi_fragment_count} compounds have >=2 fragments")
    print(f"  {pathogen_name}: {strict_pass} compounds pass strict SELECT filters (with >=2 fragments)")
    if pathogen_name == 'e_coli':
        print(f"  {pathogen_name}: {len(results) - strict_pass} compounds pass relaxed SELECT filters (with >=2 fragments)")

    return results


def rank_and_select_top_candidates(filtered_compounds, top_n=10):
    """
    Rank compounds by score and select top N.
    """
    # Sort by compound_score (descending)
    sorted_compounds = sorted(
        filtered_compounds,
        key=lambda x: x[1]['compound_score'],
        reverse=True
    )

    return sorted_compounds[:top_n]


def format_fragment_table(fragments, fragment_type):
    """
    Format fragment list as markdown table.
    """
    if not fragments:
        return "No fragments"

    lines = []
    lines.append("| Fragment Type | Fragment ID | Fragment SMILES | Avg Attribution | TP Count | Activity Rate |")
    lines.append("|---------------|-------------|-----------------|-----------------|----------|---------------|")

    for frag in fragments:
        smiles_display = frag['fragment_smiles'][:50] + "..." if len(frag['fragment_smiles']) > 50 else frag['fragment_smiles']
        lines.append(
            f"| {fragment_type} | {frag['fragment_id']} | {smiles_display} | "
            f"{frag['avg_attribution']:.3f} | {frag['tp_count']} | {frag['activity_rate_percent']:.1f}% |"
        )

    return "\n".join(lines)


def generate_candidate_report(rank, cid, data, compliance):
    """
    Generate report for a single candidate compound.
    """
    lines = []
    lines.append(f"### Top Candidate #{rank}: {cid}")
    lines.append(f"- **SMILES:** {data['smiles']}")
    lines.append(f"- **Properties:** MW {data['mw']:.1f} Da, LogP {data['logp']:.2f}")
    lines.append(f"- **Activity:** Ensemble Prediction {data['prediction']:.3f}")
    lines.append(f"- **Fragment Count:** {data['scaffold_count']} scaffolds, {data['substituent_count']} substituents")
    lines.append(f"- **Compound Score:** {data['compound_score']:.2f}")
    lines.append(f"- **SELECT Compliance:** {compliance}")
    lines.append("")
    if data['scaffolds']:
        lines.append("**High-Quality Scaffolds:**")
        lines.append("")
        lines.append(format_fragment_table(data['scaffolds'][:10], "Scaffold"))  # Limit to top 10 for readability
        lines.append("")
    if data['substituents']:
        lines.append("**High-Quality Substituents:**")
        lines.append("")
        lines.append(format_fragment_table(data['substituents'][:10], "Substituent"))
        lines.append("")

    # Add recommendation based on fragment counts and their tp_counts
    min_scaffold_tp = min([f['tp_count'] for f in data['scaffolds']]) if data['scaffolds'] else 0
    min_substituent_tp = min([f['tp_count'] for f in data['substituents']]) if data['substituents'] else 0

    if data['scaffold_count'] >= 2 and min_scaffold_tp >= 10:
        if data['substituent_count'] >= 1:
            recommendation = "EXCELLENT - Multiple transferable scaffolds (tp>=10) with substituents"
        else:
            recommendation = "GOOD - Multiple transferable scaffolds (tp>=10), consider adding substituents for specificity"
    elif data['scaffold_count'] == 1 and data['substituent_count'] >= 2:
        recommendation = "GOOD - Single scaffold with multiple substituents for specificity"
    elif data['scaffold_count'] == 1 and min_scaffold_tp >= 20:
        recommendation = "FAIR - Single highly transferable scaffold (tp>=20)"
    elif data['scaffold_count'] >= 1 and min_scaffold_tp < 10:
        recommendation = "POOR - Scaffolds appear in <10 active compounds (molecule-specific)"
    else:
        recommendation = "FAIR - Limited fragment coverage"

    lines.append(f"**Recommendation:** {recommendation}")
    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def main():
    base_path = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis\POSITIVE")

    # Define file paths
    files = {
        's_aureus': {
            'scaffolds': base_path / "SA_specific_positive_scaffolds.csv",
            'substituents': base_path / "SA_specific_positive_substitutents.csv"
        },
        'e_coli': {
            'scaffolds': base_path / "EC_specific_positive_scaffolds.csv",
            'substituents': base_path / "EC_specific_positive_substitutents.csv"
        },
        'c_albicans': {
            'scaffolds': base_path / "CA_specific_positive_scaffolds.csv",
            'substituents': base_path / "CA_specific_positive_substitutents.csv"
        }
    }

    # Storage for all results
    all_results = {}

    # Process each pathogen
    for pathogen, file_paths in files.items():
        print(f"\n{'='*60}")
        print(f"Processing {pathogen.upper().replace('_', ' ')}")
        print(f"{'='*60}")

        # Read CSVs
        print(f"\nReading CSV files...")
        scaffolds_df = pd.read_csv(file_paths['scaffolds'])
        substituents_df = pd.read_csv(file_paths['substituents'])
        print(f"  Scaffolds: {len(scaffolds_df)} rows")
        print(f"  Substituents: {len(substituents_df)} rows")

        # Try progressive relaxation to find compounds with ≥2 fragments
        top_candidates = []
        for relax_level in range(4):  # 0=strict, 1=relax tp_count, 2=relax attr, 3=relax consistency
            if top_candidates and len(top_candidates) >= 5:
                break  # Found enough candidates

            print(f"\nFiltering high-quality fragments (relax level {relax_level})...")
            hq_scaffolds = filter_high_quality_fragments(scaffolds_df, "Scaffolds", relax_level)
            hq_substituents = filter_high_quality_fragments(substituents_df, "Substituents", relax_level)

            # Extract compound data
            print(f"\nExtracting compound data...")
            compounds = extract_compound_data(hq_scaffolds, hq_substituents, pathogen)

            # Apply SELECT filters (only includes compounds with >=2 fragments)
            print(f"\nApplying SELECT filters (requires >=2 fragments)...")
            filtered_compounds = apply_select_filters(compounds, pathogen)

            if filtered_compounds:
                # Rank and select top candidates
                print(f"\nSelecting top 10 candidates...")
                top_candidates = rank_and_select_top_candidates(filtered_compounds, top_n=10)
                print(f"  Selected {len(top_candidates)} candidates")
                break
            else:
                print(f"  No compounds found with >=2 fragments at relax level {relax_level}")

        if not top_candidates:
            print(f"\nWARNING: No compounds found with >=2 fragments for {pathogen} even with relaxed criteria!")
            print(f"  Trying fallback: showing best multi-fragment compounds regardless of SELECT...")

            # Fallback: show best multi-fragment compounds even if they don't meet SELECT
            for fallback_level in range(4):
                hq_scaffolds = filter_high_quality_fragments(scaffolds_df, "Scaffolds", fallback_level)
                hq_substituents = filter_high_quality_fragments(substituents_df, "Substituents", fallback_level)
                compounds = extract_compound_data(hq_scaffolds, hq_substituents, pathogen)

                # Get multi-fragment compounds without SELECT filters
                multi_frag = []
                for cid, data in compounds.items():
                    if data['mw'] is not None and data['logp'] is not None:
                        total_fragments = data['scaffold_count'] + data['substituent_count']
                        if total_fragments >= 2:
                            # Apply very relaxed MW filter (exclude very large compounds)
                            if data['mw'] <= 400:
                                multi_frag.append((cid, data, 'X Does NOT meet SELECT criteria'))

                if multi_frag:
                    print(f"  Found {len(multi_frag)} multi-fragment compounds (ignoring SELECT criteria)")
                    top_candidates = rank_and_select_top_candidates(multi_frag, top_n=10)
                    break

        all_results[pathogen] = top_candidates

    # Generate final report
    print(f"\n{'='*60}")
    print("Generating Final Report")
    print(f"{'='*60}")

    report = []
    report.append("# EXEMPLAR COMPOUND RECOMMENDATIONS")
    report.append("")

    # Generate reports for each pathogen
    for pathogen in ['s_aureus', 'e_coli', 'c_albicans']:
        pathogen_display = {
            's_aureus': 'S. AUREUS (SELECT-G⁺)',
            'e_coli': 'E. COLI (SELECT-G⁻)',
            'c_albicans': 'C. ALBICANS (SELECT-CA)'
        }

        report.append(f"## {pathogen_display[pathogen]}")
        report.append("")

        candidates = all_results[pathogen]
        if not candidates:
            report.append("**No candidates found meeting all criteria.**")
            report.append("")
            continue

        for i, (cid, data, compliance) in enumerate(candidates, 1):
            report.append(generate_candidate_report(i, cid, data, compliance))

    # Generate summary comparison table
    report.append("# SUMMARY COMPARISON")
    report.append("")
    report.append("| Pathogen | Best Exemplar | MW | LogP | Pred | # Scaffolds | # Substituents | Score | SELECT Compliance |")
    report.append("|----------|--------------|-----|------|------|-------------|----------------|-------|-------------------|")

    for pathogen in ['s_aureus', 'e_coli', 'c_albicans']:
        pathogen_display = {
            's_aureus': 'S. aureus',
            'e_coli': 'E. coli',
            'c_albicans': 'C. albicans'
        }

        candidates = all_results[pathogen]
        if candidates:
            cid, data, compliance = candidates[0]  # Best exemplar
            report.append(
                f"| {pathogen_display[pathogen]} | {cid} | {data['mw']:.1f} | {data['logp']:.2f} | "
                f"{data['prediction']:.3f} | {data['scaffold_count']} | {data['substituent_count']} | "
                f"{data['compound_score']:.1f} | {compliance} |"
            )

    # Save report
    report_text = "\n".join(report)
    output_file = base_path / "EXEMPLAR_RECOMMENDATIONS.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nReport saved to: {output_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
