"""
Find and visualize the best SA+CA exemplar molecule that fits within BOTH SELECT zones.

SELECT Zone Overlap (must fit both S. aureus AND C. albicans):
- MW: 250-450 (CA min, SA max)
- LogP: 1.9-4.1 (SA stricter)
- TPSA: 34-80 (CA min, SA max)
- HBD: 0-2 (SA stricter)
- HBA: 2-8 (same)
- Rings: 2-5 (SA stricter)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import matplotlib
import os
import re

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw, rdDepictor

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis\DUAL_ACTIVE_POSITIVE"
OUTPUT_DIR = os.path.join(BASE_DIR, 'radar_plots')

# SELECT zones based on literature
SELECT_ZONES = {
    'Gram_positive': {  # S. aureus
        'MW': (200, 450),
        'LogP': (1.9, 4.1),
        'TPSA': (17, 80),
        'HBD': (0, 2),
        'HBA': (2, 8),
        'Rings': (2, 5),
    },
    'Fungi': {  # C. albicans
        'MW': (250, 550),
        'LogP': (1.7, 5.0),
        'TPSA': (34, 100),
        'HBD': (0, 3),
        'HBA': (2, 8),
        'Rings': (2, 6),
    }
}

# Overlap zone (fits BOTH S. aureus AND C. albicans)
OVERLAP_ZONE = {
    'MW': (250, 450),
    'LogP': (1.9, 4.1),
    'TPSA': (34, 80),
    'HBD': (0, 2),
    'HBA': (2, 8),
    'Rings': (2, 5),
}

# Radar plot configuration
RADAR_CONFIG = {
    'categories': ['LogP', 'MW', 'TPSA', 'HBD', 'HBA', 'Rings'],
    'ranges': {
        'LogP': (-2, 6),
        'MW': (100, 600),
        'TPSA': (0, 150),
        'HBD': (0, 6),
        'HBA': (0, 12),
        'Rings': (0, 8),
    }
}

# Colors
COLORS = {
    'SA_CA': {'primary': '#8B008B', 'fill': '#DDA0DD'},  # Purple (Gram+ & Fungi)
}


# ============================================================================
# FUNCTIONS
# ============================================================================

def calculate_properties(smiles):
    """Calculate physicochemical properties from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        props = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': rdMolDescriptors.CalcTPSA(mol),
            'HBD': rdMolDescriptors.CalcNumHBD(mol),
            'HBA': rdMolDescriptors.CalcNumHBA(mol),
            'Rings': rdMolDescriptors.CalcNumRings(mol),
            'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'ArRings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'Fsp3': rdMolDescriptors.CalcFractionCSP3(mol),
        }
        return props
    except Exception as e:
        print(f"Error: {e}")
        return None


def check_zone_fit(props, zone):
    """Check if properties fit within a zone. Returns (fits_all, violations)."""
    violations = []
    for prop, (min_val, max_val) in zone.items():
        if prop in props:
            val = props[prop]
            if val < min_val:
                violations.append(f"{prop}={val:.2f} < {min_val}")
            elif val > max_val:
                violations.append(f"{prop}={val:.2f} > {max_val}")
    return len(violations) == 0, violations


def score_compound(props, zone):
    """Score how well a compound fits within a zone (0-100, higher is better)."""
    score = 0
    n_props = 0

    for prop, (min_val, max_val) in zone.items():
        if prop in props:
            val = props[prop]
            range_size = max_val - min_val
            mid = (min_val + max_val) / 2

            if min_val <= val <= max_val:
                # Within range - score based on distance from center
                distance_from_mid = abs(val - mid) / (range_size / 2)
                prop_score = 100 * (1 - distance_from_mid * 0.5)  # Max 100, min 50 if at edge
            else:
                # Outside range - penalty based on how far outside
                if val < min_val:
                    distance_outside = (min_val - val) / range_size
                else:
                    distance_outside = (val - max_val) / range_size
                prop_score = max(0, 50 - distance_outside * 100)

            score += prop_score
            n_props += 1

    return score / n_props if n_props > 0 else 0


def normalize_value(value, min_val, max_val):
    """Normalize value to 0-1 range."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def parse_pathogen_examples(example_str):
    """Extract compound examples from pathogen_examples column."""
    if pd.isna(example_str) or not example_str:
        return []

    compounds = []
    pattern = r'(\w+)\s+Example:\s+(\w+)\s*\|\s*SMILES:\s*([^\|]+)\s*\|'
    matches = re.findall(pattern, example_str)

    seen = set()
    for match in matches:
        pathogen, compound_id, smiles = match
        if compound_id not in seen:
            seen.add(compound_id)
            compounds.append({
                'pathogen': pathogen,
                'compound_id': compound_id,
                'smiles': smiles.strip()
            })
    return compounds


def create_radar_plot(compound_data, output_path):
    """Create radar plot showing compound against both SELECT zones."""
    categories = RADAR_CONFIG['categories']
    ranges = RADAR_CONFIG['ranges']

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],
               ['20%', '40%', '60%', '80%', '100%'],
               color='grey', size=10, alpha=0.7)
    plt.ylim(0, 1.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=16, fontweight='bold', fontfamily='Arial')
    ax.tick_params(axis='x', pad=30)

    # Draw SELECT zones
    zone_configs = [
        ('Gram_positive', '#DC143C', 'S. aureus (G+) SELECT zone'),
        ('Fungi', '#228B22', 'C. albicans SELECT zone'),
    ]

    for zone_name, zone_color, zone_label in zone_configs:
        zone_data = SELECT_ZONES[zone_name]

        zone_max_values = []
        for cat in categories:
            if cat in zone_data:
                zone_min, zone_max = zone_data[cat]
                range_min, range_max = ranges[cat]
                zone_max_values.append(normalize_value(zone_max, range_min, range_max))
            else:
                zone_max_values.append(1)
        zone_max_values += zone_max_values[:1]

        ax.fill(angles, zone_max_values, color=zone_color, alpha=0.1)
        ax.plot(angles, zone_max_values, color=zone_color, linewidth=2,
                linestyle='--', alpha=0.6)

    # Plot compound profile
    compound_values = []
    for cat in categories:
        value = compound_data.get(cat, 0)
        range_min, range_max = ranges[cat]
        normalized = normalize_value(value, range_min, range_max)
        compound_values.append(max(0, min(1.1, normalized)))
    compound_values += compound_values[:1]

    color_scheme = COLORS['SA_CA']
    ax.plot(angles, compound_values, 'o-', linewidth=3, markersize=12,
            color=color_scheme['primary'], label="Exemplar Molecule")
    ax.fill(angles, compound_values, alpha=0.25, color=color_scheme['fill'])

    # Legend
    legend_elements = [
        Line2D([0], [0], color=color_scheme['primary'], linewidth=3,
               marker='o', markersize=10, label='Exemplar Molecule'),
        Line2D([0], [0], color='#DC143C', linewidth=2,
               linestyle='--', alpha=0.6, label='S. aureus (G+) SELECT zone'),
        Line2D([0], [0], color='#228B22', linewidth=2,
               linestyle='--', alpha=0.6, label='C. albicans SELECT zone'),
    ]

    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1.45, 1.15), fontsize=12, framealpha=0.9)

    plt.title("S. aureus + C. albicans Exemplar\nPhysicochemical Properties",
              size=18, fontweight='bold', y=1.25, pad=20)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.5)
    plt.close()

    print(f"Saved radar plot: {output_path}")


def generate_structure_image(smiles, output_path, legend=None):
    """Generate 2D molecular structure image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Could not parse SMILES: {smiles}")
        return

    rdDepictor.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=(700, 600), legend=legend, fitImage=True)
    img.save(output_path)
    print(f"Saved structure: {output_path}")


def main():
    print("=" * 70)
    print("FINDING BEST SA+CA EXEMPLAR WITHIN SELECT ZONE OVERLAP")
    print("=" * 70)

    print("\nOVERLAP ZONE (must fit BOTH S. aureus AND C. albicans):")
    for prop, (min_val, max_val) in OVERLAP_ZONE.items():
        print(f"  {prop}: {min_val} - {max_val}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load SA+CA scaffolds
    scaffold_path = os.path.join(BASE_DIR, 'dual_SA_CA_positive_scaffolds.csv')
    df = pd.read_csv(scaffold_path)
    print(f"\nLoaded {len(df)} SA+CA scaffolds")

    # Filter for high-quality scaffolds
    df_good = df[
        (df['avg_activity_rate_percent'] >= 90) &
        (df['total_compounds_both_pathogens'] >= 5)
    ].copy()
    print(f"High-quality scaffolds (>=90% activity, >=5 compounds): {len(df_good)}")

    # Search for compounds that fit the overlap zone
    print("\nSearching for compounds fitting overlap zone...")

    candidates = []

    for idx, row in df_good.iterrows():
        examples = parse_pathogen_examples(row.get('pathogen_examples', ''))

        for ex in examples:
            props = calculate_properties(ex['smiles'])
            if props is None:
                continue

            fits, violations = check_zone_fit(props, OVERLAP_ZONE)
            score = score_compound(props, OVERLAP_ZONE)

            candidates.append({
                'fragment_id': row['fragment_id'],
                'fragment_smiles': row['fragment_smiles'],
                'compound_id': ex['compound_id'],
                'compound_smiles': ex['smiles'],
                'pathogen': ex['pathogen'],
                'activity_rate': row['avg_activity_rate_percent'],
                'total_compounds': row['total_compounds_both_pathogens'],
                'fits_overlap': fits,
                'n_violations': len(violations),
                'violations': '; '.join(violations) if violations else 'None',
                'fit_score': score,
                **props
            })

    df_candidates = pd.DataFrame(candidates)
    print(f"\nTotal compound examples found: {len(df_candidates)}")

    # Separate perfect fits and near-fits
    df_perfect = df_candidates[df_candidates['fits_overlap'] == True].copy()
    df_near = df_candidates[df_candidates['n_violations'] <= 1].copy()

    print(f"Perfect fits (all properties in range): {len(df_perfect)}")
    print(f"Near fits (0-1 violations): {len(df_near)}")

    # Sort by fit score and select best
    if len(df_perfect) > 0:
        df_best = df_perfect.sort_values(['fit_score', 'total_compounds'], ascending=[False, False])
        source = "perfect fit"
    elif len(df_near) > 0:
        df_best = df_near.sort_values(['n_violations', 'fit_score', 'total_compounds'], ascending=[True, False, False])
        source = "near fit"
    else:
        df_best = df_candidates.sort_values(['n_violations', 'fit_score'], ascending=[True, False])
        source = "best available"

    # Show top candidates
    print("\n" + "=" * 70)
    print("TOP SA+CA EXEMPLAR CANDIDATES")
    print("=" * 70)

    for i, (_, row) in enumerate(df_best.head(10).iterrows()):
        fit_status = "[PERFECT]" if row['fits_overlap'] else f"[{row['n_violations']} violations]"
        print(f"\n{i+1}. {row['compound_id']} {fit_status}")
        print(f"   Score: {row['fit_score']:.1f}/100")
        print(f"   From scaffold: {row['fragment_id']} ({row['total_compounds']} compounds, {row['activity_rate']:.1f}% activity)")
        print(f"   MW: {row['MW']:.1f}, LogP: {row['LogP']:.2f}, TPSA: {row['TPSA']:.1f}")
        print(f"   HBD: {row['HBD']}, HBA: {row['HBA']}, Rings: {row['Rings']}")
        if row['violations'] != 'None':
            print(f"   Violations: {row['violations']}")

    # Select the best exemplar
    best = df_best.iloc[0]

    print("\n" + "=" * 70)
    print(f"SELECTED SA+CA EXEMPLAR ({source})")
    print("=" * 70)
    print(f"\nCompound ID: {best['compound_id']}")
    print(f"SMILES: {best['compound_smiles']}")
    print(f"\nPhysicochemical Properties:")
    print(f"  MW: {best['MW']:.2f} (target: 250-450)")
    print(f"  LogP: {best['LogP']:.2f} (target: 1.9-4.1)")
    print(f"  TPSA: {best['TPSA']:.2f} (target: 34-80)")
    print(f"  HBD: {best['HBD']} (target: 0-2)")
    print(f"  HBA: {best['HBA']} (target: 2-8)")
    print(f"  Rings: {best['Rings']} (target: 2-5)")
    print(f"\nAdditional Properties:")
    print(f"  RotBonds: {best['RotBonds']}")
    print(f"  ArRings: {best['ArRings']}")
    print(f"  Fsp3: {best['Fsp3']:.3f}")
    print(f"\nSource Scaffold:")
    print(f"  Fragment ID: {best['fragment_id']}")
    print(f"  Total compounds: {best['total_compounds']}")
    print(f"  Activity rate: {best['activity_rate']:.1f}%")
    print(f"\nFit Assessment:")
    print(f"  Fit Score: {best['fit_score']:.1f}/100")
    print(f"  Violations: {best['violations']}")

    # Create outputs
    print("\n" + "=" * 70)
    print("GENERATING OUTPUTS")
    print("=" * 70)

    # Prepare compound data for radar plot
    compound_data = {
        'MW': best['MW'],
        'LogP': best['LogP'],
        'TPSA': best['TPSA'],
        'HBD': best['HBD'],
        'HBA': best['HBA'],
        'Rings': best['Rings'],
    }

    # Create radar plot
    radar_path = os.path.join(OUTPUT_DIR, 'sa_ca_exemplar_radar.png')
    create_radar_plot(compound_data, radar_path)

    # Create structure image
    structure_path = os.path.join(OUTPUT_DIR, 'sa_ca_exemplar_structure.png')
    generate_structure_image(best['compound_smiles'], structure_path,
                            legend=f"SA+CA Exemplar: {best['compound_id']}")

    # Save exemplar details to CSV
    exemplar_data = {
        'combination': 'S. aureus + C. albicans',
        'compound_id': best['compound_id'],
        'compound_smiles': best['compound_smiles'],
        'fragment_id': best['fragment_id'],
        'fragment_smiles': best['fragment_smiles'],
        'MW': best['MW'],
        'LogP': best['LogP'],
        'TPSA': best['TPSA'],
        'HBD': best['HBD'],
        'HBA': best['HBA'],
        'Rings': best['Rings'],
        'RotBonds': best['RotBonds'],
        'ArRings': best['ArRings'],
        'Fsp3': best['Fsp3'],
        'activity_rate': best['activity_rate'],
        'total_compounds': best['total_compounds'],
        'fit_score': best['fit_score'],
        'fits_sa_zone': check_zone_fit(compound_data, SELECT_ZONES['Gram_positive'])[0],
        'fits_ca_zone': check_zone_fit(compound_data, SELECT_ZONES['Fungi'])[0],
        'fits_overlap': best['fits_overlap'],
        'violations': best['violations'],
    }

    df_exemplar = pd.DataFrame([exemplar_data])
    csv_path = os.path.join(OUTPUT_DIR, 'sa_ca_exemplar_details.csv')
    df_exemplar.to_csv(csv_path, index=False)
    print(f"\nSaved exemplar details: {csv_path}")

    # Also save all candidates for reference
    candidates_path = os.path.join(OUTPUT_DIR, 'sa_ca_all_candidates.csv')
    df_best.head(50).to_csv(candidates_path, index=False)
    print(f"Saved top 50 candidates: {candidates_path}")

    print("\n" + "=" * 70)
    print("SA+CA EXEMPLAR ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files in {OUTPUT_DIR}:")
    print("  - sa_ca_exemplar_radar.png")
    print("  - sa_ca_exemplar_structure.png")
    print("  - sa_ca_exemplar_details.csv")
    print("  - sa_ca_all_candidates.csv")


if __name__ == '__main__':
    main()
