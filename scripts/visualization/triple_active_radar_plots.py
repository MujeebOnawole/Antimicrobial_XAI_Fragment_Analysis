"""
Triple-Active (Broad Spectrum) Radar Plot Generator
====================================================
Generates radar plots for triple-active antimicrobial exemplar molecules
active against S. aureus + E. coli + C. albicans.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import os
import re

# Set Arial font for publication quality
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 18

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
    from rdkit.Chem import rdDepictor
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis\TRIPLE_ACTIVE_POSITIVE"

FILES = {
    'scaffolds': os.path.join(BASE_DIR, 'Multi_positive_scaffolds.csv'),
    'substituents': os.path.join(BASE_DIR, 'Multi_positive_substituents.csv'),
}

OUTPUT_DIR = os.path.join(BASE_DIR, 'radar_plots')

# Color scheme for triple-active (orange/gold - represents broad spectrum)
COLORS = {
    'triple': {'primary': '#FF8C00', 'fill': '#FFD700'},  # Dark orange / Gold
}

# SELECT zones for all three pathogens
SELECT_ZONES = {
    'Gram_positive': {
        'MW': (200, 450),
        'LogP': (1.9, 4.1),
        'TPSA': (17, 80),
        'HBD': (0, 2),
        'HBA': (2, 8),
        'Rings': (2, 5),
    },
    'Gram_negative': {
        'MW': (200, 500),
        'LogP': (-2, 3.4),
        'TPSA': (80, 150),
        'HBD': (2, 5),
        'HBA': (4, 10),
        'Rings': (1, 4),
    },
    'Fungi': {
        'MW': (250, 550),
        'LogP': (1.7, 5.0),
        'TPSA': (34, 100),
        'HBD': (0, 3),
        'HBA': (2, 8),
        'Rings': (2, 6),
    }
}

RADAR_CONFIG = {
    'categories': ['LogP', 'MW', 'TPSA', 'HBD', 'HBA', 'Rings'],
    'ranges': {
        'LogP': (-2, 6),
        'MW': (100, 600),
        'TPSA': (0, 200),
        'HBD': (0, 6),
        'HBA': (0, 12),
        'Rings': (0, 8),
    }
}


def calculate_properties(smiles):
    """Calculate physicochemical properties from SMILES."""
    if not RDKIT_AVAILABLE:
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
            'Rings': rdMolDescriptors.CalcNumRings(mol),
            'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'ArRings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'Fsp3': rdMolDescriptors.CalcFractionCSP3(mol),
            'HeavyAtoms': rdMolDescriptors.CalcNumHeavyAtoms(mol),
        }
    except:
        return None


def parse_pathogen_examples(example_str):
    """Extract compound examples from pathogen_examples column."""
    if pd.isna(example_str) or not example_str:
        return []

    compounds = []
    pattern = r'(\w+)\s+Example:\s+(\w+)\s*\|\s*SMILES:\s*([^\|]+)\s*\|\s*Target:\s*(\d+)\s*\|\s*Attribution:\s*([\d\.\-]+)\s*\|\s*Prediction:\s*([\d\.]+)'
    matches = re.findall(pattern, example_str)

    seen = set()
    for match in matches:
        pathogen, compound_id, smiles, target, attribution, prediction = match
        if compound_id not in seen:
            seen.add(compound_id)
            compounds.append({
                'pathogen': pathogen,
                'compound_id': compound_id,
                'smiles': smiles.strip(),
                'prediction': float(prediction),
                'attribution': float(attribution)
            })
    return compounds


def parse_pathogen_breakdown(breakdown_str):
    """Parse pathogen breakdown to get per-pathogen statistics."""
    if pd.isna(breakdown_str) or not breakdown_str:
        return {}

    result = {}
    pattern = r'(\w+):\s*(\d+)\s*compounds\s*\((\d+)\s*TP,\s*(\d+)\s*TN\)\s*\|\s*Avg Attr:\s*([\d\.\-]+)\s*\|\s*Activity:\s*([\d\.]+)%'
    matches = re.findall(pattern, breakdown_str)

    for match in matches:
        pathogen, total, tp, tn, avg_attr, activity = match
        pathogen_clean = pathogen.lower()
        if pathogen_clean not in result:
            result[pathogen_clean] = {
                'total_compounds': int(total),
                'tp_count': int(tp),
                'tn_count': int(tn),
                'activity_rate': float(activity)
            }
    return result


def normalize_value(value, min_val, max_val):
    """Normalize a value to 0-1 range."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def create_triple_radar_plot(compound_data, output_path):
    """
    Create radar plot for triple-active molecule showing all 3 SELECT zones.
    SELECT zones shown as outline boundaries only (no filled regions).
    """
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
               color='grey', size=9, alpha=0.7)
    plt.ylim(0, 1.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=18, fontweight='bold', fontfamily='Arial')
    ax.tick_params(axis='x', pad=25)

    # Zone colors and labels - species names italicized per biological convention
    zone_info = {
        'Gram_positive': ('#DC143C', r'$\it{S. aureus}$ ($G^+$)'),
        'Gram_negative': ('#1E90FF', r'$\it{E. coli}$ ($G^-$)'),
        'Fungi': ('#228B22', r'$\it{C. albicans}$'),
    }

    # Draw all 3 SELECT zones with filled regions and outer boundary only
    for zone_name, (zone_color, zone_label) in zone_info.items():
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

        # Fill the zone with semi-transparent color
        ax.fill(angles, zone_max_values, color=zone_color, alpha=0.08)
        # Only draw outer boundary line (no inner boundary)
        ax.plot(angles, zone_max_values, color=zone_color, linewidth=2.5,
                linestyle='--', alpha=0.7)

    # Plot compound data - this is the ONLY filled region
    compound_values = []
    for cat in categories:
        value = compound_data.get(cat, 0)
        range_min, range_max = ranges[cat]
        normalized = normalize_value(value, range_min, range_max)
        compound_values.append(max(0, min(1, normalized)))
    compound_values += compound_values[:1]

    ax.plot(angles, compound_values, 'o-', linewidth=3, markersize=10,
            color=COLORS['triple']['primary'], label="Broad-Spectrum Exemplar")
    ax.fill(angles, compound_values, alpha=0.35, color=COLORS['triple']['fill'])

    # Legend with clear distinction
    legend_elements = [
        Line2D([0], [0], color=COLORS['triple']['primary'], linewidth=3,
               marker='o', markersize=8, label='Broad-Spectrum Exemplar')
    ]
    for zone_name, (zone_color, zone_label) in zone_info.items():
        legend_elements.append(
            Line2D([0], [0], color=zone_color, linewidth=2.5,
                   linestyle='--', alpha=0.7, label=f'{zone_label} SELECT zone')
        )

    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1.45, 1.15), fontsize=11, framealpha=0.9)

    plt.title("Broad-Spectrum (Triple-Active) Exemplar\nPhysicochemical Properties",
              size=18, fontweight='bold', y=1.25, pad=20)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.5)
    plt.close()
    print(f"Saved: {output_path}")


def generate_structure_image(smiles, output_path, legend=None):
    """Generate 2D molecular structure image."""
    if not RDKIT_AVAILABLE:
        return
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    rdDepictor.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=(700, 600), legend=legend, fitImage=True)
    img.save(output_path)
    print(f"Saved structure: {output_path}")


def main():
    print("=" * 70)
    print("TRIPLE-ACTIVE (BROAD SPECTRUM) RADAR PLOT GENERATOR")
    print("=" * 70)

    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit required")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load scaffolds
    df_scaffolds = pd.read_csv(FILES['scaffolds'])
    print(f"\nLoaded {len(df_scaffolds)} scaffolds")

    # Filter for high-quality scaffolds
    # - Activity >= 90%
    # - At least 20 compounds
    # - More than 5 heavy atoms (exclude trivial fragments like benzene)
    df_filtered = df_scaffolds[
        (df_scaffolds['avg_activity_rate_percent'] >= 90) &
        (df_scaffolds['total_compounds_all_pathogens'] >= 20)
    ].copy()

    # Calculate properties for filtered scaffolds
    print(f"Filtering for activity >= 90% and compounds >= 20...")
    df_filtered['props'] = df_filtered['fragment_smiles'].apply(calculate_properties)
    df_filtered = df_filtered[df_filtered['props'].notna()]

    # Filter out trivial scaffolds (< 8 heavy atoms)
    df_filtered['heavy_atoms'] = df_filtered['props'].apply(lambda x: x.get('HeavyAtoms', 0) if x else 0)
    df_filtered = df_filtered[df_filtered['heavy_atoms'] >= 8]

    print(f"Scaffolds after filtering: {len(df_filtered)}")

    # Sort by compound count and activity
    df_sorted = df_filtered.sort_values(
        by=['total_compounds_all_pathogens', 'avg_activity_rate_percent'],
        ascending=[False, False]
    )

    print("\nTop 10 Triple-Active Scaffolds:")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        props = row['props']
        print(f"\n  Rank {i+1}: Fragment {row['fragment_id']}")
        print(f"    SMILES: {row['fragment_smiles'][:50]}...")
        print(f"    Compounds: {row['total_compounds_all_pathogens']}")
        print(f"    Activity: {row['avg_activity_rate_percent']:.1f}%")
        print(f"    MW: {props['MW']:.1f}, LogP: {props['LogP']:.2f}")

    # Select best exemplar
    if len(df_sorted) == 0:
        print("No suitable scaffolds found")
        return

    best_scaffold = df_sorted.iloc[0]
    best_props = best_scaffold['props']

    print("\n" + "=" * 50)
    print("SELECTED TRIPLE-ACTIVE EXEMPLAR")
    print("=" * 50)
    print(f"Fragment ID: {best_scaffold['fragment_id']}")
    print(f"SMILES: {best_scaffold['fragment_smiles']}")
    print(f"Total Compounds: {best_scaffold['total_compounds_all_pathogens']}")
    print(f"Activity Rate: {best_scaffold['avg_activity_rate_percent']:.1f}%")
    print(f"\nPhysicochemical Properties:")
    print(f"  MW: {best_props['MW']:.1f}")
    print(f"  LogP: {best_props['LogP']:.2f}")
    print(f"  TPSA: {best_props['TPSA']:.1f}")
    print(f"  HBD: {best_props['HBD']}, HBA: {best_props['HBA']}")
    print(f"  Rings: {best_props['Rings']}")

    # Try to get compound examples
    examples = parse_pathogen_examples(best_scaffold.get('pathogen_examples', ''))
    compound_for_plot = None

    if examples:
        print(f"\nCompound examples found: {len(examples)}")
        for ex in examples[:3]:
            ex_props = calculate_properties(ex['smiles'])
            if ex_props:
                print(f"  - {ex['compound_id']}: MW={ex_props['MW']:.1f}, LogP={ex_props['LogP']:.2f}")
                if compound_for_plot is None:
                    compound_for_plot = {
                        'compound_id': ex['compound_id'],
                        'smiles': ex['smiles'],
                        **ex_props
                    }

    # Use scaffold if no compound examples
    if compound_for_plot is None:
        compound_for_plot = {
            'compound_id': f"Fragment_{best_scaffold['fragment_id']}",
            'smiles': best_scaffold['fragment_smiles'],
            **best_props
        }

    # Generate radar plot
    radar_path = os.path.join(OUTPUT_DIR, 'radar_triple_active_molecule.png')
    create_triple_radar_plot(compound_for_plot, radar_path)

    # Generate structure image
    structure_path = os.path.join(OUTPUT_DIR, 'structure_triple_active_molecule.png')
    generate_structure_image(
        compound_for_plot['smiles'],
        structure_path,
        legend="Broad-Spectrum Exemplar"
    )

    # Parse pathogen breakdown for per-pathogen stats
    breakdown = parse_pathogen_breakdown(best_scaffold.get('pathogen_breakdown', ''))
    if breakdown:
        print("\nPer-Pathogen Activity:")
        for pathogen, stats in breakdown.items():
            print(f"  {pathogen}: {stats['activity_rate']:.1f}% ({stats['tp_count']} TP)")

    # Save exemplar details
    exemplar_data = {
        'fragment_id': best_scaffold['fragment_id'],
        'fragment_smiles': best_scaffold['fragment_smiles'],
        'compound_id': compound_for_plot['compound_id'],
        'compound_smiles': compound_for_plot['smiles'],
        'total_compounds': best_scaffold['total_compounds_all_pathogens'],
        'activity_rate': best_scaffold['avg_activity_rate_percent'],
        'MW': compound_for_plot['MW'],
        'LogP': compound_for_plot['LogP'],
        'TPSA': compound_for_plot['TPSA'],
        'HBD': compound_for_plot['HBD'],
        'HBA': compound_for_plot['HBA'],
        'Rings': compound_for_plot['Rings'],
    }

    # Add per-pathogen stats
    for pathogen, stats in breakdown.items():
        exemplar_data[f'{pathogen}_activity'] = stats['activity_rate']
        exemplar_data[f'{pathogen}_tp_count'] = stats['tp_count']

    df_exemplar = pd.DataFrame([exemplar_data])
    csv_path = os.path.join(OUTPUT_DIR, 'triple_active_exemplar.csv')
    df_exemplar.to_csv(csv_path, index=False)
    print(f"\nSaved exemplar data: {csv_path}")

    # Also save top 10 scaffolds for reference
    top_scaffolds = []
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        props = row['props']
        top_scaffolds.append({
            'rank': i + 1,
            'fragment_id': row['fragment_id'],
            'fragment_smiles': row['fragment_smiles'],
            'total_compounds': row['total_compounds_all_pathogens'],
            'activity_rate': row['avg_activity_rate_percent'],
            'MW': props['MW'],
            'LogP': props['LogP'],
            'TPSA': props['TPSA'],
            'HBD': props['HBD'],
            'HBA': props['HBA'],
            'Rings': props['Rings'],
        })

    df_top = pd.DataFrame(top_scaffolds)
    top_path = os.path.join(OUTPUT_DIR, 'triple_active_top10_scaffolds.csv')
    df_top.to_csv(top_path, index=False)
    print(f"Saved top 10 scaffolds: {top_path}")

    print("\n" + "=" * 70)
    print("TRIPLE-ACTIVE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
