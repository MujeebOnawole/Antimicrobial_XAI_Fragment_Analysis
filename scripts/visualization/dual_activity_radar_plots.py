"""
Dual-Activity Radar Plot Generator
===================================
Generates publication-quality radar plots for dual-active antimicrobial scaffolds
across pathogen combinations (SA+CA, SA+EC, CA+EC).

Calculates physicochemical properties and visualizes them against SELECT zones
for Gram-positive, Gram-negative, and fungal targets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import os

# Try importing RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
    from rdkit.Chem import rdDepictor
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available. Please install with: pip install rdkit")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis\DUAL_ACTIVE_POSITIVE"

# Input files
FILES = {
    'SA_CA': os.path.join(BASE_DIR, 'dual_SA_CA_positive_scaffolds.csv'),
    'SA_EC': os.path.join(BASE_DIR, 'dual_SA_EC_positive_scaffolds.csv'),
    'CA_EC': os.path.join(BASE_DIR, 'dual_CA_EC_positive_scaffolds.csv'),
}

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, 'radar_plots')

# Color schemes for each pathogen combination
COLORS = {
    'SA_CA': {'primary': '#8B008B', 'fill': '#DDA0DD', 'zone': '#FFB6C1'},  # Purple/magenta (Gram+ & Fungi)
    'SA_EC': {'primary': '#1E90FF', 'fill': '#ADD8E6', 'zone': '#87CEEB'},  # Blue (Gram+ & Gram-)
    'CA_EC': {'primary': '#228B22', 'fill': '#90EE90', 'zone': '#98FB98'},  # Green (Fungi & Gram-)
}

# SELECT zones based on literature for antimicrobial design
# These are approximate ranges from SELECT rules
SELECT_ZONES = {
    'Gram_positive': {  # S. aureus
        'MW': (200, 450),
        'LogP': (1.9, 4.1),
        'TPSA': (17, 80),  # Can tolerate higher TPSA
        'HBD': (0, 2),
        'HBA': (2, 8),
        'Rings': (2, 5),
        'RotBonds': (0, 8),
        'ArRings': (1, 4),
    },
    'Gram_negative': {  # E. coli
        'MW': (200, 500),
        'LogP': (-2, 3.4),  # Lower LogP preferred
        'TPSA': (80, 150),  # Higher polarity
        'HBD': (2, 5),
        'HBA': (4, 10),
        'Rings': (1, 4),
        'RotBonds': (0, 10),
        'ArRings': (0, 3),
    },
    'Fungi': {  # C. albicans
        'MW': (250, 550),
        'LogP': (1.7, 5.0),  # Moderate to high
        'TPSA': (34, 100),
        'HBD': (0, 3),
        'HBA': (2, 8),
        'Rings': (2, 6),
        'RotBonds': (0, 10),
        'ArRings': (1, 5),
    }
}

# Radar plot axes configuration
RADAR_AXES = {
    'primary': {
        'categories': ['LogP', 'MW', 'TPSA', 'HBD', 'HBA', 'Rings'],
        'ranges': {
            'LogP': (-2, 6),
            'MW': (100, 600),
            'TPSA': (0, 200),
            'HBD': (0, 6),
            'HBA': (0, 12),
            'Rings': (0, 8),
        }
    },
    'extended': {
        'categories': ['LogP', 'MW', 'TPSA', 'HBD', 'HBA', 'Rings', 'RotBonds', 'Fsp3'],
        'ranges': {
            'LogP': (-2, 6),
            'MW': (100, 600),
            'TPSA': (0, 200),
            'HBD': (0, 6),
            'HBA': (0, 12),
            'Rings': (0, 8),
            'RotBonds': (0, 15),
            'Fsp3': (0, 1),
        }
    }
}


# ============================================================================
# PROPERTY CALCULATION FUNCTIONS
# ============================================================================

def calculate_properties(smiles):
    """Calculate physicochemical properties from SMILES using RDKit."""
    if not RDKIT_AVAILABLE:
        return None

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
            'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
            'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol),
            'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),
            'LipinskiViolations': sum([
                Descriptors.MolWt(mol) > 500,
                Descriptors.MolLogP(mol) > 5,
                rdMolDescriptors.CalcNumHBD(mol) > 5,
                rdMolDescriptors.CalcNumHBA(mol) > 10,
            ]),
            'MembranePermeabilityScore': Descriptors.MolLogP(mol) - 0.01 * rdMolDescriptors.CalcTPSA(mol)
        }
        return props
    except Exception as e:
        print(f"Error calculating properties for {smiles}: {e}")
        return None


def load_and_process_data(filepath, top_n=5):
    """Load scaffold data and calculate properties for top compounds."""
    df = pd.read_csv(filepath)

    # Filter for high-quality scaffolds
    df_filtered = df[
        (df['avg_activity_rate_percent'] >= 90) &
        (df['total_compounds_both_pathogens'] >= 10)
    ].copy()

    # Sort by total compounds and activity rate
    df_sorted = df_filtered.sort_values(
        by=['total_compounds_both_pathogens', 'avg_activity_rate_percent', 'overall_avg_attribution'],
        ascending=[False, False, False]
    ).head(top_n)

    # Calculate properties for each scaffold
    results = []
    for idx, row in df_sorted.iterrows():
        smiles = row['fragment_smiles']
        props = calculate_properties(smiles)
        if props:
            result = {
                'fragment_id': row['fragment_id'],
                'smiles': smiles,
                'pathogen_combination': row['pathogen_combination'],
                'total_compounds': row['total_compounds_both_pathogens'],
                'activity_rate': row['avg_activity_rate_percent'],
                'avg_attribution': row['overall_avg_attribution'],
                **props
            }
            results.append(result)

    return pd.DataFrame(results)


# ============================================================================
# RADAR PLOT FUNCTIONS
# ============================================================================

def normalize_value(value, min_val, max_val):
    """Normalize a value to 0-1 range."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def create_radar_plot(compound_data, combination_name, select_zones, output_path,
                     axes_config='primary'):
    """
    Create a radar plot for a dual-active compound showing SELECT zones.

    Parameters:
    -----------
    compound_data : dict
        Dictionary containing compound properties
    combination_name : str
        Name of the pathogen combination (SA_CA, SA_EC, or CA_EC)
    select_zones : dict
        Dictionary of SELECT zones to display
    output_path : str
        Path to save the plot
    axes_config : str
        Which axes configuration to use ('primary' or 'extended')
    """
    config = RADAR_AXES[axes_config]
    categories = config['categories']
    ranges = config['ranges']

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Set the starting angle at the top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw gridlines
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],
               ['20%', '40%', '60%', '80%', '100%'],
               color='grey', size=10)
    plt.ylim(0, 1.1)

    # Set category labels
    plt.xticks(angles[:-1], categories, size=14, fontweight='bold')

    # Determine which SELECT zones to show based on combination
    zone_mapping = {
        'SA_CA': ['Gram_positive', 'Fungi'],
        'SA_EC': ['Gram_positive', 'Gram_negative'],
        'CA_EC': ['Fungi', 'Gram_negative'],
    }

    # Species names italicized per biological convention
    zone_colors = {
        'Gram_positive': ('#DC143C', r'$\it{S. aureus}$ ($G^+$)'),
        'Gram_negative': ('#1E90FF', r'$\it{E. coli}$ ($G^-$)'),
        'Fungi': ('#228B22', r'$\it{C. albicans}$'),
    }

    # Draw SELECT zones as shaded regions
    zones_to_draw = zone_mapping.get(combination_name, [])

    for zone_name in zones_to_draw:
        zone_data = SELECT_ZONES[zone_name]
        zone_color, zone_label = zone_colors[zone_name]

        # Calculate normalized zone boundaries
        zone_min_values = []
        zone_max_values = []

        for cat in categories:
            if cat in zone_data:
                zone_min, zone_max = zone_data[cat]
                range_min, range_max = ranges[cat]
                zone_min_values.append(normalize_value(zone_min, range_min, range_max))
                zone_max_values.append(normalize_value(zone_max, range_min, range_max))
            else:
                # Default to full range if not specified
                zone_min_values.append(0)
                zone_max_values.append(1)

        # Complete the loop
        zone_min_values += zone_min_values[:1]
        zone_max_values += zone_max_values[:1]

        # Draw zone as shaded band
        ax.fill(angles, zone_max_values, color=zone_color, alpha=0.1)
        ax.plot(angles, zone_max_values, color=zone_color, linewidth=1.5,
                linestyle='--', alpha=0.7)
        ax.plot(angles, zone_min_values, color=zone_color, linewidth=1.5,
                linestyle='--', alpha=0.7)

    # Plot compound data
    compound_values = []
    for cat in categories:
        value = compound_data.get(cat, 0)
        range_min, range_max = ranges[cat]
        normalized = normalize_value(value, range_min, range_max)
        compound_values.append(max(0, min(1, normalized)))  # Clip to 0-1
    compound_values += compound_values[:1]

    # Plot compound profile
    color_scheme = COLORS[combination_name]
    ax.plot(angles, compound_values, 'o-', linewidth=3,
            color=color_scheme['primary'], label=f"Exemplar Scaffold")
    ax.fill(angles, compound_values, alpha=0.3, color=color_scheme['fill'])

    # Add value labels at each point
    for i, (angle, value, cat) in enumerate(zip(angles[:-1], compound_values[:-1], categories)):
        actual_value = compound_data.get(cat, 0)
        if cat == 'MW':
            label = f'{actual_value:.0f}'
        elif cat in ['HBD', 'HBA', 'Rings', 'RotBonds', 'ArRings']:
            label = f'{int(actual_value)}'
        else:
            label = f'{actual_value:.2f}'

        # Position label slightly outside the point
        label_r = value + 0.1 if value < 0.9 else value - 0.1
        ax.annotate(label, xy=(angle, value), xytext=(angle, label_r),
                   fontsize=10, ha='center', va='center',
                   color=color_scheme['primary'], fontweight='bold')

    # Create legend
    legend_elements = [
        Line2D([0], [0], color=color_scheme['primary'], linewidth=3,
               marker='o', label='Exemplar Scaffold')
    ]

    for zone_name in zones_to_draw:
        zone_color, zone_label = zone_colors[zone_name]
        legend_elements.append(
            Line2D([0], [0], color=zone_color, linewidth=1.5,
                   linestyle='--', alpha=0.7, label=f'{zone_label} SELECT zone')
        )

    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1.35, 1.1), fontsize=11)

    # Title - species names italicized per biological convention
    combination_labels = {
        'SA_CA': r'$\it{S. aureus}$ + $\it{C. albicans}$',
        'SA_EC': r'$\it{S. aureus}$ + $\it{E. coli}$',
        'CA_EC': r'$\it{C. albicans}$ + $\it{E. coli}$',
    }

    title = f"{combination_labels[combination_name]} Exemplar\nPhysicochemical Properties"
    plt.title(title, size=16, fontweight='bold', y=1.15)

    # Add compound info as text box
    info_text = (f"Fragment ID: {compound_data.get('fragment_id', 'N/A')}\n"
                f"Activity: {compound_data.get('activity_rate', 0):.1f}%\n"
                f"Compounds: {compound_data.get('total_compounds', 0)}")

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, -0.12, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")


def generate_structure_image(smiles, output_path, legend=None):
    """Generate 2D molecular structure image."""
    if not RDKIT_AVAILABLE:
        print("RDKit not available for structure generation")
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Could not parse SMILES: {smiles}")
        return

    # Generate 2D coordinates
    rdDepictor.Compute2DCoords(mol)

    # Create image
    img = Draw.MolToImage(mol, size=(600, 500), legend=legend, fitImage=True)
    img.save(output_path)
    print(f"Saved structure: {output_path}")


def create_comparison_radar(all_compounds, output_path):
    """
    Create a single radar plot comparing exemplars from all three combinations.
    """
    config = RADAR_AXES['primary']
    categories = config['categories']
    ranges = config['ranges']

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],
               ['20%', '40%', '60%', '80%', '100%'],
               color='grey', size=10)
    plt.ylim(0, 1.1)

    plt.xticks(angles[:-1], categories, size=14, fontweight='bold')

    # Plot each compound - species names italicized per biological convention
    combination_labels = {
        'SA_CA': r'$\it{S. aureus}$ + $\it{C. albicans}$',
        'SA_EC': r'$\it{S. aureus}$ + $\it{E. coli}$',
        'CA_EC': r'$\it{C. albicans}$ + $\it{E. coli}$',
    }

    legend_elements = []

    for combo_name, compound_data in all_compounds.items():
        compound_values = []
        for cat in categories:
            value = compound_data.get(cat, 0)
            range_min, range_max = ranges[cat]
            normalized = normalize_value(value, range_min, range_max)
            compound_values.append(max(0, min(1, normalized)))
        compound_values += compound_values[:1]

        color_scheme = COLORS[combo_name]
        ax.plot(angles, compound_values, 'o-', linewidth=3,
                color=color_scheme['primary'], label=combination_labels[combo_name])
        ax.fill(angles, compound_values, alpha=0.15, color=color_scheme['fill'])

        legend_elements.append(
            Line2D([0], [0], color=color_scheme['primary'], linewidth=3,
                   marker='o', label=combination_labels[combo_name])
        )

    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1.35, 1.1), fontsize=12)

    plt.title("Dual-Active Exemplar Comparison\nPhysicochemical Profiles",
              size=18, fontweight='bold', y=1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved comparison plot: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("DUAL-ACTIVITY RADAR PLOT GENERATOR")
    print("=" * 70)

    # Check RDKit availability
    if not RDKIT_AVAILABLE:
        print("\nERROR: RDKit is required for property calculations.")
        print("Please install with: pip install rdkit")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Process each pathogen combination
    all_exemplars = {}
    all_property_data = []

    # Species names italicized per biological convention
    combination_names = {
        'SA_CA': r'$\it{S. aureus}$ + $\it{C. albicans}$ (NOT $\it{E. coli}$)',
        'SA_EC': r'$\it{S. aureus}$ + $\it{E. coli}$ (NOT $\it{C. albicans}$)',
        'CA_EC': r'$\it{C. albicans}$ + $\it{E. coli}$ (NOT $\it{S. aureus}$)',
    }

    for combo_key, filepath in FILES.items():
        print(f"\n{'='*50}")
        print(f"Processing: {combination_names[combo_key]}")
        print(f"{'='*50}")

        if not os.path.exists(filepath):
            print(f"WARNING: File not found: {filepath}")
            continue

        # Load and process data
        df_processed = load_and_process_data(filepath, top_n=3)

        if df_processed.empty:
            print("No scaffolds met filtering criteria")
            continue

        print(f"\nTop exemplars found: {len(df_processed)}")

        # Select best exemplar (rank 1)
        best_exemplar = df_processed.iloc[0].to_dict()
        all_exemplars[combo_key] = best_exemplar

        # Print exemplar info
        print(f"\n  Best Exemplar:")
        print(f"    Fragment ID: {best_exemplar['fragment_id']}")
        print(f"    SMILES: {best_exemplar['smiles']}")
        print(f"    Activity: {best_exemplar['activity_rate']:.1f}%")
        print(f"    Total Compounds: {best_exemplar['total_compounds']}")
        print(f"    MW: {best_exemplar['MW']:.1f}")
        print(f"    LogP: {best_exemplar['LogP']:.2f}")
        print(f"    TPSA: {best_exemplar['TPSA']:.1f}")
        print(f"    HBD: {best_exemplar['HBD']}, HBA: {best_exemplar['HBA']}")
        print(f"    Rings: {best_exemplar['Rings']}")

        # Generate individual radar plot
        radar_path = os.path.join(OUTPUT_DIR, f'radar_{combo_key}_exemplar.png')
        create_radar_plot(best_exemplar, combo_key, SELECT_ZONES, radar_path)

        # Generate structure image
        structure_path = os.path.join(OUTPUT_DIR, f'structure_{combo_key}_exemplar.png')
        generate_structure_image(
            best_exemplar['smiles'],
            structure_path,
            legend=f"{combination_names[combo_key]} Exemplar"
        )

        # Collect all processed data
        for _, row in df_processed.iterrows():
            row_dict = row.to_dict()
            row_dict['combination'] = combo_key
            all_property_data.append(row_dict)

    # Generate comparison radar plot
    if len(all_exemplars) >= 2:
        comparison_path = os.path.join(OUTPUT_DIR, 'radar_comparison_all_exemplars.png')
        create_comparison_radar(all_exemplars, comparison_path)

    # Save property comparison table
    if all_property_data:
        df_all = pd.DataFrame(all_property_data)
        csv_path = os.path.join(OUTPUT_DIR, 'dual_activity_property_comparison.csv')
        df_all.to_csv(csv_path, index=False)
        print(f"\nSaved property comparison: {csv_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")

    # Summary
    print("\nGenerated files:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")


if __name__ == '__main__':
    main()
