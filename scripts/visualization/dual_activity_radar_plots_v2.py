"""
Dual-Activity Radar Plot Generator v2
=====================================
Generates publication-quality radar plots for dual-active antimicrobial MOLECULES
(not scaffolds) across pathogen combinations (SA+CA, SA+EC, CA+EC).

Changes from v1:
- Uses full exemplar molecules instead of scaffold fragments
- Improved spacing for easier cropping
- Removed value annotations from within plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import matplotlib
import os

# Set Arial font for publication quality
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12

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
OUTPUT_DIR = os.path.join(BASE_DIR, 'radar_plots')

# Input file with exemplar compound details
EXEMPLAR_CSV = os.path.join(OUTPUT_DIR, 'exemplar_compounds_detailed.csv')

# Color schemes for each pathogen combination
COLORS = {
    'SA_CA': {'primary': '#8B008B', 'fill': '#DDA0DD'},  # Purple/magenta (Gram+ & Fungi)
    'SA_EC': {'primary': '#1E90FF', 'fill': '#ADD8E6'},  # Blue (Gram+ & Gram-)
    'CA_EC': {'primary': '#228B22', 'fill': '#90EE90'},  # Green (Fungi & Gram-)
}

# SELECT zones based on literature for antimicrobial design
SELECT_ZONES = {
    'Gram_positive': {  # S. aureus
        'MW': (200, 450),
        'LogP': (1.9, 4.1),
        'TPSA': (17, 80),
        'HBD': (0, 2),
        'HBA': (2, 8),
        'Rings': (2, 5),
    },
    'Gram_negative': {  # E. coli
        'MW': (200, 500),
        'LogP': (-2, 3.4),
        'TPSA': (80, 150),
        'HBD': (2, 5),
        'HBA': (4, 10),
        'Rings': (1, 4),
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

# Radar plot axes configuration
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
        }
        return props
    except Exception as e:
        print(f"Error calculating properties for {smiles}: {e}")
        return None


def normalize_value(value, min_val, max_val):
    """Normalize a value to 0-1 range."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


# ============================================================================
# RADAR PLOT FUNCTIONS
# ============================================================================

def create_radar_plot(compound_data, combination_key, output_path):
    """
    Create a clean radar plot for a dual-active molecule showing SELECT zones.

    - No value annotations on the plot
    - Improved spacing for cropping
    - Uses full molecule properties
    """
    categories = RADAR_CONFIG['categories']
    ranges = RADAR_CONFIG['ranges']

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    # Create figure with extra padding
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    # Adjust subplot position for more margin
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

    # Set the starting angle at the top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw gridlines - cleaner style
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],
               ['20%', '40%', '60%', '80%', '100%'],
               color='grey', size=9, alpha=0.7)
    plt.ylim(0, 1.15)

    # Set category labels with more distance from plot - Arial size 12
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold', fontfamily='Arial')
    ax.tick_params(axis='x', pad=25)  # Add padding to labels

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
    zones_to_draw = zone_mapping.get(combination_key, [])

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
                zone_min_values.append(0)
                zone_max_values.append(1)

        # Complete the loop
        zone_min_values += zone_min_values[:1]
        zone_max_values += zone_max_values[:1]

        # Draw zone as shaded band (only outer boundary line to avoid confusion)
        ax.fill(angles, zone_max_values, color=zone_color, alpha=0.1)
        ax.plot(angles, zone_max_values, color=zone_color, linewidth=2,
                linestyle='--', alpha=0.6)

    # Plot compound data
    compound_values = []
    for cat in categories:
        value = compound_data.get(cat, 0)
        range_min, range_max = ranges[cat]
        normalized = normalize_value(value, range_min, range_max)
        compound_values.append(max(0, min(1, normalized)))  # Clip to 0-1
    compound_values += compound_values[:1]

    # Plot compound profile - clean line without value labels
    color_scheme = COLORS[combination_key]
    ax.plot(angles, compound_values, 'o-', linewidth=3, markersize=10,
            color=color_scheme['primary'], label="Exemplar Molecule")
    ax.fill(angles, compound_values, alpha=0.25, color=color_scheme['fill'])

    # Create legend - positioned further from plot
    legend_elements = [
        Line2D([0], [0], color=color_scheme['primary'], linewidth=3,
               marker='o', markersize=8, label='Exemplar Molecule')
    ]

    for zone_name in zones_to_draw:
        zone_color, zone_label = zone_colors[zone_name]
        legend_elements.append(
            Line2D([0], [0], color=zone_color, linewidth=2,
                   linestyle='--', alpha=0.6, label=f'{zone_label} SELECT zone')
        )

    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1.45, 1.15), fontsize=12, framealpha=0.9)

    # Title - positioned further above plot, species names italicized
    combination_labels = {
        'SA_CA': r'$\it{S. aureus}$ + $\it{C. albicans}$',
        'SA_EC': r'$\it{S. aureus}$ + $\it{E. coli}$',
        'CA_EC': r'$\it{C. albicans}$ + $\it{E. coli}$',
    }

    title = f"{combination_labels[combination_key]} Exemplar\nPhysicochemical Properties"
    plt.title(title, size=18, fontweight='bold', y=1.25, pad=20)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.5)
    plt.close()

    print(f"Saved: {output_path}")


def create_comparison_radar(all_compounds, output_path):
    """
    Create a single radar plot comparing exemplar molecules from all three combinations.
    Clean version without value annotations.
    """
    categories = RADAR_CONFIG['categories']
    ranges = RADAR_CONFIG['ranges']

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],
               ['20%', '40%', '60%', '80%', '100%'],
               color='grey', size=9, alpha=0.7)
    plt.ylim(0, 1.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold', fontfamily='Arial')
    ax.tick_params(axis='x', pad=25)

    # Plot each compound - species names italicized
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
        ax.plot(angles, compound_values, 'o-', linewidth=3, markersize=10,
                color=color_scheme['primary'], label=combination_labels[combo_name])
        ax.fill(angles, compound_values, alpha=0.12, color=color_scheme['fill'])

        legend_elements.append(
            Line2D([0], [0], color=color_scheme['primary'], linewidth=3,
                   marker='o', markersize=8, label=combination_labels[combo_name])
        )

    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1.45, 1.15), fontsize=13, framealpha=0.9)

    plt.title("Dual-Active Exemplar Comparison\nPhysicochemical Profiles",
              size=20, fontweight='bold', y=1.25, pad=20)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.5)
    plt.close()

    print(f"Saved comparison plot: {output_path}")


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

    # Create image with more padding
    img = Draw.MolToImage(mol, size=(700, 600), legend=legend, fitImage=True)
    img.save(output_path)
    print(f"Saved structure: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("DUAL-ACTIVITY RADAR PLOT GENERATOR v2")
    print("Using Full Exemplar Molecules")
    print("=" * 70)

    if not RDKIT_AVAILABLE:
        print("\nERROR: RDKit is required for property calculations.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load exemplar compound data
    if not os.path.exists(EXEMPLAR_CSV):
        print(f"ERROR: Exemplar CSV not found: {EXEMPLAR_CSV}")
        print("Please run extract_exemplar_compounds.py first")
        return

    df_exemplars = pd.read_csv(EXEMPLAR_CSV)
    print(f"\nLoaded {len(df_exemplars)} exemplar compound records")

    # Map combination names
    combo_map = {
        'S. aureus + C. albicans': 'SA_CA',
        'S. aureus + E. coli': 'SA_EC',
        'C. albicans + E. coli': 'CA_EC',
    }

    all_exemplars = {}
    property_records = []

    for combo_label, combo_key in combo_map.items():
        print(f"\n{'='*50}")
        print(f"Processing: {combo_label}")
        print(f"{'='*50}")

        # Get rank 1 exemplar for this combination
        df_combo = df_exemplars[
            (df_exemplars['combination'] == combo_label) &
            (df_exemplars['exemplar_rank'] == 1)
        ]

        if df_combo.empty:
            print(f"No rank 1 exemplar found for {combo_label}")
            continue

        # Get the first compound with valid SMILES
        best_compound = None
        for _, row in df_combo.iterrows():
            compound_smiles = row['compound_smiles']
            compound_id = row['compound_id']

            # Skip if no valid compound SMILES
            if pd.isna(compound_smiles) or compound_id == 'N/A (see source data)':
                # Use fragment SMILES as fallback
                compound_smiles = row['fragment_smiles']
                compound_id = f"Fragment_{row['fragment_id']}"

            props = calculate_properties(compound_smiles)
            if props:
                best_compound = {
                    'compound_id': compound_id,
                    'compound_smiles': compound_smiles,
                    'fragment_id': row['fragment_id'],
                    'fragment_smiles': row['fragment_smiles'],
                    **props
                }
                break

        if not best_compound:
            print(f"Could not calculate properties for {combo_label}")
            continue

        all_exemplars[combo_key] = best_compound

        # Print info
        print(f"\n  Exemplar Molecule:")
        print(f"    Compound ID: {best_compound['compound_id']}")
        print(f"    SMILES: {best_compound['compound_smiles'][:60]}...")
        print(f"    MW: {best_compound['MW']:.1f}")
        print(f"    LogP: {best_compound['LogP']:.2f}")
        print(f"    TPSA: {best_compound['TPSA']:.1f}")
        print(f"    HBD: {best_compound['HBD']}, HBA: {best_compound['HBA']}")
        print(f"    Rings: {best_compound['Rings']}")

        # Generate individual radar plot
        radar_path = os.path.join(OUTPUT_DIR, f'radar_{combo_key}_molecule.png')
        create_radar_plot(best_compound, combo_key, radar_path)

        # Generate structure image
        structure_path = os.path.join(OUTPUT_DIR, f'structure_{combo_key}_molecule.png')
        generate_structure_image(
            best_compound['compound_smiles'],
            structure_path,
            legend=f"{combo_label} Exemplar"
        )

        # Store for property comparison
        record = {
            'combination': combo_label,
            'compound_id': best_compound['compound_id'],
            'compound_smiles': best_compound['compound_smiles'],
            **{k: v for k, v in best_compound.items() if k in RADAR_CONFIG['categories'] + ['RotBonds', 'ArRings', 'Fsp3']}
        }
        property_records.append(record)

    # Generate comparison radar plot
    if len(all_exemplars) >= 2:
        comparison_path = os.path.join(OUTPUT_DIR, 'radar_comparison_molecules.png')
        create_comparison_radar(all_exemplars, comparison_path)

    # Save property comparison
    if property_records:
        df_props = pd.DataFrame(property_records)
        props_path = os.path.join(OUTPUT_DIR, 'exemplar_molecule_properties.csv')
        df_props.to_csv(props_path, index=False)
        print(f"\nSaved property comparison: {props_path}")

    print("\n" + "=" * 70)
    print("RADAR PLOTS GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nOutput files in: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if 'molecule' in f:
            print(f"  - {f}")


if __name__ == '__main__':
    main()
