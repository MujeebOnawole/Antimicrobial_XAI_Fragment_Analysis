"""
Generate Radar Plots for Exemplar Compounds - Version 2
Creates two versions:
1. Combined (structure + radar) - for tracking/reference
2. Radar-only - with legend at bottom right for PowerPoint cropping

Three exemplar compounds:
1. CHEMBL4548986 - S. aureus (SELECT-G+)
2. CHEMBL4203270 - E. coli (SELECT-G-)
3. CS000245342 - C. albicans (SELECT-CA)
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pathlib import Path

# Try to import RDKit for structure generation
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Combined plots will not include structures.")

# Set paths
BASE_DIR = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis")
OUTPUT_DIR = BASE_DIR / "figures" / "radar_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# COMPOUND DATA
# =============================================================================

compounds = {
    'CHEMBL4548986': {
        'smiles': 'CCN1C2=CC=CC=C2C2=CC(C)=C3C(=O)C4OC4C(=O)C3=C21',
        'name': 'S. aureus exemplar',
        'select_rule': 'SELECT-G+',
        'pathogen': 'SA',
        'color': '#DC143C',  # Crimson red
        'zone_color': '#FFB6C1',  # Light pink
        'provided': {'MW': 305.3, 'LogP': 3.27, 'TPSA': 51.6, 'HBD': 0, 'HBA': 4, 'Rings': 4}
    },
    'CHEMBL4203270': {
        'smiles': 'CN(C)CC1=CC2=CC3=C(C=C2N1C)OCCC1=C3NC(=O)C(C(=O)O)=C1O',
        'name': 'E. coli exemplar',
        'select_rule': 'SELECT-G-',
        'pathogen': 'EC',
        'color': '#1E90FF',  # Dodger blue
        'zone_color': '#ADD8E6',  # Light blue
        'provided': {'MW': 383.4, 'LogP': 1.93, 'TPSA': 107.8, 'HBD': 3, 'HBA': 8, 'Rings': 4}
    },
    'CS000245342': {
        'smiles': 'S=c1oc(-c2ccccc2)nn1CN1CCCC1',
        'name': 'C. albicans exemplar',
        'select_rule': 'SELECT-CA',
        'pathogen': 'CA',
        'color': '#228B22',  # Forest green
        'zone_color': '#90EE90',  # Light green
        'provided': {'MW': 261.4, 'LogP': 2.93, 'TPSA': 34.2, 'HBD': 0, 'HBA': 4, 'Rings': 3}
    }
}

# =============================================================================
# SELECT RANGES (mean +/- SD from statistical analysis)
# =============================================================================

select_ranges = {
    'SA': {
        'MW': {'mean': 268.20, 'sd': 84.99},
        'LogP': {'mean': 3.00, 'sd': 1.61},
        'TPSA': {'mean': 35.09, 'sd': 21.73},
        'HBD': {'mean': 1.05, 'sd': 1.10},
        'HBA': {'mean': 3.18, 'sd': 1.69},
        'Rings': {'mean': 1.83, 'sd': 1.09}
    },
    'EC': {
        'MW': {'mean': 256.91, 'sd': 81.94},
        'LogP': {'mean': 2.33, 'sd': 1.68},
        'TPSA': {'mean': 42.17, 'sd': 23.47},
        'HBD': {'mean': 1.33, 'sd': 1.10},
        'HBA': {'mean': 3.67, 'sd': 1.71},
        'Rings': {'mean': 2.87, 'sd': 1.20}
    },
    'CA': {
        'MW': {'mean': 260.53, 'sd': 80.90},
        'LogP': {'mean': 2.69, 'sd': 1.48},
        'TPSA': {'mean': 38.02, 'sd': 23.83},
        'HBD': {'mean': 0.70, 'sd': 0.89},
        'HBA': {'mean': 3.54, 'sd': 2.21},
        'Rings': {'mean': 3.03, 'sd': 1.15}
    }
}

# Property scales for radar plot axes
PROPERTY_SCALES = {
    'LogP': {'min': 0, 'max': 6, 'unit': ''},
    'MW': {'min': 0, 'max': 500, 'unit': 'Da'},
    'TPSA': {'min': 0, 'max': 150, 'unit': 'A^2'},
    'HBD': {'min': 0, 'max': 5, 'unit': ''},
    'HBA': {'min': 0, 'max': 10, 'unit': ''},
    'Rings': {'min': 0, 'max': 6, 'unit': ''}
}

PROPERTIES = ['LogP', 'MW', 'TPSA', 'HBD', 'HBA', 'Rings']


def normalize_value(value, prop_name):
    """Normalize value to 0-100 scale based on property range."""
    scale = PROPERTY_SCALES[prop_name]
    normalized = (value - scale['min']) / (scale['max'] - scale['min']) * 100
    return max(0, min(100, normalized))


def create_radar_only(compound_id, compound_data, select_data, output_path):
    """
    Create radar-only plot with legend at bottom right.
    For use in PowerPoint with ChemDraw structures.
    """
    compound_props = compound_data['provided']
    pathogen = compound_data['pathogen']
    select = select_data[pathogen]

    # Calculate normalized values
    compound_norm = [normalize_value(compound_props.get(prop, 0), prop) for prop in PROPERTIES]

    select_lower = [normalize_value(max(0, select[prop]['mean'] - select[prop]['sd']), prop) for prop in PROPERTIES]
    select_upper = [normalize_value(select[prop]['mean'] + select[prop]['sd'], prop) for prop in PROPERTIES]
    select_mean_norm = [normalize_value(select[prop]['mean'], prop) for prop in PROPERTIES]

    # Set up angles
    num_props = len(PROPERTIES)
    angles = [n / num_props * 2 * pi for n in range(num_props)]
    angles += angles[:1]

    # Close loops
    compound_norm += compound_norm[:1]
    select_lower += select_lower[:1]
    select_upper += select_upper[:1]
    select_mean_norm += select_mean_norm[:1]

    # Create figure - square format for radar
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')

    # Plot SELECT zone as shaded area
    ax.fill_between(angles, select_lower, select_upper,
                    alpha=0.3, color=compound_data['zone_color'],
                    label=f'{compound_data["select_rule"]} range')

    # Plot SELECT mean as dashed line
    ax.plot(angles, select_mean_norm, '--', linewidth=1.5,
            color='gray', alpha=0.7, label=f'{compound_data["select_rule"]} mean')

    # Plot compound values
    ax.plot(angles, compound_norm, 'o-', linewidth=2.5, markersize=8,
            color=compound_data['color'], label=compound_id)
    ax.fill(angles, compound_norm, alpha=0.1, color=compound_data['color'])

    # Axis labels with property scales
    labels = []
    for prop in PROPERTIES:
        scale = PROPERTY_SCALES[prop]
        unit = f" ({scale['unit']})" if scale['unit'] else ""
        labels.append(f"{prop}{unit}\n[0-{scale['max']}]")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    # Radial limits and labels
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)

    # Gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)

    # Legend at BOTTOM RIGHT - outside the plot area
    ax.legend(loc='lower right', bbox_to_anchor=(1.25, -0.05), fontsize=10,
              framealpha=0.9, edgecolor='gray')

    # Title
    ax.set_title(f"Physicochemical Profile: {compound_id}\nvs {compound_data['select_rule']}",
                 fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return True


def create_combined_plot(compound_id, compound_data, select_data, output_path):
    """
    Create combined plot (structure + radar) for tracking/reference.
    """
    compound_props = compound_data['provided']
    pathogen = compound_data['pathogen']
    select = select_data[pathogen]

    # Calculate normalized values
    compound_norm = [normalize_value(compound_props.get(prop, 0), prop) for prop in PROPERTIES]

    select_lower = [normalize_value(max(0, select[prop]['mean'] - select[prop]['sd']), prop) for prop in PROPERTIES]
    select_upper = [normalize_value(select[prop]['mean'] + select[prop]['sd'], prop) for prop in PROPERTIES]
    select_mean_norm = [normalize_value(select[prop]['mean'], prop) for prop in PROPERTIES]

    # Set up angles
    num_props = len(PROPERTIES)
    angles = [n / num_props * 2 * pi for n in range(num_props)]
    angles += angles[:1]

    # Close loops
    compound_norm += compound_norm[:1]
    select_lower += select_lower[:1]
    select_upper += select_upper[:1]
    select_mean_norm += select_mean_norm[:1]

    # Create figure with two panels
    fig = plt.figure(figsize=(14, 6))

    # Panel (a): Molecular structure
    ax1 = fig.add_subplot(121)

    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(compound_data['smiles'])
        if mol:
            rdDepictor.Compute2DCoords(mol)
            img = Draw.MolToImage(mol, size=(400, 400))
            ax1.imshow(img)

    ax1.axis('off')
    ax1.set_title(f"(a) Molecular Structure", fontsize=12, fontweight='bold', pad=10)
    ax1.text(0.5, -0.05, f"Exemplar: {compound_id}", transform=ax1.transAxes,
             ha='center', fontsize=11, style='italic')

    # Panel (b): Radar plot
    ax2 = fig.add_subplot(122, projection='polar')

    # Plot SELECT zone
    ax2.fill_between(angles, select_lower, select_upper,
                     alpha=0.3, color=compound_data['zone_color'],
                     label=f'{compound_data["select_rule"]} range')

    # Plot SELECT mean
    ax2.plot(angles, select_mean_norm, '--', linewidth=1.5,
             color='gray', alpha=0.7, label=f'{compound_data["select_rule"]} mean')

    # Plot compound
    ax2.plot(angles, compound_norm, 'o-', linewidth=2.5, markersize=8,
             color=compound_data['color'], label=compound_id)
    ax2.fill(angles, compound_norm, alpha=0.1, color=compound_data['color'])

    # Axis labels
    labels = []
    for prop in PROPERTIES:
        scale = PROPERTY_SCALES[prop]
        unit = f" ({scale['unit']})" if scale['unit'] else ""
        labels.append(f"{prop}{unit}\n[0-{scale['max']}]")

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([20, 40, 60, 80, 100])
    ax2.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)

    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax2.xaxis.grid(True, linestyle='-', alpha=0.3)

    # Legend at bottom right
    ax2.legend(loc='lower right', bbox_to_anchor=(1.25, -0.05), fontsize=9,
               framealpha=0.9, edgecolor='gray')

    ax2.set_title(f"(b) Physicochemical Profile vs {compound_data['select_rule']}",
                  fontsize=12, fontweight='bold', pad=20)

    # Main title
    fig.suptitle(f"{compound_data['name'].title()}: {compound_id}",
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING EXEMPLAR RADAR PLOTS - VERSION 2")
    print("=" * 70)

    print("\nCreating two versions per compound:")
    print("  1. Combined (structure + radar) - for tracking")
    print("  2. Radar-only (legend at bottom right) - for PowerPoint")
    print()

    for compound_id, data in compounds.items():
        print(f"\n{compound_id} ({data['name']}):")

        # Create radar-only version
        radar_file = OUTPUT_DIR / f"{compound_id}_radar_only.png"
        success = create_radar_only(compound_id, data, select_ranges, radar_file)
        if success:
            print(f"  [OK] Radar-only: {radar_file.name}")

        # Create combined version
        combined_file = OUTPUT_DIR / f"{compound_id}_combined.png"
        success = create_combined_plot(compound_id, data, select_ranges, combined_file)
        if success:
            print(f"  [OK] Combined:   {combined_file.name}")

    print("\n" + "=" * 70)
    print("FILES GENERATED:")
    print("=" * 70)

    print("\nFor PowerPoint (radar-only, legend at bottom right):")
    for compound_id in compounds.keys():
        print(f"  - {compound_id}_radar_only.png")

    print("\nFor tracking/reference (combined structure + radar):")
    for compound_id in compounds.keys():
        print(f"  - {compound_id}_combined.png")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
