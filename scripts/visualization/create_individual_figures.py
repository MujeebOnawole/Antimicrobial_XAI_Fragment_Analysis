import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem import Descriptors, rdMolDescriptors
from math import pi
from io import BytesIO
from PIL import Image

# Compound data
compounds = {
    'CHEMBL4548986': {
        'smiles': 'CCN1C2=CC=CC=C2C2=CC(C)=C3C(=O)C4OC4C(=O)C3=C21',
        'name': 'S. aureus exemplar',
        'pathogen': 'S. aureus',
        'select_type': 'SELECT-G+',
        'measured': {
            'MW': 305.3,
            'LogP': 3.27,
            'TPSA': 51.6,
            'HBD': 0,
            'HBA': 4,
            'Rings': 4
        }
    },
    'CHEMBL4203270': {
        'smiles': 'CN(C)CC1=CC2=CC3=C(C=C2N1C)OCCC1=C3NC(=O)C(C(=O)O)=C1O',
        'name': 'E. coli exemplar',
        'pathogen': 'E. coli',
        'select_type': 'SELECT-G-',
        'measured': {
            'MW': 383.4,
            'LogP': 1.93,
            'TPSA': 107.8,
            'HBD': 3,
            'HBA': 8,
            'Rings': 4
        }
    },
    'CS000245342': {
        'smiles': 'S=c1oc(-c2ccccc2)nn1CN1CCCC1',
        'name': 'C. albicans exemplar',
        'pathogen': 'C. albicans',
        'select_type': 'SELECT-CA',
        'measured': {
            'MW': 261.4,
            'LogP': 2.93,
            'TPSA': 34.2,
            'HBD': 0,
            'HBA': 4,
            'Rings': 3
        }
    }
}

# SELECT zone definitions for radar plot
SELECT_ZONES = {
    'SELECT-G+': {
        'LogP': (1.9, 4.1),
        'MW': (200, 450),  # Approximate range
        'TPSA': (17, 43),
        'HBD': (0, 1),
        'HBA': (2, 6),  # Approximate
        'Rings': (2, 5)  # Approximate
    },
    'SELECT-G-': {
        'LogP': (1.3, 3.4),
        'MW': (200, 450),
        'TPSA': (25, 50),
        'HBD': (3, 5),
        'HBA': (4, 8),
        'Rings': (2, 5)
    },
    'SELECT-CA': {
        'LogP': (1.7, 3.7),
        'MW': (207, 303),
        'TPSA': (34, 38),
        'HBD': (0, 2),
        'HBA': (2, 6),
        'Rings': (3, 3)
    }
}

# Property ranges for normalization
PROPERTY_RANGES = {
    'LogP': (0, 5),
    'MW': (150, 450),
    'TPSA': (0, 150),
    'HBD': (0, 6),
    'HBA': (0, 10),
    'Rings': (0, 6)
}

def normalize_value(value, prop_name):
    """Normalize property value to 0-1 range"""
    min_val, max_val = PROPERTY_RANGES[prop_name]
    normalized = (value - min_val) / (max_val - min_val)
    return max(0, min(1, normalized))  # Clamp to [0, 1]

def get_select_zone_polygon(select_type):
    """Generate normalized polygon coordinates for SELECT zone"""
    categories = ['LogP', 'MW', 'TPSA', 'HBD', 'HBA', 'Rings']
    zone_def = SELECT_ZONES[select_type]

    # For polygon, we need vertices at each axis
    # We'll take the max value from the range to define the zone boundary
    values = []
    for cat in categories:
        if cat in zone_def:
            _, max_range = zone_def[cat]
            normalized = normalize_value(max_range, cat)
        else:
            normalized = 0.7  # Default if not specified
        values.append(normalized)

    return values

def generate_structure_image(smiles, size=(600, 500)):
    """Generate molecular structure image from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Generate 2D coordinates
    rdDepictor.Compute2DCoords(mol)

    # Generate image
    img = Draw.MolToImage(mol, size=size, fitImage=True)
    return img

def create_radar_plot(ax, compound_data, select_type):
    """Create radar plot for a single compound"""
    categories = ['LogP', 'MW', 'TPSA', 'HBD', 'HBA', 'Rings']
    N = len(categories)

    # Compute angles for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Setup polar plot
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines and labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, weight='bold')

    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=10, color='gray')

    # Add gridlines
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Plot SELECT zone as shaded region
    zone_values = get_select_zone_polygon(select_type)
    zone_values += zone_values[:1]  # Complete the circle
    ax.fill(angles, zone_values, color='green', alpha=0.15, label=f'{select_type} zone')
    ax.plot(angles, zone_values, color='green', linewidth=2, linestyle='--', alpha=0.5)

    # Plot compound values
    values = []
    for cat in categories:
        prop_value = compound_data['measured'][cat]
        normalized = normalize_value(prop_value, cat)
        values.append(normalized)

    values += values[:1]  # Complete the circle

    # Use uniform blue color for all compounds
    ax.plot(angles, values, color='#1E90FF', linewidth=3, label=compound_data['name'])
    ax.fill(angles, values, color='#1E90FF', alpha=0.25)
    ax.scatter(angles[:-1], values[:-1], color='#1E90FF', s=100, zorder=5, edgecolor='white', linewidth=2)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, fontsize=10)

def create_combined_figure(compound_id):
    """Create combined figure with structure and radar plot"""
    compound_data = compounds[compound_id]

    # Create figure with gridspec
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

    # Add title
    fig.suptitle(f"{compound_data['name']} - {compound_data['pathogen']}",
                 fontsize=20, fontweight='bold', y=0.98)

    # Left panel: Structure
    ax_struct = fig.add_subplot(gs[0])
    ax_struct.axis('off')

    # Generate structure image
    struct_img = generate_structure_image(compound_data['smiles'])
    if struct_img:
        ax_struct.imshow(struct_img)

        # Add SMILES below structure
        smiles_text = f"SMILES: {compound_data['smiles']}"
        if len(smiles_text) > 80:
            # Wrap long SMILES
            mid = len(smiles_text) // 2
            wrap_pos = smiles_text.rfind(' ', 0, mid + 20)
            if wrap_pos == -1:
                wrap_pos = mid
            smiles_text = smiles_text[:wrap_pos] + '\n' + smiles_text[wrap_pos:]

        ax_struct.text(0.5, -0.05, smiles_text,
                      transform=ax_struct.transAxes,
                      ha='center', va='top', fontsize=10,
                      family='monospace', style='italic')

    # Right panel: Radar plot
    ax_radar = fig.add_subplot(gs[1], projection='polar')
    create_radar_plot(ax_radar, compound_data, compound_data['select_type'])

    # Save figure
    output_filename = f"{compound_id}_combined.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Generated: {output_filename}")

    plt.close()

def main():
    """Generate individual combined figures for all compounds"""
    print("=" * 80)
    print("Generating Individual Compound Figures")
    print("=" * 80)

    for compound_id in compounds.keys():
        print(f"\nProcessing {compound_id}...")
        create_combined_figure(compound_id)

    print("\n" + "=" * 80)
    print("COMPLETE! Generated files:")
    for compound_id in compounds.keys():
        print(f"  - {compound_id}_combined.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
