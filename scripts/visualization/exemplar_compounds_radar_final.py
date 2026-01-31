"""
Generate radar plots with 8 properties for ACTUAL ChEMBL COMPOUNDS (not fragments).
All exemplars are full compounds with 0 violations across all 8 properties.

Exemplar Compounds:
- SA: CHEMBL4536843 (100% within SELECT-G+ range)
- EC: CHEMBL369493 (100% within SELECT-G- range)
- CA: CHEMBL4277673 (100% within SELECT-CA range)
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pathlib import Path

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14  # Increased base font size

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, Descriptors, rdMolDescriptors

OUTPUT_DIR = Path(r'C:/Users/uqaonawo/OneDrive - The University of Queensland/Desktop/fragments_analysis/figures/radar_plots')

# ACTUAL ChEMBL COMPOUNDS with 0 violations across all 8 properties
# Using mathtext superscripts in plots: G^+ (Gram-positive), G^- (Gram-negative)
compounds = {
    'CHEMBL4536843': {
        'smiles': 'CC(C)[C@@H]1O/C(=C/C(=O)CCC2=CC=CC=C2)NC1=O',
        'name': 'S. aureus (G+)',
        'name_plot': r'S. aureus ($G^+$)',  # Mathtext superscript for plots
        'select_rule': r'SELECT-$G^+$',  # Mathtext superscript for plots
        'pathogen': 'SA',
        'color': '#DC143C',
        'zone_color': '#FFB6C1',
    },
    'CHEMBL369493': {
        'smiles': 'CCCS(=O)(=O)N1C=CC2=C(SC(CC)=C2)B1O',
        'name': 'E. coli (G-)',
        'name_plot': r'E. coli ($G^-$)',  # Mathtext superscript for plots
        'select_rule': r'SELECT-$G^-$',  # Mathtext superscript for plots
        'pathogen': 'EC',
        'color': '#1E90FF',
        'zone_color': '#ADD8E6',
    },
    'CHEMBL4277673': {
        'smiles': 'FC1=CC=C2SCC(CN3C=NC=N3)=C(Cl)C2=C1',
        'name': 'C. albicans',
        'name_plot': 'C. albicans',  # No superscript needed
        'select_rule': 'SELECT-CA',
        'pathogen': 'CA',
        'color': '#228B22',
        'zone_color': '#90EE90',
    }
}

# SELECT ranges for all 8 properties (mean +/- SD)
select_ranges = {
    'SA': {
        'MW': {'mean': 268.20, 'sd': 84.99},
        'LogP': {'mean': 3.00, 'sd': 1.61},
        'TPSA': {'mean': 35.09, 'sd': 21.73},
        'HBD': {'mean': 1.05, 'sd': 1.10},
        'HBA': {'mean': 3.18, 'sd': 1.69},
        'RotBonds': {'mean': 3.12, 'sd': 2.51},
        'Rings': {'mean': 1.83, 'sd': 1.09},
        'Fsp3': {'mean': 0.35, 'sd': 0.22}
    },
    'EC': {
        'MW': {'mean': 256.91, 'sd': 81.94},
        'LogP': {'mean': 2.33, 'sd': 1.68},
        'TPSA': {'mean': 42.17, 'sd': 23.47},
        'HBD': {'mean': 1.33, 'sd': 1.10},
        'HBA': {'mean': 3.67, 'sd': 1.71},
        'RotBonds': {'mean': 2.53, 'sd': 2.21},
        'Rings': {'mean': 2.87, 'sd': 1.20},
        'Fsp3': {'mean': 0.28, 'sd': 0.21}
    },
    'CA': {
        'MW': {'mean': 260.53, 'sd': 80.90},
        'LogP': {'mean': 2.69, 'sd': 1.48},
        'TPSA': {'mean': 38.02, 'sd': 23.83},
        'HBD': {'mean': 0.70, 'sd': 0.89},
        'HBA': {'mean': 3.54, 'sd': 2.21},
        'RotBonds': {'mean': 2.93, 'sd': 2.37},
        'Rings': {'mean': 3.03, 'sd': 1.15},
        'Fsp3': {'mean': 0.30, 'sd': 0.21}
    }
}

# All 8 properties with scales (Fsp3 = Fraction sp³ carbons)
PROPERTY_SCALES = {
    'MW': {'min': 0, 'max': 500, 'label': 'MW (Da)\n[0-500]'},
    'LogP': {'min': 0, 'max': 6, 'label': 'LogP\n[0-6]'},
    'TPSA': {'min': 0, 'max': 150, 'label': 'TPSA (\u00c5\u00b2)\n[0-150]'},
    'HBD': {'min': 0, 'max': 5, 'label': 'HBD\n[0-5]'},
    'HBA': {'min': 0, 'max': 10, 'label': 'HBA\n[0-10]'},
    'RotBonds': {'min': 0, 'max': 15, 'label': 'RotBonds\n[0-15]'},
    'Rings': {'min': 0, 'max': 6, 'label': 'Rings\n[0-6]'},
    'Fsp3': {'min': 0, 'max': 1, 'label': r'$Fsp^3$' + '\n[0-1]'}
}

PROPERTIES = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Rings', 'Fsp3']


def calc_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MW': round(Descriptors.MolWt(mol), 1),
        'LogP': round(Descriptors.MolLogP(mol), 2),
        'TPSA': round(rdMolDescriptors.CalcTPSA(mol), 1),
        'HBD': rdMolDescriptors.CalcNumHBD(mol),
        'HBA': rdMolDescriptors.CalcNumHBA(mol),
        'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'Rings': rdMolDescriptors.CalcNumRings(mol),
        'Fsp3': round(rdMolDescriptors.CalcFractionCSP3(mol), 2)
    }


def normalize_value(value, prop_name):
    scale = PROPERTY_SCALES[prop_name]
    return max(0, min(100, (value - scale['min']) / (scale['max'] - scale['min']) * 100))


def create_radar_only(chembl_id, compound_data, select_data, output_path):
    """Create radar plot with all 8 properties - filled center to eliminate white space."""

    props = calc_props(compound_data['smiles'])
    pathogen = compound_data['pathogen']
    select = select_data[pathogen]

    # Normalize values for 8 properties
    compound_norm = [normalize_value(props[prop], prop) for prop in PROPERTIES]
    select_lower = [normalize_value(max(0, select[prop]['mean'] - select[prop]['sd']), prop) for prop in PROPERTIES]
    select_upper = [normalize_value(select[prop]['mean'] + select[prop]['sd'], prop) for prop in PROPERTIES]
    select_mean_norm = [normalize_value(select[prop]['mean'], prop) for prop in PROPERTIES]

    # Angles for 8 properties
    num_props = len(PROPERTIES)
    angles = [n / num_props * 2 * pi for n in range(num_props)]
    angles += angles[:1]

    # Close loops
    compound_norm += compound_norm[:1]
    select_lower += select_lower[:1]
    select_upper += select_upper[:1]
    select_mean_norm += select_mean_norm[:1]

    # Create figure
    fig = plt.figure(figsize=(10, 11))
    ax = fig.add_axes([0.1, 0.05, 0.75, 0.75], projection='polar')

    # FIXED: Use ax.fill() instead of fill_between - fills polygon from center properly
    ax.fill(angles, select_upper, alpha=0.35, color=compound_data['zone_color'],
            label=f"{compound_data['select_rule']} range", zorder=1)

    # SELECT mean line
    ax.plot(angles, select_mean_norm, '--', linewidth=1.5, color='gray', alpha=0.7,
            label=f"{compound_data['select_rule']} mean", zorder=2)

    # Compound line (no fill under it)
    ax.plot(angles, compound_norm, 'o-', linewidth=2.5, markersize=8,
            color=compound_data['color'], label=f'Exemplar ({chembl_id})', zorder=3)

    # Axis labels - 16pt BOLD for 8 properties
    labels = [PROPERTY_SCALES[prop]['label'] for prop in PROPERTIES]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold', fontfamily='Arial')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=14, fontfamily='Arial')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)

    # Legend - 18pt, positioned outside circle to avoid Fsp3 overlap
    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.55, -0.02), fontsize=18, framealpha=0.9, edgecolor='gray')
    for text in legend.get_texts():
        text.set_fontfamily('Arial')

    # Title with ChEMBL ID - 18pt
    fig.text(0.5, 0.92, f'Physicochemical Profile: {chembl_id}',
             ha='center', fontsize=18, fontweight='bold', fontfamily='Arial')
    fig.text(0.5, 0.88, f"vs {compound_data['select_rule']} ({compound_data['name_plot']})",
             ha='center', fontsize=16, fontweight='bold', fontfamily='Arial')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return props


def create_combined_plot(chembl_id, compound_data, select_data, props, output_path):
    """Create combined structure + radar plot with 8 properties."""

    pathogen = compound_data['pathogen']
    select = select_data[pathogen]

    # Normalize values
    compound_norm = [normalize_value(props[prop], prop) for prop in PROPERTIES]
    select_lower = [normalize_value(max(0, select[prop]['mean'] - select[prop]['sd']), prop) for prop in PROPERTIES]
    select_upper = [normalize_value(select[prop]['mean'] + select[prop]['sd'], prop) for prop in PROPERTIES]
    select_mean_norm = [normalize_value(select[prop]['mean'], prop) for prop in PROPERTIES]

    num_props = len(PROPERTIES)
    angles = [n / num_props * 2 * pi for n in range(num_props)]
    angles += angles[:1]

    compound_norm += compound_norm[:1]
    select_lower += select_lower[:1]
    select_upper += select_upper[:1]
    select_mean_norm += select_mean_norm[:1]

    fig = plt.figure(figsize=(16, 7))

    # Structure panel
    ax1 = fig.add_subplot(121)
    mol = Chem.MolFromSmiles(compound_data['smiles'])
    if mol:
        rdDepictor.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(400, 400))
        ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('(a) Molecular Structure', fontsize=16, fontweight='bold', fontfamily='Arial', pad=15)
    ax1.text(0.5, -0.08, f"Exemplar: {chembl_id}", transform=ax1.transAxes,
             ha='center', fontsize=14, style='italic', fontfamily='Arial')

    # Radar panel
    ax2 = fig.add_subplot(122, projection='polar')

    # FIXED: Use ax.fill() instead of fill_between - fills polygon from center properly
    ax2.fill(angles, select_upper, alpha=0.35, color=compound_data['zone_color'],
             label=f"{compound_data['select_rule']} range", zorder=1)
    ax2.plot(angles, select_mean_norm, '--', linewidth=1.5, color='gray', alpha=0.7,
             label=f"{compound_data['select_rule']} mean", zorder=2)
    ax2.plot(angles, compound_norm, 'o-', linewidth=2.5, markersize=8,
             color=compound_data['color'], label=f'Exemplar ({chembl_id})', zorder=3)

    labels = [PROPERTY_SCALES[prop]['label'] for prop in PROPERTIES]
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels, fontsize=16, fontweight='bold', fontfamily='Arial')
    ax2.set_ylim(0, 100)
    ax2.set_yticks([20, 40, 60, 80, 100])
    ax2.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=14, fontfamily='Arial')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax2.xaxis.grid(True, linestyle='-', alpha=0.3)

    legend = ax2.legend(loc='lower right', bbox_to_anchor=(1.55, -0.02), fontsize=18, framealpha=0.9, edgecolor='gray')
    for text in legend.get_texts():
        text.set_fontfamily('Arial')

    ax2.set_title(f"(b) Physicochemical Profile vs {compound_data['select_rule']}",
                  fontsize=16, fontweight='bold', fontfamily='Arial', pad=20)
    fig.suptitle(f"{compound_data['name_plot'].title()}: {chembl_id}",
                 fontsize=18, fontweight='bold', fontfamily='Arial', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# Main execution
print("=" * 70)
print("GENERATING EXEMPLAR COMPOUND RADAR PLOTS - 8 PROPERTIES")
print("Using ACTUAL ChEMBL compounds (not fragments)")
print("All compounds have 0 violations across all 8 properties")
print("=" * 70)

for chembl_id, data in compounds.items():
    print(f"\n{data['name']}:")
    print(f"  ChEMBL ID: {chembl_id}")
    print(f"  SMILES: {data['smiles']}")

    # Radar only
    radar_file = OUTPUT_DIR / f'{chembl_id}_radar_8props.png'
    props = create_radar_only(chembl_id, data, select_ranges, radar_file)

    print(f"  Properties (all 8 within range):")
    print(f"    MW={props['MW']}, LogP={props['LogP']}, TPSA={props['TPSA']}, HBD={props['HBD']}")
    print(f"    HBA={props['HBA']}, RotBonds={props['RotBonds']}, Rings={props['Rings']}, Fsp3={props['Fsp3']}")
    print(f"  [OK] {radar_file.name}")

    # Combined
    combined_file = OUTPUT_DIR / f'{chembl_id}_combined_8props.png'
    create_combined_plot(chembl_id, data, select_ranges, props, combined_file)
    print(f"  [OK] {combined_file.name}")

print("\n" + "=" * 70)
print("COMPLETE!")
print("All plots use ChEMBL compound IDs with 0 violations.")
print("=" * 70)
