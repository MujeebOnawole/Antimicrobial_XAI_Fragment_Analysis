"""
Regenerate radar plots with SINGLE SHADE only (SELECT range).
No fill under compound line - just solid line with markers.
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pathlib import Path

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

OUTPUT_DIR = Path(r'C:/Users/uqaonawo/OneDrive - The University of Queensland/Desktop/fragments_analysis/figures/radar_plots')

compounds = {
    'CHEMBL4548986': {
        'smiles': 'CCN1C2=CC=CC=C2C2=CC(C)=C3C(=O)C4OC4C(=O)C3=C21',
        'name': 'S. aureus exemplar',
        'select_rule_plain': 'SELECT-G+',
        'color': '#DC143C',
        'zone_color': '#FFB6C1',
        'provided': {'MW': 305.3, 'LogP': 3.27, 'TPSA': 51.6, 'HBD': 0, 'HBA': 4, 'Rings': 4}
    },
    'CHEMBL4203270': {
        'smiles': 'CN(C)CC1=CC2=CC3=C(C=C2N1C)OCCC1=C3NC(=O)C(C(=O)O)=C1O',
        'name': 'E. coli exemplar',
        'select_rule_plain': 'SELECT-G-',
        'color': '#1E90FF',
        'zone_color': '#ADD8E6',
        'provided': {'MW': 383.4, 'LogP': 1.93, 'TPSA': 107.8, 'HBD': 3, 'HBA': 8, 'Rings': 4}
    },
    'CS000245342': {
        'smiles': 'S=c1oc(-c2ccccc2)nn1CN1CCCC1',
        'name': 'C. albicans exemplar',
        'select_rule_plain': 'SELECT-CA',
        'color': '#228B22',
        'zone_color': '#90EE90',
        'provided': {'MW': 261.4, 'LogP': 2.93, 'TPSA': 34.2, 'HBD': 0, 'HBA': 4, 'Rings': 3}
    }
}

select_ranges = {
    'CHEMBL4548986': {
        'MW': {'mean': 268.20, 'sd': 84.99}, 'LogP': {'mean': 3.00, 'sd': 1.61},
        'TPSA': {'mean': 35.09, 'sd': 21.73}, 'HBD': {'mean': 1.05, 'sd': 1.10},
        'HBA': {'mean': 3.18, 'sd': 1.69}, 'Rings': {'mean': 1.83, 'sd': 1.09}
    },
    'CHEMBL4203270': {
        'MW': {'mean': 256.91, 'sd': 81.94}, 'LogP': {'mean': 2.33, 'sd': 1.68},
        'TPSA': {'mean': 42.17, 'sd': 23.47}, 'HBD': {'mean': 1.33, 'sd': 1.10},
        'HBA': {'mean': 3.67, 'sd': 1.71}, 'Rings': {'mean': 2.87, 'sd': 1.20}
    },
    'CS000245342': {
        'MW': {'mean': 260.53, 'sd': 80.90}, 'LogP': {'mean': 2.69, 'sd': 1.48},
        'TPSA': {'mean': 38.02, 'sd': 23.83}, 'HBD': {'mean': 0.70, 'sd': 0.89},
        'HBA': {'mean': 3.54, 'sd': 2.21}, 'Rings': {'mean': 3.03, 'sd': 1.15}
    }
}

PROPERTY_SCALES = {
    'LogP': {'min': 0, 'max': 6, 'label': 'LogP\n[0-6]'},
    'MW': {'min': 0, 'max': 500, 'label': 'MW (Da)\n[0-500]'},
    'TPSA': {'min': 0, 'max': 150, 'label': 'TPSA (\u00C5\u00B2)\n[0-150]'},
    'HBD': {'min': 0, 'max': 5, 'label': 'HBD\n[0-5]'},
    'HBA': {'min': 0, 'max': 10, 'label': 'HBA\n[0-10]'},
    'Rings': {'min': 0, 'max': 6, 'label': 'Rings\n[0-6]'}
}
PROPERTIES = ['LogP', 'MW', 'TPSA', 'HBD', 'HBA', 'Rings']


def normalize_value(value, prop_name):
    scale = PROPERTY_SCALES[prop_name]
    return max(0, min(100, (value - scale['min']) / (scale['max'] - scale['min']) * 100))


for compound_id, data in compounds.items():
    select = select_ranges[compound_id]
    compound_props = data['provided']

    compound_norm = [normalize_value(compound_props.get(prop, 0), prop) for prop in PROPERTIES]
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

    # ===== RADAR ONLY =====
    fig = plt.figure(figsize=(9, 10))
    ax = fig.add_axes([0.1, 0.05, 0.75, 0.75], projection='polar')

    # Only shade the SELECT range - NO fill under compound line
    ax.fill_between(angles, select_lower, select_upper, alpha=0.3, color=data['zone_color'],
                    label=f"{data['select_rule_plain']} range")
    ax.plot(angles, select_mean_norm, '--', linewidth=1.5, color='gray', alpha=0.7,
            label=f"{data['select_rule_plain']} mean")

    # Compound as solid line with markers only - NO FILL
    ax.plot(angles, compound_norm, 'o-', linewidth=2.5, markersize=8,
            color=data['color'], label=compound_id)

    labels = [PROPERTY_SCALES[prop]['label'] for prop in PROPERTIES]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontfamily='Arial')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=12, fontfamily='Arial')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)

    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.35, -0.1), fontsize=12, framealpha=0.9, edgecolor='gray')
    for text in legend.get_texts():
        text.set_fontfamily('Arial')

    fig.text(0.5, 0.92, f'Physicochemical Profile: {compound_id}', ha='center', fontsize=12, fontweight='bold', fontfamily='Arial')
    fig.text(0.5, 0.88, f"vs {data['select_rule_plain']}", ha='center', fontsize=12, fontweight='bold', fontfamily='Arial')

    plt.savefig(OUTPUT_DIR / f'{compound_id}_radar_only.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[OK] {compound_id}_radar_only.png')

    # ===== COMBINED =====
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121)
    mol = Chem.MolFromSmiles(data['smiles'])
    if mol:
        rdDepictor.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(400, 400))
        ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('(a) Molecular Structure', fontsize=12, fontweight='bold', fontfamily='Arial', pad=15)
    ax1.text(0.5, -0.08, f'Exemplar: {compound_id}', transform=ax1.transAxes, ha='center', fontsize=12, style='italic', fontfamily='Arial')

    ax2 = fig.add_subplot(122, projection='polar')
    ax2.fill_between(angles, select_lower, select_upper, alpha=0.3, color=data['zone_color'],
                     label=f"{data['select_rule_plain']} range")
    ax2.plot(angles, select_mean_norm, '--', linewidth=1.5, color='gray', alpha=0.7,
             label=f"{data['select_rule_plain']} mean")
    ax2.plot(angles, compound_norm, 'o-', linewidth=2.5, markersize=8,
             color=data['color'], label=compound_id)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels, fontsize=12, fontfamily='Arial')
    ax2.set_ylim(0, 100)
    ax2.set_yticks([20, 40, 60, 80, 100])
    ax2.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=12, fontfamily='Arial')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax2.xaxis.grid(True, linestyle='-', alpha=0.3)

    legend = ax2.legend(loc='lower right', bbox_to_anchor=(1.35, -0.1), fontsize=12, framealpha=0.9, edgecolor='gray')
    for text in legend.get_texts():
        text.set_fontfamily('Arial')

    ax2.set_title(f"(b) Physicochemical Profile vs {data['select_rule_plain']}", fontsize=12, fontweight='bold', fontfamily='Arial', pad=20)
    fig.suptitle(f"{data['name'].title()}: {compound_id}", fontsize=14, fontweight='bold', fontfamily='Arial', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / f'{compound_id}_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[OK] {compound_id}_combined.png')

print('\nAll plots regenerated - SINGLE SHADE only (SELECT range)')
print('Compound shown as solid line with markers (no fill)')
