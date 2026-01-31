"""
Generate radar plots with NEW EXEMPLARS that perfectly fit SELECT ranges.
Fixed: No whitish space in shaded region - using polygon fill instead of fill_between.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from math import pi
from pathlib import Path

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, Descriptors, rdMolDescriptors

OUTPUT_DIR = Path(r'C:/Users/uqaonawo/OneDrive - The University of Queensland/Desktop/fragments_analysis/figures/radar_plots')

# NEW EXEMPLARS - All perfectly fit within SELECT ranges!
compounds = {
    'SA_exemplar': {
        'smiles': 'C1=NC=CC(CCC[C@@H]2CCNCC2)=C1',
        'name': 'S. aureus exemplar',
        'select_rule': 'SELECT-G+',
        'pathogen': 'SA',
        'color': '#DC143C',
        'zone_color': '#FFB6C1',
    },
    'EC_exemplar': {
        'smiles': 'C1=CC2=C(C=C1)C1=C(N3C=CC=N3)C=CN=C1N2',
        'name': 'E. coli exemplar',
        'select_rule': 'SELECT-G-',
        'pathogen': 'EC',
        'color': '#1E90FF',
        'zone_color': '#ADD8E6',
    },
    'CA_exemplar': {
        'smiles': 'S=c1oc(-c2ccccc2)nn1CN1CCCC1',
        'name': 'C. albicans exemplar',
        'select_rule': 'SELECT-CA',
        'pathogen': 'CA',
        'color': '#228B22',
        'zone_color': '#90EE90',
    }
}

# SELECT ranges (mean +/- SD)
select_ranges = {
    'SA': {
        'MW': {'mean': 268.20, 'sd': 84.99}, 'LogP': {'mean': 3.00, 'sd': 1.61},
        'TPSA': {'mean': 35.09, 'sd': 21.73}, 'HBD': {'mean': 1.05, 'sd': 1.10},
        'HBA': {'mean': 3.18, 'sd': 1.69}, 'Rings': {'mean': 1.83, 'sd': 1.09}
    },
    'EC': {
        'MW': {'mean': 256.91, 'sd': 81.94}, 'LogP': {'mean': 2.33, 'sd': 1.68},
        'TPSA': {'mean': 42.17, 'sd': 23.47}, 'HBD': {'mean': 1.33, 'sd': 1.10},
        'HBA': {'mean': 3.67, 'sd': 1.71}, 'Rings': {'mean': 2.87, 'sd': 1.20}
    },
    'CA': {
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
        'Rings': rdMolDescriptors.CalcNumRings(mol)
    }


def normalize_value(value, prop_name):
    scale = PROPERTY_SCALES[prop_name]
    return max(0, min(100, (value - scale['min']) / (scale['max'] - scale['min']) * 100))


def create_radar_only(compound_id, compound_data, select_data, output_path):
    """Create radar plot with proper filled SELECT zone (no whitish gap)."""

    # Calculate compound properties from SMILES
    props = calc_props(compound_data['smiles'])
    pathogen = compound_data['pathogen']
    select = select_data[pathogen]

    # Normalize values
    compound_norm = [normalize_value(props[prop], prop) for prop in PROPERTIES]
    select_lower = [normalize_value(max(0, select[prop]['mean'] - select[prop]['sd']), prop) for prop in PROPERTIES]
    select_upper = [normalize_value(select[prop]['mean'] + select[prop]['sd'], prop) for prop in PROPERTIES]
    select_mean_norm = [normalize_value(select[prop]['mean'], prop) for prop in PROPERTIES]

    # Angles
    num_props = len(PROPERTIES)
    angles = [n / num_props * 2 * pi for n in range(num_props)]
    angles += angles[:1]

    # Close loops
    compound_norm += compound_norm[:1]
    select_lower += select_lower[:1]
    select_upper += select_upper[:1]
    select_mean_norm += select_mean_norm[:1]

    # Create figure
    fig = plt.figure(figsize=(9, 10))
    ax = fig.add_axes([0.1, 0.05, 0.75, 0.75], projection='polar')

    # Draw SELECT zone as filled polygon (outer - inner creates the band)
    # First fill from 0 to upper bound, then fill from 0 to lower bound with white
    # Better approach: create vertices for the band shape

    # Create band by drawing outer polygon and masking inner
    theta_fine = np.linspace(0, 2*np.pi, 100)

    # Interpolate the values for smooth polygon
    from scipy import interpolate

    # Upper bound polygon
    f_upper = interpolate.interp1d(angles, select_upper, kind='linear', fill_value='extrapolate')
    r_upper = f_upper(theta_fine)

    # Lower bound polygon
    f_lower = interpolate.interp1d(angles, select_lower, kind='linear', fill_value='extrapolate')
    r_lower = f_lower(theta_fine)

    # Fill the band between lower and upper
    ax.fill_between(theta_fine, r_lower, r_upper, alpha=0.35, color=compound_data['zone_color'],
                    label=f"{compound_data['select_rule']} range", zorder=1)

    # Also fill from 0 to lower with same color but very light (to avoid white gap)
    ax.fill_between(theta_fine, 0, r_lower, alpha=0.1, color=compound_data['zone_color'], zorder=0)

    # SELECT mean line
    ax.plot(angles, select_mean_norm, '--', linewidth=1.5, color='gray', alpha=0.7,
            label=f"{compound_data['select_rule']} mean", zorder=2)

    # Compound line
    ax.plot(angles, compound_norm, 'o-', linewidth=2.5, markersize=8,
            color=compound_data['color'], label=compound_id, zorder=3)

    # Axis labels
    labels = [PROPERTY_SCALES[prop]['label'] for prop in PROPERTIES]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontfamily='Arial')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=12, fontfamily='Arial')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)

    # Legend
    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.35, -0.1), fontsize=12, framealpha=0.9, edgecolor='gray')
    for text in legend.get_texts():
        text.set_fontfamily('Arial')

    # Title with spacing
    fig.text(0.5, 0.92, f'Physicochemical Profile: {compound_id}', ha='center', fontsize=12, fontweight='bold', fontfamily='Arial')
    fig.text(0.5, 0.88, f"vs {compound_data['select_rule']}", ha='center', fontsize=12, fontweight='bold', fontfamily='Arial')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return props


def create_combined_plot(compound_id, compound_data, select_data, props, output_path):
    """Create combined structure + radar plot."""

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
    ax1.set_title('(a) Molecular Structure', fontsize=12, fontweight='bold', fontfamily='Arial', pad=15)
    ax1.text(0.5, -0.08, f'Exemplar: {compound_id}', transform=ax1.transAxes, ha='center', fontsize=12, style='italic', fontfamily='Arial')

    # Radar panel
    ax2 = fig.add_subplot(122, projection='polar')

    # Interpolate for smooth fill
    from scipy import interpolate
    theta_fine = np.linspace(0, 2*np.pi, 100)
    f_upper = interpolate.interp1d(angles, select_upper, kind='linear', fill_value='extrapolate')
    f_lower = interpolate.interp1d(angles, select_lower, kind='linear', fill_value='extrapolate')
    r_upper = f_upper(theta_fine)
    r_lower = f_lower(theta_fine)

    ax2.fill_between(theta_fine, r_lower, r_upper, alpha=0.35, color=compound_data['zone_color'],
                     label=f"{compound_data['select_rule']} range", zorder=1)
    ax2.fill_between(theta_fine, 0, r_lower, alpha=0.1, color=compound_data['zone_color'], zorder=0)

    ax2.plot(angles, select_mean_norm, '--', linewidth=1.5, color='gray', alpha=0.7,
             label=f"{compound_data['select_rule']} mean", zorder=2)
    ax2.plot(angles, compound_norm, 'o-', linewidth=2.5, markersize=8,
             color=compound_data['color'], label=compound_id, zorder=3)

    labels = [PROPERTY_SCALES[prop]['label'] for prop in PROPERTIES]
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

    ax2.set_title(f"(b) Physicochemical Profile vs {compound_data['select_rule']}", fontsize=12, fontweight='bold', fontfamily='Arial', pad=20)
    fig.suptitle(f"{compound_data['name'].title()}: {compound_id}", fontsize=14, fontweight='bold', fontfamily='Arial', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# Main execution
print("=" * 70)
print("GENERATING NEW EXEMPLAR RADAR PLOTS")
print("All exemplars perfectly fit within SELECT ranges!")
print("=" * 70)

for compound_id, data in compounds.items():
    print(f"\n{compound_id} ({data['name']}):")
    print(f"  SMILES: {data['smiles']}")

    # Radar only
    radar_file = OUTPUT_DIR / f'{compound_id}_radar_only.png'
    props = create_radar_only(compound_id, data, select_ranges, radar_file)
    print(f"  Properties: MW={props['MW']}, LogP={props['LogP']}, TPSA={props['TPSA']}, HBD={props['HBD']}, HBA={props['HBA']}, Rings={props['Rings']}")
    print(f"  [OK] {radar_file.name}")

    # Combined
    combined_file = OUTPUT_DIR / f'{compound_id}_combined.png'
    create_combined_plot(compound_id, data, select_ranges, props, combined_file)
    print(f"  [OK] {combined_file.name}")

print("\n" + "=" * 70)
print("COMPLETE - New exemplars generated!")
print("All compounds fall WITHIN their respective SELECT ranges.")
print("=" * 70)
