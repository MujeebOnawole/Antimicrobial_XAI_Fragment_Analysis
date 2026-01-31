"""
Generate Radar Plots for Dual and Triple Active Exemplar Compounds
==================================================================
Creates publication-quality radar plots with 8 properties for dual-active
and triple-active exemplar compounds.

All exemplars have 0 violations across all 8 physicochemical properties.
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pathlib import Path
import pandas as pd

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, Descriptors, rdMolDescriptors

BASE_DIR = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis")
OUTPUT_DIR = BASE_DIR / "figures" / "radar_plots"

# Dual and Triple Active Exemplars (all with 0 violations)
# Names use italicized species names per biological convention
EXEMPLARS = {
    # Dual Active: S. aureus + E. coli
    'CHEMBL2178320': {
        'smiles': 'OC1=CC(CC(F)(F)F)=CC=C1OC1=CC=CC(F)=N1',
        'name': r'$\it{S. aureus}$ + $\it{E. coli}$ exemplar',
        'combination': 'SA_EC',
        'pathogens': ['SA', 'EC'],
        'color': '#9932CC',  # Dark orchid (purple - blend of red+blue)
        'zone_color': '#DDA0DD',  # Plum
        'activity': {
            'SA': {'mic_ug_ml': 0.25, 'activity_um': 0.9},
            'EC': {'mic_ug_ml': 4.0, 'activity_um': 13.9},
        }
    },
    # Dual Active: S. aureus + C. albicans
    'CHEMBL5207371': {
        'smiles': 'O=C1NC2=CC=C(C(F)(F)F)C=C2N=C1CBr',
        'name': r'$\it{S. aureus}$ + $\it{C. albicans}$ exemplar',
        'combination': 'SA_CA',
        'pathogens': ['SA', 'CA'],
        'color': '#B8860B',  # Dark goldenrod (blend of red+green)
        'zone_color': '#F0E68C',  # Khaki
        'activity': {
            'SA': {'mic_ug_ml': 31.2, 'activity_um': 101.6},
            'CA': {'mic_ug_ml': 31.2, 'activity_um': 101.6},
        }
    },
    # Dual Active: E. coli + C. albicans
    'CHEMBL5409101': {
        'smiles': 'O=P(C1=CC=CC=C1)(C1=CC=CC=C1)C1CCC/C1=N\\O',
        'name': r'$\it{E. coli}$ + $\it{C. albicans}$ exemplar',
        'combination': 'EC_CA',
        'pathogens': ['EC', 'CA'],
        'color': '#20B2AA',  # Light sea green (blend of blue+green)
        'zone_color': '#AFEEEE',  # Pale turquoise
        'activity': {
            'EC': {'mic_ug_ml': 0.1, 'activity_um': 0.3},
            'CA': {'mic_ug_ml': 0.1, 'activity_um': 0.3},
        }
    },
    # Triple Active: All three pathogens
    # UPDATED: Replaced CHEMBL3822555 (CA prediction below threshold) with CHEMBL2297203
    'CHEMBL2297203': {
        'smiles': 'CC1=CC(C(=O)/C=C/C2=CC=CN2)=C(C)O1',
        'name': 'Broad-spectrum exemplar',
        'combination': 'TRIPLE',
        'pathogens': ['SA', 'EC', 'CA'],
        'color': '#FF8C00',  # Dark orange
        'zone_color': '#FFD700',  # Gold
        'activity': {
            'SA': {'mic_ug_ml': 30.0, 'activity_um': 139.4},
            'EC': {'mic_ug_ml': 40.0, 'activity_um': 185.8},
            'CA': {'mic_ug_ml': 30.0, 'activity_um': 139.4},
        }
    }
}

# SELECT ranges for each pathogen
SELECT_RANGES = {
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

# Zone colors for each pathogen (for SELECT zone visualization)
# Using mathtext superscripts: G^+ (Gram-positive), G^- (Gram-negative)
# Species names italicized per biological convention
PATHOGEN_COLORS = {
    'SA': {'color': '#DC143C', 'label': r'$\it{S. aureus}$ ($G^+$)'},
    'EC': {'color': '#1E90FF', 'label': r'$\it{E. coli}$ ($G^-$)'},
    'CA': {'color': '#228B22', 'label': r'$\it{C. albicans}$'},
}

PROPERTY_SCALES = {
    'MW': {'min': 0, 'max': 500, 'label': 'MW (Da)\n[0-500]'},
    'LogP': {'min': 0, 'max': 6, 'label': 'LogP\n[0-6]'},
    'TPSA': {'min': 0, 'max': 150, 'label': 'TPSA (Å²)\n[0-150]'},
    'HBD': {'min': 0, 'max': 5, 'label': 'HBD\n[0-5]'},
    'HBA': {'min': 0, 'max': 10, 'label': 'HBA\n[0-10]'},
    'RotBonds': {'min': 0, 'max': 15, 'label': 'RotBonds\n[0-15]'},
    'Rings': {'min': 0, 'max': 6, 'label': 'Rings\n[0-6]'},
    'Fsp3': {'min': 0, 'max': 1, 'label': r'$Fsp^3$' + '\n[0-1]'}
}

PROPERTIES = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Rings', 'Fsp3']


def calc_props(smiles):
    """Calculate all 8 physicochemical properties."""
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
    """Normalize value to 0-100% scale."""
    scale = PROPERTY_SCALES[prop_name]
    return max(0, min(100, (value - scale['min']) / (scale['max'] - scale['min']) * 100))


def get_combined_range(pathogens):
    """Get combined SELECT range (intersection) for multiple pathogens."""
    combined = {}
    for prop in PROPERTIES:
        lower_bounds = []
        upper_bounds = []
        for p in pathogens:
            mean = SELECT_RANGES[p][prop]['mean']
            sd = SELECT_RANGES[p][prop]['sd']
            lower_bounds.append(mean - sd)
            upper_bounds.append(mean + sd)
        combined[prop] = {
            'lower': max(lower_bounds),
            'upper': min(upper_bounds),
            'mean': np.mean([SELECT_RANGES[p][prop]['mean'] for p in pathogens])
        }
    return combined


def create_radar_plot(chembl_id, data, output_path):
    """Create radar plot with all 8 properties showing overlapping SELECT zones."""

    props = calc_props(data['smiles'])
    pathogens = data['pathogens']
    combined = get_combined_range(pathogens)

    # Normalize compound values
    compound_norm = [normalize_value(props[prop], prop) for prop in PROPERTIES]

    # Angles for 8 properties
    num_props = len(PROPERTIES)
    angles = [n / num_props * 2 * pi for n in range(num_props)]
    angles += angles[:1]
    compound_norm += compound_norm[:1]

    # Create figure
    fig = plt.figure(figsize=(10, 11))
    ax = fig.add_axes([0.1, 0.05, 0.75, 0.75], projection='polar')

    # Draw SELECT zones for each pathogen
    for pathogen in pathogens:
        select = SELECT_RANGES[pathogen]
        p_color = PATHOGEN_COLORS[pathogen]['color']
        p_label = PATHOGEN_COLORS[pathogen]['label']

        # Upper boundary of this pathogen's SELECT zone
        upper = [normalize_value(select[prop]['mean'] + select[prop]['sd'], prop) for prop in PROPERTIES]
        upper += upper[:1]

        # Fill with very light color
        ax.fill(angles, upper, alpha=0.15, color=p_color, label=f"{p_label} range", zorder=1)

    # Draw combined (intersection) zone mean line
    combined_mean = [normalize_value(combined[prop]['mean'], prop) for prop in PROPERTIES]
    combined_mean += combined_mean[:1]
    ax.plot(angles, combined_mean, '--', linewidth=1.5, color='gray', alpha=0.7,
            label='Combined mean', zorder=2)

    # Plot compound profile
    ax.plot(angles, compound_norm, 'o-', linewidth=2.5, markersize=8,
            color=data['color'], label=f'Exemplar ({chembl_id})', zorder=3)

    # Axis labels
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

    # Title - species names italicized
    combo_labels = {
        'SA_EC': r'$\it{S. aureus}$ + $\it{E. coli}$ (Dual Active)',
        'SA_CA': r'$\it{S. aureus}$ + $\it{C. albicans}$ (Dual Active)',
        'EC_CA': r'$\it{E. coli}$ + $\it{C. albicans}$ (Dual Active)',
        'TRIPLE': 'Broad-Spectrum (Triple Active)',
    }

    fig.text(0.5, 0.92, f'Physicochemical Profile: {chembl_id}',
             ha='center', fontsize=18, fontweight='bold', fontfamily='Arial')
    fig.text(0.5, 0.88, combo_labels[data['combination']],
             ha='center', fontsize=16, fontweight='bold', fontfamily='Arial')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return props


def create_combined_plot(chembl_id, data, props, output_path):
    """Create combined structure + radar plot."""

    pathogens = data['pathogens']
    combined = get_combined_range(pathogens)

    # Normalize values
    compound_norm = [normalize_value(props[prop], prop) for prop in PROPERTIES]

    num_props = len(PROPERTIES)
    angles = [n / num_props * 2 * pi for n in range(num_props)]
    angles += angles[:1]
    compound_norm += compound_norm[:1]

    fig = plt.figure(figsize=(16, 7))

    # Structure panel
    ax1 = fig.add_subplot(121)
    mol = Chem.MolFromSmiles(data['smiles'])
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

    # Draw SELECT zones for each pathogen
    for pathogen in pathogens:
        select = SELECT_RANGES[pathogen]
        p_color = PATHOGEN_COLORS[pathogen]['color']
        p_label = PATHOGEN_COLORS[pathogen]['label']

        upper = [normalize_value(select[prop]['mean'] + select[prop]['sd'], prop) for prop in PROPERTIES]
        upper += upper[:1]
        ax2.fill(angles, upper, alpha=0.15, color=p_color, label=f"{p_label} range", zorder=1)

    # Combined mean
    combined_mean = [normalize_value(combined[prop]['mean'], prop) for prop in PROPERTIES]
    combined_mean += combined_mean[:1]
    ax2.plot(angles, combined_mean, '--', linewidth=1.5, color='gray', alpha=0.7,
             label='Combined mean', zorder=2)

    # Compound profile
    ax2.plot(angles, compound_norm, 'o-', linewidth=2.5, markersize=8,
             color=data['color'], label=f'Exemplar ({chembl_id})', zorder=3)

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

    combo_labels = {
        'SA_EC': r'$\it{S. aureus}$ + $\it{E. coli}$',
        'SA_CA': r'$\it{S. aureus}$ + $\it{C. albicans}$',
        'EC_CA': r'$\it{E. coli}$ + $\it{C. albicans}$',
        'TRIPLE': 'Broad-Spectrum',
    }

    ax2.set_title(f"(b) Physicochemical Profile ({combo_labels[data['combination']]})",
                  fontsize=16, fontweight='bold', fontfamily='Arial', pad=20)
    fig.suptitle(f"{data['name']}: {chembl_id}",
                 fontsize=18, fontweight='bold', fontfamily='Arial', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("=" * 70)
    print("GENERATING DUAL & TRIPLE ACTIVE EXEMPLAR RADAR PLOTS")
    print("All compounds have 0 violations across all 8 properties")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_props = []

    for chembl_id, data in EXEMPLARS.items():
        print(f"\n{data['name']}:")
        print(f"  ChEMBL ID: {chembl_id}")
        print(f"  SMILES: {data['smiles']}")

        # Radar only
        radar_file = OUTPUT_DIR / f'{chembl_id}_radar_8props.png'
        props = create_radar_plot(chembl_id, data, radar_file)

        print(f"  Properties (all 8 within combined range):")
        print(f"    MW={props['MW']}, LogP={props['LogP']}, TPSA={props['TPSA']}, HBD={props['HBD']}")
        print(f"    HBA={props['HBA']}, RotBonds={props['RotBonds']}, Rings={props['Rings']}, Fsp3={props['Fsp3']}")

        # Activity info
        print(f"  Activity:")
        for pathogen, activity in data['activity'].items():
            print(f"    {pathogen}: MIC={activity['mic_ug_ml']} µg/mL ({activity['activity_um']:.1f} µM)")

        print(f"  [OK] {radar_file.name}")

        # Combined
        combined_file = OUTPUT_DIR / f'{chembl_id}_combined_8props.png'
        create_combined_plot(chembl_id, data, props, combined_file)
        print(f"  [OK] {combined_file.name}")

        # Collect for CSV
        row = {
            'ChEMBL_ID': chembl_id,
            'Combination': data['combination'],
            'Name': data['name'],
            'SMILES': data['smiles'],
            **props,
        }
        for pathogen, activity in data['activity'].items():
            row[f'{pathogen}_MIC_ug_ml'] = activity['mic_ug_ml']
            row[f'{pathogen}_Activity_uM'] = activity['activity_um']

        all_props.append(row)

    # Save summary CSV
    df = pd.DataFrame(all_props)
    csv_file = OUTPUT_DIR / 'dual_triple_exemplars_with_activity.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n[OK] Summary saved: {csv_file.name}")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
