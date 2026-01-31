"""
Positive vs Negative Fragment Comparison Radar Plots
=====================================================
Creates radar plots comparing positive (active-promoting) vs negative
(inactive-promoting) fragment property profiles for each pathogen.

This addresses the question: Do negative fragments have different property
profiles that can discriminate them from positive (active) fragments?
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

# Paths
BASE_DIR = Path(r'C:/Users/uqaonawo/OneDrive - The University of Queensland/Desktop/fragments_analysis')
OUTPUT_DIR = BASE_DIR / 'figures' / 'radar_plots'
DATA_DIR = BASE_DIR / 'data' / 'single_pathogen'

# Property scales (same as positive plot for fair comparison)
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

# Pathogen display info
PATHOGEN_INFO = {
    'SA': {'name': r'$\it{S. aureus}$ ($G^+$)', 'color': '#DC143C', 'color_light': '#FFB6C1'},
    'EC': {'name': r'$\it{E. coli}$ ($G^-$)', 'color': '#1E90FF', 'color_light': '#ADD8E6'},
    'CA': {'name': r'$\it{C. albicans}$', 'color': '#228B22', 'color_light': '#90EE90'},
}

# POSITIVE fragment statistics (from summary_statistics_table.csv)
POSITIVE_STATS = {
    'SA': {
        'MW': {'mean': 268.20, 'sd': 84.99},
        'LogP': {'mean': 3.00, 'sd': 1.61},
        'TPSA': {'mean': 35.09, 'sd': 21.73},
        'HBD': {'mean': 1.05, 'sd': 1.10},
        'HBA': {'mean': 3.18, 'sd': 1.69},
        'RotBonds': {'mean': 3.95, 'sd': 3.03},
        'Rings': {'mean': 1.83, 'sd': 1.09},
        'Fsp3': {'mean': 0.35, 'sd': 0.22}
    },
    'EC': {
        'MW': {'mean': 256.91, 'sd': 81.94},
        'LogP': {'mean': 2.33, 'sd': 1.68},
        'TPSA': {'mean': 42.17, 'sd': 23.47},
        'HBD': {'mean': 1.33, 'sd': 1.10},
        'HBA': {'mean': 3.67, 'sd': 1.71},
        'RotBonds': {'mean': 3.94, 'sd': 2.67},
        'Rings': {'mean': 1.72, 'sd': 1.05},
        'Fsp3': {'mean': 0.28, 'sd': 0.21}
    },
    'CA': {
        'MW': {'mean': 260.53, 'sd': 80.90},
        'LogP': {'mean': 2.69, 'sd': 1.48},
        'TPSA': {'mean': 38.02, 'sd': 23.83},
        'HBD': {'mean': 0.70, 'sd': 0.89},
        'HBA': {'mean': 3.54, 'sd': 2.21},
        'RotBonds': {'mean': 4.17, 'sd': 2.71},
        'Rings': {'mean': 1.84, 'sd': 1.25},
        'Fsp3': {'mean': 0.30, 'sd': 0.21}
    }
}

# Sample counts
POSITIVE_N = {'SA': 2332, 'EC': 537, 'CA': 1234}
NEGATIVE_N = {'SA': 398, 'EC': 161, 'CA': 227}


def calculate_properties_from_smiles(smiles):
    """Calculate physicochemical properties from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
        'HBD': rdMolDescriptors.CalcNumHBD(mol),
        'HBA': rdMolDescriptors.CalcNumHBA(mol),
        'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'Rings': rdMolDescriptors.CalcNumRings(mol),
        'Fsp3': rdMolDescriptors.CalcFractionCSP3(mol)
    }


def calculate_negative_statistics():
    """Calculate statistics for negative fragments from CSV files."""
    negative_stats = {}

    pathogen_files = {
        'SA': 'SA_specific_negative_scaffolds.csv',
        'EC': 'EC_specific_negative_scaffolds.csv',
        'CA': 'CA_specific_negative_scaffolds.csv'
    }

    for pathogen, filename in pathogen_files.items():
        filepath = DATA_DIR / 'negative' / filename

        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            continue

        df = pd.read_csv(filepath)

        # Calculate properties for each fragment
        all_props = []
        for smiles in df['fragment_smiles']:
            props = calculate_properties_from_smiles(smiles)
            if props:
                all_props.append(props)

        if not all_props:
            print(f"Warning: No valid SMILES for {pathogen}")
            continue

        props_df = pd.DataFrame(all_props)

        # Calculate mean and SD for each property
        negative_stats[pathogen] = {}
        prop_mapping = {
            'MW': 'MW', 'LogP': 'LogP', 'TPSA': 'TPSA',
            'HBD': 'HBD', 'HBA': 'HBA', 'RotBonds': 'RotBonds',
            'Rings': 'Rings', 'Fsp3': 'Fsp3'
        }

        for prop_name, col_name in prop_mapping.items():
            negative_stats[pathogen][prop_name] = {
                'mean': props_df[col_name].mean(),
                'sd': props_df[col_name].std()
            }

        negative_stats[pathogen]['n'] = len(props_df)

        print(f"{pathogen} negative fragments: n={len(props_df)}")

    return negative_stats


def normalize_value(value, prop_name):
    """Normalize value to 0-100% scale."""
    scale = PROPERTY_SCALES[prop_name]
    return max(0, min(100, (value - scale['min']) / (scale['max'] - scale['min']) * 100))


def create_comparison_radar_single_pathogen(pathogen, positive_stats, negative_stats):
    """Create radar plot comparing positive vs negative for a single pathogen."""

    num_props = len(PROPERTIES)
    angles = [n / num_props * 2 * pi for n in range(num_props)]
    angles += angles[:1]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.75, 0.75], projection='polar')

    info = PATHOGEN_INFO[pathogen]
    pos = positive_stats[pathogen]
    neg = negative_stats[pathogen]

    # Positive fragment mean values (solid line)
    pos_means = [normalize_value(pos[prop]['mean'], prop) for prop in PROPERTIES]
    pos_means += pos_means[:1]

    # Negative fragment mean values (dashed line)
    neg_means = [normalize_value(neg[prop]['mean'], prop) for prop in PROPERTIES]
    neg_means += neg_means[:1]

    # Plot positive (solid, darker)
    ax.plot(angles, pos_means, 'o-',
            linewidth=3, markersize=10,
            color=info['color'],
            label=f'Positive (n={POSITIVE_N[pathogen]})',
            zorder=3)

    # Plot negative (dashed, lighter)
    ax.plot(angles, neg_means, 's--',
            linewidth=3, markersize=10,
            color=info['color'], alpha=0.6,
            label=f'Negative (n={neg["n"]})',
            zorder=2)

    # Axis labels
    labels = [PROPERTY_SCALES[prop]['label'] for prop in PROPERTIES]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold', fontfamily='Arial')

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)

    # Legend
    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.55, -0.02),
                       fontsize=16, framealpha=0.95, edgecolor='gray')

    # Title
    pathogen_name = PATHOGEN_INFO[pathogen]['name']
    fig.text(0.5, 0.95, f'{pathogen_name}: Positive vs Negative Fragments',
             ha='center', fontsize=20, fontweight='bold', fontfamily='Arial')
    fig.text(0.5, 0.91, 'Comparison of mean property profiles',
             ha='center', fontsize=16, fontfamily='Arial')

    # Save
    output_path = OUTPUT_DIR / f'{pathogen}_positive_negative_comparison_radar.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def create_overlay_comparison_radar(positive_stats, negative_stats):
    """Create overlay radar plot showing all pathogens, positive vs negative."""

    num_props = len(PROPERTIES)
    angles = [n / num_props * 2 * pi for n in range(num_props)]
    angles += angles[:1]

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75], projection='polar')

    for pathogen in ['SA', 'EC', 'CA']:
        info = PATHOGEN_INFO[pathogen]
        pos = positive_stats[pathogen]
        neg = negative_stats[pathogen]

        # Positive fragment mean values
        pos_means = [normalize_value(pos[prop]['mean'], prop) for prop in PROPERTIES]
        pos_means += pos_means[:1]

        # Negative fragment mean values
        neg_means = [normalize_value(neg[prop]['mean'], prop) for prop in PROPERTIES]
        neg_means += neg_means[:1]

        # Plot positive (solid)
        ax.plot(angles, pos_means, 'o-',
                linewidth=3, markersize=8,
                color=info['color'],
                label=f'{info["name"]} Positive',
                zorder=3)

        # Plot negative (dashed)
        ax.plot(angles, neg_means, 's--',
                linewidth=2, markersize=6,
                color=info['color'], alpha=0.5,
                label=f'{info["name"]} Negative',
                zorder=2)

    # Axis labels
    labels = [PROPERTY_SCALES[prop]['label'] for prop in PROPERTIES]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold', fontfamily='Arial')

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)

    # Legend (lower right corner)
    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.45, -0.05),
                       fontsize=14, framealpha=0.95, edgecolor='gray')

    # Note
    fig.text(0.5, 0.02, 'Solid lines = Positive (active-promoting); Dashed lines = Negative (inactive-promoting)',
             ha='center', fontsize=12, style='italic', fontfamily='Arial')

    # Title
    fig.text(0.5, 0.95, 'SELECT Rules: Positive vs Negative Fragment Profiles',
             ha='center', fontsize=20, fontweight='bold', fontfamily='Arial')
    fig.text(0.5, 0.91, 'Comparison across all three pathogens',
             ha='center', fontsize=16, fontfamily='Arial')

    # Save
    output_path = OUTPUT_DIR / 'all_pathogens_positive_negative_comparison_radar.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def create_negative_only_overlay_radar(negative_stats):
    """Create overlay radar plot showing just negative fragments (equivalent to positive plot)."""

    num_props = len(PROPERTIES)
    angles = [n / num_props * 2 * pi for n in range(num_props)]
    angles += angles[:1]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.75, 0.75], projection='polar')

    for pathogen in ['SA', 'EC', 'CA']:
        info = PATHOGEN_INFO[pathogen]
        neg = negative_stats[pathogen]

        # Negative fragment mean values
        neg_means = [normalize_value(neg[prop]['mean'], prop) for prop in PROPERTIES]
        neg_means += neg_means[:1]

        # Upper bound (mean + SD)
        upper_values = [normalize_value(neg[prop]['mean'] + neg[prop]['sd'], prop) for prop in PROPERTIES]
        upper_values += upper_values[:1]

        # Lower bound (mean - SD)
        lower_values = [normalize_value(max(0, neg[prop]['mean'] - neg[prop]['sd']), prop) for prop in PROPERTIES]
        lower_values += lower_values[:1]

        # Plot mean line (solid, thick) with markers
        ax.plot(angles, neg_means, 'o-',
                linewidth=3, markersize=8,
                color=info['color'],
                label=info['name'],
                zorder=3)

        # Plot upper bound (dashed, thin)
        ax.plot(angles, upper_values,
                linewidth=1.5, linestyle='--',
                color=info['color'], alpha=0.6,
                zorder=2)

        # Plot lower bound (dashed, thin)
        ax.plot(angles, lower_values,
                linewidth=1.5, linestyle='--',
                color=info['color'], alpha=0.6,
                zorder=2)

    # Axis labels
    labels = [PROPERTY_SCALES[prop]['label'] for prop in PROPERTIES]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold', fontfamily='Arial')

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)

    # Legend
    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.55, -0.02),
                       fontsize=18, framealpha=0.95, edgecolor='gray')

    # Note
    fig.text(0.5, 0.02, 'Solid lines = mean; Dashed lines = ± 1 SD',
             ha='center', fontsize=12, style='italic', fontfamily='Arial')

    # Title
    fig.text(0.5, 0.95, 'NEGATIVE Fragments: Pathogen-Specific Property Profiles',
             ha='center', fontsize=20, fontweight='bold', fontfamily='Arial')
    fig.text(0.5, 0.91, 'Comparison of mean ± SD across all 8 properties',
             ha='center', fontsize=16, fontfamily='Arial')

    # Save
    output_path = OUTPUT_DIR / 'single_pathogen_NEGATIVE_overlay_radar_with_range.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def print_comparison_table(positive_stats, negative_stats):
    """Print comparison table of positive vs negative statistics."""

    print("\n" + "=" * 100)
    print("POSITIVE vs NEGATIVE FRAGMENT COMPARISON")
    print("=" * 100)

    for pathogen in ['SA', 'EC', 'CA']:
        pos = positive_stats[pathogen]
        neg = negative_stats[pathogen]

        print(f"\n{PATHOGEN_INFO[pathogen]['name'].replace('$', '').replace('it{', '').replace('}', '')}")
        print(f"Positive n={POSITIVE_N[pathogen]}, Negative n={neg['n']}")
        print("-" * 80)
        print(f"{'Property':<12} {'Positive Mean':<15} {'Negative Mean':<15} {'Difference':<15} {'Direction':<15}")
        print("-" * 80)

        for prop in PROPERTIES:
            pos_mean = pos[prop]['mean']
            neg_mean = neg[prop]['mean']
            diff = pos_mean - neg_mean
            direction = "Pos higher" if diff > 0 else "Neg higher" if diff < 0 else "Equal"

            print(f"{prop:<12} {pos_mean:>12.2f}   {neg_mean:>12.2f}   {diff:>+12.2f}   {direction:<15}")

    print("\n" + "=" * 100)


def save_statistics_csv(positive_stats, negative_stats):
    """Save comparison statistics to CSV."""

    rows = []
    for pathogen in ['SA', 'EC', 'CA']:
        pos = positive_stats[pathogen]
        neg = negative_stats[pathogen]

        for prop in PROPERTIES:
            rows.append({
                'Pathogen': pathogen,
                'Property': prop,
                'Positive_Mean': pos[prop]['mean'],
                'Positive_SD': pos[prop]['sd'],
                'Positive_N': POSITIVE_N[pathogen],
                'Negative_Mean': neg[prop]['mean'],
                'Negative_SD': neg[prop]['sd'],
                'Negative_N': neg['n'],
                'Difference': pos[prop]['mean'] - neg[prop]['mean']
            })

    df = pd.DataFrame(rows)
    output_path = BASE_DIR / 'results' / 'statistics' / 'positive_negative_comparison_statistics.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved statistics: {output_path}")
    return output_path


def main():
    print("=" * 70)
    print("GENERATING POSITIVE vs NEGATIVE FRAGMENT COMPARISON PLOTS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate negative fragment statistics
    print("\nCalculating negative fragment statistics from SMILES...")
    negative_stats = calculate_negative_statistics()

    if not negative_stats:
        print("ERROR: Could not calculate negative statistics. Exiting.")
        return

    # Print comparison table
    print_comparison_table(POSITIVE_STATS, negative_stats)

    # Save statistics to CSV
    save_statistics_csv(POSITIVE_STATS, negative_stats)

    # Create individual pathogen comparison plots
    print("\nGenerating individual pathogen comparison plots...")
    for pathogen in ['SA', 'EC', 'CA']:
        if pathogen in negative_stats:
            create_comparison_radar_single_pathogen(pathogen, POSITIVE_STATS, negative_stats)

    # Create overlay comparison plot
    print("\nGenerating overlay comparison plot...")
    create_overlay_comparison_radar(POSITIVE_STATS, negative_stats)

    # Create negative-only overlay plot (equivalent to the positive one)
    print("\nGenerating negative-only overlay plot (equivalent to positive plot)...")
    create_negative_only_overlay_radar(negative_stats)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. SA_positive_negative_comparison_radar.png")
    print("  2. EC_positive_negative_comparison_radar.png")
    print("  3. CA_positive_negative_comparison_radar.png")
    print("  4. all_pathogens_positive_negative_comparison_radar.png")
    print("  5. single_pathogen_NEGATIVE_overlay_radar_with_range.png")
    print("  6. positive_negative_comparison_statistics.csv")


if __name__ == '__main__':
    main()
