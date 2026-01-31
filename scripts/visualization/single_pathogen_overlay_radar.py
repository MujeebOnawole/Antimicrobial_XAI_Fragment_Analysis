"""
Single Radar Plot Overlaying All Three Pathogen SELECT Ranges
=============================================================
Creates a single radar/spider plot showing SA, EC, and CA SELECT ranges
as lines (not shaded) for easy visual comparison.

Technical names: radar plot, spider plot, star plot, web chart
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pathlib import Path

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

OUTPUT_DIR = Path(r'C:/Users/uqaonawo/OneDrive - The University of Queensland/Desktop/fragments_analysis/figures/radar_plots')

# SELECT ranges for all 8 properties (mean +/- SD)
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

# Pathogen display info with mathtext superscripts and italicized species names
PATHOGEN_INFO = {
    'SA': {'name': r'$\it{S. aureus}$ ($G^+$)', 'color': '#DC143C'},
    'EC': {'name': r'$\it{E. coli}$ ($G^-$)', 'color': '#1E90FF'},
    'CA': {'name': r'$\it{C. albicans}$', 'color': '#228B22'},
}

# Property scales (Fsp3 = Fraction sp³ carbons, using mathtext for superscript)
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


def normalize_value(value, prop_name):
    """Normalize value to 0-100% scale."""
    scale = PROPERTY_SCALES[prop_name]
    return max(0, min(100, (value - scale['min']) / (scale['max'] - scale['min']) * 100))


def create_overlay_radar():
    """Create radar plot with all three pathogen ranges overlaid as lines."""

    num_props = len(PROPERTIES)
    angles = [n / num_props * 2 * pi for n in range(num_props)]
    angles += angles[:1]

    # Create figure
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.75, 0.75], projection='polar')

    # Plot each pathogen's SELECT range as a line
    for pathogen, info in PATHOGEN_INFO.items():
        select = SELECT_RANGES[pathogen]

        # Use mean values for the line
        mean_values = [normalize_value(select[prop]['mean'], prop) for prop in PROPERTIES]
        mean_values += mean_values[:1]

        # Plot the mean line with markers
        ax.plot(angles, mean_values, 'o-',
                linewidth=3,
                markersize=10,
                color=info['color'],
                label=info['name'],
                zorder=3)

    # Axis labels - 16pt BOLD
    labels = [PROPERTY_SCALES[prop]['label'] for prop in PROPERTIES]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold', fontfamily='Arial')

    # Y-axis settings
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=12, fontfamily='Arial')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)

    # Legend - 18pt, positioned outside circle to avoid Fsp3 overlap
    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.55, -0.02),
                       fontsize=18, framealpha=0.95, edgecolor='gray')
    for text in legend.get_texts():
        text.set_fontfamily('Arial')

    # Title
    fig.text(0.5, 0.95, 'SELECT Rules: Pathogen-Specific Property Profiles',
             ha='center', fontsize=20, fontweight='bold', fontfamily='Arial')
    fig.text(0.5, 0.91, 'Mean values across all 8 physicochemical properties',
             ha='center', fontsize=16, fontfamily='Arial')

    # Save
    output_path = OUTPUT_DIR / 'single_pathogen_overlay_radar.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def create_overlay_radar_with_range():
    """Create radar plot with ranges shown as upper/lower bounds."""

    num_props = len(PROPERTIES)
    angles = [n / num_props * 2 * pi for n in range(num_props)]
    angles += angles[:1]

    # Create figure
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.75, 0.75], projection='polar')

    # Plot each pathogen's SELECT range
    for pathogen, info in PATHOGEN_INFO.items():
        select = SELECT_RANGES[pathogen]

        # Mean values
        mean_values = [normalize_value(select[prop]['mean'], prop) for prop in PROPERTIES]
        mean_values += mean_values[:1]

        # Upper bound (mean + SD)
        upper_values = [normalize_value(select[prop]['mean'] + select[prop]['sd'], prop) for prop in PROPERTIES]
        upper_values += upper_values[:1]

        # Lower bound (mean - SD)
        lower_values = [normalize_value(max(0, select[prop]['mean'] - select[prop]['sd']), prop) for prop in PROPERTIES]
        lower_values += lower_values[:1]

        # Plot mean line (solid, thick) with markers
        ax.plot(angles, mean_values, 'o-',
                linewidth=3,
                markersize=8,
                color=info['color'],
                label=info['name'],
                zorder=3)

        # Plot upper bound (dashed, thin) - no markers
        ax.plot(angles, upper_values,
                linewidth=1.5,
                linestyle='--',
                color=info['color'],
                alpha=0.6,
                zorder=2)

        # Plot lower bound (dashed, thin) - no markers
        ax.plot(angles, lower_values,
                linewidth=1.5,
                linestyle='--',
                color=info['color'],
                alpha=0.6,
                zorder=2)

    # Axis labels
    labels = [PROPERTY_SCALES[prop]['label'] for prop in PROPERTIES]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold', fontfamily='Arial')

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=12, fontfamily='Arial')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)

    # Legend - 18pt, positioned outside circle to avoid Fsp3 overlap
    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.55, -0.02),
                       fontsize=18, framealpha=0.95, edgecolor='gray')
    for text in legend.get_texts():
        text.set_fontfamily('Arial')

    # Add note about dashed lines
    fig.text(0.5, 0.02, 'Solid lines = mean; Dashed lines = ± 1 SD',
             ha='center', fontsize=12, style='italic', fontfamily='Arial')

    # Title
    fig.text(0.5, 0.95, 'SELECT Rules: Pathogen-Specific Property Profiles',
             ha='center', fontsize=20, fontweight='bold', fontfamily='Arial')
    fig.text(0.5, 0.91, 'Comparison of mean ± SD across all 8 properties',
             ha='center', fontsize=16, fontfamily='Arial')

    # Save
    output_path = OUTPUT_DIR / 'single_pathogen_overlay_radar_with_range.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def main():
    print("=" * 70)
    print("GENERATING SINGLE-PATHOGEN OVERLAY RADAR PLOTS")
    print("Using superscripts: G+ (Gram-positive), G- (Gram-negative)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Version 1: Just mean lines
    print("\n1. Creating overlay with mean lines only...")
    create_overlay_radar()

    # Version 2: Mean lines with SD range bounds
    print("\n2. Creating overlay with mean +/- SD bounds...")
    create_overlay_radar_with_range()

    # Print summary of SELECT ranges
    print("\n" + "=" * 60)
    print("SELECT RANGE SUMMARY (Mean +/- SD)")
    print("=" * 60)
    print(f"\n{'Property':<12} {'S. aureus (G+)':<18} {'E. coli (G-)':<18} {'C. albicans':<18}")
    print("-" * 66)

    for prop in PROPERTIES:
        sa = SELECT_RANGES['SA'][prop]
        ec = SELECT_RANGES['EC'][prop]
        ca = SELECT_RANGES['CA'][prop]

        sa_str = f"{sa['mean']:.2f} +/- {sa['sd']:.2f}"
        ec_str = f"{ec['mean']:.2f} +/- {ec['sd']:.2f}"
        ca_str = f"{ca['mean']:.2f} +/- {ca['sd']:.2f}"

        print(f"{prop:<12} {sa_str:<18} {ec_str:<18} {ca_str:<18}")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
