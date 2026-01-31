import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from math import pi

# Set Arial font for publication quality
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 18

# Compound data
compounds = {
    'S aureus': {
        'name': 'S. aureus exemplar',
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
    'E coli': {
        'name': 'E. coli exemplar',
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
    'C albicans': {
        'name': 'C. albicans exemplar',
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
        'MW': (200, 450),
        'TPSA': (17, 43),
        'HBD': (0, 1),
        'HBA': (2, 6),
        'Rings': (2, 5)
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

def create_standalone_radar_plot(compound_key, compound_data):
    """Create standalone radar plot for a single compound"""
    categories = ['LogP', 'MW', 'TPSA', 'HBD', 'HBA', 'Rings']
    N = len(categories)

    # Compute angles for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Setup polar plot
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines and labels - Arial size 18
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=18, weight='bold', fontfamily='Arial')
    ax.tick_params(axis='x', pad=25)

    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=11, color='gray')

    # Add gridlines
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Plot SELECT zone as shaded region (no dashed line to avoid confusion)
    select_type = compound_data['select_type']
    zone_values = get_select_zone_polygon(select_type)
    zone_values += zone_values[:1]  # Complete the circle
    ax.fill(angles, zone_values, color='green', alpha=0.15, label=f'{select_type} zone')

    # Plot compound values
    values = []
    for cat in categories:
        prop_value = compound_data['measured'][cat]
        normalized = normalize_value(prop_value, cat)
        values.append(normalized)

    values += values[:1]  # Complete the circle

    # Use uniform blue color
    ax.plot(angles, values, color='#1E90FF', linewidth=3, label=compound_data['name'])
    ax.fill(angles, values, color='#1E90FF', alpha=0.25)
    ax.scatter(angles[:-1], values[:-1], color='#1E90FF', s=120, zorder=5, edgecolor='white', linewidth=2)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, fontsize=12)

    # Add title
    plt.title(f"{compound_data['name']}\nPhysicochemical Properties",
              size=16, weight='bold', pad=20)

    # Save figure
    output_filename = f"{compound_key} radar plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Generated: {output_filename}")

    plt.close()

def main():
    """Generate standalone radar plots for all compounds"""
    print("=" * 80)
    print("Generating Standalone Radar Plots")
    print("=" * 80)

    for compound_key, compound_data in compounds.items():
        print(f"\nProcessing {compound_key}...")
        create_standalone_radar_plot(compound_key, compound_data)

    print("\n" + "=" * 80)
    print("COMPLETE! Generated files:")
    for compound_key in compounds.keys():
        print(f"  - {compound_key} radar plot.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
