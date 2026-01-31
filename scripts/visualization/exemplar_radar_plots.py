"""
Generate Publication-Quality Radar Plots for Exemplar Compounds
Shows compound properties against SELECT acceptable ranges (mean +/- SD)

Three exemplar compounds:
1. CHEMBL4548986 - S. aureus (SELECT-G+)
2. CHEMBL4203270 - E. coli (SELECT-G-)
3. CS000245342 - C. albicans (SELECT-CA)
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pathlib import Path
import pandas as pd

# Try to import RDKit for structure generation
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, rdDepictor
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Structure images will not be generated.")

# Set paths
BASE_DIR = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis")
OUTPUT_DIR = BASE_DIR / "figures" / "radar_plots"
STRUCTURE_DIR = BASE_DIR / "figures" / "molecular_structures"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# COMPOUND DATA
# =============================================================================

compounds = {
    'CHEMBL4548986': {
        'smiles': 'CCN1C2=CC=CC=C2C2=CC(C)=C3C(=O)C4OC4C(=O)C3=C21',
        'name': 'S. aureus exemplar',
        'select_rule': 'SELECT-G$^+$',  # LaTeX superscript for matplotlib
        'select_rule_plain': 'SELECT-G+',
        'pathogen': 'SA',
        'color': '#DC143C',  # Crimson red
        'zone_color': '#FFB6C1',  # Light pink
        'provided': {
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
        'select_rule': 'SELECT-G$^-$',  # LaTeX superscript
        'select_rule_plain': 'SELECT-G-',
        'pathogen': 'EC',
        'color': '#1E90FF',  # Dodger blue
        'zone_color': '#ADD8E6',  # Light blue
        'provided': {
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
        'select_rule': 'SELECT-CA',
        'select_rule_plain': 'SELECT-CA',
        'pathogen': 'CA',
        'color': '#228B22',  # Forest green
        'zone_color': '#90EE90',  # Light green
        'provided': {
            'MW': 261.4,
            'LogP': 2.93,
            'TPSA': 34.2,
            'HBD': 0,
            'HBA': 4,
            'Rings': 3
        }
    }
}

# =============================================================================
# SELECT RANGES (from statistical analysis - mean +/- SD)
# =============================================================================

# From summary_statistics_table.csv (already calculated)
select_ranges = {
    'SA': {  # S. aureus (SELECT-G+)
        'MW': {'mean': 268.20, 'sd': 84.99},
        'LogP': {'mean': 3.00, 'sd': 1.61},
        'TPSA': {'mean': 35.09, 'sd': 21.73},
        'HBD': {'mean': 1.05, 'sd': 1.10},
        'HBA': {'mean': 3.18, 'sd': 1.69},
        'Rings': {'mean': 1.83, 'sd': 1.09},
        'ArRings': {'mean': 1.83, 'sd': 1.09},
        'RotBonds': {'mean': 3.95, 'sd': 3.03}
    },
    'EC': {  # E. coli (SELECT-G-)
        'MW': {'mean': 256.91, 'sd': 81.94},
        'LogP': {'mean': 2.33, 'sd': 1.68},
        'TPSA': {'mean': 42.17, 'sd': 23.47},
        'HBD': {'mean': 1.33, 'sd': 1.10},
        'HBA': {'mean': 3.67, 'sd': 1.71},
        'Rings': {'mean': 2.87, 'sd': 1.20},  # Approximate
        'ArRings': {'mean': 1.72, 'sd': 1.05},
        'RotBonds': {'mean': 3.94, 'sd': 2.67}
    },
    'CA': {  # C. albicans (SELECT-CA)
        'MW': {'mean': 260.53, 'sd': 80.90},
        'LogP': {'mean': 2.69, 'sd': 1.48},
        'TPSA': {'mean': 38.02, 'sd': 23.83},
        'HBD': {'mean': 0.70, 'sd': 0.89},
        'HBA': {'mean': 3.54, 'sd': 2.21},
        'Rings': {'mean': 3.03, 'sd': 1.15},  # Approximate
        'ArRings': {'mean': 1.84, 'sd': 1.25},
        'RotBonds': {'mean': 4.17, 'sd': 2.71}
    }
}

# Actual property scales for radar plot axes
PROPERTY_SCALES = {
    'LogP': {'min': 0, 'max': 6, 'unit': ''},
    'MW': {'min': 0, 'max': 500, 'unit': 'Da'},
    'TPSA': {'min': 0, 'max': 150, 'unit': 'A^2'},
    'HBD': {'min': 0, 'max': 5, 'unit': ''},
    'HBA': {'min': 0, 'max': 10, 'unit': ''},
    'Rings': {'min': 0, 'max': 6, 'unit': ''}
}

# Properties for radar plot (in order, clockwise from top)
PROPERTIES = ['LogP', 'MW', 'TPSA', 'HBD', 'HBA', 'Rings']


def calculate_properties_rdkit(smiles):
    """Calculate properties using RDKit if available."""
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        'MW': round(Descriptors.MolWt(mol), 1),
        'LogP': round(Descriptors.MolLogP(mol), 2),
        'TPSA': round(rdMolDescriptors.CalcTPSA(mol), 1),
        'HBD': rdMolDescriptors.CalcNumHBD(mol),
        'HBA': rdMolDescriptors.CalcNumHBA(mol),
        'Rings': rdMolDescriptors.CalcNumRings(mol),
        'ArRings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'Fsp3': round(rdMolDescriptors.CalcFractionCSP3(mol), 3)
    }


def normalize_value(value, prop_name):
    """Normalize value to 0-100 scale based on property range."""
    scale = PROPERTY_SCALES[prop_name]
    normalized = (value - scale['min']) / (scale['max'] - scale['min']) * 100
    return max(0, min(100, normalized))  # Clamp to 0-100


def create_radar_plot(compound_id, compound_data, select_data, output_path):
    """
    Create radar plot with compound properties and SELECT acceptable zone.

    Shows:
    - Compound properties as solid line with markers
    - SELECT range as shaded zone (mean +/- SD)
    """
    # Get compound properties (use provided values)
    compound_props = compound_data['provided']

    # Calculate normalized values for compound
    compound_norm = []
    for prop in PROPERTIES:
        val = compound_props.get(prop, 0)
        compound_norm.append(normalize_value(val, prop))

    # Get SELECT range for this pathogen
    pathogen = compound_data['pathogen']
    select = select_data[pathogen]

    # Calculate SELECT zone boundaries (mean +/- SD)
    select_lower = []
    select_upper = []
    select_mean_norm = []

    for prop in PROPERTIES:
        mean = select[prop]['mean']
        sd = select[prop]['sd']
        lower = max(0, mean - sd)  # Floor at 0
        upper = mean + sd

        select_lower.append(normalize_value(lower, prop))
        select_upper.append(normalize_value(upper, prop))
        select_mean_norm.append(normalize_value(mean, prop))

    # Set up angles for radar chart
    num_props = len(PROPERTIES)
    angles = [n / num_props * 2 * pi for n in range(num_props)]
    angles += angles[:1]  # Complete the circle

    # Close the data loops
    compound_norm += compound_norm[:1]
    select_lower += select_lower[:1]
    select_upper += select_upper[:1]
    select_mean_norm += select_mean_norm[:1]

    # Create figure with two subplots
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

    # Plot SELECT zone as shaded area
    ax2.fill_between(angles, select_lower, select_upper,
                     alpha=0.3, color=compound_data['zone_color'],
                     label=f'{compound_data["select_rule_plain"]} range')

    # Plot SELECT mean as dashed line
    ax2.plot(angles, select_mean_norm, '--', linewidth=1.5,
             color='gray', alpha=0.7, label=f'{compound_data["select_rule_plain"]} mean')

    # Plot compound values
    ax2.plot(angles, compound_norm, 'o-', linewidth=2.5, markersize=8,
             color=compound_data['color'], label=compound_id)
    ax2.fill(angles, compound_norm, alpha=0.1, color=compound_data['color'])

    # Set up axis labels with property names and scales
    labels = []
    for prop in PROPERTIES:
        scale = PROPERTY_SCALES[prop]
        unit = f" ({scale['unit']})" if scale['unit'] else ""
        labels.append(f"{prop}{unit}\n[0-{scale['max']}]")

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels, fontsize=9)

    # Set radial limits and labels
    ax2.set_ylim(0, 100)
    ax2.set_yticks([20, 40, 60, 80, 100])
    ax2.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)

    # Add gridlines
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax2.xaxis.grid(True, linestyle='-', alpha=0.3)

    # Legend
    ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)

    # Title for radar plot
    ax2.set_title(f"(b) Physicochemical Profile vs {compound_data['select_rule_plain']}",
                  fontsize=12, fontweight='bold', pad=20)

    # Main title
    fig.suptitle(f"{compound_data['name'].title()}: {compound_id}",
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return True


def create_combined_radar_plot(compounds_dict, select_data, output_path):
    """
    Create a single radar plot showing all three exemplars.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')

    num_props = len(PROPERTIES)
    angles = [n / num_props * 2 * pi for n in range(num_props)]
    angles += angles[:1]

    for compound_id, data in compounds_dict.items():
        compound_props = data['provided']
        pathogen = data['pathogen']

        # Normalize compound values
        compound_norm = []
        for prop in PROPERTIES:
            val = compound_props.get(prop, 0)
            compound_norm.append(normalize_value(val, prop))
        compound_norm += compound_norm[:1]

        # Plot compound
        ax.plot(angles, compound_norm, 'o-', linewidth=2.5, markersize=8,
                color=data['color'], label=f"{data['name']} ({compound_id})")
        ax.fill(angles, compound_norm, alpha=0.1, color=data['color'])

    # Set up axis labels
    labels = []
    for prop in PROPERTIES:
        scale = PROPERTY_SCALES[prop]
        unit = f" ({scale['unit']})" if scale['unit'] else ""
        labels.append(f"{prop}{unit}\n[0-{scale['max']}]")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)

    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("Physicochemical Profiles of Pathogen-Specific Exemplars",
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_structure_image(compound_id, smiles, name, output_path):
    """Generate high-resolution molecular structure image."""
    if not RDKIT_AVAILABLE:
        print(f"  Skipping structure for {compound_id} - RDKit not available")
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  Error: Could not parse SMILES for {compound_id}")
        return False

    # Generate 2D coordinates
    rdDepictor.Compute2DCoords(mol)

    # Create image with legend
    img = Draw.MolToImage(mol, size=(800, 600), legend=f"Exemplar: {compound_id}")
    img.save(output_path)

    return True


def generate_property_comparison_table(compounds_dict, output_path):
    """Generate CSV comparing provided vs calculated properties."""
    rows = []

    for compound_id, data in compounds_dict.items():
        provided = data['provided']

        # Calculate using RDKit if available
        if RDKIT_AVAILABLE:
            calculated = calculate_properties_rdkit(data['smiles'])
        else:
            calculated = None

        for prop in PROPERTIES:
            row = {
                'Compound_ID': compound_id,
                'Name': data['name'],
                'Pathogen': data['pathogen'],
                'SELECT_Rule': data['select_rule_plain'],
                'Property': prop,
                'Provided': provided.get(prop, 'N/A'),
            }

            if calculated:
                calc_val = calculated.get(prop, 'N/A')
                row['Calculated'] = calc_val
                if isinstance(provided.get(prop), (int, float)) and isinstance(calc_val, (int, float)):
                    diff = abs(provided[prop] - calc_val)
                    pct_diff = (diff / provided[prop] * 100) if provided[prop] != 0 else 0
                    row['Difference'] = round(diff, 2)
                    row['Pct_Diff'] = round(pct_diff, 1)
                else:
                    row['Difference'] = 'N/A'
                    row['Pct_Diff'] = 'N/A'
            else:
                row['Calculated'] = 'N/A'
                row['Difference'] = 'N/A'
                row['Pct_Diff'] = 'N/A'

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def generate_figure_captions(compounds_dict, select_data, output_path):
    """Generate figure captions for all plots."""
    captions = []

    for compound_id, data in compounds_dict.items():
        pathogen = data['pathogen']
        select = select_data[pathogen]
        props = data['provided']

        # Build property string
        prop_str = f"MW {props['MW']} Da, LogP {props['LogP']}, TPSA {props['TPSA']} A^2, HBD {props['HBD']}, HBA {props['HBA']}, Rings {props['Rings']}"

        # Build SELECT range string
        select_str = ", ".join([
            f"{prop}: {select[prop]['mean']:.1f} +/- {select[prop]['sd']:.1f}"
            for prop in ['LogP', 'MW', 'TPSA', 'HBD', 'HBA', 'Rings']
        ])

        # Determine which properties are within/outside range
        within = []
        outside = []
        for prop in PROPERTIES:
            mean = select[prop]['mean']
            sd = select[prop]['sd']
            val = props.get(prop, 0)
            lower = max(0, mean - sd)
            upper = mean + sd

            if lower <= val <= upper:
                within.append(prop)
            else:
                outside.append(prop)

        # Sample sizes
        n_samples = {'SA': 2332, 'EC': 537, 'CA': 1234}

        caption = f"""
Figure X.X. Physicochemical profile of {data['name']}.

(a) Molecular structure of {compound_id} ({prop_str}).

(b) Radar plot comparing compound properties (solid colored line) to {data['select_rule_plain']}
acceptable range (shaded zone representing mean +/- 1 standard deviation for {data['pathogen']}
positive fragments, N={n_samples[pathogen]:,}).

Property axes scaled to standard ranges: LogP (0-6), MW (0-500 Da), TPSA (0-150 A^2),
HBD (0-5), HBA (0-10), Total Rings (0-6). Percentages indicate position within each
property's scale.

{data['select_rule_plain']} ranges (mean +/- SD): {select_str}

Properties within acceptable range: {', '.join(within) if within else 'None'}
Properties outside acceptable range: {', '.join(outside) if outside else 'None'}
"""
        captions.append({
            'compound_id': compound_id,
            'name': data['name'],
            'caption': caption.strip()
        })

    # Write captions to file
    with open(output_path, 'w') as f:
        f.write("FIGURE CAPTIONS FOR EXEMPLAR COMPOUND RADAR PLOTS\n")
        f.write("=" * 70 + "\n\n")

        for cap in captions:
            f.write(f"{'='*70}\n")
            f.write(f"{cap['compound_id']} - {cap['name']}\n")
            f.write(f"{'='*70}\n\n")
            f.write(cap['caption'])
            f.write("\n\n")

    return captions


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING EXEMPLAR COMPOUND RADAR PLOTS")
    print("=" * 70)

    # Validate/calculate properties using RDKit
    print("\n1. Validating compound properties...")
    for compound_id, data in compounds.items():
        print(f"\n  {compound_id} ({data['name']}):")
        print(f"    SMILES: {data['smiles']}")

        if RDKIT_AVAILABLE:
            calc = calculate_properties_rdkit(data['smiles'])
            if calc:
                print(f"    Provided: MW={data['provided']['MW']}, LogP={data['provided']['LogP']}, TPSA={data['provided']['TPSA']}")
                print(f"    Calculated: MW={calc['MW']}, LogP={calc['LogP']}, TPSA={calc['TPSA']}")
                # Update with calculated values if significantly different
                data['calculated'] = calc

    # Generate individual radar plots
    print("\n2. Generating individual radar plots...")
    for compound_id, data in compounds.items():
        output_file = OUTPUT_DIR / f"{compound_id}_radar_plot.png"
        print(f"  Creating: {output_file.name}")
        success = create_radar_plot(compound_id, data, select_ranges, output_file)
        if success:
            print(f"    [OK] Saved")

    # Generate combined radar plot
    print("\n3. Generating combined radar plot...")
    combined_file = OUTPUT_DIR / "combined_exemplars_radar_plot.png"
    create_combined_radar_plot(compounds, select_ranges, combined_file)
    print(f"  [OK] Saved: {combined_file.name}")

    # Generate molecular structure images
    print("\n4. Generating molecular structure images...")
    for compound_id, data in compounds.items():
        structure_file = STRUCTURE_DIR / f"{compound_id}_structure.png"
        print(f"  Creating: {structure_file.name}")
        success = generate_structure_image(compound_id, data['smiles'], data['name'], structure_file)
        if success:
            print(f"    [OK] Saved")

    # Generate property comparison table
    print("\n5. Generating property comparison table...")
    comparison_file = OUTPUT_DIR / "exemplar_property_comparison.csv"
    df = generate_property_comparison_table(compounds, comparison_file)
    print(f"  [OK] Saved: {comparison_file.name}")

    # Generate figure captions
    print("\n6. Generating figure captions...")
    captions_file = OUTPUT_DIR / "figure_captions.txt"
    captions = generate_figure_captions(compounds, select_ranges, captions_file)
    print(f"  [OK] Saved: {captions_file.name}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF GENERATED FILES")
    print("=" * 70)

    print("\nRadar Plots:")
    for compound_id in compounds.keys():
        print(f"  - {OUTPUT_DIR / f'{compound_id}_radar_plot.png'}")
    print(f"  - {combined_file}")

    print("\nMolecular Structures:")
    for compound_id in compounds.keys():
        print(f"  - {STRUCTURE_DIR / f'{compound_id}_structure.png'}")

    print(f"\nSupporting Files:")
    print(f"  - {comparison_file}")
    print(f"  - {captions_file}")

    print("\n" + "=" * 70)
    print("PROPERTY SCALES USED FOR RADAR AXES:")
    print("=" * 70)
    for prop, scale in PROPERTY_SCALES.items():
        unit = f" {scale['unit']}" if scale['unit'] else ""
        print(f"  {prop}: {scale['min']} - {scale['max']}{unit}")

    print("\n" + "=" * 70)
    print("SELECT RANGES (mean +/- SD):")
    print("=" * 70)
    for pathogen, ranges in select_ranges.items():
        pathogen_name = {'SA': 'S. aureus (SELECT-G+)', 'EC': 'E. coli (SELECT-G-)', 'CA': 'C. albicans (SELECT-CA)'}
        print(f"\n{pathogen_name[pathogen]}:")
        for prop in PROPERTIES:
            mean = ranges[prop]['mean']
            sd = ranges[prop]['sd']
            lower = max(0, mean - sd)
            upper = mean + sd
            print(f"  {prop}: {mean:.2f} +/- {sd:.2f} (range: {lower:.2f} - {upper:.2f})")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
