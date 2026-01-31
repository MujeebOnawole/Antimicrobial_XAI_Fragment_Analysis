"""
Dual and Triple Active Exemplar + Fragment Visualization
=========================================================
Extracts the highest attribution fragments from prediction files and creates
combined images showing each exemplar compound alongside its top fragment.
Also creates a CSV with fragment details.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, Descriptors, rdMolDescriptors

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

BASE_DIR = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis")
RAW_DIR = BASE_DIR / "raw_data"
OUTPUT_DIR = BASE_DIR / "figures" / "exemplar_fragments"

# Prediction files
PRED_FILES = {
    'SA': RAW_DIR / "S_aureus_pred_class_murcko.csv",
    'EC': RAW_DIR / "E_coli_pred_class_murcko.csv",
    'CA': RAW_DIR / "C_albicans_pred_class_murcko.csv",
}

# Dual and Triple Active Exemplars
EXEMPLARS = {
    'CHEMBL2178320': {
        'smiles': 'OC1=CC(CC(F)(F)F)=CC=C1OC1=CC=CC(F)=N1',
        'name': r'$\it{S. aureus}$ + $\it{E. coli}$',
        'short_name': 'SA+EC Dual',
        'pathogens': ['SA', 'EC'],
        'color': '#9932CC',
        'mic': {'SA': 0.25, 'EC': 4.0},
    },
    'CHEMBL5207371': {
        'smiles': 'O=C1NC2=CC=C(C(F)(F)F)C=C2N=C1CBr',
        'name': r'$\it{S. aureus}$ + $\it{C. albicans}$',
        'short_name': 'SA+CA Dual',
        'pathogens': ['SA', 'CA'],
        'color': '#B8860B',
        'mic': {'SA': 31.2, 'CA': 31.2},
    },
    'CHEMBL5409101': {
        'smiles': 'O=P(C1=CC=CC=C1)(C1=CC=CC=C1)C1CCC/C1=N\\O',
        'name': r'$\it{E. coli}$ + $\it{C. albicans}$',
        'short_name': 'EC+CA Dual',
        'pathogens': ['EC', 'CA'],
        'color': '#20B2AA',
        'mic': {'EC': 0.1, 'CA': 0.1},
    },
    'CHEMBL2297203': {
        'smiles': 'CC1=CC(C(=O)/C=C/C2=CC=CN2)=C(C)O1',
        'name': 'Broad-spectrum',
        'short_name': 'Triple Active',
        'pathogens': ['SA', 'EC', 'CA'],
        'color': '#FF8C00',
        'mic': {'SA': 30.0, 'EC': 40.0, 'CA': 30.0},
    },
}


def extract_fragments_from_row(row):
    """Extract all fragments and their attribution scores from a prediction row."""
    fragments = []

    # Check for scaffold fragments (murcko_substructure_X)
    for i in range(40):  # Check up to 40 substructures
        smiles_col = f'murcko_substructure_{i}_smiles'
        attr_col = f'murcko_substructure_{i}_attribution'

        if smiles_col in row.index and attr_col in row.index:
            smiles = row[smiles_col]
            attr = row[attr_col]

            if pd.notna(smiles) and pd.notna(attr) and smiles != '':
                fragments.append({
                    'type': 'scaffold',
                    'index': i,
                    'smiles': smiles,
                    'attribution': float(attr),
                    'id': f'scaffold_{i}'
                })

    # Check for substituent fragments (murcko_substituent_X_Y)
    for i in range(10):  # scaffold index
        for j in range(12):  # substituent index
            smiles_col = f'murcko_substituent_{i}_{j}_smiles'
            attr_col = f'murcko_substituent_{i}_{j}_attribution'

            if smiles_col in row.index and attr_col in row.index:
                smiles = row[smiles_col]
                attr = row[attr_col]

                if pd.notna(smiles) and pd.notna(attr) and smiles != '':
                    fragments.append({
                        'type': 'substituent',
                        'index': (i, j),
                        'smiles': smiles,
                        'attribution': float(attr),
                        'id': f'substituent_{i}_{j}'
                    })

    return fragments


def get_compound_fragments(chembl_id, pathogen):
    """Get fragments for a compound from a specific pathogen's prediction file."""
    pred_file = PRED_FILES[pathogen]

    # Read only the row for this compound
    df = pd.read_csv(pred_file)
    row = df[df['COMPOUND_ID'] == chembl_id]

    if row.empty:
        return None, None

    row = row.iloc[0]
    ensemble_pred = row['ensemble_prediction']

    fragments = extract_fragments_from_row(row)

    return ensemble_pred, fragments


def get_top_fragment_for_exemplar(chembl_id, pathogens):
    """Get the top fragment across all relevant pathogen files."""
    all_fragments = []
    predictions = {}

    for pathogen in pathogens:
        pred, frags = get_compound_fragments(chembl_id, pathogen)
        if frags:
            for frag in frags:
                frag['source_pathogen'] = pathogen
            all_fragments.extend(frags)
        if pred is not None:
            predictions[pathogen] = pred

    if not all_fragments:
        return None, predictions

    # Find unique fragments by SMILES and get max attribution
    unique_frags = {}
    for frag in all_fragments:
        smiles = frag['smiles']
        if smiles not in unique_frags or frag['attribution'] > unique_frags[smiles]['attribution']:
            unique_frags[smiles] = frag

    # Sort by attribution and return top
    sorted_frags = sorted(unique_frags.values(), key=lambda x: x['attribution'], reverse=True)

    return sorted_frags, predictions


def mol_to_image(smiles, size=(400, 400)):
    """Convert SMILES to image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rdDepictor.Compute2DCoords(mol)
    return Draw.MolToImage(mol, size=size)


def calc_props(smiles):
    """Calculate basic properties."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        'MW': round(Descriptors.MolWt(mol), 1),
        'Rings': rdMolDescriptors.CalcNumRings(mol),
        'LogP': round(Descriptors.MolLogP(mol), 2),
    }


def create_combined_image(chembl_id, data, top_fragment, predictions, output_path):
    """Create image showing compound + top fragment side by side."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Compound panel
    compound_img = mol_to_image(data['smiles'], size=(450, 450))
    if compound_img:
        axes[0].imshow(compound_img)
    axes[0].axis('off')

    # Compound title and info
    pred_text = ', '.join([f"{p}={predictions.get(p, 'N/A'):.3f}" for p in data['pathogens']])
    compound_props = calc_props(data['smiles'])
    axes[0].set_title(f"Exemplar Compound: {chembl_id}\n{data['name']}",
                      fontsize=14, fontweight='bold', pad=15)

    mic_text = ' / '.join([f"{p}: {data['mic'].get(p, 'N/A')} µg/mL" for p in data['pathogens']])
    axes[0].text(0.5, -0.08, f"Predictions: {pred_text}\nMIC: {mic_text}",
                 transform=axes[0].transAxes, ha='center', fontsize=11, style='italic')

    # Fragment panel
    if top_fragment:
        frag_img = mol_to_image(top_fragment['smiles'], size=(450, 450))
        if frag_img:
            axes[1].imshow(frag_img)
        axes[1].axis('off')

        frag_props = calc_props(top_fragment['smiles'])
        axes[1].set_title(f"Top Contributing Fragment\nAttribution: {top_fragment['attribution']:.4f}",
                          fontsize=14, fontweight='bold', color=data['color'], pad=15)

        frag_info = f"SMILES: {top_fragment['smiles'][:40]}{'...' if len(top_fragment['smiles']) > 40 else ''}\n"
        frag_info += f"MW: {frag_props.get('MW', 'N/A')} Da | LogP: {frag_props.get('LogP', 'N/A')} | Rings: {frag_props.get('Rings', 'N/A')}"
        axes[1].text(0.5, -0.08, frag_info, transform=axes[1].transAxes,
                     ha='center', fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "No fragment data available", transform=axes[1].transAxes,
                     ha='center', va='center', fontsize=14)
        axes[1].axis('off')

    # Arrow between panels
    fig.text(0.5, 0.5, r'$\longleftarrow$', fontsize=50, ha='center', va='center',
             transform=fig.transFigure, color='gray')
    fig.text(0.5, 0.42, 'contains', fontsize=12, ha='center', va='center',
             transform=fig.transFigure, color='gray', style='italic')

    fig.suptitle(f"{data['short_name']} Exemplar with Top Contributing Fragment",
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return True


def create_multi_fragment_image(chembl_id, data, fragments, predictions, output_path):
    """Create image showing compound + top 2 fragments."""

    if len(fragments) < 2:
        return False

    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.3, 1, 1], hspace=0.35, wspace=0.15)

    # Compound panel (spans both rows)
    ax_compound = fig.add_subplot(gs[:, 0])
    compound_img = mol_to_image(data['smiles'], size=(450, 450))
    if compound_img:
        ax_compound.imshow(compound_img)
    ax_compound.axis('off')
    ax_compound.set_title(f"Exemplar: {chembl_id}\n{data['name']}",
                          fontsize=13, fontweight='bold', pad=10)

    # Fragment panels
    for i, frag in enumerate(fragments[:2]):
        ax_frag = fig.add_subplot(gs[i, 1:])

        frag_img = mol_to_image(frag['smiles'], size=(350, 350))
        if frag_img:
            ax_frag.imshow(frag_img)
        ax_frag.axis('off')

        frag_props = calc_props(frag['smiles'])
        title = f"Fragment {i+1}: Attribution = {frag['attribution']:.4f}"
        ax_frag.set_title(title, fontsize=12, fontweight='bold', color=data['color'])

        frag_info = f"MW: {frag_props.get('MW', 'N/A')} Da | Type: {frag['type']}"
        ax_frag.text(0.5, -0.12, frag_info, transform=ax_frag.transAxes, ha='center', fontsize=10)

    fig.suptitle(f"{data['short_name']} Exemplar with Top 2 Contributing Fragments",
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return True


def main():
    print("=" * 70)
    print("DUAL & TRIPLE ACTIVE EXEMPLAR + FRAGMENT VISUALIZATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect fragment data for CSV
    csv_data = []

    for chembl_id, data in EXEMPLARS.items():
        print(f"\n{data['short_name']} ({chembl_id}):")
        print(f"  Pathogens: {', '.join(data['pathogens'])}")

        # Get top fragments
        fragments, predictions = get_top_fragment_for_exemplar(chembl_id, data['pathogens'])

        if not fragments:
            print(f"  WARNING: No fragment data found!")
            continue

        print(f"  Predictions: {predictions}")
        print(f"  Top fragment: {fragments[0]['smiles'][:50]}...")
        print(f"  Attribution: {fragments[0]['attribution']:.4f}")

        # Store for CSV
        for i, frag in enumerate(fragments[:5]):  # Top 5 fragments
            frag_props = calc_props(frag['smiles'])
            csv_data.append({
                'ChEMBL_ID': chembl_id,
                'Combination': data['short_name'],
                'Pathogens': '+'.join(data['pathogens']),
                'Fragment_Rank': i + 1,
                'Fragment_ID': frag['id'],
                'Fragment_Type': frag['type'],
                'Fragment_SMILES': frag['smiles'],
                'Attribution_Score': round(frag['attribution'], 4),
                'Fragment_MW': frag_props.get('MW'),
                'Fragment_LogP': frag_props.get('LogP'),
                'Fragment_Rings': frag_props.get('Rings'),
                'Source_Pathogen': frag.get('source_pathogen', 'N/A'),
            })

        # Create single panel image (compound + top fragment)
        output_single = OUTPUT_DIR / f'{chembl_id}_compound_top_fragment.png'
        create_combined_image(chembl_id, data, fragments[0], predictions, output_single)
        print(f"  [OK] {output_single.name}")

        # Create multi-fragment image (compound + top 2 fragments)
        if len(fragments) >= 2:
            output_multi = OUTPUT_DIR / f'{chembl_id}_compound_fragments.png'
            create_multi_fragment_image(chembl_id, data, fragments, predictions, output_multi)
            print(f"  [OK] {output_multi.name}")

    # Save CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_file = OUTPUT_DIR / 'dual_triple_exemplar_fragments.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n[OK] Fragment data saved to: {csv_file.name}")

        # Print summary table
        print("\n" + "=" * 70)
        print("TOP FRAGMENTS SUMMARY")
        print("=" * 70)

        for chembl_id in EXEMPLARS.keys():
            subset = df[df['ChEMBL_ID'] == chembl_id]
            if not subset.empty:
                top = subset.iloc[0]
                print(f"\n{top['Combination']} ({chembl_id}):")
                print(f"  Top Fragment SMILES: {top['Fragment_SMILES']}")
                print(f"  Attribution Score: {top['Attribution_Score']}")
                print(f"  Fragment Type: {top['Fragment_Type']}")
                print(f"  Fragment MW: {top['Fragment_MW']} Da")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
