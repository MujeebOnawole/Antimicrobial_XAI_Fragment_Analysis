"""
Exemplar Compound + Fragment Visualization
==========================================
Creates images showing each exemplar compound alongside its key
contributing fragment(s) with attribution scores.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, Descriptors, rdMolDescriptors

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

BASE_DIR = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis")
RAW_DIR = BASE_DIR / "raw_data"
OUTPUT_DIR = BASE_DIR / "figures" / "exemplar_fragments"

# Exemplar data with fragments (from previous analysis)
EXEMPLARS = {
    'CHEMBL4536843': {
        'smiles': 'CC(C)[C@@H]1O/C(=C/C(=O)CCC2=CC=CC=C2)NC1=O',
        'name': 'S. aureus (G+)',
        'pathogen': 'SA',
        'color': '#DC143C',
        'prediction': 0.994,
        'fragments': [
            {'smiles': 'C(/C=C1\\NCCO1)CCC1=CC=CC=C1', 'attribution': 0.7616},
            {'smiles': 'C1COCN1', 'attribution': 0.5062},
            {'smiles': 'C1=CC=CC=C1', 'attribution': 0.0665},
        ]
    },
    'CHEMBL369493': {
        'smiles': 'CCCS(=O)(=O)N1C=CC2=C(SC(CC)=C2)B1O',
        'name': 'E. coli (G-)',
        'pathogen': 'EC',
        'color': '#1E90FF',
        'prediction': 0.922,
        'fragments': [
            {'smiles': 'B1NC=CC2=C1SC=C2', 'attribution': 0.7616},
            {'smiles': 'B1C=CC=CN1', 'attribution': 0.7347},
            {'smiles': 'C1=CSC=C1', 'attribution': 0.7223},
        ]
    },
    'CHEMBL4277673': {
        'smiles': 'FC1=CC=C2SCC(CN3C=NC=N3)=C(Cl)C2=C1',
        'name': 'C. albicans',
        'pathogen': 'CA',
        'color': '#228B22',
        'prediction': 0.989,
        'fragments': [
            {'smiles': 'C1=CCSC=C1', 'attribution': 0.7616},
            {'smiles': 'C1=CC2=CC=CC=C2SC1', 'attribution': 0.7482},
            {'smiles': 'C1=C(CN2C=NC=N2)CSC=C1', 'attribution': 0.6545},
        ]
    },
    'CHEMBL2178320': {
        'smiles': 'OC1=CC(CC(F)(F)F)=CC=C1OC1=CC=CC(F)=N1',
        'name': 'SA+EC Dual',
        'pathogen': 'SA+EC',
        'color': '#9932CC',
        'prediction': 0.985,
        'fragments': [
            {'smiles': 'C1=CC=C(OC2=CC=CC=N2)C=C1', 'attribution': 0.7616},
            {'smiles': 'C1=CC=NC=C1', 'attribution': 0.4891},
        ]
    },
    'CHEMBL5207371': {
        'smiles': 'O=C1NC2=CC=C(C(F)(F)F)C=C2N=C1CBr',
        'name': 'SA+CA Dual',
        'pathogen': 'SA+CA',
        'color': '#B8860B',
        'prediction': 0.809,
        'fragments': [
            {'smiles': 'C1=CC=CC=C1', 'attribution': 0.7616},
            {'smiles': 'C1=NC2=CC=CC=C2N1', 'attribution': 0.5234},
        ]
    },
    'CHEMBL5409101': {
        'smiles': 'O=P(C1=CC=CC=C1)(C1=CC=CC=C1)C1CCC/C1=N\\O',
        'name': 'EC+CA Dual',
        'pathogen': 'EC+CA',
        'color': '#20B2AA',
        'prediction': 0.953,
        'fragments': [
            {'smiles': 'C1CCCC1PC1=CC=CC=C1', 'attribution': 0.7616},
            {'smiles': 'C1=CC=CC=C1', 'attribution': 0.3808},
        ]
    },
    'CHEMBL3822555': {
        'smiles': 'CC(C)(C)C1=CC(=O)C=C(NCCC2=CC=CC=C2)C1=O',
        'name': 'Triple Active',
        'pathogen': 'SA+EC+CA',
        'color': '#FF8C00',
        'prediction': 0.693,
        'fragments': [
            {'smiles': 'C1C=CCC(NCCC2=CC=CC=C2)=C1', 'attribution': 0.7616},
            {'smiles': 'C1=CC=CC=C1', 'attribution': 0.3808},
        ]
    },
}


def mol_to_image(smiles, size=(350, 350)):
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
    }


def create_compound_fragment_image(chembl_id, data, output_path):
    """Create image showing compound and top 2 fragments."""

    fragments = data['fragments'][:2]  # Top 2 fragments
    n_fragments = len(fragments)

    # Create figure with compound on left, fragments on right
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.2)

    # Compound panel (spans both rows)
    ax_compound = fig.add_subplot(gs[:, 0])
    compound_img = mol_to_image(data['smiles'], size=(400, 400))
    if compound_img:
        ax_compound.imshow(compound_img)
    ax_compound.axis('off')
    ax_compound.set_title(f"Exemplar Compound\n{chembl_id}", fontsize=14, fontweight='bold', pad=10)

    # Add compound info
    compound_props = calc_props(data['smiles'])
    info_text = f"Prediction: {data['prediction']:.3f}\nMW: {compound_props.get('MW', 'N/A')} Da"
    ax_compound.text(0.5, -0.05, info_text, transform=ax_compound.transAxes,
                     ha='center', fontsize=11, style='italic')

    # Fragment panels
    for i, frag in enumerate(fragments):
        ax_frag = fig.add_subplot(gs[i, 1:])

        frag_img = mol_to_image(frag['smiles'], size=(300, 300))
        if frag_img:
            ax_frag.imshow(frag_img)
        ax_frag.axis('off')

        frag_props = calc_props(frag['smiles'])
        title = f"Fragment {i+1}: Attribution = {frag['attribution']:.4f}"
        ax_frag.set_title(title, fontsize=12, fontweight='bold', color=data['color'])

        # Fragment info
        frag_info = f"MW: {frag_props.get('MW', 'N/A')} Da | Rings: {frag_props.get('Rings', 'N/A')}"
        ax_frag.text(0.5, -0.08, frag_info, transform=ax_frag.transAxes,
                     ha='center', fontsize=10)

    # Main title
    fig.suptitle(f"{data['name']} Exemplar: Compound + Key Fragments",
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path.name}")


def create_single_panel_image(chembl_id, data, output_path):
    """Create single row image: compound + top fragment side by side."""

    top_frag = data['fragments'][0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Compound
    compound_img = mol_to_image(data['smiles'], size=(400, 400))
    if compound_img:
        axes[0].imshow(compound_img)
    axes[0].axis('off')
    axes[0].set_title(f"Compound: {chembl_id}\nPrediction: {data['prediction']:.3f}",
                      fontsize=13, fontweight='bold')

    # Top Fragment
    frag_img = mol_to_image(top_frag['smiles'], size=(400, 400))
    if frag_img:
        axes[1].imshow(frag_img)
    axes[1].axis('off')

    frag_props = calc_props(top_frag['smiles'])
    axes[1].set_title(f"Top Fragment\nAttribution: {top_frag['attribution']:.4f} | MW: {frag_props.get('MW', 'N/A')} Da",
                      fontsize=13, fontweight='bold', color=data['color'])

    # Arrow between panels
    fig.text(0.5, 0.5, r'$\rightarrow$', fontsize=40, ha='center', va='center',
             transform=fig.transFigure)

    fig.suptitle(f"{data['name']} Exemplar", fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path.name}")


def create_grid_summary(output_path):
    """Create a grid showing all exemplars with their top fragments."""

    n_exemplars = len(EXEMPLARS)
    fig, axes = plt.subplots(n_exemplars, 3, figsize=(15, n_exemplars * 3.5))

    for idx, (chembl_id, data) in enumerate(EXEMPLARS.items()):
        # Compound
        compound_img = mol_to_image(data['smiles'], size=(300, 300))
        if compound_img:
            axes[idx, 0].imshow(compound_img)
        axes[idx, 0].axis('off')
        if idx == 0:
            axes[idx, 0].set_title("Compound", fontsize=14, fontweight='bold')
        axes[idx, 0].text(0.5, -0.1, f"{chembl_id}\n{data['name']}",
                          transform=axes[idx, 0].transAxes, ha='center', fontsize=10)

        # Top Fragment
        top_frag = data['fragments'][0]
        frag_img = mol_to_image(top_frag['smiles'], size=(300, 300))
        if frag_img:
            axes[idx, 1].imshow(frag_img)
        axes[idx, 1].axis('off')
        if idx == 0:
            axes[idx, 1].set_title("Top Fragment", fontsize=14, fontweight='bold')
        frag_props = calc_props(top_frag['smiles'])
        axes[idx, 1].text(0.5, -0.1, f"Attr: {top_frag['attribution']:.4f}\nMW: {frag_props.get('MW', 'N/A')} Da",
                          transform=axes[idx, 1].transAxes, ha='center', fontsize=10, color=data['color'])

        # Second Fragment (if available)
        if len(data['fragments']) > 1:
            frag2 = data['fragments'][1]
            frag2_img = mol_to_image(frag2['smiles'], size=(300, 300))
            if frag2_img:
                axes[idx, 2].imshow(frag2_img)
            frag2_props = calc_props(frag2['smiles'])
            axes[idx, 2].text(0.5, -0.1, f"Attr: {frag2['attribution']:.4f}\nMW: {frag2_props.get('MW', 'N/A')} Da",
                              transform=axes[idx, 2].transAxes, ha='center', fontsize=10, color=data['color'])
        axes[idx, 2].axis('off')
        if idx == 0:
            axes[idx, 2].set_title("2nd Fragment", fontsize=14, fontweight='bold')

    fig.suptitle("All Exemplar Compounds with Key Contributing Fragments",
                 fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved grid summary: {output_path.name}")


def main():
    print("=" * 70)
    print("EXEMPLAR COMPOUND + FRAGMENT VISUALIZATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Individual images for each exemplar
    print("\nGenerating individual compound+fragment images...")
    for chembl_id, data in EXEMPLARS.items():
        print(f"\n{data['name']} ({chembl_id}):")

        # Two-fragment layout
        output_file = OUTPUT_DIR / f'{chembl_id}_compound_fragments.png'
        create_compound_fragment_image(chembl_id, data, output_file)

        # Simple side-by-side
        output_file2 = OUTPUT_DIR / f'{chembl_id}_compound_top_fragment.png'
        create_single_panel_image(chembl_id, data, output_file2)

    # Grid summary of all exemplars
    print("\nGenerating grid summary...")
    grid_file = OUTPUT_DIR / 'all_exemplars_with_fragments_grid.png'
    create_grid_summary(grid_file)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
