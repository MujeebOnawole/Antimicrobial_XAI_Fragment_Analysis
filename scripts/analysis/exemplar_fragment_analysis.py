"""
Exemplar Compound Fragment Analysis
===================================
For each exemplar compound, identify the key contributing fragments
and their attribution scores to link compounds back to fragment-based work.
"""

import pandas as pd
import numpy as np
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

BASE_DIR = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis")
RAW_DIR = BASE_DIR / "raw_data"
OUTPUT_DIR = BASE_DIR / "results" / "statistics"

# Single-pathogen exemplars
SINGLE_EXEMPLARS = {
    'CHEMBL4536843': {
        'smiles': 'CC(C)[C@@H]1O/C(=C/C(=O)CCC2=CC=CC=C2)NC1=O',
        'pathogen': 'SA',
        'name': 'S. aureus (G+) exemplar'
    },
    'CHEMBL369493': {
        'smiles': 'CCCS(=O)(=O)N1C=CC2=C(SC(CC)=C2)B1O',
        'pathogen': 'EC',
        'name': 'E. coli (G-) exemplar'
    },
    'CHEMBL4277673': {
        'smiles': 'FC1=CC=C2SCC(CN3C=NC=N3)=C(Cl)C2=C1',
        'pathogen': 'CA',
        'name': 'C. albicans exemplar'
    }
}

# Dual and Triple exemplars
MULTI_EXEMPLARS = {
    'CHEMBL2178320': {
        'smiles': 'OC1=CC(CC(F)(F)F)=CC=C1OC1=CC=CC(F)=N1',
        'pathogens': ['SA', 'EC'],
        'name': 'SA+EC dual exemplar'
    },
    'CHEMBL5207371': {
        'smiles': 'O=C1NC2=CC=C(C(F)(F)F)C=C2N=C1CBr',
        'pathogens': ['SA', 'CA'],
        'name': 'SA+CA dual exemplar'
    },
    'CHEMBL5409101': {
        'smiles': 'O=P(C1=CC=CC=C1)(C1=CC=CC=C1)C1CCC/C1=N\\O',
        'pathogens': ['EC', 'CA'],
        'name': 'EC+CA dual exemplar'
    },
    'CHEMBL3822555': {
        'smiles': 'CC(C)(C)C1=CC(=O)C=C(NCCC2=CC=CC=C2)C1=O',
        'pathogens': ['SA', 'EC', 'CA'],
        'name': 'Triple-active exemplar'
    }
}

# Murcko files
MURCKO_FILES = {
    'SA': RAW_DIR / 'S_aureus_pred_class_murcko.csv',
    'EC': RAW_DIR / 'E_coli_pred_class_murcko.csv',
    'CA': RAW_DIR / 'C_albicans_pred_class_murcko.csv',
}


def get_fragments_from_compound(df_murcko, compound_id):
    """Extract all fragments and their attributions for a compound."""
    row = df_murcko[df_murcko['COMPOUND_ID'] == compound_id]
    if len(row) == 0:
        return None, None

    row = row.iloc[0]
    fragments = []

    # Check for murcko_substructure columns (up to 40)
    for i in range(40):
        smiles_col = f'murcko_substructure_{i}_smiles'
        attr_col = f'murcko_substructure_{i}_attribution'

        if smiles_col in row.index and attr_col in row.index:
            smiles = row[smiles_col]
            attr = row[attr_col]

            if pd.notna(smiles) and pd.notna(attr) and smiles != '':
                fragments.append({
                    'fragment_smiles': smiles,
                    'attribution': float(attr),
                    'type': 'substructure'
                })

    # Sort by attribution (descending)
    fragments = sorted(fragments, key=lambda x: x['attribution'], reverse=True)

    # Get prediction score
    pred = row.get('ensemble_prediction', row.get('prediction', None))

    return fragments, pred


def calc_props(smiles):
    """Calculate properties for a SMILES."""
    if not RDKIT_AVAILABLE:
        return {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        'MW': round(Descriptors.MolWt(mol), 1),
        'LogP': round(Descriptors.MolLogP(mol), 2),
        'TPSA': round(rdMolDescriptors.CalcTPSA(mol), 1),
        'HBD': rdMolDescriptors.CalcNumHBD(mol),
        'HBA': rdMolDescriptors.CalcNumHBA(mol),
        'Rings': rdMolDescriptors.CalcNumRings(mol),
    }


def main():
    print("=" * 70)
    print("EXEMPLAR COMPOUND FRAGMENT ANALYSIS")
    print("Linking exemplar compounds to their key contributing fragments")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load murcko files
    murcko_data = {}
    for pathogen, filepath in MURCKO_FILES.items():
        if filepath.exists():
            murcko_data[pathogen] = pd.read_csv(filepath)
            print(f"Loaded {pathogen} murcko data: {len(murcko_data[pathogen])} compounds")
        else:
            print(f"WARNING: {filepath} not found")

    all_results = []

    # Process single-pathogen exemplars
    print("\n" + "=" * 60)
    print("SINGLE-PATHOGEN EXEMPLARS")
    print("=" * 60)

    for chembl_id, data in SINGLE_EXEMPLARS.items():
        pathogen = data['pathogen']
        print(f"\n{data['name']} ({chembl_id}):")

        if pathogen not in murcko_data:
            print(f"  No murcko data for {pathogen}")
            continue

        fragments, pred = get_fragments_from_compound(murcko_data[pathogen], chembl_id)

        if fragments is None:
            print(f"  Compound not found in {pathogen} murcko data")
            # Still add to results with N/A
            all_results.append({
                'ChEMBL_ID': chembl_id,
                'Name': data['name'],
                'Compound_SMILES': data['smiles'],
                'Pathogen(s)': pathogen,
                'Prediction_Score': 'N/A',
                'Top_Fragment_SMILES': 'N/A',
                'Top_Fragment_Attribution': 'N/A',
                'Top_Fragment_MW': 'N/A',
                'Num_Fragments': 0
            })
            continue

        print(f"  Prediction score: {pred:.3f}" if pred else "  Prediction score: N/A")
        print(f"  Number of fragments: {len(fragments)}")

        if fragments:
            top_frag = fragments[0]
            frag_props = calc_props(top_frag['fragment_smiles'])
            print(f"  Top fragment: {top_frag['fragment_smiles']}")
            print(f"    Attribution: {top_frag['attribution']:.4f}")
            print(f"    MW: {frag_props.get('MW', 'N/A')}, Rings: {frag_props.get('Rings', 'N/A')}")

            # Show top 3 fragments
            print(f"  Top 3 fragments by attribution:")
            for i, frag in enumerate(fragments[:3]):
                fp = calc_props(frag['fragment_smiles'])
                print(f"    {i+1}. {frag['fragment_smiles'][:50]}... | attr={frag['attribution']:.4f} | MW={fp.get('MW', '?')}")

            all_results.append({
                'ChEMBL_ID': chembl_id,
                'Name': data['name'],
                'Compound_SMILES': data['smiles'],
                'Pathogen(s)': pathogen,
                'Prediction_Score': f"{pred:.3f}" if pred else 'N/A',
                'Top_Fragment_SMILES': top_frag['fragment_smiles'],
                'Top_Fragment_Attribution': f"{top_frag['attribution']:.4f}",
                'Top_Fragment_MW': frag_props.get('MW', 'N/A'),
                'Num_Fragments': len(fragments),
                'Fragment_2_SMILES': fragments[1]['fragment_smiles'] if len(fragments) > 1 else '',
                'Fragment_2_Attribution': f"{fragments[1]['attribution']:.4f}" if len(fragments) > 1 else '',
                'Fragment_3_SMILES': fragments[2]['fragment_smiles'] if len(fragments) > 2 else '',
                'Fragment_3_Attribution': f"{fragments[2]['attribution']:.4f}" if len(fragments) > 2 else '',
            })

    # Process multi-pathogen exemplars
    print("\n" + "=" * 60)
    print("DUAL & TRIPLE ACTIVE EXEMPLARS")
    print("=" * 60)

    for chembl_id, data in MULTI_EXEMPLARS.items():
        pathogens = data['pathogens']
        print(f"\n{data['name']} ({chembl_id}):")

        best_result = None
        best_attr = -1

        for pathogen in pathogens:
            if pathogen not in murcko_data:
                continue

            fragments, pred = get_fragments_from_compound(murcko_data[pathogen], chembl_id)

            if fragments:
                pred_str = f"{pred:.3f}" if pred else "N/A"
                print(f"  {pathogen}: pred={pred_str}, top_attr={fragments[0]['attribution']:.4f}")

                if fragments[0]['attribution'] > best_attr:
                    best_attr = fragments[0]['attribution']
                    best_result = {
                        'pathogen': pathogen,
                        'fragments': fragments,
                        'pred': pred
                    }

        if best_result:
            top_frag = best_result['fragments'][0]
            frag_props = calc_props(top_frag['fragment_smiles'])
            print(f"  Best fragment (from {best_result['pathogen']}): {top_frag['fragment_smiles']}")
            print(f"    Attribution: {top_frag['attribution']:.4f}")

            all_results.append({
                'ChEMBL_ID': chembl_id,
                'Name': data['name'],
                'Compound_SMILES': data['smiles'],
                'Pathogen(s)': '+'.join(pathogens),
                'Prediction_Score': f"{best_result['pred']:.3f}" if best_result['pred'] else 'N/A',
                'Top_Fragment_SMILES': top_frag['fragment_smiles'],
                'Top_Fragment_Attribution': f"{top_frag['attribution']:.4f}",
                'Top_Fragment_MW': frag_props.get('MW', 'N/A'),
                'Num_Fragments': len(best_result['fragments']),
                'Fragment_2_SMILES': best_result['fragments'][1]['fragment_smiles'] if len(best_result['fragments']) > 1 else '',
                'Fragment_2_Attribution': f"{best_result['fragments'][1]['attribution']:.4f}" if len(best_result['fragments']) > 1 else '',
                'Fragment_3_SMILES': best_result['fragments'][2]['fragment_smiles'] if len(best_result['fragments']) > 2 else '',
                'Fragment_3_Attribution': f"{best_result['fragments'][2]['attribution']:.4f}" if len(best_result['fragments']) > 2 else '',
            })
        else:
            print(f"  No fragment data found")
            all_results.append({
                'ChEMBL_ID': chembl_id,
                'Name': data['name'],
                'Compound_SMILES': data['smiles'],
                'Pathogen(s)': '+'.join(pathogens),
                'Prediction_Score': 'N/A',
                'Top_Fragment_SMILES': 'N/A',
                'Top_Fragment_Attribution': 'N/A',
                'Top_Fragment_MW': 'N/A',
                'Num_Fragments': 0
            })

    # Save results
    df_results = pd.DataFrame(all_results)
    output_file = OUTPUT_DIR / 'exemplar_compounds_with_fragments.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n\nSaved: {output_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: EXEMPLAR COMPOUNDS WITH KEY FRAGMENTS")
    print("=" * 70)
    print(f"\n{'ChEMBL ID':<15} {'Pathogen':<12} {'Top Fragment':<40} {'Attribution':<12}")
    print("-" * 80)
    for r in all_results:
        frag_short = r['Top_Fragment_SMILES'][:35] + '...' if len(str(r['Top_Fragment_SMILES'])) > 35 else r['Top_Fragment_SMILES']
        print(f"{r['ChEMBL_ID']:<15} {r['Pathogen(s)']:<12} {frag_short:<40} {r['Top_Fragment_Attribution']:<12}")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
