"""
Find better SA+CA exemplar that fits within SELECT zones.
The current exemplar has LogP > 7 which exceeds both SA and CA SELECT zones.
"""

import pandas as pd
import os
import re

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

BASE_DIR = r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis\DUAL_ACTIVE_POSITIVE"

# SELECT zone LogP ranges
SELECT_LOGP = {
    'S_aureus': (1.9, 4.1),
    'C_albicans': (1.7, 5.0),
}

def calculate_logp(smiles):
    """Calculate LogP from SMILES."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolLogP(mol)

def calculate_all_props(smiles):
    """Calculate all properties from SMILES."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
        'HBD': rdMolDescriptors.CalcNumHBD(mol),
        'HBA': rdMolDescriptors.CalcNumHBA(mol),
        'Rings': rdMolDescriptors.CalcNumRings(mol),
    }

def parse_pathogen_examples(example_str):
    """Extract compound examples from pathogen_examples column."""
    if pd.isna(example_str) or not example_str:
        return []

    compounds = []
    pattern = r'(\w+)\s+Example:\s+(\w+)\s*\|\s*SMILES:\s*([^\|]+)\s*\|'
    matches = re.findall(pattern, example_str)

    seen = set()
    for match in matches:
        pathogen, compound_id, smiles = match
        if compound_id not in seen:
            seen.add(compound_id)
            compounds.append({
                'pathogen': pathogen,
                'compound_id': compound_id,
                'smiles': smiles.strip()
            })
    return compounds

def main():
    print("=" * 70)
    print("SEARCHING FOR BETTER SA+CA EXEMPLAR")
    print("Target LogP range: 1.7 - 5.0 (overlap of SA and CA SELECT zones)")
    print("=" * 70)

    # Load SA+CA scaffolds
    filepath = os.path.join(BASE_DIR, 'dual_SA_CA_positive_scaffolds.csv')
    df = pd.read_csv(filepath)

    print(f"\nLoaded {len(df)} SA+CA scaffolds")

    # Filter for high-quality scaffolds
    df_good = df[
        (df['avg_activity_rate_percent'] >= 90) &
        (df['total_compounds_both_pathogens'] >= 10)
    ].copy()

    print(f"Scaffolds with >=90% activity and >=10 compounds: {len(df_good)}")

    # Calculate LogP for each scaffold
    print("\nCalculating scaffold LogP values...")
    df_good['scaffold_logp'] = df_good['fragment_smiles'].apply(calculate_logp)

    # Find scaffolds with LogP in acceptable range (1.7 - 5.0)
    df_acceptable = df_good[
        (df_good['scaffold_logp'] >= 1.7) &
        (df_good['scaffold_logp'] <= 5.0)
    ].sort_values('total_compounds_both_pathogens', ascending=False)

    print(f"\nScaffolds with LogP 1.7-5.0: {len(df_acceptable)}")

    if len(df_acceptable) > 0:
        print("\n" + "=" * 50)
        print("TOP SA+CA SCAFFOLDS WITH ACCEPTABLE LogP:")
        print("=" * 50)

        for i, (idx, row) in enumerate(df_acceptable.head(10).iterrows()):
            print(f"\nRank {i+1}: Fragment {row['fragment_id']}")
            print(f"  SMILES: {row['fragment_smiles'][:60]}...")
            print(f"  Scaffold LogP: {row['scaffold_logp']:.2f}")
            print(f"  Activity: {row['avg_activity_rate_percent']:.1f}%")
            print(f"  Compounds: {row['total_compounds_both_pathogens']}")

            # Try to extract compound examples
            examples = parse_pathogen_examples(row.get('pathogen_examples', ''))
            if examples:
                print(f"  Example compounds:")
                for ex in examples[:2]:
                    props = calculate_all_props(ex['smiles'])
                    if props:
                        print(f"    - {ex['compound_id']}: LogP={props['LogP']:.2f}, MW={props['MW']:.1f}")

    # Also check compounds directly
    print("\n" + "=" * 50)
    print("SEARCHING FOR COMPOUNDS WITH ACCEPTABLE LogP:")
    print("=" * 50)

    good_compounds = []

    for idx, row in df_good.head(50).iterrows():  # Check top 50 scaffolds
        examples = parse_pathogen_examples(row.get('pathogen_examples', ''))
        for ex in examples:
            props = calculate_all_props(ex['smiles'])
            if props and 1.7 <= props['LogP'] <= 5.0:
                good_compounds.append({
                    'fragment_id': row['fragment_id'],
                    'fragment_smiles': row['fragment_smiles'],
                    'compound_id': ex['compound_id'],
                    'compound_smiles': ex['smiles'],
                    'activity_rate': row['avg_activity_rate_percent'],
                    'total_compounds': row['total_compounds_both_pathogens'],
                    **props
                })

    if good_compounds:
        df_good_compounds = pd.DataFrame(good_compounds)
        df_good_compounds = df_good_compounds.sort_values('total_compounds', ascending=False)

        print(f"\nFound {len(df_good_compounds)} compounds with LogP 1.7-5.0")
        print("\nTop candidates:")

        for i, row in df_good_compounds.head(5).iterrows():
            print(f"\n  {row['compound_id']}:")
            print(f"    SMILES: {row['compound_smiles'][:60]}...")
            print(f"    LogP: {row['LogP']:.2f}")
            print(f"    MW: {row['MW']:.1f}")
            print(f"    TPSA: {row['TPSA']:.1f}")
            print(f"    HBD: {row['HBD']}, HBA: {row['HBA']}")
            print(f"    From scaffold: {row['fragment_id']} ({row['total_compounds']} compounds)")

        # Save best candidate
        best = df_good_compounds.iloc[0]
        print("\n" + "=" * 50)
        print("RECOMMENDED SA+CA EXEMPLAR:")
        print("=" * 50)
        print(f"Compound ID: {best['compound_id']}")
        print(f"SMILES: {best['compound_smiles']}")
        print(f"LogP: {best['LogP']:.2f}")
        print(f"MW: {best['MW']:.1f}")
        print(f"TPSA: {best['TPSA']:.1f}")
        print(f"HBD: {best['HBD']}, HBA: {best['HBA']}, Rings: {best['Rings']}")

        # Save to CSV
        output_path = os.path.join(BASE_DIR, 'radar_plots', 'sa_ca_alternative_exemplars.csv')
        df_good_compounds.to_csv(output_path, index=False)
        print(f"\nSaved alternatives to: {output_path}")
    else:
        print("\nNo compounds found with LogP in acceptable range.")
        print("The SA+CA dual-active compounds may use non-traditional mechanisms.")

if __name__ == '__main__':
    main()
