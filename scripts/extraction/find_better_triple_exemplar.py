"""
Find Better Triple-Active Exemplar
===================================
Finds triple-active compounds that:
1. Have EXPERIMENTAL activity against all three pathogens (from source data)
2. Have PREDICTED activity (ensemble_prediction >= 0.5) for all three pathogens
3. Have 0 violations across all 8 physicochemical properties
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
    print("ERROR: RDKit required")

# Paths
BASE_DIR = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\fragments_analysis")
RAW_DATA_DIR = BASE_DIR / "raw_data"
SOURCE_DATA_DIR = RAW_DATA_DIR / "source_data"
OUTPUT_DIR = BASE_DIR / "figures" / "radar_plots"

# Files
SOURCE_FILES = {
    'SA': SOURCE_DATA_DIR / "S_aureus_raw_data.csv",
    'EC': SOURCE_DATA_DIR / "E_coli_raw_data.csv",
    'CA': SOURCE_DATA_DIR / "C_albicans_raw_data.csv",
}

PRED_FILES = {
    'SA': RAW_DATA_DIR / "S_aureus_pred_class_murcko.csv",
    'EC': RAW_DATA_DIR / "E_coli_pred_class_murcko.csv",
    'CA': RAW_DATA_DIR / "C_albicans_pred_class_murcko.csv",
}

# SELECT ranges (mean +/- SD)
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

PROPERTIES = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Rings', 'Fsp3']
# Use binary prediction column (0 or 1) - each pathogen has its own threshold


def calculate_properties(smiles):
    """Calculate all 8 physicochemical properties from SMILES."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
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
    except:
        return None


def get_triple_range():
    """Get combined SELECT range for triple-active (intersection of all three)."""
    combined = {}
    for prop in PROPERTIES:
        lower_bounds = []
        upper_bounds = []
        for p in ['SA', 'EC', 'CA']:
            mean = SELECT_RANGES[p][prop]['mean']
            sd = SELECT_RANGES[p][prop]['sd']
            lower_bounds.append(mean - sd)
            upper_bounds.append(mean + sd)
        combined[prop] = {
            'lower': max(lower_bounds),
            'upper': min(upper_bounds)
        }
    return combined


def count_violations(props, combined_range):
    """Count property violations."""
    violations = 0
    details = []
    for prop in PROPERTIES:
        value = props[prop]
        lower = combined_range[prop]['lower']
        upper = combined_range[prop]['upper']
        if value < lower or value > upper:
            violations += 1
            details.append(f"{prop}: {value} (range: {lower:.2f}-{upper:.2f})")
    return violations, details


def main():
    print("=" * 70)
    print("FINDING BETTER TRIPLE-ACTIVE EXEMPLAR")
    print("Criteria: Experimental + Predicted active for all 3 pathogens")
    print("         + 0 violations across all 8 properties")
    print("=" * 70)

    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit is required")
        return

    # Load experimental activity data
    print("\n1. Loading experimental activity data...")
    exp_active = {}
    exp_mic = {}

    for pathogen, filepath in SOURCE_FILES.items():
        print(f"   Loading {pathogen} from {filepath.name}...")
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.lower()

        # Filter for active compounds
        df_active = df[df['activity'].isin(['active', 1, '1'])].copy()

        for _, row in df_active.iterrows():
            cid = row['compound_id']
            if cid not in exp_active:
                exp_active[cid] = set()
                exp_mic[cid] = {}
            exp_active[cid].add(pathogen)

            # Get MIC
            mic = row.get('mic_ug_ml')
            if pd.notna(mic):
                exp_mic[cid][pathogen] = mic

        print(f"      {len(df_active)} active compounds")

    # Find experimentally triple-active
    triple_exp = {cid for cid, pathogens in exp_active.items()
                  if pathogens == {'SA', 'EC', 'CA'}}
    print(f"\n   Experimentally triple-active compounds: {len(triple_exp)}")

    # Load prediction data (using binary prediction column, not ensemble)
    print("\n2. Loading prediction data (binary prediction column)...")
    predictions = {}
    ensemble_preds = {}
    smiles_map = {}

    for pathogen, filepath in PRED_FILES.items():
        print(f"   Loading {pathogen} predictions from {filepath.name}...")
        df = pd.read_csv(filepath, usecols=['COMPOUND_ID', 'SMILES', 'prediction', 'ensemble_prediction'])

        for _, row in df.iterrows():
            cid = row['COMPOUND_ID']
            if cid not in predictions:
                predictions[cid] = {}
                ensemble_preds[cid] = {}
                smiles_map[cid] = row['SMILES']
            predictions[cid][pathogen] = row['prediction']  # Binary: 0 or 1
            ensemble_preds[cid][pathogen] = row['ensemble_prediction']

    # Filter for predicted active (prediction == 1) in all three
    print("\n3. Filtering for predicted active (prediction == 1) in all three...")
    triple_pred = set()
    for cid, preds in predictions.items():
        if len(preds) == 3:
            if all(p == 1 for p in preds.values()):
                triple_pred.add(cid)
    print(f"   Predicted triple-active compounds: {len(triple_pred)}")

    # Intersection: experimentally AND predicted triple-active
    candidates = triple_exp & triple_pred
    print(f"\n4. Compounds with BOTH experimental AND predicted triple activity: {len(candidates)}")

    if not candidates:
        print("\nNo compounds meet both criteria. Relaxing to experimental only...")
        candidates = triple_exp

    # Calculate properties and check violations
    print("\n5. Checking property violations...")
    triple_range = get_triple_range()

    print("\n   Triple SELECT range (intersection):")
    for prop in PROPERTIES:
        r = triple_range[prop]
        print(f"      {prop}: {r['lower']:.2f} - {r['upper']:.2f}")

    results = []
    for cid in candidates:
        smiles = smiles_map.get(cid)
        if not smiles:
            continue

        props = calculate_properties(smiles)
        if props is None:
            continue

        violations, details = count_violations(props, triple_range)

        # Get predictions (binary and ensemble)
        binary_preds = predictions.get(cid, {})
        ens_preds = ensemble_preds.get(cid, {})

        results.append({
            'compound_id': cid,
            'smiles': smiles,
            'violations': violations,
            'violation_details': details,
            'SA_binary': binary_preds.get('SA', 0),
            'EC_binary': binary_preds.get('EC', 0),
            'CA_binary': binary_preds.get('CA', 0),
            'SA_ensemble': ens_preds.get('SA', 0),
            'EC_ensemble': ens_preds.get('EC', 0),
            'CA_ensemble': ens_preds.get('CA', 0),
            'SA_mic': exp_mic.get(cid, {}).get('SA'),
            'EC_mic': exp_mic.get(cid, {}).get('EC'),
            'CA_mic': exp_mic.get(cid, {}).get('CA'),
            **props
        })

    # Sort by violations, then by average ensemble prediction score
    results.sort(key=lambda x: (x['violations'],
                                -(x['SA_ensemble'] + x['EC_ensemble'] + x['CA_ensemble'])/3))

    # Show results
    print(f"\n6. Results (showing top 10):")
    print("-" * 70)

    zero_violation = [r for r in results if r['violations'] == 0]
    print(f"\n   Compounds with 0 violations: {len(zero_violation)}")

    for i, r in enumerate(results[:10]):
        print(f"\n   {i+1}. {r['compound_id']}")
        print(f"      Violations: {r['violations']}")
        if r['violations'] > 0:
            print(f"      Details: {r['violation_details']}")
        print(f"      Binary predictions: SA={r['SA_binary']}, EC={r['EC_binary']}, CA={r['CA_binary']}")
        print(f"      Ensemble scores: SA={r['SA_ensemble']:.3f}, EC={r['EC_ensemble']:.3f}, CA={r['CA_ensemble']:.3f}")
        print(f"      MIC (ug/mL): SA={r['SA_mic']}, EC={r['EC_mic']}, CA={r['CA_mic']}")
        print(f"      Properties: MW={r['MW']}, LogP={r['LogP']}, TPSA={r['TPSA']}")
        print(f"                  HBD={r['HBD']}, HBA={r['HBA']}, RotBonds={r['RotBonds']}")
        print(f"                  Rings={r['Rings']}, Fsp3={r['Fsp3']}")
        print(f"      SMILES: {r['smiles'][:60]}...")

    # Save full results
    if results:
        df_results = pd.DataFrame(results)
        output_file = OUTPUT_DIR / "triple_active_candidates.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n\nFull results saved to: {output_file}")

    # Highlight best candidate
    if zero_violation:
        best = zero_violation[0]
        print("\n" + "=" * 70)
        print("BEST TRIPLE-ACTIVE EXEMPLAR CANDIDATE:")
        print("=" * 70)
        print(f"   Compound: {best['compound_id']}")
        print(f"   SMILES: {best['smiles']}")
        print(f"   Binary predictions: SA={best['SA_binary']}, EC={best['EC_binary']}, CA={best['CA_binary']}")
        print(f"   Ensemble scores: SA={best['SA_ensemble']:.3f}, EC={best['EC_ensemble']:.3f}, CA={best['CA_ensemble']:.3f}")
        print(f"   MIC (ug/mL): SA={best['SA_mic']}, EC={best['EC_mic']}, CA={best['CA_mic']}")
        print(f"   All binary predictions = 1: {all(best[f'{p}_binary'] == 1 for p in ['SA', 'EC', 'CA'])}")


if __name__ == '__main__':
    main()
