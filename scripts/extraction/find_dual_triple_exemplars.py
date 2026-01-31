"""
Find Dual and Triple Active Exemplar Compounds
===============================================
Finds actual ChEMBL compounds with dual/triple activity that have
0 violations across all 8 physicochemical properties.

For dual-active: compound must be active in BOTH pathogens
For triple-active: compound must be active in ALL THREE pathogens

Property ranges are based on the intersection of relevant SELECT zones.
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
RAW_DATA_DIR = BASE_DIR / "raw_data" / "source_data"
OUTPUT_DIR = BASE_DIR / "figures" / "radar_plots"

# Source files
FILES = {
    'SA': RAW_DATA_DIR / "S_aureus_raw_data.csv",
    'EC': RAW_DATA_DIR / "E_coli_raw_data.csv",
    'CA': RAW_DATA_DIR / "C_albicans_raw_data.csv",
}

# SELECT ranges for single pathogens (mean +/- SD from single-pathogen analysis)
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


def get_combined_range(pathogens):
    """
    Get combined SELECT range for multiple pathogens.
    Uses the INTERSECTION of ranges (overlap region).
    For each property: max of lower bounds, min of upper bounds.
    """
    combined = {}
    for prop in PROPERTIES:
        lower_bounds = []
        upper_bounds = []
        for p in pathogens:
            mean = SELECT_RANGES[p][prop]['mean']
            sd = SELECT_RANGES[p][prop]['sd']
            lower_bounds.append(mean - sd)
            upper_bounds.append(mean + sd)

        # Intersection: take max of lower bounds, min of upper bounds
        combined[prop] = {
            'lower': max(lower_bounds),
            'upper': min(upper_bounds),
            'mean': np.mean([SELECT_RANGES[p][prop]['mean'] for p in pathogens]),
            'sd': np.mean([SELECT_RANGES[p][prop]['sd'] for p in pathogens])
        }
    return combined


def count_violations(props, combined_range):
    """Count how many properties are outside the combined range."""
    violations = 0
    violation_details = []
    for prop in PROPERTIES:
        value = props[prop]
        lower = combined_range[prop]['lower']
        upper = combined_range[prop]['upper']
        if value < lower or value > upper:
            violations += 1
            violation_details.append(f"{prop}: {value} (range: {lower:.1f}-{upper:.1f})")
    return violations, violation_details


def load_activity_data():
    """Load activity data from all three pathogen files."""
    data = {}

    for pathogen, filepath in FILES.items():
        print(f"Loading {pathogen} data from {filepath.name}...")
        df = pd.read_csv(filepath)

        # Standardize column names
        df.columns = df.columns.str.lower()

        # Get compound ID and activity columns
        if 'compound_id' in df.columns:
            id_col = 'compound_id'
        else:
            print(f"  Warning: No compound_id column found in {filepath.name}")
            continue

        # Get activity column
        if 'activity' in df.columns:
            activity_col = 'activity'
        elif 'target' in df.columns:
            activity_col = 'target'
        else:
            print(f"  Warning: No activity column found in {filepath.name}")
            continue

        # Get SMILES column
        smiles_col = None
        for col in ['standardized_smiles', 'processed_smiles', 'canonical_smiles', 'smiles']:
            if col in df.columns:
                smiles_col = col
                break

        if smiles_col is None:
            print(f"  Warning: No SMILES column found in {filepath.name}")
            continue

        # Get MIC columns
        mic_col = None
        mic_um_col = None
        for col in ['mic_ug_ml']:
            if col in df.columns:
                mic_col = col
                break
        for col in ['activity_um', 'activity_μm']:
            if col in df.columns:
                mic_um_col = col
                break

        # Filter for active compounds
        df_active = df[df[activity_col].isin(['active', 1, '1'])].copy()

        print(f"  Total compounds: {len(df)}, Active: {len(df_active)}")

        # Store data
        for _, row in df_active.iterrows():
            compound_id = row[id_col]
            if compound_id not in data:
                data[compound_id] = {
                    'smiles': row[smiles_col],
                    'pathogens': {},
                }

            # Store activity info for this pathogen
            data[compound_id]['pathogens'][pathogen] = {
                'active': True,
                'mic_ug_ml': row[mic_col] if mic_col and pd.notna(row.get(mic_col)) else None,
                'activity_um': row[mic_um_col] if mic_um_col and pd.notna(row.get(mic_um_col)) else None,
            }

    return data


def find_multi_active_compounds(data, target_pathogens, exclude_pathogens=None):
    """
    Find compounds active against all target_pathogens but NOT against exclude_pathogens.
    """
    if exclude_pathogens is None:
        exclude_pathogens = []

    results = []
    for compound_id, info in data.items():
        active_pathogens = set(info['pathogens'].keys())

        # Check if active against all target pathogens
        if not all(p in active_pathogens for p in target_pathogens):
            continue

        # Check if NOT active against excluded pathogens
        if any(p in active_pathogens for p in exclude_pathogens):
            continue

        results.append({
            'compound_id': compound_id,
            'smiles': info['smiles'],
            'pathogens': info['pathogens']
        })

    return results


def find_best_exemplar(compounds, target_pathogens, max_violations=0):
    """
    Find the best exemplar compound with minimum violations.
    """
    combined_range = get_combined_range(target_pathogens)

    candidates = []
    for compound in compounds:
        props = calculate_properties(compound['smiles'])
        if props is None:
            continue

        violations, details = count_violations(props, combined_range)

        # Collect MIC values
        mic_values = {}
        for p in target_pathogens:
            if p in compound['pathogens']:
                mic_values[f'{p}_mic_ug_ml'] = compound['pathogens'][p].get('mic_ug_ml')
                mic_values[f'{p}_activity_um'] = compound['pathogens'][p].get('activity_um')

        candidates.append({
            'compound_id': compound['compound_id'],
            'smiles': compound['smiles'],
            'violations': violations,
            'violation_details': details,
            **props,
            **mic_values
        })

    # Sort by violations (ascending)
    candidates.sort(key=lambda x: x['violations'])

    # Filter to those with acceptable violations
    best = [c for c in candidates if c['violations'] <= max_violations]

    return best, combined_range, candidates


def main():
    print("=" * 70)
    print("FINDING DUAL AND TRIPLE ACTIVE EXEMPLAR COMPOUNDS")
    print("Criteria: 0 violations across all 8 physicochemical properties")
    print("=" * 70)

    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit is required")
        return

    # Load all activity data
    data = load_activity_data()
    print(f"\nTotal unique compounds loaded: {len(data)}")

    # Define combinations to search
    combinations = {
        'SA_EC': {'target': ['SA', 'EC'], 'exclude': ['CA'], 'label': 'S. aureus + E. coli'},
        'SA_CA': {'target': ['SA', 'CA'], 'exclude': ['EC'], 'label': 'S. aureus + C. albicans'},
        'EC_CA': {'target': ['EC', 'CA'], 'exclude': ['SA'], 'label': 'E. coli + C. albicans'},
        'TRIPLE': {'target': ['SA', 'EC', 'CA'], 'exclude': [], 'label': 'Triple Active (All Three)'},
    }

    all_results = []

    for combo_key, combo_info in combinations.items():
        print(f"\n{'=' * 60}")
        print(f"Searching for: {combo_info['label']}")
        print(f"{'=' * 60}")

        # Find compounds
        compounds = find_multi_active_compounds(
            data,
            combo_info['target'],
            combo_info['exclude']
        )
        print(f"Found {len(compounds)} compounds active against {combo_info['target']}")

        if not compounds:
            print("  No compounds found for this combination")
            continue

        # Find best exemplars
        best, combined_range, all_candidates = find_best_exemplar(
            compounds,
            combo_info['target'],
            max_violations=0
        )

        print(f"\nCombined SELECT range for {combo_info['label']}:")
        for prop in PROPERTIES:
            r = combined_range[prop]
            print(f"  {prop}: {r['lower']:.2f} - {r['upper']:.2f}")

        print(f"\nCandidates with 0 violations: {len(best)}")

        if best:
            # Show top 5
            print("\nTop 5 exemplars:")
            for i, exemplar in enumerate(best[:5]):
                print(f"\n  {i+1}. {exemplar['compound_id']}")
                print(f"     SMILES: {exemplar['smiles'][:60]}...")
                print(f"     MW={exemplar['MW']}, LogP={exemplar['LogP']}, TPSA={exemplar['TPSA']}")
                print(f"     HBD={exemplar['HBD']}, HBA={exemplar['HBA']}, RotBonds={exemplar['RotBonds']}")
                print(f"     Rings={exemplar['Rings']}, Fsp3={exemplar['Fsp3']}")

                # Show MIC values
                for p in combo_info['target']:
                    mic = exemplar.get(f'{p}_mic_ug_ml')
                    um = exemplar.get(f'{p}_activity_um')
                    if mic:
                        print(f"     {p} MIC: {mic} ug/mL ({um:.1f} uM)" if um else f"     {p} MIC: {mic} ug/mL")

            # Store best one for output
            best_one = best[0]
            best_one['combination'] = combo_info['label']
            best_one['combination_key'] = combo_key
            all_results.append(best_one)
        else:
            # Find compound with minimum violations
            if all_candidates:
                min_violations = min(c['violations'] for c in all_candidates)
                best_available = [c for c in all_candidates if c['violations'] == min_violations]
                print(f"\nNo 0-violation compounds found. Best available has {min_violations} violations:")

                for i, exemplar in enumerate(best_available[:3]):
                    print(f"\n  {i+1}. {exemplar['compound_id']}")
                    print(f"     Violations ({exemplar['violations']}): {exemplar['violation_details']}")

                best_one = best_available[0]
                best_one['combination'] = combo_info['label']
                best_one['combination_key'] = combo_key
                all_results.append(best_one)

    # Save results
    if all_results:
        df_results = pd.DataFrame(all_results)
        output_file = OUTPUT_DIR / "dual_triple_exemplars.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n{'=' * 70}")
        print(f"Results saved to: {output_file}")
        print(f"{'=' * 70}")

        # Print summary
        print("\nSUMMARY OF SELECTED EXEMPLARS:")
        print("-" * 70)
        for _, row in df_results.iterrows():
            print(f"\n{row['combination']}:")
            print(f"  Compound: {row['compound_id']}")
            print(f"  Violations: {row['violations']}")
            print(f"  MW={row['MW']}, LogP={row['LogP']}, TPSA={row['TPSA']}")


if __name__ == '__main__':
    main()
