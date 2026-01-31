#!/usr/bin/env python3
"""
Extract representative chemical structures for pathogen-specific design principles
from antimicrobial fragment analysis data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# Define file paths
base_path = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\Pathogen_specific_analysis")

# Key files to analyze
files_to_analyze = {
    'single_pathogen': {
        'xai_fragments': base_path / "POSITIVE" / "xai_fragments_with_properties.csv",
        'properties': base_path / "POSITIVE" / "physicochemical_properties_matrix.csv"
    },
    'dual_active': base_path / "DUAL_ACTIVE_POSITIVE" / "robust_dual_activity_fragments_with_properties.csv",
    'triple_active': base_path / "TRIPLE_ACTIVE_POSITIVE" / "triple_active_fragments_with_properties.csv"
}

def load_data_chunks(filepath, chunksize=1000):
    """Load large CSV files in chunks and apply filtering"""
    chunks = []
    
    try:
        for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False):
            # Apply basic filtering criteria
            if 'avg_attribution' in chunk.columns and 'activity_rate_percent' in chunk.columns:
                filtered_chunk = chunk[
                    (chunk['avg_attribution'] >= 0.6) & 
                    (chunk['activity_rate_percent'] >= 95)
                ]
                if not filtered_chunk.empty:
                    chunks.append(filtered_chunk)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()
    
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    else:
        return pd.DataFrame()

def extract_parent_smiles(text):
    """Extract SMILES string from the highest_attribution_example field"""
    if pd.isna(text) or not isinstance(text, str):
        return None
    
    # Look for SMILES pattern after "SMILES: "
    match = re.search(r'SMILES:\s*([^\s|]+)', text)
    if match:
        return match.group(1).strip()
    return None

def categorize_pathogen_specificity(row):
    """Determine pathogen specificity based on available data"""
    if 'pathogen' in row and pd.notna(row['pathogen']):
        return row['pathogen']
    
    # For multi-pathogen data, check column names or content
    columns = row.index.tolist() if hasattr(row, 'index') else []
    
    # Check for dual activity indicators
    dual_indicators = [col for col in columns if any(pathogen in col.lower() for pathogen in ['sa_', 'ec_', 'ca_'])]
    if len(dual_indicators) >= 2:
        return "dual_active"
    
    # Check for triple activity
    if len(dual_indicators) >= 3:
        return "triple_active"
    
    return "unknown"

def analyze_property_discrimination(df):
    """Find examples showing key property discrimination patterns"""
    discrimination_examples = []
    
    if df.empty:
        return discrimination_examples
    
    # SA vs EC: LogP differences (high LogP for SA, low for EC)
    sa_fragments = df[df.get('pathogen', '') == 'SA'].copy() if 'pathogen' in df.columns else pd.DataFrame()
    ec_fragments = df[df.get('pathogen', '') == 'EC'].copy() if 'pathogen' in df.columns else pd.DataFrame()
    
    if not sa_fragments.empty and not ec_fragments.empty and 'logp' in df.columns:
        # High LogP SA fragments
        high_logp_sa = sa_fragments.nlargest(2, 'logp')
        for _, row in high_logp_sa.iterrows():
            discrimination_examples.append({
                'discrimination_type': 'SA_high_LogP',
                'fragment_data': row,
                'rationale': f"High LogP ({row.get('logp', 'N/A'):.2f}) favors SA selectivity"
            })
        
        # Low LogP EC fragments
        low_logp_ec = ec_fragments.nsmallest(2, 'logp')
        for _, row in low_logp_ec.iterrows():
            discrimination_examples.append({
                'discrimination_type': 'EC_low_LogP',
                'fragment_data': row,
                'rationale': f"Low LogP ({row.get('logp', 'N/A'):.2f}) favors EC selectivity"
            })
    
    # EC vs CA: HBD differences (high HBD for EC, low for CA)
    ca_fragments = df[df.get('pathogen', '') == 'CA'].copy() if 'pathogen' in df.columns else pd.DataFrame()
    
    if not ec_fragments.empty and not ca_fragments.empty and 'num_hbd' in df.columns:
        # High HBD EC fragments
        high_hbd_ec = ec_fragments.nlargest(2, 'num_hbd')
        for _, row in high_hbd_ec.iterrows():
            discrimination_examples.append({
                'discrimination_type': 'EC_high_HBD',
                'fragment_data': row,
                'rationale': f"High HBD ({row.get('num_hbd', 'N/A')}) favors EC selectivity"
            })
        
        # Low HBD CA fragments
        low_hbd_ca = ca_fragments.nsmallest(2, 'num_hbd')
        for _, row in low_hbd_ca.iterrows():
            discrimination_examples.append({
                'discrimination_type': 'CA_low_HBD',
                'fragment_data': row,
                'rationale': f"Low HBD ({row.get('num_hbd', 'N/A')}) favors CA selectivity"
            })
    
    # TPSA differences (high TPSA for EC, low for SA)
    if not ec_fragments.empty and not sa_fragments.empty and 'tpsa' in df.columns:
        # High TPSA EC fragments
        high_tpsa_ec = ec_fragments.nlargest(2, 'tpsa')
        for _, row in high_tpsa_ec.iterrows():
            discrimination_examples.append({
                'discrimination_type': 'EC_high_TPSA',
                'fragment_data': row,
                'rationale': f"High TPSA ({row.get('tpsa', 'N/A'):.1f}) favors EC selectivity"
            })
        
        # Low TPSA SA fragments
        low_tpsa_sa = sa_fragments.nsmallest(2, 'tpsa')
        for _, row in low_tpsa_sa.iterrows():
            discrimination_examples.append({
                'discrimination_type': 'SA_low_TPSA',
                'fragment_data': row,
                'rationale': f"Low TPSA ({row.get('tpsa', 'N/A'):.1f}) favors SA selectivity"
            })
    
    return discrimination_examples

def extract_representative_fragments():
    """Main function to extract representative fragments"""
    
    print("Loading and analyzing antimicrobial fragment data...")
    print("=" * 60)
    
    all_fragments = []
    
    # 1. Load single pathogen specific fragments
    print("1. Analyzing single pathogen-specific fragments...")
    single_pathogen_df = load_data_chunks(files_to_analyze['single_pathogen']['xai_fragments'])
    
    if not single_pathogen_df.empty:
        print(f"   Loaded {len(single_pathogen_df)} high-quality single pathogen fragments")
        
        # Get top fragments for each pathogen
        for pathogen in ['SA', 'EC', 'CA']:
            pathogen_fragments = single_pathogen_df[
                single_pathogen_df.get('pathogen', '') == pathogen
            ].copy()
            
            if not pathogen_fragments.empty:
                # Sort by avg_attribution and activity_rate_percent
                pathogen_fragments = pathogen_fragments.sort_values(
                    ['avg_attribution', 'activity_rate_percent'], 
                    ascending=[False, False]
                )
                
                # Take top 4 fragments per pathogen
                top_fragments = pathogen_fragments.head(4)
                
                for _, row in top_fragments.iterrows():
                    parent_smiles = extract_parent_smiles(row.get('highest_attribution_example', ''))
                    
                    fragment_data = {
                        'category': 'pathogen_specific',
                        'pathogen_specificity': pathogen,
                        'fragment_id': row.get('fragment_id', ''),
                        'fragment_smiles': row.get('fragment_smiles', ''),
                        'molecular_weight': row.get('molecular_weight', ''),
                        'logp': row.get('logp', ''),
                        'tpsa': row.get('tpsa', ''),
                        'num_hbd': row.get('num_hbd', ''),
                        'num_hba': row.get('num_hba', ''),
                        'avg_attribution': row.get('avg_attribution', ''),
                        'activity_rate_percent': row.get('activity_rate_percent', ''),
                        'highest_attribution_example': parent_smiles,
                        'chemical_rationale': f"{pathogen}-specific fragment with high XAI confidence"
                    }
                    all_fragments.append(fragment_data)
        
        # Analyze property discrimination patterns
        print("   Analyzing property discrimination patterns...")
        discrimination_examples = analyze_property_discrimination(single_pathogen_df)
        
        for example in discrimination_examples[:8]:  # Limit to 8 discrimination examples
            fragment_data = example['fragment_data']
            parent_smiles = extract_parent_smiles(fragment_data.get('highest_attribution_example', ''))
            
            discrimination_fragment = {
                'category': 'property_discrimination',
                'pathogen_specificity': example['discrimination_type'],
                'fragment_id': fragment_data.get('fragment_id', ''),
                'fragment_smiles': fragment_data.get('fragment_smiles', ''),
                'molecular_weight': fragment_data.get('molecular_weight', ''),
                'logp': fragment_data.get('logp', ''),
                'tpsa': fragment_data.get('tpsa', ''),
                'num_hbd': fragment_data.get('num_hbd', ''),
                'num_hba': fragment_data.get('num_hba', ''),
                'avg_attribution': fragment_data.get('avg_attribution', ''),
                'activity_rate_percent': fragment_data.get('activity_rate_percent', ''),
                'highest_attribution_example': parent_smiles,
                'chemical_rationale': example['rationale']
            }
            all_fragments.append(discrimination_fragment)
    
    # 2. Load dual active fragments
    print("2. Analyzing dual-active fragments...")
    dual_df = load_data_chunks(files_to_analyze['dual_active'])
    
    if not dual_df.empty:
        print(f"   Loaded {len(dual_df)} high-quality dual-active fragments")
        
        # Sort by avg_attribution
        dual_df = dual_df.sort_values('avg_attribution', ascending=False)
        top_dual = dual_df.head(4)
        
        for _, row in top_dual.iterrows():
            parent_smiles = extract_parent_smiles(row.get('highest_attribution_example', ''))
            
            fragment_data = {
                'category': 'dual_active',
                'pathogen_specificity': 'broad_spectrum_dual',
                'fragment_id': row.get('fragment_id', ''),
                'fragment_smiles': row.get('fragment_smiles', ''),
                'molecular_weight': row.get('molecular_weight', ''),
                'logp': row.get('logp', ''),
                'tpsa': row.get('tpsa', ''),
                'num_hbd': row.get('num_hbd', ''),
                'num_hba': row.get('num_hba', ''),
                'avg_attribution': row.get('avg_attribution', ''),
                'activity_rate_percent': row.get('activity_rate_percent', ''),
                'highest_attribution_example': parent_smiles,
                'chemical_rationale': "Dual-pathogen active fragment with balanced properties"
            }
            all_fragments.append(fragment_data)
    
    # 3. Load triple active fragments
    print("3. Analyzing triple-active fragments...")
    try:
        # Try to load triple active data in smaller chunks due to size
        triple_chunks = []
        chunk_size = 500
        
        for chunk in pd.read_csv(files_to_analyze['triple_active'], chunksize=chunk_size, low_memory=False):
            # Apply stricter filtering for triple active due to large dataset
            if 'avg_attribution' in chunk.columns and 'activity_rate_percent' in chunk.columns:
                filtered = chunk[
                    (chunk['avg_attribution'] >= 0.7) &  # Higher threshold for triple active
                    (chunk['activity_rate_percent'] >= 98)  # Higher threshold for triple active
                ]
                if not filtered.empty:
                    triple_chunks.append(filtered)
                    if len(triple_chunks) >= 10:  # Limit chunks to avoid memory issues
                        break
        
        if triple_chunks:
            triple_df = pd.concat(triple_chunks, ignore_index=True)
            print(f"   Loaded {len(triple_df)} high-quality triple-active fragments")
            
            # Sort by avg_attribution
            triple_df = triple_df.sort_values('avg_attribution', ascending=False)
            top_triple = triple_df.head(4)
            
            for _, row in top_triple.iterrows():
                parent_smiles = extract_parent_smiles(row.get('highest_attribution_example', ''))
                
                fragment_data = {
                    'category': 'triple_active',
                    'pathogen_specificity': 'broad_spectrum_triple',
                    'fragment_id': row.get('fragment_id', ''),
                    'fragment_smiles': row.get('fragment_smiles', ''),
                    'molecular_weight': row.get('molecular_weight', ''),
                    'logp': row.get('logp', ''),
                    'tpsa': row.get('tpsa', ''),
                    'num_hbd': row.get('num_hbd', ''),
                    'num_hba': row.get('num_hba', ''),
                    'avg_attribution': row.get('avg_attribution', ''),
                    'activity_rate_percent': row.get('activity_rate_percent', ''),
                    'highest_attribution_example': parent_smiles,
                    'chemical_rationale': "Triple-pathogen active fragment with optimal broad-spectrum properties"
                }
                all_fragments.append(fragment_data)
        else:
            print("   No high-quality triple-active fragments found with strict criteria")
            
    except Exception as e:
        print(f"   Error loading triple active data: {e}")
    
    # Create results DataFrame
    if all_fragments:
        results_df = pd.DataFrame(all_fragments)
        
        # Save to CSV
        output_file = base_path / "representative_fragments_for_manuscript.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved {len(results_df)} representative fragments to:")
        print(f"  {output_file}")
        
        # Create summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        category_counts = results_df['category'].value_counts()
        print("Fragments by category:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")
        
        pathogen_counts = results_df['pathogen_specificity'].value_counts()
        print("\nFragments by pathogen specificity:")
        for pathogen, count in pathogen_counts.items():
            print(f"  {pathogen}: {count}")
        
        # Property ranges
        numeric_cols = ['molecular_weight', 'logp', 'tpsa', 'num_hbd', 'num_hba', 'avg_attribution', 'activity_rate_percent']
        print("\nProperty ranges:")
        for col in numeric_cols:
            if col in results_df.columns:
                col_data = pd.to_numeric(results_df[col], errors='coerce')
                if not col_data.isna().all():
                    print(f"  {col}: {col_data.min():.2f} - {col_data.max():.2f}")
        
        return results_df
    else:
        print("No fragments found meeting the criteria!")
        return pd.DataFrame()

def create_structure_visualization_guide(results_df):
    """Create a guide for structure visualization"""
    
    if results_df.empty:
        return
    
    guide_content = """
# Representative Antimicrobial Fragments - Structure Visualization Guide

## Fragment Categories and Chemical Rationales

### 1. Pathogen-Specific Fragments
"""
    
    # Pathogen-specific fragments
    pathogen_specific = results_df[results_df['category'] == 'pathogen_specific']
    
    for pathogen in ['SA', 'EC', 'CA']:
        pathogen_frags = pathogen_specific[pathogen_specific['pathogen_specificity'] == pathogen]
        if not pathogen_frags.empty:
            guide_content += f"\n#### {pathogen} (Staphylococcus aureus) Specific:\n"
            for _, row in pathogen_frags.iterrows():
                guide_content += f"- **Fragment {row['fragment_id']}**: `{row['fragment_smiles']}`\n"
                guide_content += f"  - MW: {row['molecular_weight']}, LogP: {row['logp']}, TPSA: {row['tpsa']}\n"
                guide_content += f"  - XAI Attribution: {row['avg_attribution']}, Activity Rate: {row['activity_rate_percent']}%\n"
                guide_content += f"  - Rationale: {row['chemical_rationale']}\n\n"
    
    # Property discrimination examples
    discrimination = results_df[results_df['category'] == 'property_discrimination']
    if not discrimination.empty:
        guide_content += "\n### 2. Property Discrimination Examples\n"
        
        for _, row in discrimination.iterrows():
            guide_content += f"- **{row['pathogen_specificity']}**: `{row['fragment_smiles']}`\n"
            guide_content += f"  - MW: {row['molecular_weight']}, LogP: {row['logp']}, TPSA: {row['tpsa']}\n"
            guide_content += f"  - Rationale: {row['chemical_rationale']}\n\n"
    
    # Multi-pathogen examples
    multi_pathogen = results_df[results_df['category'].isin(['dual_active', 'triple_active'])]
    if not multi_pathogen.empty:
        guide_content += "\n### 3. Broad-Spectrum Fragments\n"
        
        for _, row in multi_pathogen.iterrows():
            guide_content += f"- **{row['category'].title().replace('_', '-')}**: `{row['fragment_smiles']}`\n"
            guide_content += f"  - MW: {row['molecular_weight']}, LogP: {row['logp']}, TPSA: {row['tpsa']}\n"
            guide_content += f"  - XAI Attribution: {row['avg_attribution']}, Activity Rate: {row['activity_rate_percent']}%\n"
            guide_content += f"  - Rationale: {row['chemical_rationale']}\n\n"
    
    guide_content += """
## Key Design Principles

1. **SA Selectivity**: Higher LogP, lower TPSA
2. **EC Selectivity**: Lower LogP, higher TPSA, higher HBD
3. **CA Selectivity**: Lower HBD, moderate properties
4. **Broad-Spectrum**: Balanced properties across all parameters

## Recommended Visualization Tools
- ChemDraw or ChemSketch for 2D structures
- RDKit for programmatic visualization
- PyMOL for 3D representations if needed
"""
    
    # Save guide
    guide_file = base_path / "fragment_visualization_guide.md"
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"Created visualization guide: {guide_file}")

if __name__ == "__main__":
    try:
        results = extract_representative_fragments()
        if not results.empty:
            create_structure_visualization_guide(results)
            print(f"\nAnalysis complete! Found {len(results)} representative fragments.")
        else:
            print("\nNo fragments found meeting the criteria.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()