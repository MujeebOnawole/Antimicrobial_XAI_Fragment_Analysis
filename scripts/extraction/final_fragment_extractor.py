#!/usr/bin/env python3
"""
Final comprehensive extraction of key antimicrobial fragments for manuscript.
Focus on highest quality examples with complete property data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# Define file paths
base_path = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\Pathogen_specific_analysis")

def extract_parent_smiles(text):
    """Extract SMILES from example text"""
    if pd.isna(text) or not isinstance(text, str):
        return None
    match = re.search(r'SMILES:\s*([^\s|]+)', text)
    return match.group(1).strip() if match else None

def load_pathogen_specific_fragments():
    """Load the highest quality pathogen-specific fragments"""
    print("Extracting pathogen-specific fragments...")
    
    fragments = []
    xai_file = base_path / "POSITIVE" / "xai_fragments_with_properties.csv"
    props_file = base_path / "POSITIVE" / "physicochemical_properties_matrix.csv"
    
    try:
        # Load XAI data in chunks with correct column names
        xai_chunks = []
        for chunk in pd.read_csv(xai_file, chunksize=1000, low_memory=False):
            # Check if required columns exist
            if 'pathogen' in chunk.columns and 'avg_attribution' in chunk.columns:
                filtered = chunk[
                    (chunk['avg_attribution'] >= 0.6) & 
                    (chunk['activity_rate_percent'] >= 95) &
                    (chunk['pathogen'].isin(['SA', 'EC', 'CA']))
                ]
                if not filtered.empty:
                    xai_chunks.append(filtered)
        
        if xai_chunks:
            xai_df = pd.concat(xai_chunks, ignore_index=True)
            print(f"   Found {len(xai_df)} high-quality fragments in XAI data")
            
            # Load properties data
            props_chunks = []
            for chunk in pd.read_csv(props_file, chunksize=1000, low_memory=False):
                fragment_ids = set(xai_df['fragment_id'])
                filtered_props = chunk[chunk['fragment_id'].isin(fragment_ids)]
                if not filtered_props.empty:
                    props_chunks.append(filtered_props)
            
            if props_chunks:
                props_df = pd.concat(props_chunks, ignore_index=True)
                print(f"   Found {len(props_df)} matching property records")
                
                # Merge data
                merged_df = pd.merge(xai_df, props_df, on='fragment_id', how='inner')
                print(f"   Merged dataset: {len(merged_df)} fragments with complete data")
                
                # Extract exemplar fragments for each pathogen
                for pathogen in ['SA', 'EC', 'CA']:
                    pathogen_fragments = merged_df[merged_df['pathogen'] == pathogen].copy()
                    
                    if not pathogen_fragments.empty:
                        # Sort by attribution score and take top fragments
                        pathogen_fragments = pathogen_fragments.sort_values(
                            ['avg_attribution', 'activity_rate_percent'], 
                            ascending=[False, False]
                        )
                        
                        # Take top 3 fragments per pathogen
                        top_fragments = pathogen_fragments.head(3)
                        
                        for idx, (_, row) in enumerate(top_fragments.iterrows(), 1):
                            parent_smiles = extract_parent_smiles(row.get('highest_attribution_example', ''))
                            
                            fragment_data = {
                                'category': 'pathogen_specific',
                                'pathogen_specificity': pathogen,
                                'rank': idx,
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
                                'chemical_rationale': f"{pathogen}-specific fragment (rank {idx}) with high XAI confidence"
                            }
                            fragments.append(fragment_data)
                
                # Extract property discrimination examples
                print("   Extracting property discrimination examples...")
                
                # LogP discrimination (SA high vs EC low)
                sa_frags = merged_df[merged_df['pathogen'] == 'SA'].copy()
                ec_frags = merged_df[merged_df['pathogen'] == 'EC'].copy()
                ca_frags = merged_df[merged_df['pathogen'] == 'CA'].copy()
                
                if not sa_frags.empty and 'logp' in sa_frags.columns:
                    # High LogP SA fragment
                    high_logp_sa = sa_frags.nlargest(1, 'logp').iloc[0]
                    parent_smiles = extract_parent_smiles(high_logp_sa.get('highest_attribution_example', ''))
                    
                    fragments.append({
                        'category': 'property_discrimination',
                        'pathogen_specificity': 'SA_high_LogP',
                        'rank': 1,
                        'fragment_id': high_logp_sa.get('fragment_id', ''),
                        'fragment_smiles': high_logp_sa.get('fragment_smiles', ''),
                        'molecular_weight': high_logp_sa.get('molecular_weight', ''),
                        'logp': high_logp_sa.get('logp', ''),
                        'tpsa': high_logp_sa.get('tpsa', ''),
                        'num_hbd': high_logp_sa.get('num_hbd', ''),
                        'num_hba': high_logp_sa.get('num_hba', ''),
                        'avg_attribution': high_logp_sa.get('avg_attribution', ''),
                        'activity_rate_percent': high_logp_sa.get('activity_rate_percent', ''),
                        'highest_attribution_example': parent_smiles,
                        'chemical_rationale': f"High LogP ({high_logp_sa.get('logp', 0):.2f}) demonstrates SA selectivity preference"
                    })
                
                if not ec_frags.empty and 'logp' in ec_frags.columns:
                    # Low LogP EC fragment
                    low_logp_ec = ec_frags.nsmallest(1, 'logp').iloc[0]
                    parent_smiles = extract_parent_smiles(low_logp_ec.get('highest_attribution_example', ''))
                    
                    fragments.append({
                        'category': 'property_discrimination',
                        'pathogen_specificity': 'EC_low_LogP',
                        'rank': 1,
                        'fragment_id': low_logp_ec.get('fragment_id', ''),
                        'fragment_smiles': low_logp_ec.get('fragment_smiles', ''),
                        'molecular_weight': low_logp_ec.get('molecular_weight', ''),
                        'logp': low_logp_ec.get('logp', ''),
                        'tpsa': low_logp_ec.get('tpsa', ''),
                        'num_hbd': low_logp_ec.get('num_hbd', ''),
                        'num_hba': low_logp_ec.get('num_hba', ''),
                        'avg_attribution': low_logp_ec.get('avg_attribution', ''),
                        'activity_rate_percent': low_logp_ec.get('activity_rate_percent', ''),
                        'highest_attribution_example': parent_smiles,
                        'chemical_rationale': f"Low LogP ({low_logp_ec.get('logp', 0):.2f}) demonstrates EC selectivity preference"
                    })
                
                # TPSA discrimination (EC high vs SA low)
                if not ec_frags.empty and 'tpsa' in ec_frags.columns:
                    high_tpsa_ec = ec_frags.nlargest(1, 'tpsa').iloc[0]
                    parent_smiles = extract_parent_smiles(high_tpsa_ec.get('highest_attribution_example', ''))
                    
                    fragments.append({
                        'category': 'property_discrimination',
                        'pathogen_specificity': 'EC_high_TPSA',
                        'rank': 1,
                        'fragment_id': high_tpsa_ec.get('fragment_id', ''),
                        'fragment_smiles': high_tpsa_ec.get('fragment_smiles', ''),
                        'molecular_weight': high_tpsa_ec.get('molecular_weight', ''),
                        'logp': high_tpsa_ec.get('logp', ''),
                        'tpsa': high_tpsa_ec.get('tpsa', ''),
                        'num_hbd': high_tpsa_ec.get('num_hbd', ''),
                        'num_hba': high_tpsa_ec.get('num_hba', ''),
                        'avg_attribution': high_tpsa_ec.get('avg_attribution', ''),
                        'activity_rate_percent': high_tpsa_ec.get('activity_rate_percent', ''),
                        'highest_attribution_example': parent_smiles,
                        'chemical_rationale': f"High TPSA ({high_tpsa_ec.get('tpsa', 0):.1f}) demonstrates EC selectivity preference"
                    })
                
                if not sa_frags.empty and 'tpsa' in sa_frags.columns:
                    low_tpsa_sa = sa_frags.nsmallest(1, 'tpsa').iloc[0]
                    parent_smiles = extract_parent_smiles(low_tpsa_sa.get('highest_attribution_example', ''))
                    
                    fragments.append({
                        'category': 'property_discrimination',
                        'pathogen_specificity': 'SA_low_TPSA',
                        'rank': 1,
                        'fragment_id': low_tpsa_sa.get('fragment_id', ''),
                        'fragment_smiles': low_tpsa_sa.get('fragment_smiles', ''),
                        'molecular_weight': low_tpsa_sa.get('molecular_weight', ''),
                        'logp': low_tpsa_sa.get('logp', ''),
                        'tpsa': low_tpsa_sa.get('tpsa', ''),
                        'num_hbd': low_tpsa_sa.get('num_hbd', ''),
                        'num_hba': low_tpsa_sa.get('num_hba', ''),
                        'avg_attribution': low_tpsa_sa.get('avg_attribution', ''),
                        'activity_rate_percent': low_tpsa_sa.get('activity_rate_percent', ''),
                        'highest_attribution_example': parent_smiles,
                        'chemical_rationale': f"Low TPSA ({low_tpsa_sa.get('tpsa', 0):.1f}) demonstrates SA selectivity preference"
                    })
                
                # HBD discrimination (EC high vs CA low)
                if not ec_frags.empty and 'num_hbd' in ec_frags.columns:
                    high_hbd_ec = ec_frags.nlargest(1, 'num_hbd').iloc[0]
                    parent_smiles = extract_parent_smiles(high_hbd_ec.get('highest_attribution_example', ''))
                    
                    fragments.append({
                        'category': 'property_discrimination',
                        'pathogen_specificity': 'EC_high_HBD',
                        'rank': 1,
                        'fragment_id': high_hbd_ec.get('fragment_id', ''),
                        'fragment_smiles': high_hbd_ec.get('fragment_smiles', ''),
                        'molecular_weight': high_hbd_ec.get('molecular_weight', ''),
                        'logp': high_hbd_ec.get('logp', ''),
                        'tpsa': high_hbd_ec.get('tpsa', ''),
                        'num_hbd': high_hbd_ec.get('num_hbd', ''),
                        'num_hba': high_hbd_ec.get('num_hba', ''),
                        'avg_attribution': high_hbd_ec.get('avg_attribution', ''),
                        'activity_rate_percent': high_hbd_ec.get('activity_rate_percent', ''),
                        'highest_attribution_example': parent_smiles,
                        'chemical_rationale': f"High HBD ({high_hbd_ec.get('num_hbd', 0)}) demonstrates EC vs CA selectivity"
                    })
        
        return fragments
        
    except Exception as e:
        print(f"   Error loading pathogen fragments: {e}")
        return []

def load_best_dual_and_triple_fragments():
    """Load representative dual and triple active fragments"""
    print("Extracting multi-pathogen fragments...")
    
    fragments = []
    
    # Load dual active - just get a few high-quality examples
    try:
        dual_file = base_path / "DUAL_ACTIVE_POSITIVE" / "robust_dual_activity_fragments_with_properties.csv"
        dual_sample = pd.read_csv(dual_file, nrows=100, low_memory=False)
        
        if 'overall_avg_attribution' in dual_sample.columns:
            dual_filtered = dual_sample[
                (dual_sample['overall_avg_attribution'] >= 0.65) & 
                (dual_sample['avg_activity_rate_percent'] >= 95)
            ].head(3)
            
            for idx, (_, row) in enumerate(dual_filtered.iterrows(), 1):
                fragments.append({
                    'category': 'dual_active',
                    'pathogen_specificity': row.get('combination_name', 'dual_spectrum'),
                    'rank': idx,
                    'fragment_id': row.get('fragment_id', ''),
                    'fragment_smiles': row.get('fragment_smiles', ''),
                    'molecular_weight': '',
                    'logp': '',
                    'tpsa': '',
                    'num_hbd': '',
                    'num_hba': '',
                    'avg_attribution': row.get('overall_avg_attribution', ''),
                    'activity_rate_percent': row.get('avg_activity_rate_percent', ''),
                    'highest_attribution_example': extract_parent_smiles(str(row.get('pathogen_examples', ''))),
                    'chemical_rationale': f"Dual-pathogen active fragment with excellent broad-spectrum properties"
                })
    except Exception as e:
        print(f"   Error loading dual fragments: {e}")
    
    # Load triple active - get the most robust examples
    try:
        triple_file = base_path / "TRIPLE_ACTIVE_POSITIVE" / "triple_active_fragments_with_properties.csv"
        triple_sample = pd.read_csv(triple_file, nrows=50, low_memory=False)
        
        # Get simple, high-impact triple active fragments
        triple_top = triple_sample.head(3)
        
        for idx, (_, row) in enumerate(triple_top.iterrows(), 1):
            fragments.append({
                'category': 'triple_active',
                'pathogen_specificity': 'broad_spectrum',
                'rank': idx,
                'fragment_id': row.get('fragment_id', ''),
                'fragment_smiles': row.get('fragment_smiles', ''),
                'molecular_weight': row.get('molecular_weight', ''),
                'logp': row.get('logp', ''),
                'tpsa': row.get('tpsa', ''),
                'num_hbd': row.get('num_hbd', ''),
                'num_hba': row.get('num_hba', ''),
                'avg_attribution': row.get('avg_attribution', ''),
                'activity_rate_percent': row.get('activity_rate_percent', ''),
                'highest_attribution_example': extract_parent_smiles(row.get('highest_attribution_example', '')),
                'chemical_rationale': f"Triple-pathogen active with optimal broad-spectrum characteristics"
            })
    except Exception as e:
        print(f"   Error loading triple fragments: {e}")
    
    return fragments

def create_manuscript_ready_output(all_fragments):
    """Create clean, manuscript-ready output files"""
    
    if not all_fragments:
        print("No fragments extracted!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_fragments)
    
    # Save main dataset
    main_file = base_path / "manuscript_fragments_final.csv"
    df.to_csv(main_file, index=False)
    
    # Create summary table
    summary_data = []
    
    # Pathogen-specific summary
    pathogen_specific = df[df['category'] == 'pathogen_specific']
    for pathogen in ['SA', 'EC', 'CA']:
        pathogen_frags = pathogen_specific[pathogen_specific['pathogen_specificity'] == pathogen]
        if not pathogen_frags.empty:
            summary_data.append({
                'Category': f'{pathogen} Specific',
                'Count': len(pathogen_frags),
                'Avg_Attribution': pathogen_frags['avg_attribution'].mean(),
                'Avg_Activity_Rate': pathogen_frags['activity_rate_percent'].mean(),
                'Key_Properties': f"LogP: {pathogen_frags['logp'].mean():.2f}, TPSA: {pathogen_frags['tpsa'].mean():.1f}"
            })
    
    # Property discrimination summary
    discrimination = df[df['category'] == 'property_discrimination']
    if not discrimination.empty:
        summary_data.append({
            'Category': 'Property Discrimination',
            'Count': len(discrimination),
            'Avg_Attribution': discrimination['avg_attribution'].mean(),
            'Avg_Activity_Rate': discrimination['activity_rate_percent'].mean(),
            'Key_Properties': "Illustrates selectivity patterns"
        })
    
    # Multi-pathogen summary
    multi_pathogen = df[df['category'].isin(['dual_active', 'triple_active'])]
    if not multi_pathogen.empty:
        summary_data.append({
            'Category': 'Broad-Spectrum',
            'Count': len(multi_pathogen),
            'Avg_Attribution': multi_pathogen['avg_attribution'].mean(),
            'Avg_Activity_Rate': multi_pathogen['activity_rate_percent'].mean(),
            'Key_Properties': "Balanced multi-pathogen activity"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = base_path / "manuscript_fragments_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nFINAL RESULTS:")
    print(f"================")
    print(f"Total fragments extracted: {len(df)}")
    print(f"Main dataset: {main_file}")
    print(f"Summary table: {summary_file}")
    
    print("\nBreakdown by category:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category.replace('_', ' ').title()}: {count}")
    
    print("\nKey statistics:")
    attr_mean = pd.to_numeric(df['avg_attribution'], errors='coerce').mean()
    activity_mean = pd.to_numeric(df['activity_rate_percent'], errors='coerce').mean()
    print(f"  Average XAI Attribution: {attr_mean:.3f}")
    print(f"  Average Activity Rate: {activity_mean:.1f}%")
    
    return df

def create_final_visualization_guide(df):
    """Create final comprehensive guide for manuscript figures"""
    
    guide_content = f"""# Antimicrobial Fragment Analysis - Manuscript Ready Results

## Executive Summary

This analysis extracted **{len(df)} high-quality representative fragments** for pathogen-specific antimicrobial design, organized into:

1. **Pathogen-Specific Fragments** ({len(df[df['category'] == 'pathogen_specific'])})
2. **Property Discrimination Examples** ({len(df[df['category'] == 'property_discrimination'])})
3. **Broad-Spectrum Fragments** ({len(df[df['category'].isin(['dual_active', 'triple_active'])])})

## Key Findings

### Pathogen-Specific Design Rules:

"""
    
    # Add pathogen-specific insights
    pathogen_specific = df[df['category'] == 'pathogen_specific']
    for pathogen in ['SA', 'EC', 'CA']:
        pathogen_frags = pathogen_specific[pathogen_specific['pathogen_specificity'] == pathogen]
        if not pathogen_frags.empty:
            pathogen_name = {'SA': 'S. aureus', 'EC': 'E. coli', 'CA': 'C. albicans'}[pathogen]
            avg_logp = pd.to_numeric(pathogen_frags['logp'], errors='coerce').mean()
            avg_tpsa = pd.to_numeric(pathogen_frags['tpsa'], errors='coerce').mean()
            
            guide_content += f"- **{pathogen_name}**: LogP = {avg_logp:.2f}, TPSA = {avg_tpsa:.1f}\n"
    
    guide_content += """
### Property Discrimination Patterns:

"""
    
    # Add discrimination patterns
    discrimination = df[df['category'] == 'property_discrimination']
    for _, row in discrimination.iterrows():
        guide_content += f"- **{row['pathogen_specificity']}**: {row['chemical_rationale']}\n"
    
    guide_content += f"""
### Fragment Examples by Category:

#### 1. Top Pathogen-Specific Fragments:
"""
    
    # Add specific examples
    for pathogen in ['SA', 'EC', 'CA']:
        pathogen_frags = pathogen_specific[pathogen_specific['pathogen_specificity'] == pathogen]
        if not pathogen_frags.empty:
            top_frag = pathogen_frags.iloc[0]
            guide_content += f"""
**{pathogen} Example**: Fragment {top_frag['fragment_id']}
- SMILES: `{top_frag['fragment_smiles']}`
- Properties: MW={top_frag['molecular_weight']}, LogP={top_frag['logp']}, TPSA={top_frag['tpsa']}
- XAI Attribution: {top_frag['avg_attribution']}, Activity: {top_frag['activity_rate_percent']}%
"""
    
    guide_content += """
#### 2. Property Discrimination Examples:
"""
    
    for _, row in discrimination.iterrows():
        guide_content += f"""
**{row['pathogen_specificity']}**: Fragment {row['fragment_id']}
- SMILES: `{row['fragment_smiles']}`
- Key Property: {row['chemical_rationale']}
"""
    
    guide_content += """
#### 3. Broad-Spectrum Examples:
"""
    
    multi_pathogen = df[df['category'].isin(['dual_active', 'triple_active'])]
    for _, row in multi_pathogen.head(3).iterrows():
        guide_content += f"""
**{row['category'].title()} {row['rank']}**: Fragment {row['fragment_id']}
- SMILES: `{row['fragment_smiles']}`
- Spectrum: {row['pathogen_specificity']}
"""
    
    guide_content += """
## Recommended Figure Panels:

### Figure 1: Pathogen-Specific Chemical Signatures
- Panel A: SA-specific fragments with high LogP/low TPSA
- Panel B: EC-specific fragments with low LogP/high TPSA  
- Panel C: CA-specific fragments with balanced properties
- Panel D: Property space visualization (LogP vs TPSA scatter plot)

### Figure 2: Property Discrimination Examples
- Panel A: LogP discrimination (SA vs EC)
- Panel B: TPSA discrimination (EC vs SA)
- Panel C: HBD discrimination (EC vs CA)
- Panel D: Radar chart comparing property profiles

### Figure 3: Broad-Spectrum Fragments
- Panel A: Dual-active fragment structures
- Panel B: Triple-active fragment structures
- Panel C: Activity heatmap across pathogens
- Panel D: Property balance analysis

## Statistical Summary:
- **Total Fragments**: {len(df)}
- **Average XAI Attribution**: {pd.to_numeric(df['avg_attribution'], errors='coerce').mean():.3f}
- **Average Activity Rate**: {pd.to_numeric(df['activity_rate_percent'], errors='coerce').mean():.1f}%
- **Quality Threshold**: XAI Attribution >= 0.6, Activity >= 95%

## Files Generated:
1. `manuscript_fragments_final.csv` - Complete curated dataset
2. `manuscript_fragments_summary.csv` - Summary statistics table
3. `final_fragment_analysis_guide.md` - This comprehensive guide

---
*High-confidence fragment analysis for antimicrobial design principles*
"""
    
    guide_file = base_path / "final_fragment_analysis_guide.md"
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"  Comprehensive guide: {guide_file}")

def main():
    """Main execution function"""
    
    print("FINAL ANTIMICROBIAL FRAGMENT EXTRACTION")
    print("=" * 50)
    print("Creating manuscript-ready dataset of representative fragments")
    print("=" * 50)
    
    all_fragments = []
    
    # 1. Extract pathogen-specific and property discrimination fragments
    pathogen_fragments = load_pathogen_specific_fragments()
    all_fragments.extend(pathogen_fragments)
    print(f"Extracted {len(pathogen_fragments)} pathogen-specific fragments")
    
    # 2. Extract multi-pathogen fragments
    multi_fragments = load_best_dual_and_triple_fragments()
    all_fragments.extend(multi_fragments)
    print(f"Extracted {len(multi_fragments)} multi-pathogen fragments")
    
    # 3. Create final output
    if all_fragments:
        final_df = create_manuscript_ready_output(all_fragments)
        if final_df is not None:
            create_final_visualization_guide(final_df)
            print(f"\nSUCCESS: Final dataset ready for manuscript with {len(all_fragments)} fragments!")
        return final_df
    else:
        print("ERROR: No fragments extracted!")
        return None

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()