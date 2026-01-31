#!/usr/bin/env python3
"""
Regenerate individual plots from existing analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from rdkit import Chem
from rdkit.Chem import Descriptors
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create directories
os.makedirs('CLAUDE_ANALYSIS/figures_individual', exist_ok=True)

def load_data():
    """Load existing fragment data"""
    fragment_data = {}
    
    for pathogen in ['SA', 'EC', 'CA']:
        for ftype in ['scaffolds', 'substitutents']:
            filepath = f'POSITIVE/{pathogen}_specific_positive_{ftype}.csv'
            try:
                df = pd.read_csv(filepath)
                key = f'{pathogen}_{ftype}'
                fragment_data[key] = df
                print(f"[OK] Loaded {len(df)} {pathogen} {ftype}")
            except FileNotFoundError:
                print(f"[MISSING] {filepath}")
    
    return fragment_data

def calculate_properties(fragment_data):
    """Calculate molecular properties"""
    molecular_properties = {}
    
    property_calculators = [
        ('MW', Descriptors.MolWt),
        ('LogP', Descriptors.MolLogP),
        ('TPSA', Descriptors.TPSA),
        ('HBA', Descriptors.NOCount),
        ('HBD', Descriptors.NHOHCount),
        ('NumAromaticRings', Descriptors.NumAromaticRings),
        ('BertzCT', Descriptors.BertzCT)
    ]
    
    for key, df in fragment_data.items():
        if 'dual' in key or 'triple' in key:
            continue
            
        print(f"Processing {key}...")
        properties_df = df.copy()
        
        for prop_name, _ in property_calculators:
            properties_df[prop_name] = np.nan
        
        for idx, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['fragment_smiles'])
                if mol is not None:
                    for prop_name, calc_func in property_calculators:
                        properties_df.at[idx, prop_name] = calc_func(mol)
            except:
                continue
                
        molecular_properties[key] = properties_df
    
    return molecular_properties

def plot_single_pathogen_counts(fragment_data):
    """Plot single pathogen fragment counts"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    single_counts = {}
    for pathogen in ['SA', 'EC', 'CA']:
        scaffolds = len(fragment_data.get(f'{pathogen}_scaffolds', []))
        substituents = len(fragment_data.get(f'{pathogen}_substitutents', []))
        single_counts[pathogen] = scaffolds + substituents
    
    bars = ax.bar(single_counts.keys(), single_counts.values(), 
                 color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_title('Single-Pathogen Fragment Counts', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Fragments', fontsize=12)
    ax.set_xlabel('Pathogen', fontsize=12)
    
    # Add count annotations
    for i, (pathogen, count) in enumerate(single_counts.items()):
        ax.text(i, count + max(single_counts.values()) * 0.02, 
               f'{count}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add pathogen names as labels
    pathogen_names = {'SA': 'S. aureus', 'EC': 'E. coli', 'CA': 'C. albicans'}
    ax.set_xticklabels([pathogen_names[p] for p in single_counts.keys()])
    
    plt.tight_layout()
    plt.savefig('CLAUDE_ANALYSIS/figures_individual/single_pathogen_counts.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Single pathogen counts plot saved")

def plot_property_distributions(molecular_properties):
    """Plot individual property distributions"""
    properties = ['MW', 'LogP', 'TPSA', 'HBA', 'HBD', 'NumAromaticRings', 'BertzCT']
    pathogen_colors = {'SA': '#1f77b4', 'EC': '#ff7f0e', 'CA': '#2ca02c'}
    pathogen_names = {'SA': 'S. aureus', 'EC': 'E. coli', 'CA': 'C. albicans'}
    
    for prop in properties:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_for_violin = []
        labels = []
        colors = []
        
        for pathogen in ['SA', 'EC', 'CA']:
            key = f'{pathogen}_scaffolds'
            if key in molecular_properties:
                prop_data = molecular_properties[key][prop].dropna()
                if len(prop_data) > 0:
                    data_for_violin.append(prop_data.values)
                    labels.append(pathogen_names[pathogen])
                    colors.append(pathogen_colors[pathogen])
        
        if data_for_violin:
            parts = ax.violinplot(data_for_violin, positions=range(len(labels)), 
                                showmeans=True, showmedians=True, showextrema=True)
            
            # Color the violin plots
            for j, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[j])
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            # Style the other elements
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                if partname in parts:
                    parts[partname].set_edgecolor('black')
                    parts[partname].set_linewidth(1.5)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=12)
            ax.set_title(f'{prop} Distribution Across Pathogens', fontsize=16, fontweight='bold')
            ax.set_ylabel(f'{prop}', fontsize=12)
            ax.set_xlabel('Pathogen', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'CLAUDE_ANALYSIS/figures_individual/{prop}_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] {prop} distribution plot saved")

def plot_summary_statistics_from_csv():
    """Plot statistical results from existing CSV"""
    try:
        results_df = pd.read_csv('CLAUDE_ANALYSIS/statistical_summary.csv')
        
        # Volcano plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        cohens_d = results_df['cohens_d'].values
        neg_log_p = -np.log10(results_df['p_corrected'].values)
        properties = results_df['property'].values
        comparisons = [f"{row['pathogen1']}-{row['pathogen2']}" for _, row in results_df.iterrows()]
        
        # Color by comparison type
        comparison_colors = {
            'SA-EC': '#1f77b4', 'SA-CA': '#ff7f0e', 'EC-CA': '#2ca02c'
        }
        
        colors = [comparison_colors.get(comp, 'gray') for comp in comparisons]
        
        scatter = ax.scatter(cohens_d, neg_log_p, c=colors, alpha=0.7, s=80)
        
        # Add significance and effect size thresholds
        ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, 
                  linewidth=2, label='p = 0.05')
        ax.axvline(x=0.3, color='blue', linestyle='--', alpha=0.7, 
                  linewidth=2, label="Cohen's d = 0.3")
        ax.axvline(x=-0.3, color='blue', linestyle='--', alpha=0.7, linewidth=2)
        
        # Annotate significant points
        for i, (d, p, prop, comp) in enumerate(zip(cohens_d, neg_log_p, properties, comparisons)):
            if p > -np.log10(0.05) and abs(d) > 0.3:
                ax.annotate(f'{prop}\n({comp})', (d, p), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.9, fontweight='bold')
        
        ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
        ax.set_ylabel('-log10(FDR-corrected p-value)', fontsize=12)
        ax.set_title('Statistical Significance vs Effect Size\n(Volcano Plot)', 
                    fontweight='bold', fontsize=16)
        
        # Create legend
        legend_elements = [plt.scatter([], [], c=color, label=comp, s=80) 
                          for comp, color in comparison_colors.items()]
        ax.legend(handles=legend_elements + [
            plt.Line2D([0], [0], color='red', linestyle='--', label='p = 0.05'),
            plt.Line2D([0], [0], color='blue', linestyle='--', label="Cohen's d = ±0.3")
        ])
        
        plt.tight_layout()
        plt.savefig('CLAUDE_ANALYSIS/figures_individual/volcano_plot.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] Volcano plot saved")
        
    except FileNotFoundError:
        print("[SKIP] Statistical summary CSV not available")

def main():
    print("Regenerating individual plots...")
    
    # Load data and calculate properties
    fragment_data = load_data()
    molecular_properties = calculate_properties(fragment_data)
    
    # Generate individual plots
    plot_single_pathogen_counts(fragment_data)
    plot_property_distributions(molecular_properties)
    plot_summary_statistics_from_csv()
    
    print("\n[SUCCESS] Individual plots regenerated in CLAUDE_ANALYSIS/figures_individual/")

if __name__ == "__main__":
    main()