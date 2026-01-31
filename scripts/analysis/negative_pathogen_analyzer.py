#!/usr/bin/env python3
"""
PERMUTATION-BASED NEGATIVE PATHOGEN ANALYZER
Uses ALL negative fragments with permutation tests to handle sample size imbalances
Designed for XAI-derived negative fragments - features to AVOID for each pathogen
Complements positive fragment analysis by identifying activity-killing features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, RDConfig
from rdkit.Chem import rdmolops
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments, Crippen, ChemicalFeatures
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
from sklearn.utils import resample
from collections import defaultdict, Counter
import itertools
import re
import os
import warnings
warnings.filterwarnings('ignore')

class PermutationNegativePathogenAnalyzer:
    def __init__(self):
        self.pathogen_map = {'SA': 'Gram+', 'EC': 'Gram-', 'CA': 'Fungi'}
        self.pathogen_colors = {'SA': 'lightcoral', 'EC': 'lightsalmon', 'CA': 'lightpink'}  # Red-ish colors for negative
        
        # Original testing dataset sizes (for context)
        self.original_testing_sizes = {
            'SA': 54277,  # S. aureus
            'EC': 44920,  # E. coli  
            'CA': 28476   # C. albicans
        }
        
        self.all_fragments = None
        self.chemical_features = None
        self.permutation_results = {}
        self.effect_sizes = {}
        
    def load_and_prepare_data(self, file_paths):
        """Load all 6 negative fragment CSV files and prepare master dataset"""
        print("Loading XAI-derived NEGATIVE fragments (activity-killing features)...")
        print("Original compound testing sizes:")
        for pathogen, size in self.original_testing_sizes.items():
            print("  {}: {:,} compounds tested".format(pathogen, size))
        print()
        
        all_data = []
        
        for pathogen in ['SA', 'EC', 'CA']:
            for fragment_type in ['scaffold', 'substituent']:
                file_key = "{}_{}".format(pathogen, fragment_type)
                if file_key in file_paths:
                    print("Loading {} negative fragments...".format(file_key))
                    df = pd.read_csv(file_paths[file_key])
                    
                    # Convert SMILES to RDKit molecules
                    df['mol'] = df['fragment_smiles'].apply(lambda x: Chem.MolFromSmiles(x) if pd.notnull(x) else None)
                    
                    # Add metadata
                    df['pathogen'] = pathogen
                    df['pathogen_class'] = self.pathogen_map[pathogen]
                    df['fragment_type'] = fragment_type
                    
                    # Add original testing context
                    df['original_testing_size'] = self.original_testing_sizes[pathogen]
                    
                    # Calculate reliability score for NEGATIVE fragments
                    # Higher negative_consistency_percent = more reliably kills activity
                    df['negative_reliability_score'] = (df['negative_consistency_percent'] / 100) * np.log(df['total_appearances'] + 1)
                    
                    # Weight by original testing coverage
                    df['testing_weight'] = 1.0 / np.sqrt(self.original_testing_sizes[pathogen])
                    
                    # Categorize negative importance (how reliably this fragment kills activity)
                    df['negative_importance_tier'] = self._categorize_negative_importance(df)
                    
                    all_data.append(df)
        
        self.all_fragments = pd.concat(all_data, ignore_index=True)
        
        # Print sample information with testing context
        print("\nXAI NEGATIVE Fragment Distribution (features that kill activity):")
        print("-" * 70)
        for pathogen in ['SA', 'EC', 'CA']:
            count = len(self.all_fragments[self.all_fragments['pathogen'] == pathogen])
            testing_size = self.original_testing_sizes[pathogen]
            negative_rate = (count / testing_size) * 100
            print("{} ({}): {} negative fragments from {:,} tested ({:.3f}% negative-XAI rate)".format(
                pathogen, self.pathogen_map[pathogen], count, testing_size, negative_rate))
        
        print("\nTotal XAI-negative fragments (features to AVOID): {}".format(len(self.all_fragments)))
        return self.all_fragments
    
    def _categorize_negative_importance(self, df):
        """Categorize fragment negative importance based on consistency and appearances"""
        conditions = [
            (df['negative_consistency_percent'] >= 90) & (df['total_appearances'] >= 20),
            (df['negative_consistency_percent'] >= 80) & (df['total_appearances'] >= 10),
            (df['negative_consistency_percent'] >= 70) & (df['total_appearances'] >= 5)
        ]
        choices = ['Highly_Detrimental', 'Reliably_Negative', 'Moderately_Negative']
        return np.select(conditions, choices, default='Limited_Evidence')
    
    def extract_physicochemical_properties(self):
        """Extract key physicochemical properties for comparative analysis"""
        print("Extracting physicochemical properties from all XAI NEGATIVE fragments...")
        
        properties_list = []
        
        for index, row in self.all_fragments.iterrows():
            if index % 1000 == 0:
                print("Processing negative fragment {}/{}".format(index+1, len(self.all_fragments)))
            
            mol = row['mol']
            if mol is None:
                continue
            
            props = {
                'fragment_id': row['fragment_id'],
                'pathogen': row['pathogen'],
                'pathogen_class': row['pathogen_class'],
                'fragment_type': row['fragment_type'],
                'fragment_smiles': row['fragment_smiles'],
                'testing_weight': row['testing_weight']
            }
            
            try:
                # Core physicochemical properties
                props['molecular_weight'] = Descriptors.MolWt(mol)
                props['logp'] = Descriptors.MolLogP(mol)
                props['tpsa'] = Descriptors.TPSA(mol)
                props['num_hbd'] = Descriptors.NumHDonors(mol)
                props['num_hba'] = Descriptors.NumHAcceptors(mol)
                props['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
                
                # Structural properties
                props['num_atoms'] = mol.GetNumAtoms()
                props['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
                props['num_rings'] = Descriptors.RingCount(mol)
                props['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
                props['num_aliphatic_rings'] = Descriptors.NumAliphaticRings(mol)
                props['num_saturated_rings'] = Descriptors.NumSaturatedRings(mol)
                props['num_heteroatoms'] = Descriptors.NumHeteroatoms(mol)
                
                # Electronic properties
                props['formal_charge'] = rdmolops.GetFormalCharge(mol)
                props['fraction_csp3'] = Descriptors.FractionCSP3(mol)
                
                # Complexity measures
                props['bertz_complexity'] = Descriptors.BertzCT(mol)
                props['balaban_j'] = Descriptors.BalabanJ(mol)
                
                # Atom type counts
                props['carbon_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
                props['nitrogen_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
                props['oxygen_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
                props['sulfur_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S')
                props['halogen_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ['F', 'Cl', 'Br', 'I'])
                
                # Fragment counts (key functional groups)
                props['aromatic_carbocycles'] = Fragments.fr_benzene(mol)
                props['aromatic_heterocycles'] = Fragments.fr_pyridine(mol) + Fragments.fr_furan(mol)
                props['aliphatic_hydroxyl'] = Fragments.fr_Al_OH(mol)
                props['aromatic_hydroxyl'] = Fragments.fr_Ar_OH(mol)
                props['carboxyl'] = Fragments.fr_COO(mol)
                props['ester'] = Fragments.fr_ester(mol)
                props['amide'] = Fragments.fr_amide(mol)
                props['amine'] = Fragments.fr_NH2(mol) + Fragments.fr_NH1(mol) + Fragments.fr_NH0(mol)
                props['halogen_subst'] = Fragments.fr_halogen(mol)
                
            except Exception as e:
                print("Error processing {}: {}".format(row['fragment_smiles'], e))
                continue
            
            properties_list.append(props)
        
        self.chemical_features = pd.DataFrame(properties_list)
        
        # Merge with original data, preserving all columns
        self.all_fragments = self.all_fragments.merge(
            self.chemical_features, 
            on=['fragment_id', 'pathogen'], 
            how='left',
            suffixes=('', '_chem')
        )
        
        print("Extracted {} physicochemical properties from all negative fragments".format(
            len(self.chemical_features.columns)-6))
        return self.chemical_features
    
    def permutation_test(self, group1, group2, n_permutations=10000, weights1=None, weights2=None):
        """
        Perform permutation test for two groups with optional weights
        Returns p-value and effect size
        """
        # Calculate observed difference in means
        if weights1 is not None and weights2 is not None:
            mean1 = np.average(group1, weights=weights1)
            mean2 = np.average(group2, weights=weights2)
        else:
            mean1 = np.mean(group1)
            mean2 = np.mean(group2)
        
        observed_diff = mean1 - mean2
        
        # Combine data for permutation
        combined_data = np.concatenate([group1, group2])
        combined_weights = None
        if weights1 is not None and weights2 is not None:
            combined_weights = np.concatenate([weights1, weights2])
        
        n1, n2 = len(group1), len(group2)
        
        # Permutation test
        permuted_diffs = []
        
        for _ in range(n_permutations):
            # Randomly permute the combined data
            if combined_weights is not None:
                indices = np.random.permutation(len(combined_data))
                perm_data = combined_data[indices]
                perm_weights = combined_weights[indices]
                
                perm_group1 = perm_data[:n1]
                perm_group2 = perm_data[n1:]
                perm_weights1 = perm_weights[:n1]
                perm_weights2 = perm_weights[n1:]
                
                perm_mean1 = np.average(perm_group1, weights=perm_weights1)
                perm_mean2 = np.average(perm_group2, weights=perm_weights2)
            else:
                perm_data = np.random.permutation(combined_data)
                perm_group1 = perm_data[:n1]
                perm_group2 = perm_data[n1:]
                
                perm_mean1 = np.mean(perm_group1)
                perm_mean2 = np.mean(perm_group2)
            
            permuted_diffs.append(perm_mean1 - perm_mean2)
        
        # Calculate p-value (two-tailed)
        permuted_diffs = np.array(permuted_diffs)
        p_value = np.sum(np.abs(permuted_diffs) >= np.abs(observed_diff)) / n_permutations
        
        # Calculate effect size (Cohen's d with pooled standard deviation)
        pooled_std = np.sqrt(((n1-1)*np.var(group1) + (n2-1)*np.var(group2)) / (n1+n2-2))
        effect_size = abs(observed_diff) / pooled_std if pooled_std > 0 else 0
        
        return p_value, effect_size, observed_diff
    
    def perform_permutation_analysis(self, n_permutations=10000, use_weights=True):
        """
        Perform permutation-based analysis using all negative fragments
        """
        print("Performing permutation-based analysis using ALL {} NEGATIVE fragments...".format(
            len(self.chemical_features)))
        print("Permutations per test: {:,}".format(n_permutations))
        print("Weighting by original testing size: {}".format(use_weights))
        
        if self.chemical_features is None:
            print("No chemical features available.")
            return pd.DataFrame()
        
        # Get numeric feature columns
        feature_cols = [col for col in self.chemical_features.columns 
                       if col not in ['fragment_id', 'pathogen', 'pathogen_class', 
                                    'fragment_type', 'fragment_smiles', 'testing_weight']]
        
        # Remove features with insufficient variation
        valid_features = []
        for feature in feature_cols:
            values = self.chemical_features[feature].dropna()
            if len(values) > 10 and values.std() > 0:
                valid_features.append(feature)
        
        print("Analyzing {} valid features for negative fragment patterns...".format(len(valid_features)))
        
        results = []
        
        # Pairwise permutation tests
        pathogen_pairs = [('SA', 'EC'), ('SA', 'CA'), ('EC', 'CA')]
        
        for i, feature in enumerate(valid_features):
            if (i + 1) % 5 == 0:
                print("  Processed {}/{} features...".format(i + 1, len(valid_features)))
            
            for pathogen1, pathogen2 in pathogen_pairs:
                try:
                    # Get data for both pathogens
                    data1 = self.chemical_features[self.chemical_features['pathogen'] == pathogen1][feature].dropna()
                    data2 = self.chemical_features[self.chemical_features['pathogen'] == pathogen2][feature].dropna()
                    
                    if len(data1) < 5 or len(data2) < 5:
                        continue
                    
                    # Get weights if using weighted analysis
                    weights1 = weights2 = None
                    if use_weights:
                        weights1 = self.chemical_features[
                            (self.chemical_features['pathogen'] == pathogen1) & 
                            (self.chemical_features[feature].notna())
                        ]['testing_weight'].values
                        weights2 = self.chemical_features[
                            (self.chemical_features['pathogen'] == pathogen2) & 
                            (self.chemical_features[feature].notna())
                        ]['testing_weight'].values
                        
                        # Ensure same length as data
                        weights1 = weights1[:len(data1)]
                        weights2 = weights2[:len(data2)]
                    
                    # Perform permutation test
                    p_value, effect_size, mean_diff = self.permutation_test(
                        data1.values, data2.values, n_permutations, weights1, weights2
                    )
                    
                    # Calculate fold change
                    mean1 = np.average(data1, weights=weights1) if weights1 is not None else np.mean(data1)
                    mean2 = np.average(data2, weights=weights2) if weights2 is not None else np.mean(data2)
                    fold_change = mean1 / mean2 if mean2 != 0 else np.inf
                    
                    results.append({
                        'feature': feature,
                        'pathogen1': pathogen1,
                        'pathogen2': pathogen2,
                        'comparison': '{}_vs_{}'.format(pathogen1, pathogen2),
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'mean_diff': mean_diff,
                        'fold_change': fold_change,
                        'mean1': mean1,
                        'mean2': mean2,
                        'n1': len(data1),
                        'n2': len(data2),
                        'direction': 'higher' if mean1 > mean2 else 'lower',
                        'weighted': use_weights,
                        'interpretation': 'more_problematic' if mean1 > mean2 else 'less_problematic'
                    })
                
                except Exception as e:
                    continue
        
        results_df = pd.DataFrame(results)
        
        # Apply FDR correction
        if len(results_df) > 0:
            from statsmodels.stats.multitest import multipletests
            _, corrected_p, _, _ = multipletests(results_df['p_value'], alpha=0.05, method='fdr_bh')
            results_df['corrected_p_value'] = corrected_p
        
        self.permutation_results = results_df
        
        # Filter significant results
        significant_results = results_df[
            (results_df['corrected_p_value'] <= 0.05) &
            (results_df['effect_size'] >= 0.3)
        ].copy()
        
        print("Negative fragment permutation analysis complete!")
        print("Total tests: {}".format(len(results_df)))
        print("Significant negative patterns (corrected p≤0.05, effect≥0.3): {}".format(len(significant_results)))
        
        return significant_results
    
    def create_negative_visualizations(self, output_dir='negative_plots'):
        """Create comprehensive visualizations for negative fragment analysis"""
        print("Creating negative fragment analysis visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if len(self.permutation_results) == 0:
            print("No permutation results to visualize.")
            return
        
        # Set up the style with red/orange theme for negative fragments
        plt.style.use('default')
        
        # 1. Enhanced Volcano plot for negative fragments
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get significant results for highlighting
        significant_results = self.permutation_results[
            (self.permutation_results['corrected_p_value'] <= 0.05) &
            (self.permutation_results['effect_size'] >= 0.3)
        ]
        
        # Color by significance and effect size (red theme for negative)
        colors = []
        for _, row in self.permutation_results.iterrows():
            if row['corrected_p_value'] <= 0.05 and row['effect_size'] >= 0.3:
                colors.append('darkred')  # Significant negative pattern
            elif row['corrected_p_value'] <= 0.05:
                colors.append('orange')  # Significant but small effect
            elif row['effect_size'] >= 0.3:
                colors.append('coral')  # Large effect but not significant
            else:
                colors.append('lightgray')  # Neither
        
        scatter = ax.scatter(self.permutation_results['effect_size'], 
                           -np.log10(self.permutation_results['corrected_p_value']),
                           c=colors, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        
        # Add threshold lines
        ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, linewidth=2, label='p=0.05')
        ax.axvline(x=0.3, color='darkred', linestyle='--', alpha=0.7, linewidth=2, label='Effect=0.3')
        
        # Annotate significant points
        for _, row in significant_results.head(5).iterrows():
            ax.annotate('{}\n({} vs {})'.format(row['feature'].replace('_', ' '), row['pathogen1'], row['pathogen2']),
                       xy=(row['effect_size'], -np.log10(row['corrected_p_value'])),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', fc='mistyrose', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                       fontsize=9)
        
        ax.set_xlabel('Effect Size (Cohen\'s d)', fontsize=14, fontweight='bold')
        ax.set_ylabel('-log10(Corrected P-Value)', fontsize=14, fontweight='bold')
        ax.set_title('NEGATIVE Fragment Chemical Pattern Significance\nFeatures to AVOID - Analysis of {} XAI Negative Fragments'.format(
            len(self.chemical_features)), fontsize=16, fontweight='bold', color='darkred')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add text box with summary
        textstr = 'Significant negative patterns: {}\nTotal tests: {}\nEffect threshold: 0.3\nSignificance: p<0.05\n\n** FEATURES TO AVOID **'.format(
            len(significant_results), len(self.permutation_results))
        props = dict(boxstyle='round', facecolor='mistyrose', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig('{}/negative_volcano_plot.png'.format(output_dir), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Negative vs Positive comparison plot (if both analyses are available)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Negative hit rates
        pathogen_info = []
        for pathogen in ['SA', 'EC', 'CA']:
            frag_count = len(self.all_fragments[self.all_fragments['pathogen'] == pathogen])
            test_count = self.original_testing_sizes[pathogen]
            negative_rate = (frag_count / test_count) * 100
            pathogen_info.append({
                'pathogen': pathogen,
                'pathogen_full': '{} ({})'.format(pathogen, self.pathogen_map[pathogen]),
                'negative_fragments': frag_count,
                'tested': test_count,
                'negative_rate': negative_rate
            })
        
        pathogen_df = pd.DataFrame(pathogen_info)
        
        # Negative fragment counts
        bars1 = axes[0].bar(pathogen_df['pathogen'], pathogen_df['negative_fragments'], 
                           color=[self.pathogen_colors[p] for p in pathogen_df['pathogen']])
        axes[0].set_title('XAI-NEGATIVE Fragments\n(Features to AVOID)', fontweight='bold', color='darkred')
        axes[0].set_ylabel('Negative Fragment Count')
        for i, v in enumerate(pathogen_df['negative_fragments']):
            axes[0].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Negative hit rates
        bars2 = axes[1].bar(pathogen_df['pathogen'], pathogen_df['negative_rate'], 
                           color=[self.pathogen_colors[p] for p in pathogen_df['pathogen']])
        axes[1].set_title('XAI Negative Hit Rates\n(Activity-Killing Rate)', fontweight='bold', color='darkred')
        axes[1].set_ylabel('Negative Hit Rate (%)')
        for i, v in enumerate(pathogen_df['negative_rate']):
            axes[1].text(i, v + 0.05, '{:.2f}%'.format(v), ha='center', va='bottom', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Negative Fragment Screening Context - FEATURES TO AVOID', fontsize=16, fontweight='bold', color='darkred')
        plt.tight_layout()
        plt.savefig('{}/negative_screening_context.png'.format(output_dir), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Top problematic features comparison
        if len(significant_results) > 0:
            top_features = significant_results.nlargest(6, 'effect_size')
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, (_, result) in enumerate(top_features.iterrows()):
                if i >= 6:
                    break
                
                feature = result['feature']
                pathogen1, pathogen2 = result['pathogen1'], result['pathogen2']
                
                # Get data for both pathogens
                data1 = self.chemical_features[self.chemical_features['pathogen'] == pathogen1][feature].dropna()
                data2 = self.chemical_features[self.chemical_features['pathogen'] == pathogen2][feature].dropna()
                
                # Create violin plot
                data_combined = pd.DataFrame({
                    'value': list(data1) + list(data2),
                    'pathogen': ['{} ({})'.format(pathogen1, self.pathogen_map[pathogen1])] * len(data1) + 
                               ['{} ({})'.format(pathogen2, self.pathogen_map[pathogen2])] * len(data2)
                })
                
                sns.violinplot(data=data_combined, x='pathogen', y='value', ax=axes[i],
                              palette=[self.pathogen_colors[pathogen1], self.pathogen_colors[pathogen2]])
                
                # Add statistical annotation
                problematic_pathogen = pathogen1 if result['direction'] == 'higher' else pathogen2
                axes[i].set_title('{}\nMORE PROBLEMATIC for {}\nEffect: {:.3f}, p: {:.2e}'.format(
                    feature.replace('_', ' ').title(), problematic_pathogen, 
                    result['effect_size'], result['corrected_p_value']),
                    fontweight='bold', color='darkred')
                axes[i].set_ylabel(feature.replace('_', ' ').title())
                axes[i].set_xlabel('')
                axes[i].grid(True, alpha=0.3)
                
                # Add mean lines
                axes[i].axhline(y=data1.mean(), color=self.pathogen_colors[pathogen1], 
                               linestyle='--', alpha=0.7, linewidth=2)
                axes[i].axhline(y=data2.mean(), color=self.pathogen_colors[pathogen2], 
                               linestyle='--', alpha=0.7, linewidth=2)
            
            # Hide unused subplots
            for j in range(i+1, 6):
                axes[j].set_visible(False)
            
            plt.suptitle('Top Problematic Chemical Features (AVOID These)', fontsize=16, fontweight='bold', color='darkred')
            plt.tight_layout()
            plt.savefig('{}/problematic_features.png'.format(output_dir), dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Negative fragment visualizations saved to {}/".format(output_dir))
        print("Generated plots:")
        print("- negative_volcano_plot.png (significance of features to avoid)")
        print("- negative_screening_context.png (negative hit rates)")
        print("- problematic_features.png (most problematic chemical features)")
    
    def generate_negative_report(self, output_file='negative_fragment_analysis_report.txt'):
        """Generate comprehensive negative fragment analysis report"""
        print("Generating negative fragment analysis report...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("NEGATIVE FRAGMENT PATHOGEN ANALYSIS\n")
            f.write("XAI-Derived Activity-Killing Features - WHAT TO AVOID\n")
            f.write("=" * 80 + "\n\n")
            
            # Methodology
            f.write("STATISTICAL METHODOLOGY\n")
            f.write("-" * 40 + "\n")
            f.write("Approach: Permutation tests using ALL negative fragments\n")
            f.write("Purpose: Identify features that consistently KILL antimicrobial activity\n")
            f.write("Testing bias correction: Weighted by original dataset sizes\n")
            f.write("Permutations per test: 10,000\n")
            f.write("Multiple testing correction: Benjamini-Hochberg FDR\n")
            f.write("Effect size threshold: 0.3 (Cohen's d)\n")
            f.write("Significance threshold: p < 0.05 (corrected)\n\n")
            
            # Original dataset context
            f.write("ORIGINAL DATASET CONTEXT\n")
            f.write("-" * 40 + "\n")
            f.write("Compounds tested in original screens:\n")
            for pathogen, size in self.original_testing_sizes.items():
                f.write("  {}: {:,} compounds\n".format(pathogen, size))
            
            f.write("\nXAI-negative fragments extracted (features that kill activity):\n")
            for pathogen in ['SA', 'EC', 'CA']:
                count = len(self.all_fragments[self.all_fragments['pathogen'] == pathogen])
                test_size = self.original_testing_sizes[pathogen]
                negative_rate = (count / test_size) * 100
                f.write("  {}: {} negative fragments ({:.3f}% negative hit rate)\n".format(pathogen, count, negative_rate))
            
            # Results
            if len(self.permutation_results) > 0:
                significant_results = self.permutation_results[
                    (self.permutation_results['corrected_p_value'] <= 0.05) &
                    (self.permutation_results['effect_size'] >= 0.3)
                ]
                
                f.write("\n\nNEGATIVE FRAGMENT PERMUTATION RESULTS\n")
                f.write("-" * 40 + "\n")
                f.write("Total pairwise tests performed: {}\n".format(len(self.permutation_results)))
                f.write("Statistically significant negative patterns: {}\n".format(len(significant_results)))
                f.write("False discovery rate: {:.1%}\n\n".format(
                    (self.permutation_results['corrected_p_value'] <= 0.05).mean()))
                
                if len(significant_results) > 0:
                    f.write("VALIDATED FEATURES TO AVOID:\n")
                    f.write("-" * 30 + "\n")
                    
                    for _, result in significant_results.nlargest(10, 'effect_size').iterrows():
                        problematic_pathogen = result['pathogen1'] if result['direction'] == 'higher' else result['pathogen2']
                        less_problematic = result['pathogen2'] if result['direction'] == 'higher' else result['pathogen1']
                        
                        f.write("AVOID {} for {}: {:.2f}x more problematic than for {} ".format(
                            result['feature'].replace('_', ' '),
                            problematic_pathogen,
                            abs(result['fold_change']),
                            less_problematic
                        ))
                        f.write("(effect: {:.3f}, p: {:.2e})\n".format(
                            result['effect_size'], result['corrected_p_value']))
                
                # Design recommendations - what to avoid
                f.write("\n\nDESIGN RECOMMENDATIONS - FEATURES TO AVOID\n")
                f.write("-" * 50 + "\n")
                f.write("Based on permutation-validated negative patterns:\n\n")
                
                for pathogen in ['SA', 'EC', 'CA']:
                    pathogen_problematic = significant_results[
                        ((significant_results['pathogen1'] == pathogen) & (significant_results['direction'] == 'higher')) |
                        ((significant_results['pathogen2'] == pathogen) & (significant_results['direction'] == 'lower'))
                    ]
                    
                    if len(pathogen_problematic) > 0:
                        f.write("{} ({}) - Features to AVOID:\n".format(
                            pathogen, self.pathogen_map[pathogen]))
                        
                        for _, feature in pathogen_problematic.nlargest(3, 'effect_size').iterrows():
                            other_pathogen = feature['pathogen2'] if feature['pathogen1'] == pathogen else feature['pathogen1']
                            f.write("  - MINIMIZE {}: {:.2f}x more detrimental than for {}\n".format(
                                feature['feature'].replace('_', ' '),
                                abs(feature['fold_change']),
                                other_pathogen
                            ))
                        f.write("\n")
                
                # Combined strategy recommendations
                f.write("\nINTEGRATED DESIGN STRATEGY:\n")
                f.write("-" * 30 + "\n")
                f.write("To design effective antimicrobials:\n")
                f.write("1. Use POSITIVE fragment analysis to identify features to ENHANCE\n")
                f.write("2. Use this NEGATIVE fragment analysis to identify features to AVOID\n")
                f.write("3. Balance both insights for optimal pathogen-specific design\n\n")
                
                f.write("Example strategic approach:\n")
                for pathogen in ['SA', 'EC', 'CA']:
                    f.write("For {}: Enhance [positive features] while avoiding [negative features]\n".format(pathogen))
            
            f.write("\nNote: These results use ALL {} XAI-derived negative fragments\n".format(len(self.all_fragments)))
            f.write("and account for original testing bias through permutation weighting.\n")
            f.write("\n** USE WITH POSITIVE FRAGMENT ANALYSIS FOR COMPLETE PICTURE **\n")
        
        print("Negative fragment analysis report saved to {}".format(output_file))

    def extract_negative_examples_with_compounds(self, output_file='negative_fragment_examples.csv'):
        """Extract example negative fragments with their source compounds"""
        print("Extracting negative fragment examples with compound structures...")
        
        if len(self.permutation_results) == 0:
            print("No significant negative patterns to extract examples from.")
            return
        
        # Get significant results
        significant_results = self.permutation_results[
            (self.permutation_results['corrected_p_value'] <= 0.05) &
            (self.permutation_results['effect_size'] >= 0.3)
        ].nlargest(10, 'effect_size')
        
        example_data = []
        
        for _, result in significant_results.iterrows():
            feature = result['feature']
            pathogen1, pathogen2 = result['pathogen1'], result['pathogen2']
            
            # Identify which pathogen this feature is more problematic for
            if result['direction'] == 'higher':
                problematic_pathogen = pathogen1
            else:
                problematic_pathogen = pathogen2
            
            # Get fragments from problematic pathogen
            pathogen_fragments = self.all_fragments[
                self.all_fragments['pathogen'] == problematic_pathogen
            ].copy()
            
            # Sort by negative reliability score and get top examples
            top_fragments = pathogen_fragments.nlargest(5, 'negative_reliability_score')
            
            for _, fragment in top_fragments.iterrows():
                try:
                    # Extract example compound information from negative fragment data
                    lowest_example = fragment.get('lowest_attribution_example', '')
                    highest_example = fragment.get('highest_attribution_example', '')
                    active_negative = fragment.get('active_negative_attribution_example', '')
                    inactive_positive = fragment.get('inactive_positive_attribution_example', '')
                    
                    # Parse compound information
                    lowest_info = self._parse_compound_info(lowest_example)
                    highest_info = self._parse_compound_info(highest_example)
                    active_neg_info = self._parse_compound_info(active_negative)
                    inactive_pos_info = self._parse_compound_info(inactive_positive)
                    
                    example_data.append({
                        'problematic_feature': feature,
                        'feature_description': feature.replace('_', ' ').title(),
                        'most_problematic_for': problematic_pathogen,
                        'pathogen_class': self.pathogen_map[problematic_pathogen],
                        'effect_size': result['effect_size'],
                        'p_value': result['corrected_p_value'],
                        'fold_change': result['fold_change'],
                        
                        # Fragment information
                        'fragment_id': fragment['fragment_id'],
                        'fragment_smiles': fragment['fragment_smiles'],
                        'fragment_type': fragment['fragment_type'],
                        'negative_reliability_score': fragment['negative_reliability_score'],
                        'avg_attribution': fragment['avg_attribution'],
                        'negative_consistency_percent': fragment['negative_consistency_percent'],
                        'inactivity_rate_percent': fragment['inactivity_rate_percent'],
                        'total_appearances': fragment['total_appearances'],
                        'negative_appearances': fragment['negative_appearances'],
                        
                        # Lowest attribution example (most negative)
                        'lowest_compound_id': lowest_info.get('id', ''),
                        'lowest_compound_smiles': lowest_info.get('smiles', ''),
                        'lowest_attribution': lowest_info.get('attribution', ''),
                        'lowest_prediction': lowest_info.get('prediction', ''),
                        'lowest_mw': lowest_info.get('mw', ''),
                        'lowest_logp': lowest_info.get('logp', ''),
                        
                        # Highest attribution example
                        'highest_compound_id': highest_info.get('id', ''),
                        'highest_compound_smiles': highest_info.get('smiles', ''),
                        'highest_attribution': highest_info.get('attribution', ''),
                        'highest_prediction': highest_info.get('prediction', ''),
                        'highest_mw': highest_info.get('mw', ''),
                        'highest_logp': highest_info.get('logp', ''),
                        
                        # Active compound with negative attribution (problematic case)
                        'active_neg_compound_id': active_neg_info.get('id', ''),
                        'active_neg_smiles': active_neg_info.get('smiles', ''),
                        'active_neg_attribution': active_neg_info.get('attribution', ''),
                        'active_neg_prediction': active_neg_info.get('prediction', ''),
                        
                        # Inactive compound with positive attribution (confounding case)
                        'inactive_pos_compound_id': inactive_pos_info.get('id', ''),
                        'inactive_pos_smiles': inactive_pos_info.get('smiles', ''),
                        'inactive_pos_attribution': inactive_pos_info.get('attribution', ''),
                        'inactive_pos_prediction': inactive_pos_info.get('prediction', ''),
                        
                        # Chemical property value for this fragment
                        'feature_value': self.chemical_features[
                            self.chemical_features['fragment_id'] == fragment['fragment_id']
                        ][feature].iloc[0] if feature in self.chemical_features.columns else 'N/A'
                    })
                    
                except Exception as e:
                    print("Error processing negative fragment {}: {}".format(fragment['fragment_id'], e))
                    continue
        
        # Create DataFrame and save
        examples_df = pd.DataFrame(example_data)
        
        if len(examples_df) > 0:
            # Sort by effect size for better presentation
            examples_df = examples_df.sort_values(['effect_size', 'negative_reliability_score'], ascending=[False, False])
            
            # Save to CSV
            examples_df.to_csv(output_file, index=False, encoding='utf-8')
            print("Negative fragment examples saved to {}".format(output_file))
            print("Found {} negative example fragments across {} problematic features".format(
                len(examples_df), examples_df['problematic_feature'].nunique()))
            
            # Create summary of features to avoid
            self._create_negative_summary_table(examples_df)
            
        else:
            print("No negative example fragments could be extracted.")
        
        return examples_df
    
    def _parse_compound_info(self, example_string):
        """Parse compound information from example strings"""
        if not example_string or pd.isna(example_string):
            return {}
        
        info = {}
        try:
            # Extract CHEMBL ID
            chembl_match = re.search(r'CHEMBL\d+', str(example_string))
            if chembl_match:
                info['id'] = chembl_match.group()
            
            # Extract SMILES
            smiles_match = re.search(r'SMILES:\s*([^\s|]+)', str(example_string))
            if smiles_match:
                info['smiles'] = smiles_match.group(1)
            
            # Extract Attribution
            attr_match = re.search(r'Attribution:\s*([-\d.]+)', str(example_string))
            if attr_match:
                info['attribution'] = float(attr_match.group(1))
            
            # Extract Prediction
            pred_match = re.search(r'Prediction:\s*([\d.-]+)', str(example_string))
            if pred_match:
                info['prediction'] = float(pred_match.group(1))
            
            # Extract MW
            mw_match = re.search(r'MW:\s*([\d.-]+)', str(example_string))
            if mw_match:
                info['mw'] = float(mw_match.group(1))
            
            # Extract LogP
            logp_match = re.search(r'LogP:\s*([-\d.]+)', str(example_string))
            if logp_match:
                info['logp'] = float(logp_match.group(1))
                
        except Exception as e:
            print("Error parsing compound info: {}".format(e))
        
        return info
    
    def _create_negative_summary_table(self, examples_df):
        """Create a summary table of key features to avoid"""
        print("Creating negative fragment summary table...")
        
        # Get top 3 problematic features per pathogen
        summary_data = []
        
        for pathogen in ['SA', 'EC', 'CA']:
            pathogen_examples = examples_df[examples_df['most_problematic_for'] == pathogen]
            
            if len(pathogen_examples) > 0:
                top_examples = pathogen_examples.head(3)
                
                for _, example in top_examples.iterrows():
                    summary_data.append({
                        'Pathogen': '{} ({})'.format(pathogen, self.pathogen_map[pathogen]),
                        'Feature to AVOID': example['feature_description'],
                        'Effect Size': '{:.3f}'.format(example['effect_size']),
                        'Problematic Fragment': example['fragment_smiles'],
                        'Fragment Type': example['fragment_type'].title(),
                        'Negative Reliability': '{:.3f}'.format(example['negative_reliability_score']),
                        'Inactivity Rate': '{:.1f}%'.format(example['inactivity_rate_percent']),
                        'Example Problem Compound': example['lowest_compound_id'],
                        'Problem Compound SMILES': example['lowest_compound_smiles'][:50] + '...' if len(str(example['lowest_compound_smiles'])) > 50 else example['lowest_compound_smiles'],
                        'Negative Attribution': '{:.3f}'.format(example['lowest_attribution']) if example['lowest_attribution'] else 'N/A'
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv('negative_fragment_summary_table.csv', index=False, encoding='utf-8')
            print("Negative fragment summary table saved to negative_fragment_summary_table.csv")
            
            # Print a preview
            print("\nFeatures to AVOID - Summary Preview:")
            print("=" * 80)
            for i, row in summary_df.head(6).iterrows():
                print("{}. {} - AVOID {}".format(i+1, row['Pathogen'], row['Feature to AVOID']))
                print("   Problematic Fragment: {}".format(row['Problematic Fragment']))
                print("   Problem Example: {} (Attribution: {})".format(row['Example Problem Compound'], row['Negative Attribution']))
                print("   Effect Size: {} | Inactivity Rate: {}".format(row['Effect Size'], row['Inactivity Rate']))
                print()
    
    def create_avoidance_guide(self, output_file='chemical_avoidance_guide.txt'):
        """Create a practical guide for avoiding problematic features"""
        print("Creating chemical avoidance guide...")
        
        if len(self.permutation_results) == 0:
            print("No significant negative patterns to create guide from.")
            return
        
        significant_results = self.permutation_results[
            (self.permutation_results['corrected_p_value'] <= 0.05) &
            (self.permutation_results['effect_size'] >= 0.3)
        ].nlargest(10, 'effect_size')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("CHEMICAL AVOIDANCE GUIDE FOR ANTIMICROBIAL DESIGN\n")
            f.write("=" * 60 + "\n\n")
            f.write("Top Chemical Features to AVOID for Each Pathogen\n")
            f.write("-" * 50 + "\n\n")
            
            for i, (_, result) in enumerate(significant_results.iterrows(), 1):
                problematic_pathogen = result['pathogen1'] if result['direction'] == 'higher' else result['pathogen2']
                less_problematic = result['pathogen2'] if result['direction'] == 'higher' else result['pathogen1']
                
                f.write("{}. AVOID {} for {} ({})\n".format(
                    i, result['feature'].replace('_', ' ').title(), 
                    problematic_pathogen, self.pathogen_map[problematic_pathogen]))
                f.write("   Problem Level: {:.2f}x more detrimental than for {}\n".format(
                    abs(result['fold_change']), less_problematic))
                f.write("   Statistical Confidence: Effect {:.3f}, p = {:.2e}\n".format(
                    result['effect_size'], result['corrected_p_value']))
                
                # Add avoidance strategy
                avoidance_strategies = {
                    'logp': 'Reduce lipophilicity - avoid long alkyl chains, use polar substituents',
                    'molecular_weight': 'Keep molecules smaller - avoid bulky substituents and extended scaffolds',
                    'num_hbd': 'Reduce H-bond donors - minimize OH, NH, COOH groups',
                    'aromatic_rings': 'Reduce aromatic content - use aliphatic alternatives where possible',
                    'sulfur_count': 'Avoid sulfur atoms - no thiols, sulfides, or sulfonamides',
                    'halogen_count': 'Minimize halogens - reduce F, Cl, Br, I substitutions',
                    'num_rotatable_bonds': 'Increase rigidity - use cyclic structures, reduce flexible linkers'
                }
                
                strategy = avoidance_strategies.get(result['feature'], 'Minimize this structural feature')
                f.write("   Avoidance Strategy: {}\n".format(strategy))
                f.write("\n")
            
            f.write("\nPATHOGEN-SPECIFIC AVOIDANCE STRATEGIES:\n")
            f.write("-" * 40 + "\n")
            
            # Compile pathogen-specific recommendations
            pathogen_avoid = {pathogen: [] for pathogen in ['SA', 'EC', 'CA']}
            
            for _, result in significant_results.iterrows():
                problematic_pathogen = result['pathogen1'] if result['direction'] == 'higher' else result['pathogen2']
                pathogen_avoid[problematic_pathogen].append({
                    'feature': result['feature'].replace('_', ' '),
                    'fold_change': abs(result['fold_change'])
                })
            
            for pathogen in ['SA', 'EC', 'CA']:
                if pathogen_avoid[pathogen]:
                    f.write("\n{} ({}) - PRIMARY AVOIDANCE LIST:\n".format(pathogen, self.pathogen_map[pathogen]))
                    for avoid_item in pathogen_avoid[pathogen][:3]:
                        f.write("  ❌ AVOID: {} ({:.1f}x more problematic)\n".format(
                            avoid_item['feature'], avoid_item['fold_change']))
            
            f.write("\n\nINTEGRATED DESIGN WORKFLOW:\n")
            f.write("-" * 30 + "\n")
            f.write("1. Start with positive fragment insights (features to ENHANCE)\n")
            f.write("2. Apply this avoidance guide (features to MINIMIZE)\n")
            f.write("3. Balance enhancement vs avoidance for optimal design\n")
            f.write("4. Prioritize pathogen-specific strategies\n")
            f.write("5. Test designs against both positive and negative patterns\n\n")
            
            f.write("EXAMPLE DESIGN PROCESS:\n")
            f.write("-" * 25 + "\n")
            f.write("Target: S. aureus antimicrobial\n")
            f.write("✅ ENHANCE: [From positive analysis] - lipophilicity, H-bond donors\n")
            f.write("❌ AVOID: [From this analysis] - excessive aromatic rings, high MW\n")
            f.write("🎯 RESULT: Moderately lipophilic, H-bonding, small molecule design\n")
        
        print("Chemical avoidance guide saved to {}".format(output_file))

def main():
    """Main negative fragment analysis pipeline"""
    print("Starting NEGATIVE Fragment XAI Analysis...")
    print("This approach identifies features that KILL antimicrobial activity.")
    print("Use with positive fragment analysis for complete design insights!\n")
    
    # File paths for NEGATIVE fragment CSV files - UPDATE THESE
    file_paths = {
        'SA_scaffold': 'SA_specific_negative_scaffolds.csv',        # Your negative scaffold files
        'SA_substituent': 'SA_specific_negative_substitutents.csv', # Your negative substituent files
        'EC_scaffold': 'EC_specific_negative_scaffolds.csv', 
        'EC_substituent': 'EC_specific_negative_substitutents.csv',
        'CA_scaffold': 'CA_specific_negative_scaffolds.csv',
        'CA_substituent': 'CA_specific_negative_substitutents.csv'
    }
    
    # Initialize negative fragment analyzer
    analyzer = PermutationNegativePathogenAnalyzer()
    
    try:
        # Load XAI negative fragments with testing context
        all_negative_fragments = analyzer.load_and_prepare_data(file_paths)
        
        # Extract physicochemical properties from ALL negative fragments
        negative_chemical_features = analyzer.extract_physicochemical_properties()
        
        # Perform permutation-based analysis on negative fragments
        print("\nRunning negative fragment permutation tests (5-10 minutes)...")
        significant_negative_patterns = analyzer.perform_permutation_analysis(
            n_permutations=10000,  # High precision
            use_weights=True       # Weight by original testing size
        )
        
        # Create negative-specific visualizations
        analyzer.create_negative_visualizations()
        
        # Extract negative fragment examples
        negative_examples_df = analyzer.extract_negative_examples_with_compounds()
        
        # Create practical avoidance guide
        analyzer.create_avoidance_guide()
        
        # Generate comprehensive negative analysis report
        analyzer.generate_negative_report()
        
        # Save all negative results
        print("\nSaving negative fragment analysis results...")
        
        if len(significant_negative_patterns) > 0:
            significant_negative_patterns.to_csv('negative_significant_patterns.csv', 
                                                index=False, encoding='utf-8')
            print("Significant negative patterns saved to negative_significant_patterns.csv")
        
        analyzer.permutation_results.to_csv('complete_negative_permutation_results.csv',
                                          index=False, encoding='utf-8')
        print("Complete negative results saved to complete_negative_permutation_results.csv")
        
        all_negative_fragments.to_csv('xai_negative_fragments_with_properties.csv', 
                                     index=False, encoding='utf-8')
        print("Enhanced negative fragments saved to xai_negative_fragments_with_properties.csv")
        
        print("\n" + "="*80)
        print("NEGATIVE FRAGMENT ANALYSIS COMPLETE!")
        print("="*80)
        print("Key insights from this AVOIDANCE analysis:")
        print("✅ Uses ALL {} XAI-derived negative fragments".format(len(all_negative_fragments)))
        print("✅ Identifies features that consistently KILL activity")
        print("✅ Accounts for original testing bias")
        print("✅ Provides pathogen-specific avoidance strategies")
        print("✅ Complements positive fragment analysis")
        
        print("\nGenerated NEGATIVE analysis files:")
        print("- negative_fragment_analysis_report.txt (complete avoidance guide)")
        print("- negative_plots/negative_volcano_plot.png (features to avoid)")
        print("- negative_plots/problematic_features.png (most detrimental features)")
        print("- negative_significant_patterns.csv (validated features to avoid)")
        print("- negative_fragment_examples.csv (example problematic fragments)")
        print("- chemical_avoidance_guide.txt (practical design guide)")
        
        # Enhanced summary
        print("\nNegative Fragment Analysis Summary:")
        print("Activity-killing fragment rates from original XAI:")
        for pathogen in ['SA', 'EC', 'CA']:
            count = len(all_negative_fragments[all_negative_fragments['pathogen'] == pathogen])
            testing_size = analyzer.original_testing_sizes[pathogen]
            negative_rate = (count / testing_size) * 100
            print("  {}: {} negative fragments from {:,} tested ({:.3f}% negative rate)".format(
                pathogen, count, testing_size, negative_rate))
        
        print("\nStatistical validation:")
        print("  - Negative pattern tests: {}".format(len(analyzer.permutation_results)))
        print("  - Validated avoidance features: {}".format(len(significant_negative_patterns)))
        print("  - Permutation tests: 10,000 per comparison")
        print("  - FDR correction: Applied")
        
        if len(significant_negative_patterns) > 0:
            print("\nTop Features to AVOID (bias-corrected):")
            
            # Show most problematic features by pathogen
            for pathogen in ['SA', 'EC', 'CA']:
                pathogen_problems = significant_negative_patterns[
                    ((significant_negative_patterns['pathogen1'] == pathogen) & 
                     (significant_negative_patterns['direction'] == 'higher')) |
                    ((significant_negative_patterns['pathogen2'] == pathogen) & 
                     (significant_negative_patterns['direction'] == 'lower'))
                ]
                
                if len(pathogen_problems) > 0:
                    top_problem = pathogen_problems.nlargest(1, 'effect_size').iloc[0]
                    other_pathogen = top_problem['pathogen2'] if top_problem['pathogen1'] == pathogen else top_problem['pathogen1']
                    print("  {} AVOID: {} ({:.1f}x more problematic than for {})".format(
                        pathogen,
                        top_problem['feature'].replace('_', ' '),
                        abs(top_problem['fold_change']),
                        other_pathogen
                    ))
            
            print("\n🎯 DESIGN STRATEGY:")
            print("Combine this NEGATIVE analysis with your POSITIVE fragment analysis:")
            print("1. Use positive fragments to identify features to ENHANCE")
            print("2. Use these negative fragments to identify features to AVOID")
            print("3. Design molecules that maximize good features while minimizing bad ones")
            print("4. Balance enhancement vs avoidance for optimal antimicrobial activity")
            
        else:
            print("\nNo statistically validated negative differences after bias correction.")
            print("This could indicate:")
            print("  - Negative fragments are similar across pathogens")
            print("  - Original testing bias masked true similarities")
            print("  - Need for more specific negative fragment criteria")
        
        print("\n🔄 NEXT STEPS:")
        print("1. Compare positive vs negative fragment insights")
        print("2. Identify features that appear in both analyses")
        print("3. Design balanced molecules using both enhancement and avoidance strategies")
        print("4. Validate designs against both positive and negative patterns")
        
        print("\n⚠️  IMPORTANT: This negative analysis is most powerful when used")
        print("   together with positive fragment analysis for complete design guidance!")
        
    except Exception as e:
        print("Error during negative fragment analysis: {}".format(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()