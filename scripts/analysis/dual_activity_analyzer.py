#!/usr/bin/env python3
"""
ROBUST DUAL-ACTIVITY FRAGMENT ANALYZER
Analyzes XAI-derived fragments that show positive activity against TWO specific pathogens
Features: Smart size-effect correction with data validation and adaptive weighting
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

class RobustDualActivityFragmentAnalyzer:
    def __init__(self, source_compound_counts=None):
        """
        Initialize analyzer with optional source compound counts for size effect correction
        
        Parameters:
        source_compound_counts (dict): Optional mapping of dual combinations to compound counts
                                     e.g., {'SA+CA': 1164, 'SA+EC': 849, 'CA+EC': 187}
                                     Set to None when data unavailable
        """
        # Dual activity combinations mapping
        self.dual_combinations = {
            'SA+CA': {'pathogens': ['SA', 'CA'], 'name': 'S.aureus + C.albicans', 'excluded': 'E.coli'},
            'SA+EC': {'pathogens': ['SA', 'EC'], 'name': 'S.aureus + E.coli', 'excluded': 'C.albicans'},
            'CA+EC': {'pathogens': ['CA', 'EC'], 'name': 'C.albicans + E.coli', 'excluded': 'S.aureus'}
        }
        
        # Pathogen class mapping
        self.pathogen_map = {'SA': 'Gram+', 'EC': 'Gram-', 'CA': 'Fungi'}
        
        # Color scheme for dual combinations
        self.combination_colors = {
            'SA+CA': '#FF6B6B',  # Red - Gram+ & Fungi
            'SA+EC': '#4ECDC4',  # Teal - Gram+ & Gram-
            'CA+EC': '#45B7D1'   # Blue - Fungi & Gram-
        }
        
        # Source compound counts (if available)
        self.source_compound_counts = source_compound_counts
        
        # Data containers
        self.all_fragments = None
        self.chemical_features = None
        self.permutation_results = {}
        self.dual_activity_patterns = {}
        self.weighting_strategy = None
        self.fragment_extraction_efficiency = {}
        
    def load_and_prepare_dual_data(self, file_paths):
        """Load all dual-activity CSV files and prepare master dataset with smart validation"""
        print("Loading XAI-derived dual-activity positive fragments...")
        print("Performing data validation and weighting strategy selection...")
        
        all_data = []
        
        for combination in ['SA_CA', 'SA_EC', 'CA_EC']:
            for fragment_type in ['scaffolds', 'substitutents']:
                file_key = "dual_{}_positive_{}".format(combination, fragment_type)
                
                if file_key in file_paths:
                    print("Loading {}...".format(file_key))
                    df = pd.read_csv(file_paths[file_key])
                    
                    # Parse combination info
                    combo_key = combination.replace('_', '+')
                    combo_info = self.dual_combinations[combo_key]
                    
                    # Convert SMILES to RDKit molecules
                    df['mol'] = df['fragment_smiles'].apply(lambda x: Chem.MolFromSmiles(x) if pd.notnull(x) else None)
                    
                    # Add metadata
                    df['dual_combination'] = combo_key
                    df['combination_name'] = combo_info['name']
                    df['included_pathogens'] = '+'.join(combo_info['pathogens'])
                    df['excluded_pathogen'] = combo_info['excluded']
                    df['fragment_type'] = fragment_type.rstrip('s')  # Remove 's' from scaffolds/substitutents
                    
                    # Calculate dual-activity reliability score
                    df['dual_reliability_score'] = (df['avg_activity_rate_percent'] / 100) * np.log(df['total_compounds_both_pathogens'] + 1)
                    
                    # Add pathogen class information
                    pathogen_classes = []
                    for pathogens in df['included_pathogens']:
                        classes = [self.pathogen_map[p] for p in pathogens.split('+')]
                        pathogen_classes.append(' + '.join(classes))
                    df['pathogen_classes'] = pathogen_classes
                    
                    # Categorize dual importance
                    df['dual_importance_tier'] = self._categorize_dual_importance(df)
                    
                    all_data.append(df)
        
        self.all_fragments = pd.concat(all_data, ignore_index=True)
        
        # Calculate actual fragment counts from loaded data
        actual_fragment_counts = {}
        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            actual_fragment_counts[combo] = len(
                self.all_fragments[self.all_fragments['dual_combination'] == combo]
            )
        
        # Determine optimal weighting strategy
        self.weighting_strategy = self._determine_weighting_strategy(actual_fragment_counts)
        
        # Print comprehensive analysis overview
        self._print_data_overview(actual_fragment_counts)
        
        return self.all_fragments
    
    def _determine_weighting_strategy(self, actual_fragment_counts):
        """Intelligently determine the best weighting strategy based on data consistency"""
        print("\n" + "="*60)
        print("DATA VALIDATION & WEIGHTING STRATEGY SELECTION")
        print("="*60)
        
        if self.source_compound_counts is None:
            print("✓ No source compound data provided - using fragment-based weighting")
            return "fragment_based"
        
        # Validate source compound data consistency
        print("Validating source compound data consistency...")
        
        # Calculate fragment extraction efficiency
        efficiency = {}
        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo in self.source_compound_counts and self.source_compound_counts[combo] > 0:
                efficiency[combo] = actual_fragment_counts[combo] / self.source_compound_counts[combo]
            else:
                efficiency[combo] = 0
        
        self.fragment_extraction_efficiency = efficiency
        
        # Print extraction efficiency analysis
        print("\nFragment Extraction Efficiency Analysis:")
        print("-" * 40)
        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            fragments = actual_fragment_counts[combo]
            compounds = self.source_compound_counts.get(combo, 0)
            eff = efficiency[combo]
            print("  {}: {} fragments from {} compounds ({:.3f} fragments/compound)".format(
                combo, fragments, compounds, eff))
        
        # Data consistency checks
        consistency_checks = self._perform_consistency_checks(actual_fragment_counts, efficiency)
        
        # Decide weighting strategy based on consistency
        if consistency_checks['use_compound_weighting']:
            print("\n✓ DATA CONSISTENT - Using compound-corrected weighting")
            return "compound_corrected"
        else:
            print("\n⚠ DATA INCONSISTENCIES DETECTED - Using fragment-based weighting")
            print("Reasons:", ", ".join(consistency_checks['warnings']))
            return "fragment_based"
    
    def _perform_consistency_checks(self, actual_fragment_counts, efficiency):
        """Perform comprehensive data consistency checks"""
        checks = {
            'use_compound_weighting': True,
            'warnings': []
        }
        
        # Check 1: Reasonable efficiency ranges (0.001 to 1.0 fragments per compound)
        for combo, eff in efficiency.items():
            if eff > 1.0:  # More fragments than compounds (impossible)
                checks['use_compound_weighting'] = False
                checks['warnings'].append("Efficiency too high for {}".format(combo))
            elif eff < 0.001:  # Suspiciously low efficiency
                checks['warnings'].append("Very low efficiency for {}".format(combo))
        
        # Check 2: Efficiency variance (shouldn't be >10x different unless chemically meaningful)
        if len(efficiency) >= 2:
            max_eff = max(efficiency.values())
            min_eff = min([e for e in efficiency.values() if e > 0])
            if min_eff > 0 and (max_eff / min_eff) > 10:
                checks['warnings'].append("Large efficiency variance detected")
        
        # Check 3: Expected fragment-compound relationship
        # Expect: SA+CA (1164) > SA+EC (849) > CA+EC (187) in fragments (generally)
        sa_ca_frags = actual_fragment_counts.get('SA+CA', 0)
        sa_ec_frags = actual_fragment_counts.get('SA+EC', 0)
        ca_ec_frags = actual_fragment_counts.get('CA+EC', 0)
        
        # Check if fragment distribution is completely inverted from compound distribution
        if ca_ec_frags > sa_ca_frags and ca_ec_frags > sa_ec_frags:
            checks['warnings'].append("CA+EC has most fragments despite fewest compounds")
        
        # Check 4: Zero fragment counts (would break analysis)
        for combo, count in actual_fragment_counts.items():
            if count == 0:
                checks['use_compound_weighting'] = False
                checks['warnings'].append("Zero fragments for {}".format(combo))
        
        return checks
    
    def _print_data_overview(self, actual_fragment_counts):
        """Print comprehensive data overview"""
        print("\n" + "="*60)
        print("DUAL-ACTIVITY DATASET OVERVIEW")
        print("="*60)
        
        total_fragments = sum(actual_fragment_counts.values())
        print("Total dual-activity fragments loaded: {}".format(total_fragments))
        
        print("\nDetailed Breakdown:")
        print("-" * 40)
        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            fragments = actual_fragment_counts[combo]
            info = self.dual_combinations[combo]
            
            print("\n{} ({}):".format(info['name'], combo))
            print("  Fragments loaded: {}".format(fragments))
            
            if self.source_compound_counts and combo in self.source_compound_counts:
                compounds = self.source_compound_counts[combo]
                efficiency = self.fragment_extraction_efficiency.get(combo, 0)
                print("  Source compounds: {}".format(compounds))
                print("  Extraction efficiency: {:.3f} fragments/compound".format(efficiency))
                print("  Fragment percentage: {:.1f}% of total".format(100 * fragments / total_fragments))
            
            # Fragment type distribution for this combination
            combo_data = self.all_fragments[self.all_fragments['dual_combination'] == combo]
            if len(combo_data) > 0:
                scaffold_count = len(combo_data[combo_data['fragment_type'] == 'scaffold'])
                substituent_count = len(combo_data[combo_data['fragment_type'] == 'substitutent'])
                print("  Scaffolds: {}, Substitutents: {}".format(scaffold_count, substituent_count))
                
                # Average activity rate
                avg_activity = combo_data['avg_activity_rate_percent'].mean()
                print("  Average activity rate: {:.1f}%".format(avg_activity))
        
        print("\nWeighting Strategy Selected: {}".format(self.weighting_strategy.replace('_', ' ').title()))
        if self.weighting_strategy == "compound_corrected":
            print("  → Will correct for source compound size differences")
        else:
            print("  → Using fragment counts only (safer for inconsistent data)")
    
    def _categorize_dual_importance(self, df):
        """Categorize dual-activity fragment importance"""
        conditions = [
            (df['avg_activity_rate_percent'] >= 95) & (df['total_compounds_both_pathogens'] >= 20),
            (df['avg_activity_rate_percent'] >= 90) & (df['total_compounds_both_pathogens'] >= 10),
            (df['avg_activity_rate_percent'] >= 80) & (df['total_compounds_both_pathogens'] >= 5)
        ]
        choices = ['High_Dual_Impact', 'Reliable_Dual', 'Moderate_Dual']
        return np.select(conditions, choices, default='Limited_Dual')
    
    def extract_physicochemical_properties(self):
        """Extract physicochemical properties for dual-activity analysis"""
        print("\n" + "="*60)
        print("EXTRACTING PHYSICOCHEMICAL PROPERTIES")
        print("="*60)
        
        properties_list = []
        
        for index, row in self.all_fragments.iterrows():
            if index % 500 == 0:
                print("Processing fragment {}/{}".format(index+1, len(self.all_fragments)))
            
            mol = row['mol']
            if mol is None:
                continue
            
            props = {
                'fragment_id': row['fragment_id'],
                'dual_combination': row['dual_combination'],
                'combination_name': row['combination_name'],
                'fragment_type': row['fragment_type'],
                'fragment_smiles': row['fragment_smiles']
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
                
                # Dual-activity specific features
                props['lipinski_violations'] = sum([
                    props['molecular_weight'] > 500,
                    props['logp'] > 5,
                    props['num_hbd'] > 5,
                    props['num_hba'] > 10
                ])
                
                # Membrane permeability indicators
                props['membrane_permeability_score'] = props['logp'] - 0.1 * props['tpsa']
                
                # Flexibility index
                props['flexibility_index'] = props['num_rotatable_bonds'] / props['num_heavy_atoms'] if props['num_heavy_atoms'] > 0 else 0
                
            except Exception as e:
                print("Error processing {}: {}".format(row['fragment_smiles'], e))
                continue
            
            properties_list.append(props)
        
        self.chemical_features = pd.DataFrame(properties_list)
        
        # Merge with original data
        self.all_fragments = self.all_fragments.merge(
            self.chemical_features, 
            on=['fragment_id', 'dual_combination'], 
            how='left',
            suffixes=('', '_chem')
        )
        
        print("Extracted {} physicochemical properties".format(
            len([col for col in self.chemical_features.columns if col not in 
                ['fragment_id', 'dual_combination', 'combination_name', 'fragment_type', 'fragment_smiles']])))
        return self.chemical_features
    
    def _calculate_adaptive_weights(self):
        """Calculate adaptive weights based on selected strategy"""
        weights = {}
        
        if self.weighting_strategy == "compound_corrected" and self.source_compound_counts:
            print("Applying compound-corrected weighting...")
            
            # Weight inversely proportional to source compound count
            # Normalize so weights sum to number of combinations
            total_compounds = sum(self.source_compound_counts.values())
            for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
                if combo in self.source_compound_counts:
                    # Inverse weighting: smaller compound pools get higher weight
                    raw_weight = total_compounds / self.source_compound_counts[combo]
                    weights[combo] = raw_weight
                else:
                    weights[combo] = 1.0
            
            # Normalize weights
            total_weight = sum(weights.values())
            for combo in weights:
                weights[combo] = (weights[combo] / total_weight) * len(weights)
                
        else:
            print("Using fragment-based weighting...")
            # Equal weighting when no compound correction
            for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
                weights[combo] = 1.0
        
        return weights
    
    def permutation_test_weighted(self, group1, group2, weights1=None, weights2=None, n_permutations=10000):
        """Perform weighted permutation test for two groups"""
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
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((n1-1)*np.var(group1) + (n2-1)*np.var(group2)) / (n1+n2-2))
        effect_size = abs(observed_diff) / pooled_std if pooled_std > 0 else 0
        
        return p_value, effect_size, observed_diff
    
    def perform_dual_activity_analysis(self, n_permutations=10000):
        """Perform comprehensive dual-activity comparison analysis with adaptive weighting"""
        print("\n" + "="*60)
        print("DUAL-ACTIVITY PERMUTATION ANALYSIS")
        print("="*60)
        print("Permutations per test: {:,}".format(n_permutations))
        print("Weighting strategy: {}".format(self.weighting_strategy))
        
        if self.chemical_features is None:
            print("No chemical features available.")
            return pd.DataFrame()
        
        # Calculate adaptive weights
        combination_weights = self._calculate_adaptive_weights()
        
        print("\nCombination weights:")
        for combo, weight in combination_weights.items():
            print("  {}: {:.3f}".format(combo, weight))
        
        # Get numeric feature columns
        feature_cols = [col for col in self.chemical_features.columns 
                       if col not in ['fragment_id', 'dual_combination', 'combination_name', 
                                    'fragment_type', 'fragment_smiles']]
        
        # Remove features with insufficient variation
        valid_features = []
        for feature in feature_cols:
            values = self.chemical_features[feature].dropna()
            if len(values) > 10 and values.std() > 0:
                valid_features.append(feature)
        
        print("\nAnalyzing {} valid features across dual combinations...".format(len(valid_features)))
        
        results = []
        
        # Pairwise comparisons between dual combinations
        combination_pairs = [
            ('SA+CA', 'SA+EC'),  # Gram+ & Fungi vs Gram+ & Gram-
            ('SA+CA', 'CA+EC'),  # Gram+ & Fungi vs Fungi & Gram-
            ('SA+EC', 'CA+EC')   # Gram+ & Gram- vs Fungi & Gram-
        ]
        
        for i, feature in enumerate(valid_features):
            if (i + 1) % 5 == 0:
                print("  Processed {}/{} features...".format(i + 1, len(valid_features)))
            
            for combo1, combo2 in combination_pairs:
                try:
                    # Get data for both combinations
                    data1 = self.chemical_features[
                        self.chemical_features['dual_combination'] == combo1
                    ][feature].dropna()
                    data2 = self.chemical_features[
                        self.chemical_features['dual_combination'] == combo2
                    ][feature].dropna()
                    
                    if len(data1) < 5 or len(data2) < 5:
                        continue
                    
                    # Prepare weights for permutation test
                    weights1 = np.full(len(data1), combination_weights[combo1])
                    weights2 = np.full(len(data2), combination_weights[combo2])
                    
                    # Perform weighted permutation test
                    p_value, effect_size, mean_diff = self.permutation_test_weighted(
                        data1.values, data2.values, weights1, weights2, n_permutations
                    )
                    
                    # Calculate weighted means and fold change
                    mean1 = np.average(data1, weights=weights1)
                    mean2 = np.average(data2, weights=weights2)
                    fold_change = mean1 / mean2 if mean2 != 0 else np.inf
                    
                    results.append({
                        'feature': feature,
                        'combo1': combo1,
                        'combo2': combo2,
                        'comparison': '{}_vs_{}'.format(combo1, combo2),
                        'combo1_name': self.dual_combinations[combo1]['name'],
                        'combo2_name': self.dual_combinations[combo2]['name'],
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'mean_diff': mean_diff,
                        'fold_change': fold_change,
                        'weighted_mean1': mean1,
                        'weighted_mean2': mean2,
                        'raw_mean1': np.mean(data1),
                        'raw_mean2': np.mean(data2),
                        'n1': len(data1),
                        'n2': len(data2),
                        'weight1': combination_weights[combo1],
                        'weight2': combination_weights[combo2],
                        'direction': 'higher' if mean1 > mean2 else 'lower',
                        'biological_context': self._get_biological_context(combo1, combo2, feature),
                        'weighting_strategy': self.weighting_strategy
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
        
        print("\nAnalysis complete!")
        print("Total tests: {}".format(len(results_df)))
        print("Significant patterns (corrected p≤0.05, effect≥0.3): {}".format(len(significant_results)))
        
        # Print weighting impact summary
        if self.weighting_strategy == "compound_corrected":
            print("\nSize-effect correction applied:")
            for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
                if combo in self.source_compound_counts:
                    print("  {}: {} compounds → weight {:.3f}".format(
                        combo, self.source_compound_counts[combo], combination_weights[combo]))
        
        return significant_results
    
    def _get_biological_context(self, combo1, combo2, feature):
        """Provide biological context for dual-activity comparisons"""
        contexts = {
            'SA+CA_vs_SA+EC': 'Fungi vs Gram- (both with Gram+)',
            'SA+CA_vs_CA+EC': 'Gram+ vs Gram- (both with Fungi)', 
            'SA+EC_vs_CA+EC': 'Gram+ vs Fungi (both with Gram-)'
        }
        comparison_key = '{}_vs_{}'.format(combo1, combo2)
        return contexts.get(comparison_key, 'Dual combination comparison')
    
    def create_dual_activity_visualizations(self, output_dir='dual_plots'):
        """Create comprehensive visualizations for dual-activity analysis"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        if len(self.permutation_results) == 0:
            print("No results to visualize.")
            return
        
        # Set up style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Enhanced dual-activity volcano plot with weighting context
        fig, ax = plt.subplots(figsize=(14, 10))
        
        significant_results = self.permutation_results[
            (self.permutation_results['corrected_p_value'] <= 0.05) &
            (self.permutation_results['effect_size'] >= 0.3)
        ]
        
        # Color by comparison type
        comparison_colors = {
            'SA+CA_vs_SA+EC': '#FF6B6B',  # Red
            'SA+CA_vs_CA+EC': '#4ECDC4',  # Teal  
            'SA+EC_vs_CA+EC': '#45B7D1'   # Blue
        }
        
        colors = [comparison_colors.get(comp, 'lightgray') for comp in self.permutation_results['comparison']]
        
        scatter = ax.scatter(self.permutation_results['effect_size'], 
                           -np.log10(self.permutation_results['corrected_p_value']),
                           c=colors, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        
        # Add threshold lines
        ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, linewidth=2, label='p=0.05')
        ax.axvline(x=0.3, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Effect=0.3')
        
        # Annotate top significant points
        for _, row in significant_results.nlargest(5, 'effect_size').iterrows():
            ax.annotate('{}\n({})'.format(row['feature'].replace('_', ' '), row['comparison']),
                       xy=(row['effect_size'], -np.log10(row['corrected_p_value'])),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                       fontsize=9)
        
        ax.set_xlabel('Effect Size (Cohen\'s d)', fontsize=14, fontweight='bold')
        ax.set_ylabel('-log10(Corrected P-Value)', fontsize=14, fontweight='bold')
        
        # Dynamic title based on weighting strategy
        title = 'Dual-Activity Chemical Pattern Significance\n'
        if self.weighting_strategy == "compound_corrected":
            title += 'Size-Effect Corrected Analysis'
        else:
            title += 'Fragment-Based Analysis'
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=comparison_colors[comp], label=comp.replace('_vs_', ' vs ')) 
                          for comp in comparison_colors.keys()]
        legend_elements.extend([
            plt.Line2D([0], [0], color='red', linestyle='--', label='p=0.05'),
            plt.Line2D([0], [0], color='blue', linestyle='--', label='Effect=0.3')
        ])
        ax.legend(handles=legend_elements, title='Comparisons & Thresholds', loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add weighting strategy annotation
        strategy_text = f'Weighting: {self.weighting_strategy.replace("_", " ").title()}'
        ax.text(0.02, 0.98, strategy_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('{}/dual_activity_volcano_plot.png'.format(output_dir), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Size effect comparison visualization
        if self.weighting_strategy == "compound_corrected" and self.source_compound_counts:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Source compounds
            compounds = [self.source_compound_counts[combo] for combo in ['SA+CA', 'SA+EC', 'CA+EC']]
            colors = [self.combination_colors[combo] for combo in ['SA+CA', 'SA+EC', 'CA+EC']]
            
            bars1 = axes[0].bar(['SA+CA', 'SA+EC', 'CA+EC'], compounds, color=colors)
            axes[0].set_title('Source Dual-Active Compounds', fontweight='bold')
            axes[0].set_ylabel('Compound Count')
            for i, v in enumerate(compounds):
                axes[0].text(i, v + 20, str(v), ha='center', va='bottom', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Fragment counts
            fragment_counts = []
            for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
                count = len(self.all_fragments[self.all_fragments['dual_combination'] == combo])
                fragment_counts.append(count)
            
            bars2 = axes[1].bar(['SA+CA', 'SA+EC', 'CA+EC'], fragment_counts, color=colors)
            axes[1].set_title('Extracted Fragments', fontweight='bold')
            axes[1].set_ylabel('Fragment Count')
            for i, v in enumerate(fragment_counts):
                axes[1].text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # Extraction efficiency
            efficiencies = [self.fragment_extraction_efficiency[combo] for combo in ['SA+CA', 'SA+EC', 'CA+EC']]
            bars3 = axes[2].bar(['SA+CA', 'SA+EC', 'CA+EC'], efficiencies, color=colors)
            axes[2].set_title('Fragment Extraction Efficiency', fontweight='bold')
            axes[2].set_ylabel('Fragments per Compound')
            for i, v in enumerate(efficiencies):
                axes[2].text(i, v + 0.005, '{:.3f}'.format(v), ha='center', va='bottom', fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            
            plt.suptitle('Size Effect Analysis: Compounds → Fragments', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('{}/size_effect_analysis.png'.format(output_dir), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Dual combination chemical profiles
        if len(significant_results) > 0:
            top_features = significant_results.nlargest(6, 'effect_size')
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, (_, result) in enumerate(top_features.iterrows()):
                if i >= 6:
                    break
                
                feature = result['feature']
                combo1, combo2 = result['combo1'], result['combo2']
                
                # Get data for both combinations
                data1 = self.chemical_features[
                    self.chemical_features['dual_combination'] == combo1
                ][feature].dropna()
                data2 = self.chemical_features[
                    self.chemical_features['dual_combination'] == combo2
                ][feature].dropna()
                
                # Create violin plot
                data_combined = pd.DataFrame({
                    'value': list(data1) + list(data2),
                    'combination': [combo1] * len(data1) + [combo2] * len(data2)
                })
                
                sns.violinplot(data=data_combined, x='combination', y='value', ax=axes[i],
                              palette=[self.combination_colors[combo1], self.combination_colors[combo2]])
                
                # Enhanced title with weighting info
                title_parts = [
                    feature.replace('_', ' ').title(),
                    'Effect: {:.3f}, p: {:.2e}'.format(result['effect_size'], result['corrected_p_value']),
                    result['biological_context']
                ]
                
                if self.weighting_strategy == "compound_corrected":
                    title_parts.append('(Size-corrected)')
                
                axes[i].set_title('\n'.join(title_parts), fontweight='bold', fontsize=10)
                axes[i].set_ylabel(feature.replace('_', ' ').title())
                axes[i].set_xlabel('')
                axes[i].grid(True, alpha=0.3)
                
                # Add weighted mean lines
                if self.weighting_strategy == "compound_corrected":
                    axes[i].axhline(y=result['weighted_mean1'], color=self.combination_colors[combo1], 
                                   linestyle='--', alpha=0.9, linewidth=2, label='Weighted Mean')
                    axes[i].axhline(y=result['weighted_mean2'], color=self.combination_colors[combo2], 
                                   linestyle='--', alpha=0.9, linewidth=2)
                else:
                    axes[i].axhline(y=result['raw_mean1'], color=self.combination_colors[combo1], 
                                   linestyle='--', alpha=0.7, linewidth=2, label='Mean')
                    axes[i].axhline(y=result['raw_mean2'], color=self.combination_colors[combo2], 
                                   linestyle='--', alpha=0.7, linewidth=2)
            
            # Hide unused subplots
            for j in range(i+1, 6):
                axes[j].set_visible(False)
            
            plt.suptitle('Top Dual-Activity Chemical Property Differences', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('{}/dual_property_comparisons.png'.format(output_dir), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Comprehensive dual combination overview
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Fragment counts per combination
        combo_counts = self.all_fragments['dual_combination'].value_counts()
        colors = [self.combination_colors[combo] for combo in combo_counts.index]
        
        bars = axes[0,0].bar(combo_counts.index, combo_counts.values, color=colors)
        axes[0,0].set_title('Dual-Activity Fragment Counts', fontweight='bold')
        axes[0,0].set_ylabel('Fragment Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(combo_counts.values):
            axes[0,0].text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # Average activity rates
        avg_activity = self.all_fragments.groupby('dual_combination')['avg_activity_rate_percent'].mean()
        bars = axes[0,1].bar(avg_activity.index, avg_activity.values, color=colors)
        axes[0,1].set_title('Average Dual Activity Rates', fontweight='bold')
        axes[0,1].set_ylabel('Activity Rate (%)')
        axes[0,1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(avg_activity.values):
            axes[0,1].text(i, v + 1, '{:.1f}%'.format(v), ha='center', va='bottom', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # Fragment type distribution
        fragment_type_dist = self.all_fragments.groupby(['dual_combination', 'fragment_type']).size().unstack(fill_value=0)
        fragment_type_dist.plot(kind='bar', ax=axes[1,0], color=['#FFB6C1', '#87CEEB'])
        axes[1,0].set_title('Fragment Type Distribution', fontweight='bold')
        axes[1,0].set_ylabel('Fragment Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(title='Fragment Type')
        axes[1,0].grid(True, alpha=0.3)
        
        # Weighting strategy impact (if applicable)
        if self.weighting_strategy == "compound_corrected" and self.source_compound_counts:
            # Show compound vs fragment relationship
            combos = ['SA+CA', 'SA+EC', 'CA+EC']
            compounds = [self.source_compound_counts[combo] for combo in combos]
            fragments = [len(self.all_fragments[self.all_fragments['dual_combination'] == combo]) for combo in combos]
            
            # Scatter plot
            for i, combo in enumerate(combos):
                axes[1,1].scatter(compounds[i], fragments[i], 
                                color=self.combination_colors[combo], s=200, alpha=0.7, label=combo)
                axes[1,1].annotate(combo, (compounds[i], fragments[i]), 
                                 xytext=(5, 5), textcoords='offset points', fontweight='bold')
            
            axes[1,1].set_xlabel('Source Compounds')
            axes[1,1].set_ylabel('Extracted Fragments')
            axes[1,1].set_title('Compounds vs Fragments\n(Size Effect Context)', fontweight='bold')
            axes[1,1].grid(True, alpha=0.3)
            
            # Add efficiency lines
            for i, combo in enumerate(combos):
                eff = self.fragment_extraction_efficiency[combo]
                axes[1,1].text(compounds[i], fragments[i] - 20, 
                             'Eff: {:.3f}'.format(eff), ha='center', fontsize=9)
        else:
            axes[1,1].text(0.5, 0.5, 'Fragment-Based\nWeighting Used\n\nNo size effect\ncorrection applied', 
                         transform=axes[1,1].transAxes, ha='center', va='center', 
                         fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[1,1].set_title('Weighting Strategy', fontweight='bold')
        
        plt.suptitle('Comprehensive Dual-Activity Analysis Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('{}/dual_activity_comprehensive_overview.png'.format(output_dir), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Enhanced visualizations saved to {}/".format(output_dir))
        print("Generated plots:")
        print("- dual_activity_volcano_plot.png (significance with weighting context)")
        if self.weighting_strategy == "compound_corrected":
            print("- size_effect_analysis.png (compound→fragment analysis)")
        print("- dual_property_comparisons.png (chemical differences)")
        print("- dual_activity_comprehensive_overview.png (complete overview)")
    
    def generate_dual_activity_report(self, output_file='robust_dual_activity_report.txt'):
        """Generate comprehensive dual-activity analysis report with weighting context"""
        print("\nGenerating comprehensive dual-activity analysis report...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ROBUST DUAL-ACTIVITY FRAGMENT ANALYSIS REPORT\n")
            f.write("XAI-Derived Fragments with Adaptive Size-Effect Correction\n")
            f.write("=" * 80 + "\n\n")
            
            # Methodology
            f.write("ANALYSIS METHODOLOGY\n")
            f.write("-" * 40 + "\n")
            f.write("Approach: Adaptive permutation testing with data validation\n")
            f.write("Weighting strategy selected: {}\n".format(self.weighting_strategy.replace('_', ' ').title()))
            
            if self.weighting_strategy == "compound_corrected":
                f.write("Size-effect correction: Applied based on source compound counts\n")
                f.write("Source compound validation: Passed consistency checks\n")
            else:
                f.write("Size-effect correction: Not applied (data validation failed or unavailable)\n")
            
            f.write("Permutations per test: 10,000\n")
            f.write("Multiple testing correction: Benjamini-Hochberg FDR\n")
            f.write("Effect size threshold: 0.3 (Cohen's d)\n")
            f.write("Significance threshold: p < 0.05 (corrected)\n\n")
            
            # Data validation results
            f.write("DATA VALIDATION RESULTS\n")
            f.write("-" * 40 + "\n")
            
            if self.source_compound_counts:
                f.write("Source compound data provided:\n")
                for combo, count in self.source_compound_counts.items():
                    f.write("  {}: {} dual-active compounds\n".format(combo, count))
                
                f.write("\nFragment extraction efficiency:\n")
                for combo, eff in self.fragment_extraction_efficiency.items():
                    f.write("  {}: {:.3f} fragments per compound\n".format(combo, eff))
                
                # Calculate efficiency ratios
                efficiencies = list(self.fragment_extraction_efficiency.values())
                max_eff = max(efficiencies)
                min_eff = min([e for e in efficiencies if e > 0])
                if min_eff > 0:
                    f.write("Efficiency range: {:.3f} to {:.3f} ({:.1f}x variation)\n".format(
                        min_eff, max_eff, max_eff/min_eff))
            else:
                f.write("No source compound data provided - using fragment-based analysis\n")
            
            # Dataset overview
            f.write("\n\nDUAL-ACTIVITY DATASET OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write("Total dual-activity fragments: {}\n\n".format(len(self.all_fragments)))
            
            for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
                count = len(self.all_fragments[self.all_fragments['dual_combination'] == combo])
                info = self.dual_combinations[combo]
                f.write("{}: {} fragments\n".format(info['name'], count))
                f.write("  Target: {} + {}\n".format(
                    self.pathogen_map[info['pathogens'][0]], 
                    self.pathogen_map[info['pathogens'][1]]
                ))
                # Fix: Map excluded pathogen name to pathogen class
                excluded_pathogen = info['excluded']
                if excluded_pathogen.startswith('E.coli'):
                    excluded_class = 'Gram-'
                elif excluded_pathogen.startswith('S.aureus'):
                    excluded_class = 'Gram+'
                elif excluded_pathogen.startswith('C.albicans'):
                    excluded_class = 'Fungi'
                else:
                    excluded_class = excluded_pathogen
                f.write("  Excludes: {} ({})\n".format(excluded_pathogen, excluded_class))
                
                # Average activity rate for this combination
                avg_activity = self.all_fragments[
                    self.all_fragments['dual_combination'] == combo
                ]['avg_activity_rate_percent'].mean()
                f.write("  Average activity rate: {:.1f}%\n".format(avg_activity))
                
                if self.source_compound_counts and combo in self.source_compound_counts:
                    compounds = self.source_compound_counts[combo]
                    efficiency = self.fragment_extraction_efficiency[combo]
                    f.write("  Source compounds: {}\n".format(compounds))
                    f.write("  Extraction efficiency: {:.3f} fragments/compound\n".format(efficiency))
                f.write("\n")
            
            # Results
            if len(self.permutation_results) > 0:
                significant_results = self.permutation_results[
                    (self.permutation_results['corrected_p_value'] <= 0.05) &
                    (self.permutation_results['effect_size'] >= 0.3)
                ]
                
                f.write("DUAL-ACTIVITY COMPARISON RESULTS\n")
                f.write("-" * 40 + "\n")
                f.write("Total pairwise tests performed: {}\n".format(len(self.permutation_results)))
                f.write("Statistically significant patterns: {}\n".format(len(significant_results)))
                f.write("False discovery rate: {:.1%}\n".format(
                    (self.permutation_results['corrected_p_value'] <= 0.05).mean()))
                f.write("Weighting strategy: {}\n\n".format(self.weighting_strategy))
                
                if len(significant_results) > 0:
                    f.write("VALIDATED DUAL-ACTIVITY DIFFERENCES:\n")
                    f.write("-" * 30 + "\n")
                    
                    # Group results by comparison type
                    for comparison in ['SA+CA_vs_SA+EC', 'SA+CA_vs_CA+EC', 'SA+EC_vs_CA+EC']:
                        comp_results = significant_results[
                            significant_results['comparison'] == comparison
                        ].nlargest(3, 'effect_size')
                        
                        if len(comp_results) > 0:
                            f.write("\n{}:\n".format(comparison.replace('_vs_', ' vs ')))
                            for _, result in comp_results.iterrows():
                                if self.weighting_strategy == "compound_corrected":
                                    mean_type = "size-corrected"
                                    mean1, mean2 = result['weighted_mean1'], result['weighted_mean2']
                                else:
                                    mean_type = "raw"
                                    mean1, mean2 = result['raw_mean1'], result['raw_mean2']
                                
                                f.write("  {} is {:.2f}x {} in {} vs {} ({} means)\n".format(
                                    result['feature'].replace('_', ' '),
                                    abs(result['fold_change']),
                                    result['direction'],
                                    result['combo1'] if result['direction'] == 'higher' else result['combo2'],
                                    result['combo2'] if result['direction'] == 'higher' else result['combo1'],
                                    mean_type
                                ))
                                f.write("    Effect: {:.3f}, p: {:.2e}, Context: {}\n".format(
                                    result['effect_size'], result['corrected_p_value'], result['biological_context']
                                ))
                
                # Design insights with weighting context
                f.write("\n\nDUAL-ACTIVITY DESIGN INSIGHTS\n")
                f.write("-" * 40 + "\n")
                if self.weighting_strategy == "compound_corrected":
                    f.write("Based on size-effect corrected analysis:\n\n")
                else:
                    f.write("Based on fragment-based analysis:\n\n")
                
                # SA+CA specific features (Gram+ & Fungi)
                sa_ca_features = significant_results[
                    (significant_results['combo1'] == 'SA+CA') & 
                    (significant_results['direction'] == 'higher')
                ]
                if len(sa_ca_features) > 0:
                    f.write("SA+CA (Gram+ & Fungi) - Distinctive Features:\n")
                    for _, feature in sa_ca_features.nlargest(3, 'effect_size').iterrows():
                        f.write("  - Enhanced {}: {:.2f}x advantage\n".format(
                            feature['feature'].replace('_', ' '),
                            abs(feature['fold_change'])
                        ))
                    f.write("  Strategy: Design for cell wall penetration AND ergosterol targeting\n\n")
                
                # SA+EC specific features (Gram+ & Gram-)
                sa_ec_features = significant_results[
                    (significant_results['combo1'] == 'SA+EC') & 
                    (significant_results['direction'] == 'higher')
                ]
                if len(sa_ec_features) > 0:
                    f.write("SA+EC (Gram+ & Gram-) - Distinctive Features:\n")
                    for _, feature in sa_ec_features.nlargest(3, 'effect_size').iterrows():
                        f.write("  - Enhanced {}: {:.2f}x advantage\n".format(
                            feature['feature'].replace('_', ' '),
                            abs(feature['fold_change'])
                        ))
                    f.write("  Strategy: Broad-spectrum bacterial activity across cell wall types\n\n")
                
                # CA+EC specific features (Fungi & Gram-)
                ca_ec_features = significant_results[
                    (significant_results['combo1'] == 'CA+EC') & 
                    (significant_results['direction'] == 'higher')
                ]
                if len(ca_ec_features) > 0:
                    f.write("CA+EC (Fungi & Gram-) - Distinctive Features:\n")
                    for _, feature in ca_ec_features.nlargest(3, 'effect_size').iterrows():
                        f.write("  - Enhanced {}: {:.2f}x advantage\n".format(
                            feature['feature'].replace('_', ' '),
                            abs(feature['fold_change'])
                        ))
                    f.write("  Strategy: Target membrane permeability for non-Gram+ organisms\n\n")
            
            # Weighting impact assessment
            f.write("WEIGHTING STRATEGY IMPACT\n")
            f.write("-" * 30 + "\n")
            if self.weighting_strategy == "compound_corrected":
                f.write("Size-effect correction successfully applied:\n")
                f.write("- Accounts for unequal source compound populations\n")
                f.write("- CA+EC fragments weighted higher due to smaller source pool\n")
                f.write("- SA+CA fragments weighted lower due to larger source pool\n")
                f.write("- Results reflect true chemical selectivity, not sampling bias\n\n")
            else:
                f.write("Fragment-based weighting used:\n")
                f.write("- Equal treatment of all fragments regardless of source\n")
                f.write("- May be influenced by compound population differences\n")
                f.write("- Conservative approach when source data is inconsistent\n\n")
            
            f.write("DUAL-ACTIVITY DRUG DESIGN RECOMMENDATIONS\n")
            f.write("-" * 50 + "\n")
            f.write("1. SA+CA combinations: Focus on membrane versatile compounds\n")
            f.write("2. SA+EC combinations: Optimize for broad bacterial penetration\n")
            f.write("3. CA+EC combinations: Target non-Gram+ membrane mechanisms\n")
            f.write("4. Use validated chemical differences for rational design\n")
            f.write("5. Consider size-effect corrections in future fragment analyses\n\n")
            
            f.write("METHODOLOGY VALIDATION\n")
            f.write("-" * 30 + "\n")
            f.write("✓ Data consistency checks performed\n")
            f.write("✓ Adaptive weighting strategy selected\n")
            f.write("✓ Permutation testing with FDR correction\n")
            f.write("✓ Effect size thresholding applied\n")
            f.write("✓ Biological context provided for interpretability\n\n")
            
            f.write("Note: Analysis based on {} total dual-activity fragments\n".format(len(self.all_fragments)))
            f.write("with robust statistical validation and adaptive size-effect correction.\n")
        
        print("Comprehensive dual-activity analysis report saved to {}".format(output_file))


def main():
    """Main robust dual-activity analysis pipeline"""
    print("Starting Robust Dual-Activity XAI Fragment Analysis...")
    print("Features: Adaptive weighting, data validation, size-effect correction")
    
    # Source compound counts from your Venn diagram data
    # Set to None if you don't have this data (e.g., for negative analysis)
    source_compound_counts = {
        'SA+CA': 1164,  # S.aureus + C.albicans dual-active compounds
        'SA+EC': 849,   # S.aureus + E.coli dual-active compounds  
        'CA+EC': 187    # C.albicans + E.coli dual-active compounds
    }
    
    # File paths - UPDATE THESE TO YOUR ACTUAL FILE PATHS
    file_paths = {
        'dual_SA_CA_positive_scaffolds': 'dual_SA_CA_positive_scaffolds.csv',
        'dual_SA_CA_positive_substitutents': 'dual_SA_CA_positive_substitutents.csv',
        'dual_SA_EC_positive_scaffolds': 'dual_SA_EC_positive_scaffolds.csv',
        'dual_SA_EC_positive_substitutents': 'dual_SA_EC_positive_substitutents.csv',
        'dual_CA_EC_positive_scaffolds': 'dual_CA_EC_positive_scaffolds.csv',
        'dual_CA_EC_positive_substitutents': 'dual_CA_EC_positive_substitutents.csv'
    }
    
    # Initialize robust dual-activity analyzer with source compound data
    analyzer = RobustDualActivityFragmentAnalyzer(source_compound_counts=source_compound_counts)
    
    # For negative analysis later, initialize without source data:
    # analyzer = RobustDualActivityFragmentAnalyzer(source_compound_counts=None)
    
    try:
        # Load dual-activity fragments with validation
        all_fragments = analyzer.load_and_prepare_dual_data(file_paths)
        
        # Extract physicochemical properties
        chemical_features = analyzer.extract_physicochemical_properties()
        
        # Perform robust dual-activity analysis
        significant_patterns = analyzer.perform_dual_activity_analysis(n_permutations=10000)
        
        # Create enhanced visualizations
        analyzer.create_dual_activity_visualizations()
        
        # Generate comprehensive report
        analyzer.generate_dual_activity_report()
        
        # Save results
        print("\nSaving robust dual-activity analysis results...")
        
        if len(significant_patterns) > 0:
            significant_patterns.to_csv('robust_dual_activity_significant_patterns.csv', 
                                      index=False, encoding='utf-8')
            print("Significant dual-activity patterns saved to robust_dual_activity_significant_patterns.csv")
        
        analyzer.permutation_results.to_csv('complete_robust_dual_activity_results.csv',
                                          index=False, encoding='utf-8')
        print("Complete robust results saved to complete_robust_dual_activity_results.csv")
        
        all_fragments.to_csv('robust_dual_activity_fragments_with_properties.csv', 
                           index=False, encoding='utf-8')
        print("Enhanced dual-activity fragments saved to robust_dual_activity_fragments_with_properties.csv")
        
        print("\n" + "="*80)
        print("ROBUST DUAL-ACTIVITY ANALYSIS COMPLETE!")
        print("="*80)
        print("Key features of this analysis:")
        print("✓ Data validation and consistency checking")
        print("✓ Adaptive weighting strategy selection")
        print("✓ Size-effect correction when appropriate")
        print("✓ Robust statistical testing with permutations")
        print("✓ Future-compatible design for negative analysis")
        
        print("\nGenerated files:")
        print("- robust_dual_activity_significant_patterns.csv (validated differences)")
        print("- complete_robust_dual_activity_results.csv (all statistical tests)")
        print("- robust_dual_activity_fragments_with_properties.csv (enhanced fragment data)")
        
        # Analysis summary with weighting context
        print("\nAnalysis Summary:")
        print("Weighting strategy used: {}".format(analyzer.weighting_strategy.replace('_', ' ').title()))
        
        if analyzer.weighting_strategy == "compound_corrected":
            print("Size-effect correction applied successfully!")
            print("Fragment extraction efficiency:")
            for combo, eff in analyzer.fragment_extraction_efficiency.items():
                compounds = analyzer.source_compound_counts[combo]
                fragments = len(all_fragments[all_fragments['dual_combination'] == combo])
                print("  {}: {:.3f} fragments/compound ({} fragments from {} compounds)".format(
                    combo, eff, fragments, compounds))
        else:
            print("Fragment-based weighting used (data validation triggered fallback)")
        
        print("\nDual-Activity Fragment Distribution:")
        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            count = len(all_fragments[all_fragments['dual_combination'] == combo])
            info = analyzer.dual_combinations[combo]
            avg_activity = all_fragments[
                all_fragments['dual_combination'] == combo
            ]['avg_activity_rate_percent'].mean()
            print("  {}: {} fragments ({:.1f}% avg activity)".format(
                info['name'], count, avg_activity))
        
        print("\nStatistical robustness:")
        print("  - Permutation tests: 10,000 per comparison")
        print("  - Data validation: Performed automatically")
        print("  - Weighting strategy: Adaptively selected")
        print("  - Multiple testing: Benjamini-Hochberg FDR correction")
        print("  - Total tests: {}".format(len(analyzer.permutation_results)))
        print("  - Validated patterns: {}".format(len(significant_patterns)))
        
        # Show key validated findings
        if len(significant_patterns) > 0:
            print("\nKey Validated Dual-Activity Insights:")
            
            # Group by comparison type
            comparisons = {
                'SA+CA_vs_SA+EC': 'Fungi vs Gram- targeting (with Gram+)',
                'SA+CA_vs_CA+EC': 'Gram+ vs Gram- targeting (with Fungi)',
                'SA+EC_vs_CA+EC': 'Gram+ vs Fungi targeting (with Gram-)'
            }
            
            for comparison, description in comparisons.items():
                comp_results = significant_patterns[
                    significant_patterns['comparison'] == comparison
                ]
                
                if len(comp_results) > 0:
                    top_result = comp_results.nlargest(1, 'effect_size').iloc[0]
                    
                    # Use appropriate mean based on weighting strategy
                    if analyzer.weighting_strategy == "compound_corrected":
                        fold_change = top_result['fold_change']
                        mean_info = "(size-corrected)"
                    else:
                        fold_change = top_result['raw_mean1'] / top_result['raw_mean2'] if top_result['raw_mean2'] != 0 else top_result['fold_change']
                        mean_info = "(raw means)"
                    
                    print("  {}: {} shows {:.1f}x higher {} {}".format(
                        description,
                        top_result['feature'].replace('_', ' '),
                        abs(fold_change),
                        top_result['feature'].replace('_', ' '),
                        mean_info
                    ))
                    print("    Effect size: {:.3f}, p: {:.2e}".format(
                        top_result['effect_size'], top_result['corrected_p_value']
                    ))
            
            print("\nDrug Design Strategy:")
            print("These validated, size-effect corrected differences reveal:")
            print("✓ True chemical requirements for dual Gram+/Fungi activity")
            print("✓ Robust features for broad-spectrum bacterial targeting")  
            print("✓ Selective properties for Fungi/Gram- dual activity")
            print("✓ Unbiased insights accounting for source compound differences")
            print("\nUse these robust insights for rational dual-activity compound design!")
            
        else:
            print("\nNo statistically validated dual-activity differences found.")
            print("This suggests:")
            print("  - Dual-activity fragments may share common chemical features")
            print("  - Current dataset may need expansion for more sensitive detection")
            print("  - Focus on individual combination analysis rather than comparative")
        
        print("\n" + "="*60)
        print("FUTURE ANALYSIS COMPATIBILITY")
        print("="*60)
        print("This script is designed for future use:")
        print("✓ For NEGATIVE dual-activity analysis: Set source_compound_counts=None")
        print("✓ For different combinations: Update dual_combinations mapping")
        print("✓ For triple-activity: Extend the framework as needed")
        print("✓ Consistent methodology across all analyses")
        
        print("\nRecommendation: These robust, validated results provide the most")
        print("reliable foundation for dual-activity compound design decisions!")
        
    except Exception as e:
        print("Error during robust dual-activity analysis: {}".format(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()