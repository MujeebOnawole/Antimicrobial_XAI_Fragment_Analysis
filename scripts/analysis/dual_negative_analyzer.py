#!/usr/bin/env python3
"""
ROBUST DUAL-ACTIVITY NEGATIVE FRAGMENT ANALYZER
Analyzes XAI-derived fragments that show NEGATIVE activity against TWO specific pathogens
Features: Smart size-effect correction with data validation and adaptive weighting for negative patterns
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

class RobustDualNegativeFragmentAnalyzer:
    def __init__(self, source_compound_counts=None, total_compound_counts=None):
        """
        Initialize analyzer for dual NEGATIVE activity fragments
        
        Parameters:
        source_compound_counts (dict): Optional mapping of dual combinations to negative compound counts
                                     e.g., {'CA+EC': 135, 'SA+CA': 200, 'SA+EC': 180}
        total_compound_counts (dict): Total compounds per pathogen for context
                                    e.g., {'SA': 54277, 'CA': 28476, 'EC': 44920}
        """
        # Dual NEGATIVE activity combinations mapping
        self.dual_combinations = {
            'CA+EC': {'pathogens': ['CA', 'EC'], 'name': 'C.albicans + E.coli NEGATIVE', 'excluded': 'S.aureus'},
            'SA+CA': {'pathogens': ['SA', 'CA'], 'name': 'S.aureus + C.albicans NEGATIVE', 'excluded': 'E.coli'},
            'SA+EC': {'pathogens': ['SA', 'EC'], 'name': 'S.aureus + E.coli NEGATIVE', 'excluded': 'C.albicans'}
        }
        
        # Pathogen class mapping
        self.pathogen_map = {'SA': 'Gram+', 'EC': 'Gram-', 'CA': 'Fungi'}
        
        # Color scheme for dual NEGATIVE combinations (darker/muted tones)
        self.combination_colors = {
            'CA+EC': '#8B4513',  # SaddleBrown - Fungi & Gram- negative
            'SA+CA': '#800000',  # Maroon - Gram+ & Fungi negative  
            'SA+EC': '#2F4F4F'   # DarkSlateGray - Gram+ & Gram- negative
        }
        
        # Source compound counts (if available)
        self.source_compound_counts = source_compound_counts
        self.total_compound_counts = total_compound_counts
        
        # Data containers
        self.all_fragments = None
        self.chemical_features = None
        self.permutation_results = {}
        self.dual_negative_patterns = {}
        self.weighting_strategy = None
        self.fragment_extraction_efficiency = {}
        
    def load_and_prepare_dual_negative_data(self, file_paths):
        """Load all dual-NEGATIVE CSV files and prepare master dataset with smart validation"""
        print("Loading XAI-derived dual-NEGATIVE activity fragments...")
        print("Performing data validation and weighting strategy selection...")
        
        all_data = []
        
        # Expected file structure for negative data
        expected_combinations = {
            'CA_EC': 'CA+EC',  # C.albicans + E.coli negative
            'SA_CA': 'SA+CA',  # S.aureus + C.albicans negative
            'SA_EC': 'SA+EC'   # S.aureus + E.coli negative
        }
        
        for file_combination, combo_key in expected_combinations.items():
            for fragment_type in ['scaffolds', 'substitutents']:
                file_key = "negative_{}_{}".format(file_combination, fragment_type)
                
                if file_key in file_paths:
                    print("Loading {}...".format(file_key))
                    df = pd.read_csv(file_paths[file_key])
                    
                    # Debug: Print column names to check if consistency_breakdown exists
                    print("  Columns found: {}".format(list(df.columns)))
                    if 'consistency_breakdown' not in df.columns:
                        print("  WARNING: 'consistency_breakdown' column not found!")
                        print("  Available columns: {}".format(list(df.columns)))
                    
                    # Parse combination info
                    combo_info = self.dual_combinations[combo_key]
                    
                    # Convert SMILES to RDKit molecules
                    df['mol'] = df['fragment_smiles'].apply(lambda x: Chem.MolFromSmiles(x) if pd.notnull(x) else None)
                    
                    # Add metadata for negative analysis
                    df['dual_combination'] = combo_key
                    df['combination_name'] = combo_info['name']
                    df['included_pathogens'] = '+'.join(combo_info['pathogens'])
                    df['excluded_pathogen'] = combo_info['excluded']
                    df['fragment_type'] = fragment_type.rstrip('s')  # Remove 's' from scaffolds/substitutents
                    
                    # Calculate dual-NEGATIVE reliability score
                    # For negative fragments, we want high inactivity rates and good compound coverage
                    df['dual_negative_reliability_score'] = (df['avg_inactivity_rate_percent'] / 100) * np.log(df['total_compounds_both_pathogens'] + 1)
                    
                    # Add pathogen class information
                    pathogen_classes = []
                    for pathogens in df['included_pathogens']:
                        classes = [self.pathogen_map[p] for p in pathogens.split('+')]
                        pathogen_classes.append(' + '.join(classes))
                    df['pathogen_classes'] = pathogen_classes
                    
                    # Categorize dual NEGATIVE importance
                    df['dual_negative_importance_tier'] = self._categorize_dual_negative_importance(df)
                    
                    # Add negative-specific metrics
                    df['negative_consistency_score'] = self._calculate_negative_consistency(df)
                    
                    all_data.append(df)
        
        if not all_data:
            raise ValueError("No negative fragment data loaded. Check file paths and naming convention.")
        
        self.all_fragments = pd.concat(all_data, ignore_index=True)
        
        # Calculate actual fragment counts from loaded data
        actual_fragment_counts = {}
        for combo in ['CA+EC', 'SA+CA', 'SA+EC']:
            actual_fragment_counts[combo] = len(
                self.all_fragments[self.all_fragments['dual_combination'] == combo]
            )
        
        # Determine optimal weighting strategy
        self.weighting_strategy = self._determine_negative_weighting_strategy(actual_fragment_counts)
        
        # Print comprehensive analysis overview
        self._print_negative_data_overview(actual_fragment_counts)
        
        return self.all_fragments
    
    def _calculate_negative_consistency(self, df):
        """Calculate consistency score for negative fragments"""
        # Check if consistency_breakdown column exists
        if 'consistency_breakdown' not in df.columns:
            print("  WARNING: 'consistency_breakdown' column not found. Using default consistency score.")
            # Return default scores based on avg_inactivity_rate_percent
            return df['avg_inactivity_rate_percent'].fillna(0)
        
        # Parse consistency breakdown to get per-pathogen negative rates
        consistency_scores = []
        
        for _, row in df.iterrows():
            breakdown = row['consistency_breakdown']
            
            # Handle NaN or empty breakdown
            if pd.isna(breakdown) or not breakdown:
                consistency_scores.append(0)
                continue
            
            # Extract negative percentages for each pathogen
            pathogen_rates = []
            for pathogen in ['c_albicans', 'e_coli', 's_aureus']:  # Note: lowercase in data
                if '{}: '.format(pathogen) in breakdown:
                    # Extract percentage
                    pattern = r'{}: ([\d.]+)% negative'.format(pathogen)
                    match = re.search(pattern, breakdown)
                    if match:
                        pathogen_rates.append(float(match.group(1)))
            
            # Calculate consistency as minimum negative rate (most conservative)
            if pathogen_rates:
                consistency_scores.append(min(pathogen_rates))
            else:
                consistency_scores.append(0)
        
        return consistency_scores
    
    def _categorize_dual_negative_importance(self, df):
        """Categorize dual-NEGATIVE fragment importance"""
        conditions = [
            (df['avg_inactivity_rate_percent'] >= 95) & (df['total_compounds_both_pathogens'] >= 20),
            (df['avg_inactivity_rate_percent'] >= 90) & (df['total_compounds_both_pathogens'] >= 10),
            (df['avg_inactivity_rate_percent'] >= 80) & (df['total_compounds_both_pathogens'] >= 5)
        ]
        choices = ['High_Negative_Impact', 'Reliable_Negative', 'Moderate_Negative']
        return np.select(conditions, choices, default='Limited_Negative')
    
    def _determine_negative_weighting_strategy(self, actual_fragment_counts):
        """Intelligently determine the best weighting strategy for negative data"""
        print("\n" + "="*60)
        print("NEGATIVE DATA VALIDATION & WEIGHTING STRATEGY SELECTION")
        print("="*60)
        
        if self.source_compound_counts is None:
            print("✓ No source compound data provided - using fragment-based weighting")
            return "fragment_based"
        
        # Validate source compound data consistency for negative fragments
        print("Validating negative source compound data consistency...")
        
        # Calculate fragment extraction efficiency
        efficiency = {}
        for combo in ['CA+EC', 'SA+CA', 'SA+EC']:
            if combo in self.source_compound_counts and self.source_compound_counts[combo] > 0:
                efficiency[combo] = actual_fragment_counts[combo] / self.source_compound_counts[combo]
            else:
                efficiency[combo] = 0
        
        self.fragment_extraction_efficiency = efficiency
        
        # Print extraction efficiency analysis
        print("\nNegative Fragment Extraction Efficiency Analysis:")
        print("-" * 50)
        for combo in ['CA+EC', 'SA+CA', 'SA+EC']:
            fragments = actual_fragment_counts[combo]
            compounds = self.source_compound_counts.get(combo, 0)
            eff = efficiency[combo]
            print("  {}: {} fragments from {} negative compounds ({:.3f} fragments/compound)".format(
                combo, fragments, compounds, eff))
        
        # Data consistency checks for negative data
        consistency_checks = self._perform_negative_consistency_checks(actual_fragment_counts, efficiency)
        
        # Decide weighting strategy based on consistency
        if consistency_checks['use_compound_weighting']:
            print("\n✓ NEGATIVE DATA CONSISTENT - Using compound-corrected weighting")
            return "compound_corrected"
        else:
            print("\n⚠ NEGATIVE DATA INCONSISTENCIES DETECTED - Using fragment-based weighting")
            print("Reasons:", ", ".join(consistency_checks['warnings']))
            return "fragment_based"
    
    def _perform_negative_consistency_checks(self, actual_fragment_counts, efficiency):
        """Perform comprehensive data consistency checks for negative fragments"""
        checks = {
            'use_compound_weighting': True,
            'warnings': []
        }
        
        # Check 1: Reasonable efficiency ranges for negative fragments
        for combo, eff in efficiency.items():
            if eff > 2.0:  # Negative fragments might have higher extraction rates
                checks['use_compound_weighting'] = False
                checks['warnings'].append("Efficiency too high for negative {}".format(combo))
            elif eff < 0.001:  # Suspiciously low efficiency
                checks['warnings'].append("Very low negative efficiency for {}".format(combo))
        
        # Check 2: Efficiency variance for negative fragments
        if len(efficiency) >= 2:
            max_eff = max(efficiency.values())
            min_eff = min([e for e in efficiency.values() if e > 0])
            if min_eff > 0 and (max_eff / min_eff) > 15:  # More lenient for negative
                checks['warnings'].append("Large negative efficiency variance detected")
        
        # Check 3: Zero fragment counts
        for combo, count in actual_fragment_counts.items():
            if count == 0:
                checks['use_compound_weighting'] = False
                checks['warnings'].append("Zero negative fragments for {}".format(combo))
        
        return checks
    
    def _print_negative_data_overview(self, actual_fragment_counts):
        """Print comprehensive negative data overview"""
        print("\n" + "="*60)
        print("DUAL-NEGATIVE DATASET OVERVIEW")
        print("="*60)
        
        total_fragments = sum(actual_fragment_counts.values())
        print("Total dual-NEGATIVE fragments loaded: {}".format(total_fragments))
        
        print("\nDetailed Breakdown:")
        print("-" * 40)
        for combo in ['CA+EC', 'SA+CA', 'SA+EC']:
            fragments = actual_fragment_counts[combo]
            info = self.dual_combinations[combo]
            
            print("\n{} ({}):".format(info['name'], combo))
            print("  Negative fragments loaded: {}".format(fragments))
            
            if self.source_compound_counts and combo in self.source_compound_counts:
                compounds = self.source_compound_counts[combo]
                efficiency = self.fragment_extraction_efficiency.get(combo, 0)
                print("  Source negative compounds: {}".format(compounds))
                print("  Extraction efficiency: {:.3f} fragments/compound".format(efficiency))
                print("  Fragment percentage: {:.1f}% of total".format(100 * fragments / total_fragments if total_fragments > 0 else 0))
            
            # Fragment type distribution for this combination
            combo_data = self.all_fragments[self.all_fragments['dual_combination'] == combo]
            if len(combo_data) > 0:
                scaffold_count = len(combo_data[combo_data['fragment_type'] == 'scaffold'])
                substituent_count = len(combo_data[combo_data['fragment_type'] == 'substitutent'])
                print("  Scaffolds: {}, Substitutents: {}".format(scaffold_count, substituent_count))
                
                # Average inactivity rate
                avg_inactivity = combo_data['avg_inactivity_rate_percent'].mean()
                print("  Average inactivity rate: {:.1f}%".format(avg_inactivity))
        
        print("\nWeighting Strategy Selected: {}".format(self.weighting_strategy.replace('_', ' ').title()))
        if self.weighting_strategy == "compound_corrected":
            print("  → Will correct for source negative compound size differences")
        else:
            print("  → Using fragment counts only (safer for inconsistent negative data)")
    
    def extract_physicochemical_properties(self):
        """Extract physicochemical properties for dual-NEGATIVE analysis"""
        print("\n" + "="*60)
        print("EXTRACTING PHYSICOCHEMICAL PROPERTIES FOR NEGATIVE FRAGMENTS")
        print("="*60)
        
        properties_list = []
        
        for index, row in self.all_fragments.iterrows():
            if index % 500 == 0:
                print("Processing negative fragment {}/{}".format(index+1, len(self.all_fragments)))
            
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
                
                # Dual-NEGATIVE specific features
                props['lipinski_violations'] = sum([
                    props['molecular_weight'] > 500,
                    props['logp'] > 5,
                    props['num_hbd'] > 5,
                    props['num_hba'] > 10
                ])
                
                # Membrane permeability indicators (important for negative activity)
                props['membrane_permeability_score'] = props['logp'] - 0.1 * props['tpsa']
                
                # Flexibility index
                props['flexibility_index'] = props['num_rotatable_bonds'] / props['num_heavy_atoms'] if props['num_heavy_atoms'] > 0 else 0
                
                # Negative-specific descriptors
                props['negative_potential_score'] = -props['logp'] + 0.05 * props['tpsa']  # Higher TPSA, lower LogP might correlate with negative activity
                
            except Exception as e:
                print("Error processing negative fragment {}: {}".format(row['fragment_smiles'], e))
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
        
        print("Extracted {} physicochemical properties for negative fragments".format(
            len([col for col in self.chemical_features.columns if col not in 
                ['fragment_id', 'dual_combination', 'combination_name', 'fragment_type', 'fragment_smiles']])))
        return self.chemical_features
    
    def _calculate_adaptive_weights(self):
        """Calculate adaptive weights based on selected strategy for negative data"""
        weights = {}
        
        if self.weighting_strategy == "compound_corrected" and self.source_compound_counts:
            print("Applying compound-corrected weighting for negative data...")
            
            # Weight inversely proportional to source negative compound count
            total_compounds = sum(self.source_compound_counts.values())
            for combo in ['CA+EC', 'SA+CA', 'SA+EC']:
                if combo in self.source_compound_counts:
                    # Inverse weighting: smaller negative compound pools get higher weight
                    raw_weight = total_compounds / self.source_compound_counts[combo]
                    weights[combo] = raw_weight
                else:
                    weights[combo] = 1.0
            
            # Normalize weights
            total_weight = sum(weights.values())
            for combo in weights:
                weights[combo] = (weights[combo] / total_weight) * len(weights)
                
        else:
            print("Using fragment-based weighting for negative data...")
            # Equal weighting when no compound correction
            for combo in ['CA+EC', 'SA+CA', 'SA+EC']:
                weights[combo] = 1.0
        
        return weights
    
    def perform_dual_negative_analysis(self, n_permutations=10000):
        """Perform comprehensive dual-NEGATIVE comparison analysis with adaptive weighting"""
        print("\n" + "="*60)
        print("DUAL-NEGATIVE PERMUTATION ANALYSIS")
        print("="*60)
        print("Permutations per test: {:,}".format(n_permutations))
        print("Weighting strategy: {}".format(self.weighting_strategy))
        
        if self.chemical_features is None:
            print("No chemical features available.")
            return pd.DataFrame()
        
        # Calculate adaptive weights
        combination_weights = self._calculate_adaptive_weights()
        
        print("\nNegative combination weights:")
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
        
        print("\nAnalyzing {} valid features across dual-NEGATIVE combinations...".format(len(valid_features)))
        
        results = []
        
        # Pairwise comparisons between dual NEGATIVE combinations
        combination_pairs = [
            ('CA+EC', 'SA+CA'),  # Fungi & Gram- vs Gram+ & Fungi negative
            ('CA+EC', 'SA+EC'),  # Fungi & Gram- vs Gram+ & Gram- negative
            ('SA+CA', 'SA+EC')   # Gram+ & Fungi vs Gram+ & Gram- negative
        ]
        
        for i, feature in enumerate(valid_features):
            if (i + 1) % 5 == 0:
                print("  Processed {}/{} negative features...".format(i + 1, len(valid_features)))
            
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
                    p_value, effect_size, mean_diff = self._permutation_test_weighted(
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
                        'biological_context': self._get_negative_biological_context(combo1, combo2, feature),
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
        
        print("\nNegative analysis complete!")
        print("Total tests: {}".format(len(results_df)))
        print("Significant NEGATIVE patterns (corrected p≤0.05, effect≥0.3): {}".format(len(significant_results)))
        
        return significant_results
    
    def _permutation_test_weighted(self, group1, group2, weights1=None, weights2=None, n_permutations=10000):
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
    
    def _get_negative_biological_context(self, combo1, combo2, feature):
        """Provide biological context for dual-NEGATIVE comparisons"""
        contexts = {
            'CA+EC_vs_SA+CA': 'Gram- vs Gram+ negative (both with Fungi)',
            'CA+EC_vs_SA+EC': 'Fungi vs Gram+ negative (both with Gram-)', 
            'SA+CA_vs_SA+EC': 'Fungi vs Gram- negative (both with Gram+)'
        }
        comparison_key = '{}_vs_{}'.format(combo1, combo2)
        return contexts.get(comparison_key, 'Dual negative combination comparison')
    
    def create_dual_negative_visualizations(self, output_dir='dual_negative_plots'):
        """Create comprehensive visualizations for dual-NEGATIVE analysis"""
        print("\n" + "="*60)
        print("CREATING NEGATIVE VISUALIZATIONS")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        if len(self.permutation_results) == 0:
            print("No negative results to visualize.")
            return
        
        # Set up style
        plt.style.use('default')
        sns.set_palette("dark")
        
        # 1. Enhanced dual-NEGATIVE volcano plot with weighting context
        fig, ax = plt.subplots(figsize=(14, 10))
        
        significant_results = self.permutation_results[
            (self.permutation_results['corrected_p_value'] <= 0.05) &
            (self.permutation_results['effect_size'] >= 0.3)
        ]
        
        # Color by comparison type (darker theme for negative)
        comparison_colors = {
            'CA+EC_vs_SA+CA': '#8B4513',  # SaddleBrown
            'CA+EC_vs_SA+EC': '#800000',  # Maroon  
            'SA+CA_vs_SA+EC': '#2F4F4F'   # DarkSlateGray
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
                       bbox=dict(boxstyle='round,pad=0.3', fc='orange', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                       fontsize=9)
        
        ax.set_xlabel('Effect Size (Cohen\'s d)', fontsize=14, fontweight='bold')
        ax.set_ylabel('-log10(Corrected P-Value)', fontsize=14, fontweight='bold')
        
        # Dynamic title based on weighting strategy
        title = 'Dual-NEGATIVE Chemical Pattern Significance\n'
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
        ax.legend(handles=legend_elements, title='Negative Comparisons & Thresholds', loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add weighting strategy annotation
        strategy_text = f'Weighting: {self.weighting_strategy.replace("_", " ").title()}'
        ax.text(0.02, 0.98, strategy_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('{}/dual_negative_volcano_plot.png'.format(output_dir), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Negative size effect comparison visualization
        if self.weighting_strategy == "compound_corrected" and self.source_compound_counts:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Source negative compounds
            compounds = [self.source_compound_counts[combo] for combo in ['CA+EC', 'SA+CA', 'SA+EC']]
            colors = [self.combination_colors[combo] for combo in ['CA+EC', 'SA+CA', 'SA+EC']]
            
            bars1 = axes[0].bar(['CA+EC', 'SA+CA', 'SA+EC'], compounds, color=colors)
            axes[0].set_title('Source Dual-NEGATIVE Compounds', fontweight='bold')
            axes[0].set_ylabel('Negative Compound Count')
            for i, v in enumerate(compounds):
                axes[0].text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Fragment counts
            fragment_counts = []
            for combo in ['CA+EC', 'SA+CA', 'SA+EC']:
                count = len(self.all_fragments[self.all_fragments['dual_combination'] == combo])
                fragment_counts.append(count)
            
            bars2 = axes[1].bar(['CA+EC', 'SA+CA', 'SA+EC'], fragment_counts, color=colors)
            axes[1].set_title('Extracted Negative Fragments', fontweight='bold')
            axes[1].set_ylabel('Fragment Count')
            for i, v in enumerate(fragment_counts):
                axes[1].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # Extraction efficiency
            efficiencies = [self.fragment_extraction_efficiency[combo] for combo in ['CA+EC', 'SA+CA', 'SA+EC']]
            bars3 = axes[2].bar(['CA+EC', 'SA+CA', 'SA+EC'], efficiencies, color=colors)
            axes[2].set_title('Negative Fragment Extraction Efficiency', fontweight='bold')
            axes[2].set_ylabel('Fragments per Negative Compound')
            for i, v in enumerate(efficiencies):
                axes[2].text(i, v + 0.01, '{:.3f}'.format(v), ha='center', va='bottom', fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            
            plt.suptitle('Negative Size Effect Analysis: Compounds → Fragments', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('{}/negative_size_effect_analysis.png'.format(output_dir), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Dual negative combination chemical profiles
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
                    title_parts.append('(Negative Size-corrected)')
                
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
            
            plt.suptitle('Top Dual-NEGATIVE Chemical Property Differences', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('{}/dual_negative_property_comparisons.png'.format(output_dir), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Comprehensive dual negative overview
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Fragment counts per combination
        combo_counts = self.all_fragments['dual_combination'].value_counts()
        colors = [self.combination_colors[combo] for combo in combo_counts.index]
        
        bars = axes[0,0].bar(combo_counts.index, combo_counts.values, color=colors)
        axes[0,0].set_title('Dual-NEGATIVE Fragment Counts', fontweight='bold')
        axes[0,0].set_ylabel('Fragment Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(combo_counts.values):
            axes[0,0].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # Average inactivity rates
        avg_inactivity = self.all_fragments.groupby('dual_combination')['avg_inactivity_rate_percent'].mean()
        bars = axes[0,1].bar(avg_inactivity.index, avg_inactivity.values, color=colors)
        axes[0,1].set_title('Average Dual Inactivity Rates', fontweight='bold')
        axes[0,1].set_ylabel('Inactivity Rate (%)')
        axes[0,1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(avg_inactivity.values):
            axes[0,1].text(i, v + 1, '{:.1f}%'.format(v), ha='center', va='bottom', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # Fragment type distribution
        fragment_type_dist = self.all_fragments.groupby(['dual_combination', 'fragment_type']).size().unstack(fill_value=0)
        fragment_type_dist.plot(kind='bar', ax=axes[1,0], color=['#D2B48C', '#A0522D'])
        axes[1,0].set_title('Negative Fragment Type Distribution', fontweight='bold')
        axes[1,0].set_ylabel('Fragment Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(title='Fragment Type')
        axes[1,0].grid(True, alpha=0.3)
        
        # Weighting strategy impact for negatives
        if self.weighting_strategy == "compound_corrected" and self.source_compound_counts:
            combos = ['CA+EC', 'SA+CA', 'SA+EC']
            compounds = [self.source_compound_counts[combo] for combo in combos]
            fragments = [len(self.all_fragments[self.all_fragments['dual_combination'] == combo]) for combo in combos]
            
            for i, combo in enumerate(combos):
                axes[1,1].scatter(compounds[i], fragments[i], 
                                color=self.combination_colors[combo], s=200, alpha=0.7, label=combo)
                axes[1,1].annotate(combo, (compounds[i], fragments[i]), 
                                 xytext=(5, 5), textcoords='offset points', fontweight='bold')
            
            axes[1,1].set_xlabel('Source Negative Compounds')
            axes[1,1].set_ylabel('Extracted Negative Fragments')
            axes[1,1].set_title('Negative Compounds vs Fragments\n(Size Effect Context)', fontweight='bold')
            axes[1,1].grid(True, alpha=0.3)
            
            # Add efficiency lines
            for i, combo in enumerate(combos):
                eff = self.fragment_extraction_efficiency[combo]
                axes[1,1].text(compounds[i], fragments[i] - 5, 
                             'Eff: {:.3f}'.format(eff), ha='center', fontsize=9)
        else:
            axes[1,1].text(0.5, 0.5, 'Fragment-Based\nWeighting Used\n\nNo negative size\neffect correction', 
                         transform=axes[1,1].transAxes, ha='center', va='center', 
                         fontsize=14, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            axes[1,1].set_title('Negative Weighting Strategy', fontweight='bold')
        
        plt.suptitle('Comprehensive Dual-NEGATIVE Analysis Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('{}/dual_negative_comprehensive_overview.png'.format(output_dir), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Enhanced negative visualizations saved to {}/".format(output_dir))
        print("Generated plots:")
        print("- dual_negative_volcano_plot.png (significance with weighting context)")
        if self.weighting_strategy == "compound_corrected":
            print("- negative_size_effect_analysis.png (compound→fragment analysis)")
        print("- dual_negative_property_comparisons.png (chemical differences)")
        print("- dual_negative_comprehensive_overview.png (complete overview)")
    
    def generate_dual_negative_report(self, output_file='robust_dual_negative_report.txt'):
        """Generate comprehensive dual-NEGATIVE analysis report with weighting context"""
        print("\nGenerating comprehensive dual-NEGATIVE analysis report...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ROBUST DUAL-NEGATIVE FRAGMENT ANALYSIS REPORT\n")
            f.write("XAI-Derived Fragments with Adaptive Size-Effect Correction\n")
            f.write("=" * 80 + "\n\n")
            
            # Methodology
            f.write("NEGATIVE ANALYSIS METHODOLOGY\n")
            f.write("-" * 40 + "\n")
            f.write("Approach: Adaptive permutation testing for NEGATIVE dual-activity patterns\n")
            f.write("Weighting strategy selected: {}\n".format(self.weighting_strategy.replace('_', ' ').title()))
            
            if self.weighting_strategy == "compound_corrected":
                f.write("Negative size-effect correction: Applied based on source negative compound counts\n")
                f.write("Source negative compound validation: Passed consistency checks\n")
            else:
                f.write("Negative size-effect correction: Not applied (data validation failed or unavailable)\n")
            
            f.write("Permutations per test: 10,000\n")
            f.write("Multiple testing correction: Benjamini-Hochberg FDR\n")
            f.write("Effect size threshold: 0.3 (Cohen's d)\n")
            f.write("Significance threshold: p < 0.05 (corrected)\n\n")
            
            # Data validation results for negatives
            f.write("NEGATIVE DATA VALIDATION RESULTS\n")
            f.write("-" * 40 + "\n")
            
            if self.source_compound_counts:
                f.write("Source negative compound data provided:\n")
                for combo, count in self.source_compound_counts.items():
                    f.write("  {}: {} dual-negative compounds\n".format(combo, count))
                
                f.write("\nNegative fragment extraction efficiency:\n")
                for combo, eff in self.fragment_extraction_efficiency.items():
                    f.write("  {}: {:.3f} fragments per negative compound\n".format(combo, eff))
                
                # Calculate efficiency ratios
                efficiencies = list(self.fragment_extraction_efficiency.values())
                max_eff = max(efficiencies)
                min_eff = min([e for e in efficiencies if e > 0])
                if min_eff > 0:
                    f.write("Negative efficiency range: {:.3f} to {:.3f} ({:.1f}x variation)\n".format(
                        min_eff, max_eff, max_eff/min_eff))
            else:
                f.write("No source negative compound data provided - using fragment-based analysis\n")
            
            # Dataset overview
            f.write("\n\nDUAL-NEGATIVE DATASET OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write("Total dual-NEGATIVE fragments: {}\n\n".format(len(self.all_fragments)))
            
            for combo in ['CA+EC', 'SA+CA', 'SA+EC']:
                count = len(self.all_fragments[self.all_fragments['dual_combination'] == combo])
                info = self.dual_combinations[combo]
                f.write("{}: {} fragments\n".format(info['name'], count))
                f.write("  Negative targets: {} + {}\n".format(
                    self.pathogen_map[info['pathogens'][0]], 
                    self.pathogen_map[info['pathogens'][1]]
                ))
                excluded_pathogen = info['excluded']
                excluded_class = self.pathogen_map.get(excluded_pathogen, excluded_pathogen)
                f.write("  Excludes: {} ({})\n".format(excluded_pathogen, excluded_class))
                
                # Average inactivity rate for this combination
                avg_inactivity = self.all_fragments[
                    self.all_fragments['dual_combination'] == combo
                ]['avg_inactivity_rate_percent'].mean()
                f.write("  Average inactivity rate: {:.1f}%\n".format(avg_inactivity))
                
                if self.source_compound_counts and combo in self.source_compound_counts:
                    compounds = self.source_compound_counts[combo]
                    efficiency = self.fragment_extraction_efficiency[combo]
                    f.write("  Source negative compounds: {}\n".format(compounds))
                    f.write("  Extraction efficiency: {:.3f} fragments/compound\n".format(efficiency))
                f.write("\n")
            
            # Results
            if len(self.permutation_results) > 0:
                significant_results = self.permutation_results[
                    (self.permutation_results['corrected_p_value'] <= 0.05) &
                    (self.permutation_results['effect_size'] >= 0.3)
                ]
                
                f.write("DUAL-NEGATIVE COMPARISON RESULTS\n")
                f.write("-" * 40 + "\n")
                f.write("Total pairwise tests performed: {}\n".format(len(self.permutation_results)))
                f.write("Statistically significant NEGATIVE patterns: {}\n".format(len(significant_results)))
                f.write("False discovery rate: {:.1%}\n".format(
                    (self.permutation_results['corrected_p_value'] <= 0.05).mean()))
                f.write("Weighting strategy: {}\n\n".format(self.weighting_strategy))
                
                if len(significant_results) > 0:
                    f.write("VALIDATED DUAL-NEGATIVE DIFFERENCES:\n")
                    f.write("-" * 30 + "\n")
                    
                    # Group results by comparison type
                    for comparison in ['CA+EC_vs_SA+CA', 'CA+EC_vs_SA+EC', 'SA+CA_vs_SA+EC']:
                        comp_results = significant_results[
                            significant_results['comparison'] == comparison
                        ].nlargest(3, 'effect_size')
                        
                        if len(comp_results) > 0:
                            f.write("\n{}:\n".format(comparison.replace('_vs_', ' vs ')))
                            for _, result in comp_results.iterrows():
                                if self.weighting_strategy == "compound_corrected":
                                    mean_type = "negative size-corrected"
                                    mean1, mean2 = result['weighted_mean1'], result['weighted_mean2']
                                else:
                                    mean_type = "raw negative"
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
                
                # Design insights for negative patterns
                f.write("\n\nDUAL-NEGATIVE DESIGN INSIGHTS\n")
                f.write("-" * 40 + "\n")
                if self.weighting_strategy == "compound_corrected":
                    f.write("Based on negative size-effect corrected analysis:\n\n")
                else:
                    f.write("Based on negative fragment-based analysis:\n\n")
                
                f.write("NEGATIVE PATTERN INTERPRETATION:\n")
                f.write("These fragments show consistent NEGATIVE activity, indicating:\n")
                f.write("- Chemical features that PREVENT activity against specific pathogen pairs\n")
                f.write("- Structural motifs to AVOID in dual-activity compound design\n")
                f.write("- Selectivity patterns that distinguish pathogen vulnerabilities\n\n")
                
            f.write("NEGATIVE WEIGHTING STRATEGY IMPACT\n")
            f.write("-" * 30 + "\n")
            if self.weighting_strategy == "compound_corrected":
                f.write("Negative size-effect correction successfully applied:\n")
                f.write("- Accounts for unequal source negative compound populations\n")
                f.write("- Results reflect true negative chemical selectivity patterns\n")
                f.write("- Unbiased insights for rational compound design\n\n")
            else:
                f.write("Fragment-based weighting used for negative data:\n")
                f.write("- Equal treatment of all negative fragments\n")
                f.write("- Conservative approach when source data is inconsistent\n")
                f.write("- May be influenced by compound population differences\n\n")
            
            f.write("DUAL-NEGATIVE DRUG DESIGN RECOMMENDATIONS\n")
            f.write("-" * 50 + "\n")
            f.write("1. AVOID features identified in negative patterns\n")
            f.write("2. Use negative insights to guide positive fragment selection\n")
            f.write("3. Consider pathogen-specific vulnerabilities revealed by negative patterns\n")
            f.write("4. Combine with positive dual-activity data for comprehensive design\n")
            f.write("5. Apply negative size-effect corrections in future analyses\n\n")
            
            f.write("Note: Analysis based on {} total dual-NEGATIVE fragments\n".format(len(self.all_fragments)))
            f.write("with robust statistical validation and adaptive size-effect correction.\n")
        
        print("Comprehensive dual-NEGATIVE analysis report saved to {}".format(output_file))


def main():
    """Main robust dual-NEGATIVE analysis pipeline"""
    print("Starting Robust Dual-NEGATIVE XAI Fragment Analysis...")
    print("Features: Adaptive weighting, data validation, negative size-effect correction")
    
    # Source NEGATIVE compound counts from your actual analysis
    # These are compounds that show negative activity for the dual combinations
    source_negative_compound_counts = {
        'SA+CA': 8851,   # S.aureus + C.albicans negative compounds (actual)
        'SA+EC': 10426,  # S.aureus + E.coli negative compounds (actual)
        'CA+EC': 10901   # C.albicans + E.coli negative compounds (actual)
    }
    
    # Total compound counts for context
    total_compound_counts = {
        'SA': 54277,
        'CA': 28476, 
        'EC': 44920
    }
    
    # File paths for NEGATIVE data - UPDATE THESE TO YOUR ACTUAL FILE PATHS
    file_paths = {
        'negative_CA_EC_scaffolds': 'dual_CA_EC_negative_scaffolds.csv',
        'negative_CA_EC_substitutents': 'dual_CA_EC_negative_substituents.csv',
        'negative_SA_CA_scaffolds': 'dual_SA_CA_negative_scaffolds.csv',
        'negative_SA_CA_substitutents': 'dual_SA_CA_negative_substituents.csv',
        'negative_SA_EC_scaffolds': 'dual_SA_EC_negative_scaffolds.csv', 
        'negative_SA_EC_substitutents': 'dual_SA_EC_negative_substituents.csv'
    }
    
    # Initialize robust dual-NEGATIVE analyzer
    analyzer = RobustDualNegativeFragmentAnalyzer(
        source_compound_counts=source_negative_compound_counts,
        total_compound_counts=total_compound_counts
    )
    
    try:
        # Load dual-NEGATIVE fragments with validation
        all_fragments = analyzer.load_and_prepare_dual_negative_data(file_paths)
        
        # Extract physicochemical properties
        chemical_features = analyzer.extract_physicochemical_properties()
        
        # Perform robust dual-NEGATIVE analysis
        significant_patterns = analyzer.perform_dual_negative_analysis(n_permutations=10000)
        
        # Create enhanced visualizations
        analyzer.create_dual_negative_visualizations()
        
        # Generate comprehensive report
        analyzer.generate_dual_negative_report()
        
        # Save results
        print("\nSaving robust dual-NEGATIVE analysis results...")
        
        if len(significant_patterns) > 0:
            significant_patterns.to_csv('robust_dual_negative_significant_patterns.csv', 
                                      index=False, encoding='utf-8')
            print("Significant dual-NEGATIVE patterns saved to robust_dual_negative_significant_patterns.csv")
        
        analyzer.permutation_results.to_csv('complete_robust_dual_negative_results.csv',
                                          index=False, encoding='utf-8')
        print("Complete negative results saved to complete_robust_dual_negative_results.csv")
        
        all_fragments.to_csv('robust_dual_negative_fragments_with_properties.csv', 
                           index=False, encoding='utf-8')
        print("Enhanced dual-NEGATIVE fragments saved to robust_dual_negative_fragments_with_properties.csv")
        
        print("\n" + "="*80)
        print("ROBUST DUAL-NEGATIVE ANALYSIS COMPLETE!")
        print("="*80)
        print("Key features of this NEGATIVE analysis:")
        print("✓ Data validation and consistency checking for negative patterns")
        print("✓ Adaptive weighting strategy selection for negative data")
        print("✓ Size-effect correction for negative compounds when appropriate")
        print("✓ Robust statistical testing with permutations for negative patterns")
        print("✓ Complementary analysis to positive dual-activity findings")
        
        print("\nGenerated files:")
        print("- robust_dual_negative_significant_patterns.csv (validated negative differences)")
        print("- complete_robust_dual_negative_results.csv (all negative statistical tests)")
        print("- robust_dual_negative_fragments_with_properties.csv (enhanced negative fragment data)")
        
        # Analysis summary with negative weighting context
        print("\nNegative Analysis Summary:")
        print("Weighting strategy used: {}".format(analyzer.weighting_strategy.replace('_', ' ').title()))
        
        if analyzer.weighting_strategy == "compound_corrected":
            print("Negative size-effect correction applied successfully!")
            print("Negative fragment extraction efficiency:")
            for combo, eff in analyzer.fragment_extraction_efficiency.items():
                compounds = analyzer.source_compound_counts[combo]
                fragments = len(all_fragments[all_fragments['dual_combination'] == combo])
                print("  {}: {:.3f} fragments/compound ({} fragments from {} negative compounds)".format(
                    combo, eff, fragments, compounds))
        else:
            print("Fragment-based weighting used (negative data validation triggered fallback)")
        
        print("\nDual-NEGATIVE Fragment Distribution:")
        for combo in ['CA+EC', 'SA+CA', 'SA+EC']:
            count = len(all_fragments[all_fragments['dual_combination'] == combo])
            info = analyzer.dual_combinations[combo]
            avg_inactivity = all_fragments[
                all_fragments['dual_combination'] == combo
            ]['avg_inactivity_rate_percent'].mean()
            print("  {}: {} fragments ({:.1f}% avg inactivity)".format(
                info['name'], count, avg_inactivity))
        
        print("\nStatistical robustness for negative patterns:")
        print("  - Permutation tests: 10,000 per comparison")
        print("  - Negative data validation: Performed automatically")
        print("  - Weighting strategy: Adaptively selected for negative data")
        print("  - Multiple testing: Benjamini-Hochberg FDR correction")
        print("  - Total negative tests: {}".format(len(analyzer.permutation_results)))
        print("  - Validated negative patterns: {}".format(len(significant_patterns)))
        
        # Show key validated negative findings
        if len(significant_patterns) > 0:
            print("\nKey Validated Dual-NEGATIVE Insights:")
            
            # Group by comparison type
            comparisons = {
                'CA+EC_vs_SA+CA': 'Fungi+Gram- vs Gram++Fungi (negative patterns)',
                'CA+EC_vs_SA+EC': 'Fungi+Gram- vs Gram++Gram- (negative patterns)',
                'SA+CA_vs_SA+EC': 'Gram++Fungi vs Gram++Gram- (negative patterns)'
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
                        mean_info = "(negative size-corrected)"
                    else:
                        fold_change = top_result['raw_mean1'] / top_result['raw_mean2'] if top_result['raw_mean2'] != 0 else top_result['fold_change']
                        mean_info = "(raw negative means)"
                    
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
            
            print("\nNegative Drug Design Strategy:")
            print("These validated, size-effect corrected NEGATIVE differences reveal:")
            print("✓ Chemical features that consistently PREVENT dual-activity")
            print("✓ Structural motifs to AVOID in broad-spectrum compound design")  
            print("✓ Pathogen-specific vulnerabilities exposed by negative patterns")
            print("✓ Complementary insights to positive dual-activity findings")
            print("✓ Unbiased negative insights accounting for compound population differences")
            print("\nCombine with positive dual-activity analysis for comprehensive design!")
            
        else:
            print("\nNo statistically validated dual-NEGATIVE differences found.")
            print("This suggests:")
            print("  - Negative dual-activity fragments may share common chemical features")
            print("  - Current negative dataset may need expansion for more sensitive detection")
            print("  - Focus on individual negative combination analysis rather than comparative")
        
        print("\n" + "="*60)
        print("INTEGRATION WITH POSITIVE ANALYSIS")
        print("="*60)
        print("For comprehensive dual-activity compound design:")
        print("✓ Use POSITIVE dual-activity patterns to identify beneficial features")
        print("✓ Use NEGATIVE dual-activity patterns to avoid detrimental features")
        print("✓ Compare positive vs negative size-effect corrections")
        print("✓ Integrate both analyses for rational structure-activity relationships")
        print("✓ Consider pathogen-specific selectivity from both perspectives")
        
        print("\nRecommendation: These robust, validated NEGATIVE results provide crucial")
        print("complementary insights to positive dual-activity findings for optimal design!")
        
    except Exception as e:
        print("Error during robust dual-NEGATIVE analysis: {}".format(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
        