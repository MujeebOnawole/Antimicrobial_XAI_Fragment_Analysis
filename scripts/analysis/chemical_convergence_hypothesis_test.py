#!/usr/bin/env python3
"""
CHEMICAL CONVERGENCE HYPOTHESIS TESTING SCRIPT
============================================
Tests the hypothesis: "Low fragments per compound indicates chemical convergence
(similar scaffolds being reused)" for dual-activity antimicrobial compounds.

Background:
- SA+CA: 0.248 fragments/compound from 1,164 compounds → 289 total fragments
- SA+EC: 6.655 fragments/compound from 849 compounds → 5,650 total fragments
- CA+EC: 0.791 fragments/compound from 187 compounds → 148 total fragments

If convergence is real for SA+CA, we expect:
✓ Higher inter-fragment similarity (similar scaffolds)
✓ Lower scaffold diversity (fewer unique scaffolds per compound)
✓ Higher scaffold reuse rates (same scaffolds in many compounds)
✓ Lower molecular complexity (simpler, more conserved structures)
✓ Tighter chemical clusters (less spread in chemical space)

Author: AI Agent for Chemical Analysis
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict, Counter
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")


class ChemicalConvergenceAnalyzer:
    """
    Comprehensive analyzer to test the chemical convergence hypothesis.
    """

    def __init__(self, output_dir='convergence_analysis'):
        """Initialize the analyzer with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Source compound counts from Venn diagram
        self.source_compounds = {
            'SA+CA': 1164,
            'SA+EC': 849,
            'CA+EC': 187
        }

        # Color scheme for combinations
        self.colors = {
            'SA+CA': '#DC143C',  # Crimson red
            'SA+EC': '#1E90FF',  # Dodger blue
            'CA+EC': '#228B22'   # Forest green
        }

        # Data containers
        self.data = {}
        self.fragments = {}
        self.fingerprints = {}
        self.similarity_matrices = {}
        self.results = {}

    def load_data(self, file_paths):
        """Load all fragment CSV files."""
        print("=" * 70)
        print("LOADING DUAL-ACTIVITY FRAGMENT DATA")
        print("=" * 70)

        combination_mapping = {
            'SA_CA': 'SA+CA',
            'SA_EC': 'SA+EC',
            'CA_EC': 'CA+EC'
        }

        for combo_underscore, combo_plus in combination_mapping.items():
            self.data[combo_plus] = {'scaffolds': None, 'substituents': None}

            # Load scaffolds
            scaffold_key = f'dual_{combo_underscore}_positive_scaffolds'
            if scaffold_key in file_paths:
                df = pd.read_csv(file_paths[scaffold_key])
                self.data[combo_plus]['scaffolds'] = df
                print(f"  {combo_plus} scaffolds: {len(df)} fragments")

            # Load substituents
            sub_key = f'dual_{combo_underscore}_positive_substitutents'
            if sub_key in file_paths:
                df = pd.read_csv(file_paths[sub_key])
                self.data[combo_plus]['substituents'] = df
                print(f"  {combo_plus} substituents: {len(df)} fragments")

        # Combine fragments for each combination
        print("\nCombining fragments per combination:")
        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            dfs = []
            if self.data[combo]['scaffolds'] is not None:
                scaffolds_df = self.data[combo]['scaffolds'].copy()
                scaffolds_df['fragment_type'] = 'scaffold'
                dfs.append(scaffolds_df)
            if self.data[combo]['substituents'] is not None:
                subs_df = self.data[combo]['substituents'].copy()
                subs_df['fragment_type'] = 'substituent'
                dfs.append(subs_df)

            if dfs:
                self.fragments[combo] = pd.concat(dfs, ignore_index=True)
                total = len(self.fragments[combo])
                compounds = self.source_compounds[combo]
                efficiency = total / compounds
                print(f"  {combo}: {total} total fragments from {compounds} compounds ({efficiency:.3f} fragments/compound)")

        return self.fragments

    def compute_fingerprints(self):
        """Compute Morgan fingerprints for all fragments."""
        print("\n" + "=" * 70)
        print("COMPUTING MORGAN FINGERPRINTS")
        print("=" * 70)

        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo not in self.fragments:
                continue

            df = self.fragments[combo]
            fps = []
            valid_indices = []
            smiles_list = []

            print(f"\nProcessing {combo}...")
            for idx, row in df.iterrows():
                smiles = row['fragment_smiles']
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    try:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                        fps.append(fp)
                        valid_indices.append(idx)
                        smiles_list.append(smiles)
                    except Exception as e:
                        continue

            self.fingerprints[combo] = {
                'fps': fps,
                'indices': valid_indices,
                'smiles': smiles_list
            }
            print(f"  Computed {len(fps)} fingerprints for {combo}")

        return self.fingerprints

    def compute_similarity_matrices(self):
        """Compute pairwise Tanimoto similarity matrices."""
        print("\n" + "=" * 70)
        print("COMPUTING TANIMOTO SIMILARITY MATRICES")
        print("=" * 70)

        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo not in self.fingerprints:
                continue

            fps = self.fingerprints[combo]['fps']
            n = len(fps)

            if n < 2:
                print(f"  {combo}: Too few fragments for similarity analysis")
                continue

            print(f"\n  Computing {n}x{n} similarity matrix for {combo}...")

            # Compute upper triangle of similarity matrix
            similarities = []
            sim_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(i+1, n):
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    similarities.append(sim)
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

            # Set diagonal to 1
            np.fill_diagonal(sim_matrix, 1.0)

            self.similarity_matrices[combo] = {
                'matrix': sim_matrix,
                'pairwise': np.array(similarities)
            }

            print(f"    Mean similarity: {np.mean(similarities):.4f}")
            print(f"    Median similarity: {np.median(similarities):.4f}")
            print(f"    Std similarity: {np.std(similarities):.4f}")

        return self.similarity_matrices

    def analyze_structural_similarity(self):
        """
        Analysis 1: Structural Similarity Analysis
        Hypothesis: SA+CA should show HIGHER similarity if convergent
        """
        print("\n" + "=" * 70)
        print("ANALYSIS 1: STRUCTURAL SIMILARITY")
        print("=" * 70)

        self.results['similarity'] = {}

        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo not in self.similarity_matrices:
                continue

            pairwise = self.similarity_matrices[combo]['pairwise']

            self.results['similarity'][combo] = {
                'mean': np.mean(pairwise),
                'median': np.median(pairwise),
                'std': np.std(pairwise),
                'min': np.min(pairwise),
                'max': np.max(pairwise),
                'q25': np.percentile(pairwise, 25),
                'q75': np.percentile(pairwise, 75),
                'n_pairs': len(pairwise),
                'pairwise': pairwise
            }

        # Print results
        print("\nSimilarity Statistics:")
        print("-" * 60)
        print(f"{'Combination':<15} {'Mean':>10} {'Median':>10} {'Std':>10} {'N_Pairs':>12}")
        print("-" * 60)
        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo in self.results['similarity']:
                r = self.results['similarity'][combo]
                print(f"{combo:<15} {r['mean']:>10.4f} {r['median']:>10.4f} {r['std']:>10.4f} {r['n_pairs']:>12,}")

        return self.results['similarity']

    def analyze_scaffold_diversity(self):
        """
        Analysis 2: Scaffold Diversity Metrics
        Hypothesis: SA+CA should show LOWER diversity if convergent
        """
        print("\n" + "=" * 70)
        print("ANALYSIS 2: SCAFFOLD DIVERSITY METRICS")
        print("=" * 70)

        self.results['diversity'] = {}

        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo not in self.fragments:
                continue

            df = self.fragments[combo]
            total_fragments = len(df)
            total_compounds = self.source_compounds[combo]

            # Get unique SMILES
            unique_smiles = df['fragment_smiles'].nunique()

            # Diversity = unique fragments / total compounds
            diversity = unique_smiles / total_compounds

            # Get scaffold-only data
            scaffolds_only = df[df['fragment_type'] == 'scaffold'] if 'fragment_type' in df.columns else df
            unique_scaffolds = scaffolds_only['fragment_smiles'].nunique() if len(scaffolds_only) > 0 else 0

            # Count fragment occurrences (from 'total_compounds_both_pathogens' column)
            if 'total_compounds_both_pathogens' in df.columns:
                occurrence_counts = df['total_compounds_both_pathogens'].values
                mean_reuse = np.mean(occurrence_counts)
                median_reuse = np.median(occurrence_counts)
                max_reuse = np.max(occurrence_counts)

                # Singleton analysis (fragments appearing in only 1 compound)
                singletons = np.sum(occurrence_counts == 1)
                singleton_pct = (singletons / total_fragments) * 100 if total_fragments > 0 else 0
            else:
                mean_reuse = 0
                median_reuse = 0
                max_reuse = 0
                singletons = 0
                singleton_pct = 0

            self.results['diversity'][combo] = {
                'total_fragments': total_fragments,
                'unique_fragments': unique_smiles,
                'total_compounds': total_compounds,
                'fragments_per_compound': total_fragments / total_compounds,
                'diversity_ratio': diversity,
                'unique_scaffolds': unique_scaffolds,
                'mean_reuse': mean_reuse,
                'median_reuse': median_reuse,
                'max_reuse': max_reuse,
                'singletons': singletons,
                'singleton_pct': singleton_pct
            }

        # Print results
        print("\nDiversity Statistics:")
        print("-" * 80)
        print(f"{'Combination':<12} {'Fragments':>10} {'Compounds':>10} {'Frag/Comp':>10} {'Diversity':>10} {'Mean Reuse':>12}")
        print("-" * 80)
        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo in self.results['diversity']:
                r = self.results['diversity'][combo]
                print(f"{combo:<12} {r['total_fragments']:>10} {r['total_compounds']:>10} {r['fragments_per_compound']:>10.3f} {r['diversity_ratio']:>10.3f} {r['mean_reuse']:>12.1f}")

        print("\nSingleton Analysis (fragments in only 1 compound):")
        print("-" * 50)
        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo in self.results['diversity']:
                r = self.results['diversity'][combo]
                print(f"  {combo}: {r['singletons']} singletons ({r['singleton_pct']:.1f}%)")

        return self.results['diversity']

    def analyze_fragment_reuse(self):
        """
        Analysis 3: Fragment Reuse Patterns
        Hypothesis: SA+CA fragments should appear in MORE compounds each (high reuse)
        """
        print("\n" + "=" * 70)
        print("ANALYSIS 3: FRAGMENT REUSE PATTERNS")
        print("=" * 70)

        self.results['reuse'] = {}

        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo not in self.fragments:
                continue

            df = self.fragments[combo]

            if 'total_compounds_both_pathogens' not in df.columns:
                print(f"  {combo}: Missing 'total_compounds_both_pathogens' column")
                continue

            occurrences = df['total_compounds_both_pathogens'].values

            # Distribution analysis
            self.results['reuse'][combo] = {
                'occurrences': occurrences,
                'mean': np.mean(occurrences),
                'median': np.median(occurrences),
                'std': np.std(occurrences),
                'min': np.min(occurrences),
                'max': np.max(occurrences),
                'total_fragment_instances': np.sum(occurrences),
                'unique_fragments': len(occurrences),
                # Bins for histogram
                'bin_1': np.sum(occurrences == 1),
                'bin_2_5': np.sum((occurrences >= 2) & (occurrences <= 5)),
                'bin_6_10': np.sum((occurrences >= 6) & (occurrences <= 10)),
                'bin_11_50': np.sum((occurrences >= 11) & (occurrences <= 50)),
                'bin_50_plus': np.sum(occurrences > 50)
            }

            # Top 10 most reused fragments
            top_10_idx = np.argsort(occurrences)[-10:][::-1]
            top_10 = []
            for idx in top_10_idx:
                top_10.append({
                    'smiles': df.iloc[idx]['fragment_smiles'],
                    'occurrences': int(occurrences[idx]),
                    'activity_rate': df.iloc[idx].get('avg_activity_rate_percent', 0)
                })
            self.results['reuse'][combo]['top_10'] = top_10

        # Print results
        print("\nFragment Reuse Statistics:")
        print("-" * 70)
        print(f"{'Combination':<12} {'Mean':>10} {'Median':>10} {'Max':>10} {'Total Instances':>18}")
        print("-" * 70)
        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo in self.results['reuse']:
                r = self.results['reuse'][combo]
                print(f"{combo:<12} {r['mean']:>10.1f} {r['median']:>10.1f} {r['max']:>10} {r['total_fragment_instances']:>18,}")

        print("\nTop 5 Most Reused Fragments per Combination:")
        print("-" * 70)
        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo in self.results['reuse']:
                print(f"\n{combo}:")
                for i, frag in enumerate(self.results['reuse'][combo]['top_10'][:5], 1):
                    smiles_short = frag['smiles'][:40] + '...' if len(frag['smiles']) > 40 else frag['smiles']
                    print(f"  {i}. {smiles_short:<45} (n={frag['occurrences']:>4}, activity={frag['activity_rate']:.1f}%)")

        return self.results['reuse']

    def analyze_molecular_complexity(self):
        """
        Analysis 4: Molecular Complexity Comparison
        Hypothesis: SA+CA should show LOWER complexity if convergent (simpler, conserved)
        """
        print("\n" + "=" * 70)
        print("ANALYSIS 4: MOLECULAR COMPLEXITY")
        print("=" * 70)

        self.results['complexity'] = {}

        complexity_properties = [
            'num_rings', 'num_aromatic_rings', 'num_rotatable_bonds',
            'molecular_weight', 'fraction_csp3', 'bertz_complexity',
            'num_heavy_atoms', 'num_heteroatoms', 'tpsa', 'logp'
        ]

        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo not in self.fragments:
                continue

            df = self.fragments[combo]
            props = {p: [] for p in complexity_properties}
            valid_count = 0

            print(f"\n  Computing complexity for {combo}...")

            for idx, row in df.iterrows():
                mol = Chem.MolFromSmiles(row['fragment_smiles'])
                if mol is None:
                    continue

                valid_count += 1

                try:
                    props['num_rings'].append(rdMolDescriptors.CalcNumRings(mol))
                    props['num_aromatic_rings'].append(rdMolDescriptors.CalcNumAromaticRings(mol))
                    props['num_rotatable_bonds'].append(rdMolDescriptors.CalcNumRotatableBonds(mol))
                    props['molecular_weight'].append(Descriptors.MolWt(mol))
                    props['fraction_csp3'].append(rdMolDescriptors.CalcFractionCSP3(mol))
                    props['bertz_complexity'].append(Descriptors.BertzCT(mol))
                    props['num_heavy_atoms'].append(mol.GetNumHeavyAtoms())
                    props['num_heteroatoms'].append(rdMolDescriptors.CalcNumHeteroatoms(mol))
                    props['tpsa'].append(rdMolDescriptors.CalcTPSA(mol))
                    props['logp'].append(Descriptors.MolLogP(mol))
                except Exception as e:
                    continue

            # Calculate statistics
            self.results['complexity'][combo] = {
                'valid_count': valid_count,
                'properties': {}
            }

            for prop in complexity_properties:
                values = np.array(props[prop])
                if len(values) > 0:
                    self.results['complexity'][combo]['properties'][prop] = {
                        'values': values,
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }

            print(f"    Computed for {valid_count} valid molecules")

        # Print summary table
        print("\nMolecular Complexity Summary:")
        print("-" * 90)
        print(f"{'Property':<20} {'SA+CA Mean':>15} {'SA+EC Mean':>15} {'CA+EC Mean':>15} {'SA+CA vs SA+EC':>15}")
        print("-" * 90)

        for prop in complexity_properties:
            row = f"{prop:<20}"
            values_dict = {}
            for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
                if combo in self.results['complexity']:
                    if prop in self.results['complexity'][combo]['properties']:
                        val = self.results['complexity'][combo]['properties'][prop]['mean']
                        values_dict[combo] = val
                        row += f" {val:>15.2f}"
                    else:
                        row += f" {'N/A':>15}"
                else:
                    row += f" {'N/A':>15}"

            # Calculate ratio
            if 'SA+CA' in values_dict and 'SA+EC' in values_dict and values_dict['SA+EC'] != 0:
                ratio = values_dict['SA+CA'] / values_dict['SA+EC']
                row += f" {ratio:>15.2f}x"
            else:
                row += f" {'N/A':>15}"

            print(row)

        return self.results['complexity']

    def analyze_chemical_clustering(self):
        """
        Analysis 5: Chemical Clustering Analysis
        Hypothesis: SA+CA should form TIGHTER, FEWER clusters (convergence)
        """
        print("\n" + "=" * 70)
        print("ANALYSIS 5: CHEMICAL CLUSTERING")
        print("=" * 70)

        self.results['clustering'] = {}

        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo not in self.similarity_matrices:
                continue

            sim_matrix = self.similarity_matrices[combo]['matrix']
            n = sim_matrix.shape[0]

            if n < 3:
                print(f"  {combo}: Too few fragments for clustering")
                continue

            print(f"\nClustering {combo} ({n} fragments)...")

            # Convert similarity to distance
            dist_matrix = 1 - sim_matrix
            np.fill_diagonal(dist_matrix, 0)  # Ensure diagonal is 0

            # Hierarchical clustering
            try:
                condensed_dist = squareform(dist_matrix)
                linkage_matrix = linkage(condensed_dist, method='average')

                # Find optimal number of clusters using silhouette score
                best_k = 2
                best_silhouette = -1
                silhouette_scores = {}

                max_k = min(20, n - 1)
                for k in range(2, max_k + 1):
                    clusters = fcluster(linkage_matrix, k, criterion='maxclust')
                    if len(np.unique(clusters)) >= 2:
                        try:
                            score = silhouette_score(dist_matrix, clusters, metric='precomputed')
                            silhouette_scores[k] = score
                            if score > best_silhouette:
                                best_silhouette = score
                                best_k = k
                        except:
                            continue

                # Get final clusters
                final_clusters = fcluster(linkage_matrix, best_k, criterion='maxclust')

                # Calculate cluster statistics
                cluster_sizes = Counter(final_clusters)

                # Calculate intra-cluster similarities
                intra_similarities = []
                for cluster_id in np.unique(final_clusters):
                    cluster_indices = np.where(final_clusters == cluster_id)[0]
                    if len(cluster_indices) > 1:
                        cluster_sims = []
                        for i in range(len(cluster_indices)):
                            for j in range(i+1, len(cluster_indices)):
                                cluster_sims.append(sim_matrix[cluster_indices[i], cluster_indices[j]])
                        if cluster_sims:
                            intra_similarities.extend(cluster_sims)

                avg_intra_sim = np.mean(intra_similarities) if intra_similarities else 0

                self.results['clustering'][combo] = {
                    'n_fragments': n,
                    'optimal_k': best_k,
                    'best_silhouette': best_silhouette,
                    'cluster_sizes': dict(cluster_sizes),
                    'avg_cluster_size': np.mean(list(cluster_sizes.values())),
                    'avg_intra_cluster_similarity': avg_intra_sim,
                    'linkage_matrix': linkage_matrix,
                    'clusters': final_clusters,
                    'silhouette_scores': silhouette_scores
                }

                print(f"    Optimal clusters: {best_k}")
                print(f"    Best silhouette score: {best_silhouette:.4f}")
                print(f"    Average intra-cluster similarity: {avg_intra_sim:.4f}")
                print(f"    Cluster sizes: {dict(cluster_sizes)}")

            except Exception as e:
                print(f"    Clustering error for {combo}: {e}")
                continue

        return self.results['clustering']

    def perform_statistical_tests(self):
        """Perform statistical tests comparing combinations."""
        print("\n" + "=" * 70)
        print("STATISTICAL TESTING")
        print("=" * 70)

        self.results['statistics'] = {
            'similarity_tests': {},
            'complexity_tests': {},
            'reuse_tests': {}
        }

        # Helper function for effect size (Cohen's d)
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            return abs(np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

        # 1. Similarity comparisons
        print("\n1. Similarity Comparisons (Tanimoto):")
        print("-" * 60)

        comparisons = [('SA+CA', 'SA+EC'), ('SA+CA', 'CA+EC'), ('SA+EC', 'CA+EC')]

        for combo1, combo2 in comparisons:
            if combo1 in self.results['similarity'] and combo2 in self.results['similarity']:
                data1 = self.results['similarity'][combo1]['pairwise']
                data2 = self.results['similarity'][combo2]['pairwise']

                # Mann-Whitney U test
                stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                effect = cohens_d(data1, data2)

                self.results['statistics']['similarity_tests'][f'{combo1}_vs_{combo2}'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'effect_size': effect,
                    'mean1': np.mean(data1),
                    'mean2': np.mean(data2),
                    'n1': len(data1),
                    'n2': len(data2)
                }

                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  {combo1} vs {combo2}: U={stat:.0f}, p={p_value:.2e}, d={effect:.3f} {sig}")
                print(f"    Means: {np.mean(data1):.4f} vs {np.mean(data2):.4f}")

        # 2. Complexity comparisons
        print("\n2. Complexity Comparisons (Bertz Complexity):")
        print("-" * 60)

        for combo1, combo2 in comparisons:
            try:
                if combo1 in self.results['complexity'] and combo2 in self.results['complexity']:
                    data1 = self.results['complexity'][combo1]['properties']['bertz_complexity']['values']
                    data2 = self.results['complexity'][combo2]['properties']['bertz_complexity']['values']

                    stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                    effect = cohens_d(data1, data2)

                    self.results['statistics']['complexity_tests'][f'{combo1}_vs_{combo2}'] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'effect_size': effect,
                        'mean1': np.mean(data1),
                        'mean2': np.mean(data2)
                    }

                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"  {combo1} vs {combo2}: U={stat:.0f}, p={p_value:.2e}, d={effect:.3f} {sig}")
                    print(f"    Means: {np.mean(data1):.2f} vs {np.mean(data2):.2f}")
            except Exception as e:
                print(f"  Error comparing {combo1} vs {combo2}: {e}")

        # 3. Fragment reuse comparisons
        print("\n3. Fragment Reuse Comparisons:")
        print("-" * 60)

        for combo1, combo2 in comparisons:
            if combo1 in self.results['reuse'] and combo2 in self.results['reuse']:
                data1 = self.results['reuse'][combo1]['occurrences']
                data2 = self.results['reuse'][combo2]['occurrences']

                stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                effect = cohens_d(data1, data2)

                self.results['statistics']['reuse_tests'][f'{combo1}_vs_{combo2}'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'effect_size': effect,
                    'mean1': np.mean(data1),
                    'mean2': np.mean(data2)
                }

                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  {combo1} vs {combo2}: U={stat:.0f}, p={p_value:.2e}, d={effect:.3f} {sig}")
                print(f"    Means: {np.mean(data1):.1f} vs {np.mean(data2):.1f}")

        # Kruskal-Wallis for all three groups
        print("\n4. Kruskal-Wallis Tests (all 3 combinations):")
        print("-" * 60)

        # Similarity
        if all(c in self.results['similarity'] for c in ['SA+CA', 'SA+EC', 'CA+EC']):
            stat, p_value = kruskal(
                self.results['similarity']['SA+CA']['pairwise'],
                self.results['similarity']['SA+EC']['pairwise'],
                self.results['similarity']['CA+EC']['pairwise']
            )
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"  Similarity: H={stat:.2f}, p={p_value:.2e} {sig}")

        # Complexity (Bertz)
        try:
            if all(c in self.results['complexity'] for c in ['SA+CA', 'SA+EC', 'CA+EC']):
                stat, p_value = kruskal(
                    self.results['complexity']['SA+CA']['properties']['bertz_complexity']['values'],
                    self.results['complexity']['SA+EC']['properties']['bertz_complexity']['values'],
                    self.results['complexity']['CA+EC']['properties']['bertz_complexity']['values']
                )
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  Complexity: H={stat:.2f}, p={p_value:.2e} {sig}")
        except:
            pass

        return self.results['statistics']

    def create_visualizations(self):
        """Generate all visualization plots."""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        # 1. Similarity distributions (violin plots)
        self._plot_similarity_distributions()

        # 2. Fragment reuse histogram
        self._plot_reuse_distributions()

        # 3. Molecular complexity radar plot
        self._plot_complexity_radar()

        # 4. Fragment occurrence heatmap
        self._plot_fragment_heatmap()

        # 5. Dendrograms
        self._plot_dendrograms()

        # 6. Summary dashboard
        self._plot_summary_dashboard()

        print(f"\nAll visualizations saved to {self.output_dir}/")

    def _plot_similarity_distributions(self):
        """Plot Tanimoto similarity distributions."""
        print("  Creating similarity distribution plots...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Violin plots
        data_for_violin = []
        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo in self.results['similarity']:
                sims = self.results['similarity'][combo]['pairwise']
                for s in sims[:5000]:  # Sample for large datasets
                    data_for_violin.append({'Combination': combo, 'Similarity': s})

        df_violin = pd.DataFrame(data_for_violin)

        if len(df_violin) > 0:
            sns.violinplot(data=df_violin, x='Combination', y='Similarity', ax=axes[0],
                          palette=[self.colors[c] for c in ['SA+CA', 'SA+EC', 'CA+EC'] if c in self.results['similarity']])
            axes[0].set_title('Tanimoto Similarity Distributions\n(Fragment Pairwise Similarity)', fontweight='bold')
            axes[0].set_ylabel('Tanimoto Similarity')
            axes[0].set_xlabel('')

            # Add mean lines
            for i, combo in enumerate(['SA+CA', 'SA+EC', 'CA+EC']):
                if combo in self.results['similarity']:
                    mean_val = self.results['similarity'][combo]['mean']
                    axes[0].hlines(mean_val, i-0.4, i+0.4, colors='black', linestyles='--', linewidth=2)

        # Box plots
        if len(df_violin) > 0:
            sns.boxplot(data=df_violin, x='Combination', y='Similarity', ax=axes[1],
                       palette=[self.colors[c] for c in ['SA+CA', 'SA+EC', 'CA+EC'] if c in self.results['similarity']])
            axes[1].set_title('Similarity Box Plots', fontweight='bold')
            axes[1].set_ylabel('Tanimoto Similarity')
            axes[1].set_xlabel('')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/similarity_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_reuse_distributions(self):
        """Plot fragment reuse distributions."""
        print("  Creating reuse distribution plots...")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for i, combo in enumerate(['SA+CA', 'SA+EC', 'CA+EC']):
            if combo in self.results['reuse']:
                occurrences = self.results['reuse'][combo]['occurrences']

                # Log-scale histogram
                axes[i].hist(occurrences, bins=50, color=self.colors[combo], edgecolor='black', alpha=0.7)
                axes[i].set_xlabel('Fragment Occurrences (compounds)')
                axes[i].set_ylabel('Count')
                axes[i].set_title(f'{combo}\n(Mean: {np.mean(occurrences):.1f}, Max: {np.max(occurrences)})',
                                 fontweight='bold', color=self.colors[combo])
                axes[i].axvline(np.mean(occurrences), color='red', linestyle='--', linewidth=2, label='Mean')
                axes[i].axvline(np.median(occurrences), color='blue', linestyle=':', linewidth=2, label='Median')
                axes[i].legend()

                # Add log scale if range is large
                if np.max(occurrences) > 100:
                    axes[i].set_yscale('log')

        plt.suptitle('Fragment Reuse Patterns Across Dual-Activity Combinations', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/reuse_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_complexity_radar(self):
        """Plot molecular complexity radar chart."""
        print("  Creating complexity radar plot...")

        # Properties to plot
        properties = ['num_rings', 'num_aromatic_rings', 'molecular_weight',
                     'num_heavy_atoms', 'bertz_complexity', 'tpsa']

        # Normalize data for radar plot
        normalized_data = {}
        for prop in properties:
            all_values = []
            for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
                if combo in self.results['complexity']:
                    if prop in self.results['complexity'][combo]['properties']:
                        all_values.append(self.results['complexity'][combo]['properties'][prop]['mean'])

            if all_values:
                min_val, max_val = min(all_values), max(all_values)
                for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
                    if combo not in normalized_data:
                        normalized_data[combo] = {}
                    if combo in self.results['complexity']:
                        if prop in self.results['complexity'][combo]['properties']:
                            val = self.results['complexity'][combo]['properties'][prop]['mean']
                            normalized_data[combo][prop] = (val - min_val) / (max_val - min_val + 0.001)

        # Create radar plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, len(properties), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo in normalized_data:
                values = [normalized_data[combo].get(p, 0) for p in properties]
                values += values[:1]  # Complete the circle

                ax.plot(angles, values, 'o-', linewidth=2, label=combo, color=self.colors[combo])
                ax.fill(angles, values, alpha=0.25, color=self.colors[combo])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([p.replace('_', '\n') for p in properties], fontsize=10)
        ax.set_title('Molecular Complexity Comparison\n(Normalized Values)', fontweight='bold', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/complexity_radar.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_fragment_heatmap(self):
        """Plot fragment occurrence heatmap for top fragments."""
        print("  Creating fragment heatmap...")

        # Get top 20 fragments across all combinations
        all_top_fragments = []
        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            if combo in self.results['reuse']:
                for frag in self.results['reuse'][combo]['top_10']:
                    all_top_fragments.append({
                        'smiles': frag['smiles'],
                        'occurrences': frag['occurrences'],
                        'combination': combo
                    })

        # Create heatmap data
        unique_smiles = list(set([f['smiles'] for f in all_top_fragments]))[:20]

        if len(unique_smiles) < 2:
            print("    Not enough data for heatmap")
            return

        heatmap_data = np.zeros((len(unique_smiles), 3))

        for i, smiles in enumerate(unique_smiles):
            for j, combo in enumerate(['SA+CA', 'SA+EC', 'CA+EC']):
                if combo in self.results['reuse']:
                    df = self.fragments[combo]
                    match = df[df['fragment_smiles'] == smiles]
                    if len(match) > 0:
                        heatmap_data[i, j] = match.iloc[0]['total_compounds_both_pathogens']

        # Create short labels for SMILES
        short_labels = [s[:20] + '...' if len(s) > 20 else s for s in unique_smiles]

        fig, ax = plt.subplots(figsize=(10, max(8, len(unique_smiles) * 0.4)))
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=['SA+CA', 'SA+EC', 'CA+EC'],
                   yticklabels=short_labels, ax=ax)
        ax.set_title('Top Fragment Occurrences Across Combinations', fontweight='bold', fontsize=14)
        ax.set_xlabel('Dual-Activity Combination')
        ax.set_ylabel('Fragment (truncated SMILES)')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fragment_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_dendrograms(self):
        """Plot hierarchical clustering dendrograms."""
        print("  Creating dendrograms...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 8))

        for i, combo in enumerate(['SA+CA', 'SA+EC', 'CA+EC']):
            if combo in self.results['clustering']:
                linkage_matrix = self.results['clustering'][combo]['linkage_matrix']

                dendrogram(linkage_matrix, ax=axes[i], leaf_rotation=90,
                          color_threshold=0.7 * max(linkage_matrix[:, 2]))
                axes[i].set_title(f'{combo}\n(k={self.results["clustering"][combo]["optimal_k"]} clusters, '
                                 f'silhouette={self.results["clustering"][combo]["best_silhouette"]:.3f})',
                                 fontweight='bold', color=self.colors[combo])
                axes[i].set_xlabel('Fragment Index')
                axes[i].set_ylabel('Distance (1 - Tanimoto)')

        plt.suptitle('Hierarchical Clustering Dendrograms', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/dendrograms.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_summary_dashboard(self):
        """Create summary dashboard with key metrics."""
        print("  Creating summary dashboard...")

        fig = plt.figure(figsize=(20, 16))

        # Grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Fragment extraction efficiency (bar)
        ax1 = fig.add_subplot(gs[0, 0])
        combos = ['SA+CA', 'SA+EC', 'CA+EC']
        efficiencies = [self.results['diversity'][c]['fragments_per_compound'] for c in combos if c in self.results['diversity']]
        colors_list = [self.colors[c] for c in combos if c in self.results['diversity']]
        bars = ax1.bar(combos[:len(efficiencies)], efficiencies, color=colors_list, edgecolor='black')
        ax1.set_ylabel('Fragments per Compound')
        ax1.set_title('Fragment Extraction Efficiency', fontweight='bold')
        for bar, eff in zip(bars, efficiencies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{eff:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. Mean similarity (bar)
        ax2 = fig.add_subplot(gs[0, 1])
        means = [self.results['similarity'][c]['mean'] for c in combos if c in self.results['similarity']]
        bars = ax2.bar(combos[:len(means)], means, color=colors_list[:len(means)], edgecolor='black')
        ax2.set_ylabel('Mean Tanimoto Similarity')
        ax2.set_title('Structural Similarity', fontweight='bold')
        for bar, mean in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

        # 3. Mean reuse (bar)
        ax3 = fig.add_subplot(gs[0, 2])
        reuses = [self.results['reuse'][c]['mean'] for c in combos if c in self.results['reuse']]
        bars = ax3.bar(combos[:len(reuses)], reuses, color=colors_list[:len(reuses)], edgecolor='black')
        ax3.set_ylabel('Mean Fragment Reuse (compounds)')
        ax3.set_title('Fragment Reuse Rate', fontweight='bold')
        for bar, reuse in zip(bars, reuses):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{reuse:.1f}', ha='center', va='bottom', fontweight='bold')

        # 4. Complexity comparison (grouped bar)
        ax4 = fig.add_subplot(gs[1, :2])
        props_to_show = ['num_rings', 'num_aromatic_rings', 'molecular_weight', 'bertz_complexity']
        x = np.arange(len(props_to_show))
        width = 0.25

        for i, combo in enumerate(['SA+CA', 'SA+EC', 'CA+EC']):
            if combo in self.results['complexity']:
                values = []
                for prop in props_to_show:
                    if prop in self.results['complexity'][combo]['properties']:
                        val = self.results['complexity'][combo]['properties'][prop]['mean']
                        # Normalize for display
                        if prop == 'molecular_weight':
                            val = val / 10
                        elif prop == 'bertz_complexity':
                            val = val / 50
                        values.append(val)
                    else:
                        values.append(0)
                ax4.bar(x + i*width, values, width, label=combo, color=self.colors[combo], edgecolor='black')

        ax4.set_ylabel('Value (scaled)')
        ax4.set_title('Molecular Complexity Metrics', fontweight='bold')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels([p.replace('_', ' ').title() for p in props_to_show])
        ax4.legend()

        # 5. Clustering metrics
        ax5 = fig.add_subplot(gs[1, 2])
        if any(c in self.results['clustering'] for c in combos):
            silhouettes = [self.results['clustering'][c]['best_silhouette']
                          for c in combos if c in self.results['clustering']]
            combos_with_data = [c for c in combos if c in self.results['clustering']]
            bars = ax5.bar(combos_with_data, silhouettes,
                          color=[self.colors[c] for c in combos_with_data], edgecolor='black')
            ax5.set_ylabel('Silhouette Score')
            ax5.set_title('Clustering Quality', fontweight='bold')
            for bar, sil in zip(bars, silhouettes):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{sil:.3f}', ha='center', va='bottom', fontweight='bold')

        # 6. Text summary
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')

        summary_text = self._generate_verdict_summary()
        ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')

        plt.suptitle('CHEMICAL CONVERGENCE HYPOTHESIS TESTING DASHBOARD',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_verdict_summary(self):
        """Generate text summary for verdict."""
        lines = []
        lines.append("=" * 70)
        lines.append("HYPOTHESIS VERDICT SUMMARY")
        lines.append("=" * 70)

        # Calculate evidence scores
        evidence_for = 0
        evidence_against = 0

        # 1. Similarity check
        if 'SA+CA' in self.results['similarity'] and 'SA+EC' in self.results['similarity']:
            sa_ca_sim = self.results['similarity']['SA+CA']['mean']
            sa_ec_sim = self.results['similarity']['SA+EC']['mean']
            if sa_ca_sim > sa_ec_sim:
                evidence_for += 1
                lines.append(f"[+] Higher similarity in SA+CA: {sa_ca_sim:.4f} vs {sa_ec_sim:.4f}")
            else:
                evidence_against += 1
                lines.append(f"[-] Lower similarity in SA+CA: {sa_ca_sim:.4f} vs {sa_ec_sim:.4f}")

        # 2. Diversity check
        if 'SA+CA' in self.results['diversity'] and 'SA+EC' in self.results['diversity']:
            sa_ca_div = self.results['diversity']['SA+CA']['fragments_per_compound']
            sa_ec_div = self.results['diversity']['SA+EC']['fragments_per_compound']
            if sa_ca_div < sa_ec_div:
                evidence_for += 1
                lines.append(f"[+] Lower diversity in SA+CA: {sa_ca_div:.3f} vs {sa_ec_div:.3f}")
            else:
                evidence_against += 1
                lines.append(f"[-] Higher diversity in SA+CA: {sa_ca_div:.3f} vs {sa_ec_div:.3f}")

        # 3. Reuse check
        if 'SA+CA' in self.results['reuse'] and 'SA+EC' in self.results['reuse']:
            sa_ca_reuse = self.results['reuse']['SA+CA']['mean']
            sa_ec_reuse = self.results['reuse']['SA+EC']['mean']
            if sa_ca_reuse > sa_ec_reuse:
                evidence_for += 1
                lines.append(f"[+] Higher reuse in SA+CA: {sa_ca_reuse:.1f} vs {sa_ec_reuse:.1f}")
            else:
                evidence_against += 1
                lines.append(f"[-] Lower reuse in SA+CA: {sa_ca_reuse:.1f} vs {sa_ec_reuse:.1f}")

        # 4. Complexity check
        try:
            if 'SA+CA' in self.results['complexity'] and 'SA+EC' in self.results['complexity']:
                sa_ca_comp = self.results['complexity']['SA+CA']['properties']['bertz_complexity']['mean']
                sa_ec_comp = self.results['complexity']['SA+EC']['properties']['bertz_complexity']['mean']
                if sa_ca_comp < sa_ec_comp:
                    evidence_for += 1
                    lines.append(f"[+] Lower complexity in SA+CA: {sa_ca_comp:.1f} vs {sa_ec_comp:.1f}")
                else:
                    evidence_against += 1
                    lines.append(f"[-] Higher complexity in SA+CA: {sa_ca_comp:.1f} vs {sa_ec_comp:.1f}")
        except:
            pass

        # 5. Clustering check
        if 'SA+CA' in self.results['clustering'] and 'SA+EC' in self.results['clustering']:
            sa_ca_sil = self.results['clustering']['SA+CA']['best_silhouette']
            sa_ec_sil = self.results['clustering']['SA+EC']['best_silhouette']
            if sa_ca_sil > sa_ec_sil:
                evidence_for += 1
                lines.append(f"[+] Tighter clusters in SA+CA: {sa_ca_sil:.3f} vs {sa_ec_sil:.3f}")
            else:
                evidence_against += 1
                lines.append(f"[-] Looser clusters in SA+CA: {sa_ca_sil:.3f} vs {sa_ec_sil:.3f}")

        lines.append("-" * 70)
        lines.append(f"Evidence FOR convergence: {evidence_for}/5")
        lines.append(f"Evidence AGAINST convergence: {evidence_against}/5")

        # Verdict
        if evidence_for >= 4:
            verdict = "STRONGLY SUPPORTED"
        elif evidence_for >= 3:
            verdict = "SUPPORTED"
        elif evidence_for == evidence_against:
            verdict = "INCONCLUSIVE"
        elif evidence_against >= 3:
            verdict = "REJECTED"
        else:
            verdict = "WEAKLY REJECTED"

        lines.append("")
        lines.append(f"VERDICT: Convergence Hypothesis is {verdict}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def generate_report(self):
        """Generate comprehensive text report."""
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 70)

        report_path = f'{self.output_dir}/chemical_convergence_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CHEMICAL CONVERGENCE HYPOTHESIS TESTING REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Hypothesis
            f.write("HYPOTHESIS\n")
            f.write("-" * 80 + "\n")
            f.write("Low fragments per compound indicates chemical convergence\n")
            f.write("(similar scaffolds being reused) for dual-activity antimicrobial compounds.\n\n")

            # Background
            f.write("BACKGROUND DATA\n")
            f.write("-" * 80 + "\n")
            for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
                if combo in self.results['diversity']:
                    r = self.results['diversity'][combo]
                    f.write(f"{combo}:\n")
                    f.write(f"  Source compounds: {r['total_compounds']}\n")
                    f.write(f"  Total fragments: {r['total_fragments']}\n")
                    f.write(f"  Fragments/compound: {r['fragments_per_compound']:.3f}\n\n")

            # Analysis 1: Similarity
            f.write("\n" + "=" * 80 + "\n")
            f.write("ANALYSIS 1: STRUCTURAL SIMILARITY\n")
            f.write("=" * 80 + "\n")
            f.write("Expectation if convergent: SA+CA should show HIGHER similarity\n\n")

            for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
                if combo in self.results['similarity']:
                    r = self.results['similarity'][combo]
                    f.write(f"{combo}:\n")
                    f.write(f"  Mean Tanimoto: {r['mean']:.4f}\n")
                    f.write(f"  Median: {r['median']:.4f}\n")
                    f.write(f"  Std: {r['std']:.4f}\n")
                    f.write(f"  Range: {r['min']:.4f} - {r['max']:.4f}\n")
                    f.write(f"  N pairs: {r['n_pairs']:,}\n\n")

            # Analysis 2: Diversity
            f.write("\n" + "=" * 80 + "\n")
            f.write("ANALYSIS 2: SCAFFOLD DIVERSITY\n")
            f.write("=" * 80 + "\n")
            f.write("Expectation if convergent: SA+CA should show LOWER diversity\n\n")

            for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
                if combo in self.results['diversity']:
                    r = self.results['diversity'][combo]
                    f.write(f"{combo}:\n")
                    f.write(f"  Diversity ratio: {r['diversity_ratio']:.4f}\n")
                    f.write(f"  Unique fragments: {r['unique_fragments']}\n")
                    f.write(f"  Singletons: {r['singletons']} ({r['singleton_pct']:.1f}%)\n\n")

            # Analysis 3: Reuse
            f.write("\n" + "=" * 80 + "\n")
            f.write("ANALYSIS 3: FRAGMENT REUSE\n")
            f.write("=" * 80 + "\n")
            f.write("Expectation if convergent: SA+CA should show HIGHER reuse\n\n")

            for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
                if combo in self.results['reuse']:
                    r = self.results['reuse'][combo]
                    f.write(f"{combo}:\n")
                    f.write(f"  Mean occurrences: {r['mean']:.1f}\n")
                    f.write(f"  Median: {r['median']:.1f}\n")
                    f.write(f"  Max: {r['max']}\n")
                    f.write(f"  Total instances: {r['total_fragment_instances']:,}\n\n")

            # Analysis 4: Complexity
            f.write("\n" + "=" * 80 + "\n")
            f.write("ANALYSIS 4: MOLECULAR COMPLEXITY\n")
            f.write("=" * 80 + "\n")
            f.write("Expectation if convergent: SA+CA should show LOWER complexity\n\n")

            for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
                if combo in self.results['complexity']:
                    f.write(f"{combo}:\n")
                    props = self.results['complexity'][combo]['properties']
                    for prop in ['bertz_complexity', 'num_rings', 'molecular_weight']:
                        if prop in props:
                            f.write(f"  {prop}: {props[prop]['mean']:.2f} (std: {props[prop]['std']:.2f})\n")
                    f.write("\n")

            # Analysis 5: Clustering
            f.write("\n" + "=" * 80 + "\n")
            f.write("ANALYSIS 5: CHEMICAL CLUSTERING\n")
            f.write("=" * 80 + "\n")
            f.write("Expectation if convergent: SA+CA should show TIGHTER clusters\n\n")

            for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
                if combo in self.results['clustering']:
                    r = self.results['clustering'][combo]
                    f.write(f"{combo}:\n")
                    f.write(f"  Optimal clusters: {r['optimal_k']}\n")
                    f.write(f"  Silhouette score: {r['best_silhouette']:.4f}\n")
                    f.write(f"  Avg intra-cluster sim: {r['avg_intra_cluster_similarity']:.4f}\n\n")

            # Statistical tests
            f.write("\n" + "=" * 80 + "\n")
            f.write("STATISTICAL TESTS\n")
            f.write("=" * 80 + "\n\n")

            if 'similarity_tests' in self.results['statistics']:
                f.write("Similarity Comparisons:\n")
                for key, val in self.results['statistics']['similarity_tests'].items():
                    sig = "***" if val['p_value'] < 0.001 else "**" if val['p_value'] < 0.01 else "*" if val['p_value'] < 0.05 else ""
                    f.write(f"  {key}: U={val['statistic']:.0f}, p={val['p_value']:.2e}, d={val['effect_size']:.3f} {sig}\n")

            # Verdict
            f.write("\n" + "=" * 80 + "\n")
            f.write("VERDICT\n")
            f.write("=" * 80 + "\n\n")
            f.write(self._generate_verdict_summary())

            # Biological interpretation
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("BIOLOGICAL INTERPRETATION\n")
            f.write("=" * 80 + "\n\n")

            f.write("If convergence is SUPPORTED:\n")
            f.write("-" * 40 + "\n")
            f.write("SA+CA compounds share common chemical scaffolds because:\n")
            f.write("1. Similar membrane-targeting mechanisms for Gram+ and Fungi\n")
            f.write("2. Convergent chemical requirements for dual activity\n")
            f.write("3. Conserved binding sites across pathogen types\n\n")

            f.write("If convergence is REJECTED:\n")
            f.write("-" * 40 + "\n")
            f.write("The low fragment count is an artifact due to:\n")
            f.write("1. Dataset composition or sampling bias\n")
            f.write("2. Fragment extraction methodology differences\n")
            f.write("3. Structural heterogeneity despite low fragment count\n\n")

            # Limitations
            f.write("LIMITATIONS & CAVEATS\n")
            f.write("-" * 40 + "\n")
            f.write("1. Fragment extraction efficiency varies by combination\n")
            f.write("2. Tanimoto similarity may miss subtle structural differences\n")
            f.write("3. Clustering quality depends on fragment sample size\n")
            f.write("4. Statistical power differs due to unequal group sizes\n\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"Report saved to {report_path}")
        return report_path

    def save_results_csv(self):
        """Save analysis results to CSV files."""
        print("\nSaving results to CSV...")

        # Property comparison table
        rows = []
        for combo in ['SA+CA', 'SA+EC', 'CA+EC']:
            row = {'combination': combo}

            if combo in self.results['diversity']:
                row.update({
                    'total_fragments': self.results['diversity'][combo]['total_fragments'],
                    'total_compounds': self.results['diversity'][combo]['total_compounds'],
                    'fragments_per_compound': self.results['diversity'][combo]['fragments_per_compound'],
                    'diversity_ratio': self.results['diversity'][combo]['diversity_ratio'],
                    'singleton_pct': self.results['diversity'][combo]['singleton_pct']
                })

            if combo in self.results['similarity']:
                row.update({
                    'mean_similarity': self.results['similarity'][combo]['mean'],
                    'median_similarity': self.results['similarity'][combo]['median'],
                    'std_similarity': self.results['similarity'][combo]['std']
                })

            if combo in self.results['reuse']:
                row.update({
                    'mean_reuse': self.results['reuse'][combo]['mean'],
                    'max_reuse': self.results['reuse'][combo]['max']
                })

            if combo in self.results['complexity']:
                for prop in ['bertz_complexity', 'num_rings', 'molecular_weight']:
                    if prop in self.results['complexity'][combo]['properties']:
                        row[f'mean_{prop}'] = self.results['complexity'][combo]['properties'][prop]['mean']

            if combo in self.results['clustering']:
                row.update({
                    'optimal_clusters': self.results['clustering'][combo]['optimal_k'],
                    'silhouette_score': self.results['clustering'][combo]['best_silhouette'],
                    'intra_cluster_sim': self.results['clustering'][combo]['avg_intra_cluster_similarity']
                })

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(f'{self.output_dir}/property_comparison.csv', index=False)
        print(f"  Saved property_comparison.csv")

        return df

    def run_full_analysis(self, file_paths):
        """Run the complete analysis pipeline."""
        print("\n" + "#" * 80)
        print("CHEMICAL CONVERGENCE HYPOTHESIS TESTING")
        print("#" * 80)
        print("\nHypothesis: Low fragments per compound indicates chemical convergence")
        print("Testing SA+CA (0.248 frag/comp) vs SA+EC (6.655 frag/comp)\n")

        # Load data
        self.load_data(file_paths)

        # Compute fingerprints
        self.compute_fingerprints()

        # Compute similarity matrices
        self.compute_similarity_matrices()

        # Run analyses
        self.analyze_structural_similarity()
        self.analyze_scaffold_diversity()
        self.analyze_fragment_reuse()
        self.analyze_molecular_complexity()
        self.analyze_chemical_clustering()

        # Statistical tests
        self.perform_statistical_tests()

        # Generate outputs
        self.create_visualizations()
        self.generate_report()
        self.save_results_csv()

        # Print final verdict
        print("\n" + "#" * 80)
        print("FINAL VERDICT")
        print("#" * 80)
        print(self._generate_verdict_summary())

        return self.results


def main():
    """Main execution function."""
    print("=" * 80)
    print("CHEMICAL CONVERGENCE HYPOTHESIS TESTING SCRIPT")
    print("=" * 80)
    print("Testing: Low fragments/compound = Chemical convergence?")
    print("=" * 80)

    # File paths
    file_paths = {
        'dual_SA_CA_positive_scaffolds': 'dual_SA_CA_positive_scaffolds.csv',
        'dual_SA_CA_positive_substitutents': 'dual_SA_CA_positive_substitutents.csv',
        'dual_SA_EC_positive_scaffolds': 'dual_SA_EC_positive_scaffolds.csv',
        'dual_SA_EC_positive_substitutents': 'dual_SA_EC_positive_substitutents.csv',
        'dual_CA_EC_positive_scaffolds': 'dual_CA_EC_positive_scaffolds.csv',
        'dual_CA_EC_positive_substitutents': 'dual_CA_EC_positive_substitutents.csv'
    }

    # Check files exist
    print("\nChecking input files...")
    for key, path in file_paths.items():
        if os.path.exists(path):
            print(f"  [OK] {path}")
        else:
            print(f"  [MISSING] {path}")

    # Initialize analyzer
    analyzer = ChemicalConvergenceAnalyzer(output_dir='convergence_analysis')

    # Run full analysis
    try:
        results = analyzer.run_full_analysis(file_paths)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  - convergence_analysis/chemical_convergence_report.txt")
        print("  - convergence_analysis/property_comparison.csv")
        print("  - convergence_analysis/similarity_distributions.png")
        print("  - convergence_analysis/reuse_distributions.png")
        print("  - convergence_analysis/complexity_radar.png")
        print("  - convergence_analysis/fragment_heatmap.png")
        print("  - convergence_analysis/dendrograms.png")
        print("  - convergence_analysis/summary_dashboard.png")

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
