-- Summary Statistics for Multipathogen Negative Scaffolds (Using Reliable Compounds)
-- Scaffolds present in ALL THREE pathogens with negative attribution

WITH all_pathogen_negative_scaffolds AS (
    -- Get scaffolds that appear with negative attribution in ALL three pathogens
    SELECT cf.fragment_id
    FROM compound_fragments cf
    JOIN fragments f ON cf.fragment_id = f.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE f.fragment_type = 'scaffold'
      AND cf.attribution_score <= -0.1  -- Negative threshold
      AND rc.has_mismatch = 0  -- Zero mismatch requirement
      AND rc.reliability_level = 'RELIABLE'  -- Only reliable compounds
    GROUP BY cf.fragment_id
    HAVING COUNT(DISTINCT cf.pathogen_id) = 3  -- Present in all three pathogens
),

scaffold_stats AS (
    -- Calculate statistics for each scaffold across all pathogens
    SELECT 
        apns.fragment_id,
        f.smiles as fragment_smiles,
        cf.pathogen_id,
        COUNT(*) as total_compounds,
        COUNT(CASE WHEN rc.target = 1 THEN 1 END) as tp_count,
        COUNT(CASE WHEN rc.target = 0 THEN 1 END) as tn_count,
        AVG(cf.attribution_score) as avg_attribution,
        MIN(cf.attribution_score) as min_attribution,
        ROUND(COUNT(CASE WHEN rc.target = 0 THEN 1 END)::numeric / COUNT(*)::numeric * 100, 2) as inactivity_rate_percent
    FROM all_pathogen_negative_scaffolds apns
    JOIN fragments f ON apns.fragment_id = f.fragment_id
    JOIN compound_fragments cf ON apns.fragment_id = cf.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE rc.has_mismatch = 0  -- Ensure all compounds are reliable
      AND rc.reliability_level = 'RELIABLE'
      AND cf.attribution_score <= -0.1  -- Only negative attributions
    GROUP BY apns.fragment_id, f.smiles, cf.pathogen_id
    HAVING COUNT(CASE WHEN rc.target = 0 THEN 1 END) >= 3  -- At least 3 TN cases per pathogen
),

aggregated_scaffold_stats AS (
    -- Aggregate statistics across all pathogens for each scaffold
    SELECT 
        fragment_id,
        fragment_smiles,
        COUNT(DISTINCT pathogen_id) as pathogen_count,
        SUM(total_compounds) as total_compounds_all_pathogens,
        SUM(tp_count) as total_tp_count,
        SUM(tn_count) as total_tn_count,
        AVG(avg_attribution) as overall_avg_attribution,
        MIN(min_attribution) as overall_min_attribution,
        AVG(inactivity_rate_percent) as avg_inactivity_rate_percent
    FROM scaffold_stats
    GROUP BY fragment_id, fragment_smiles
    HAVING COUNT(DISTINCT pathogen_id) = 3  -- Ensure present in all three pathogens
),

overall_stats AS (
    SELECT 
        COUNT(*) as total_multipathogen_scaffolds,
        ROUND(AVG(total_tn_count)::numeric, 1) as avg_tn_per_scaffold,
        ROUND(AVG(total_tp_count)::numeric, 1) as avg_tp_per_scaffold,
        ROUND(AVG(total_compounds_all_pathogens)::numeric, 1) as avg_compounds_per_scaffold,
        SUM(total_tn_count) as total_tn_compounds,
        SUM(total_tp_count) as total_tp_compounds,
        SUM(total_compounds_all_pathogens) as total_compounds_analyzed,
        ROUND(AVG(overall_avg_attribution)::numeric, 4) as overall_avg_attribution,
        ROUND(AVG(avg_inactivity_rate_percent)::numeric, 1) as avg_inactivity_rate_percent,
        ROUND(MAX(overall_avg_attribution)::numeric, 4) as max_scaffold_attribution,
        ROUND(MIN(overall_avg_attribution)::numeric, 4) as min_scaffold_attribution
    FROM aggregated_scaffold_stats
),

top_scaffolds AS (
    SELECT 
        COUNT(CASE WHEN total_tn_count >= 30 THEN 1 END) as scaffolds_with_30plus_tn,  -- 10+ per pathogen
        COUNT(CASE WHEN total_tn_count >= 60 THEN 1 END) as scaffolds_with_60plus_tn,  -- 20+ per pathogen
        COUNT(CASE WHEN total_tn_count >= 150 THEN 1 END) as scaffolds_with_150plus_tn, -- 50+ per pathogen
        COUNT(CASE WHEN avg_inactivity_rate_percent >= 50 THEN 1 END) as scaffolds_50plus_inactivity_rate,
        COUNT(CASE WHEN avg_inactivity_rate_percent >= 70 THEN 1 END) as scaffolds_70plus_inactivity_rate,
        COUNT(CASE WHEN overall_avg_attribution <= -0.2 THEN 1 END) as scaffolds_high_negative_attribution
    FROM aggregated_scaffold_stats
),

distribution_stats AS (
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_tn_count) as tn_count_q25,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_tn_count) as tn_count_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_tn_count) as tn_count_q75,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY overall_avg_attribution) as attribution_q25,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY overall_avg_attribution) as attribution_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY overall_avg_attribution) as attribution_q75
    FROM aggregated_scaffold_stats
),

pathogen_breakdown AS (
    SELECT 
        'S. aureus' as pathogen_id,
        COUNT(*) as scaffolds_in_pathogen,
        ROUND(AVG(total_compounds)::numeric, 1) as avg_compounds_per_scaffold,
        ROUND(AVG(tn_count)::numeric, 1) as avg_tn_per_scaffold,
        ROUND(AVG(avg_attribution)::numeric, 4) as avg_attribution
    FROM scaffold_stats
    WHERE pathogen_id = 's_aureus'
    UNION ALL
    SELECT 
        'C. albicans' as pathogen_id,
        COUNT(*) as scaffolds_in_pathogen,
        ROUND(AVG(total_compounds)::numeric, 1) as avg_compounds_per_scaffold,
        ROUND(AVG(tn_count)::numeric, 1) as avg_tn_per_scaffold,
        ROUND(AVG(avg_attribution)::numeric, 4) as avg_attribution
    FROM scaffold_stats
    WHERE pathogen_id = 'c_albicans'
    UNION ALL
    SELECT 
        'E. coli' as pathogen_id,
        COUNT(*) as scaffolds_in_pathogen,
        ROUND(AVG(total_compounds)::numeric, 1) as avg_compounds_per_scaffold,
        ROUND(AVG(tn_count)::numeric, 1) as avg_tn_per_scaffold,
        ROUND(AVG(avg_attribution)::numeric, 4) as avg_attribution
    FROM scaffold_stats
    WHERE pathogen_id = 'e_coli'
)

-- Final summary report
SELECT 
    'Multipathogen Negative Scaffolds Summary (Reliable Compounds Only)' as analysis_type,
    'ALL THREE PATHOGENS' as pathogen_combination,
    os.total_multipathogen_scaffolds,
    os.avg_tn_per_scaffold,
    os.avg_tp_per_scaffold,
    os.avg_compounds_per_scaffold,
    os.total_tn_compounds,
    os.total_tp_compounds,
    os.total_compounds_analyzed,
    os.overall_avg_attribution,
    os.avg_inactivity_rate_percent,
    os.min_scaffold_attribution,
    os.max_scaffold_attribution,
    
    -- Distribution statistics
    ROUND(ds.tn_count_q25::numeric, 1) as tn_count_q25,
    ROUND(ds.tn_count_median::numeric, 1) as tn_count_median,  
    ROUND(ds.tn_count_q75::numeric, 1) as tn_count_q75,
    ROUND(ds.attribution_q25::numeric, 4) as attribution_q25,
    ROUND(ds.attribution_median::numeric, 4) as attribution_median,
    ROUND(ds.attribution_q75::numeric, 4) as attribution_q75,
    
    -- Quality metrics
    ts.scaffolds_with_30plus_tn,
    ts.scaffolds_with_60plus_tn,
    ts.scaffolds_with_150plus_tn,
    ts.scaffolds_50plus_inactivity_rate,
    ts.scaffolds_70plus_inactivity_rate,
    ts.scaffolds_high_negative_attribution,
    
    -- Quality percentages
    ROUND((ts.scaffolds_with_30plus_tn::numeric / os.total_multipathogen_scaffolds * 100), 1) as pct_scaffolds_30plus_tn,
    ROUND((ts.scaffolds_50plus_inactivity_rate::numeric / os.total_multipathogen_scaffolds * 100), 1) as pct_scaffolds_50plus_inactivity,
    ROUND((ts.scaffolds_high_negative_attribution::numeric / os.total_multipathogen_scaffolds * 100), 1) as pct_scaffolds_high_negative_attribution,
    
    -- Per-pathogen breakdown
    STRING_AGG(
        pb.pathogen_id || ': ' || pb.scaffolds_in_pathogen || ' scaffolds | Avg: ' || 
        pb.avg_compounds_per_scaffold || ' compounds (' || pb.avg_tn_per_scaffold || ' TN) | ' ||
        'Attr: ' || pb.avg_attribution,
        ' | '
        ORDER BY pb.pathogen_id
    ) as pathogen_breakdown

FROM overall_stats os
CROSS JOIN top_scaffolds ts  
CROSS JOIN distribution_stats ds
CROSS JOIN pathogen_breakdown pb
GROUP BY os.total_multipathogen_scaffolds, os.avg_tn_per_scaffold, os.avg_tp_per_scaffold,
         os.avg_compounds_per_scaffold, os.total_tn_compounds, os.total_tp_compounds,
         os.total_compounds_analyzed, os.overall_avg_attribution, os.avg_inactivity_rate_percent,
         os.min_scaffold_attribution, os.max_scaffold_attribution, ds.tn_count_q25,
         ds.tn_count_median, ds.tn_count_q75, ds.attribution_q25, ds.attribution_median,
         ds.attribution_q75, ts.scaffolds_with_30plus_tn, ts.scaffolds_with_60plus_tn,
         ts.scaffolds_with_150plus_tn, ts.scaffolds_50plus_inactivity_rate,
         ts.scaffolds_70plus_inactivity_rate, ts.scaffolds_high_negative_attribution;