-- ====================================================================================
-- DUAL PATHOGEN SUMMARY QUERIES: S. aureus + C. albicans (but NOT E. coli)
-- ====================================================================================

-- SUMMARY 1: Dual S. aureus + C. albicans Negative substituents Summary
-- ====================================================================================

WITH target_pathogen_negative_substituents AS (
    SELECT cf.fragment_id
    FROM compound_fragments cf
    JOIN fragments f ON cf.fragment_id = f.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE f.fragment_type = 'substituent'
      AND cf.attribution_score <= -0.1
      AND rc.has_mismatch = 0
      AND rc.reliability_level = 'RELIABLE'
      AND cf.pathogen_id IN ('s_aureus', 'c_albicans')
    GROUP BY cf.fragment_id
    HAVING COUNT(DISTINCT cf.pathogen_id) = 2
),

e_coli_substituents AS (
    SELECT DISTINCT cf.fragment_id
    FROM compound_fragments cf
    JOIN fragments f ON cf.fragment_id = f.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE f.fragment_type = 'substituent'
      AND cf.pathogen_id = 'e_coli'
      AND rc.has_mismatch = 0
      AND rc.reliability_level = 'RELIABLE'
),

dual_specific_negative_substituents AS (
    SELECT fragment_id
    FROM target_pathogen_negative_substituents
    WHERE fragment_id NOT IN (SELECT fragment_id FROM e_coli_substituents)
),

substituent_stats AS (
    SELECT 
        dsns.fragment_id,
        f.smiles as fragment_smiles,
        cf.pathogen_id,
        COUNT(*) as total_compounds,
        COUNT(CASE WHEN rc.target = 1 THEN 1 END) as tp_count,
        COUNT(CASE WHEN rc.target = 0 THEN 1 END) as tn_count,
        AVG(cf.attribution_score) as avg_attribution,
        MIN(cf.attribution_score) as min_attribution,
        ROUND(COUNT(CASE WHEN rc.target = 0 THEN 1 END)::numeric / COUNT(*)::numeric * 100, 2) as inactivity_rate_percent
    FROM dual_specific_negative_substituents dsns
    JOIN fragments f ON dsns.fragment_id = f.fragment_id
    JOIN compound_fragments cf ON dsns.fragment_id = cf.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE rc.has_mismatch = 0
      AND rc.reliability_level = 'RELIABLE'
      AND cf.attribution_score <= -0.1
      AND cf.pathogen_id IN ('s_aureus', 'c_albicans')
    GROUP BY dsns.fragment_id, f.smiles, cf.pathogen_id
    HAVING COUNT(CASE WHEN rc.target = 0 THEN 1 END) >= 3
),

aggregated_substituent_stats AS (
    SELECT 
        fragment_id,
        fragment_smiles,
        COUNT(DISTINCT pathogen_id) as pathogen_count,
        SUM(total_compounds) as total_compounds_both_pathogens,
        SUM(tp_count) as total_tp_count,
        SUM(tn_count) as total_tn_count,
        AVG(avg_attribution) as overall_avg_attribution,
        MIN(min_attribution) as overall_min_attribution,
        AVG(inactivity_rate_percent) as avg_inactivity_rate_percent
    FROM substituent_stats
    GROUP BY fragment_id, fragment_smiles
    HAVING COUNT(DISTINCT pathogen_id) = 2
),

overall_stats AS (
    SELECT 
        COUNT(*) as total_dual_substituents,
        ROUND(AVG(total_tn_count)::numeric, 1) as avg_tn_per_substituent,
        ROUND(AVG(total_tp_count)::numeric, 1) as avg_tp_per_substituent,
        ROUND(AVG(total_compounds_both_pathogens)::numeric, 1) as avg_compounds_per_substituent,
        SUM(total_tn_count) as total_tn_compounds,
        SUM(total_tp_count) as total_tp_compounds,
        SUM(total_compounds_both_pathogens) as total_compounds_analyzed,
        ROUND(AVG(overall_avg_attribution)::numeric, 4) as overall_avg_attribution,
        ROUND(AVG(avg_inactivity_rate_percent)::numeric, 1) as avg_inactivity_rate_percent,
        ROUND(MAX(overall_avg_attribution)::numeric, 4) as max_substituent_attribution,
        ROUND(MIN(overall_avg_attribution)::numeric, 4) as min_substituent_attribution
    FROM aggregated_substituent_stats
),

top_substituents AS (
    SELECT 
        COUNT(CASE WHEN total_tn_count >= 20 THEN 1 END) as substituents_with_20plus_tn,   -- 10+ per pathogen
        COUNT(CASE WHEN total_tn_count >= 40 THEN 1 END) as substituents_with_40plus_tn,   -- 20+ per pathogen
        COUNT(CASE WHEN total_tn_count >= 100 THEN 1 END) as substituents_with_100plus_tn, -- 50+ per pathogen
        COUNT(CASE WHEN avg_inactivity_rate_percent >= 50 THEN 1 END) as substituents_50plus_inactivity_rate,
        COUNT(CASE WHEN avg_inactivity_rate_percent >= 70 THEN 1 END) as substituents_70plus_inactivity_rate,
        COUNT(CASE WHEN overall_avg_attribution <= -0.2 THEN 1 END) as substituents_high_negative_attribution
    FROM aggregated_substituent_stats
),

distribution_stats AS (
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_tn_count) as tn_count_q25,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_tn_count) as tn_count_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_tn_count) as tn_count_q75,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY overall_avg_attribution) as attribution_q25,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY overall_avg_attribution) as attribution_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY overall_avg_attribution) as attribution_q75
    FROM aggregated_substituent_stats
),

pathogen_breakdown AS (
    SELECT 
        'S. aureus' as pathogen_id,
        COUNT(*) as substituents_in_pathogen,
        ROUND(AVG(total_compounds)::numeric, 1) as avg_compounds_per_substituent,
        ROUND(AVG(tn_count)::numeric, 1) as avg_tn_per_substituent,
        ROUND(AVG(avg_attribution)::numeric, 4) as avg_attribution
    FROM substituent_stats
    WHERE pathogen_id = 's_aureus'
    UNION ALL
    SELECT 
        'C. albicans' as pathogen_id,
        COUNT(*) as substituents_in_pathogen,
        ROUND(AVG(total_compounds)::numeric, 1) as avg_compounds_per_substituent,
        ROUND(AVG(tn_count)::numeric, 1) as avg_tn_per_substituent,
        ROUND(AVG(avg_attribution)::numeric, 4) as avg_attribution
    FROM substituent_stats
    WHERE pathogen_id = 'c_albicans'
)

SELECT 
    'Dual S.aureus + C.albicans Negative substituents Summary (Reliable Compounds Only)' as analysis_type,
    'S. aureus + C. albicans (NOT E. coli)' as pathogen_combination,
    os.total_dual_substituents,
    os.avg_tn_per_substituent,
    os.avg_tp_per_substituent,
    os.avg_compounds_per_substituent,
    os.total_tn_compounds,
    os.total_tp_compounds,
    os.total_compounds_analyzed,
    os.overall_avg_attribution,
    os.avg_inactivity_rate_percent,
    os.min_substituent_attribution,
    os.max_substituent_attribution,
    
    -- Distribution statistics
    ROUND(ds.tn_count_q25::numeric, 1) as tn_count_q25,
    ROUND(ds.tn_count_median::numeric, 1) as tn_count_median,  
    ROUND(ds.tn_count_q75::numeric, 1) as tn_count_q75,
    ROUND(ds.attribution_q25::numeric, 4) as attribution_q25,
    ROUND(ds.attribution_median::numeric, 4) as attribution_median,
    ROUND(ds.attribution_q75::numeric, 4) as attribution_q75,
    
    -- Quality metrics
    ts.substituents_with_20plus_tn,
    ts.substituents_with_40plus_tn,
    ts.substituents_with_100plus_tn,
    ts.substituents_50plus_inactivity_rate,
    ts.substituents_70plus_inactivity_rate,
    ts.substituents_high_negative_attribution,
    
    -- Quality percentages
    ROUND((ts.substituents_with_20plus_tn::numeric / os.total_dual_substituents * 100), 1) as pct_substituents_20plus_tn,
    ROUND((ts.substituents_50plus_inactivity_rate::numeric / os.total_dual_substituents * 100), 1) as pct_substituents_50plus_inactivity,
    ROUND((ts.substituents_high_negative_attribution::numeric / os.total_dual_substituents * 100), 1) as pct_substituents_high_negative_attribution,
    
    -- Per-pathogen breakdown
    STRING_AGG(
        pb.pathogen_id || ': ' || pb.substituents_in_pathogen || ' substituents | Avg: ' || 
        pb.avg_compounds_per_substituent || ' compounds (' || pb.avg_tn_per_substituent || ' TN) | ' ||
        'Attr: ' || pb.avg_attribution,
        ' | '
        ORDER BY pb.pathogen_id
    ) as pathogen_breakdown

FROM overall_stats os
CROSS JOIN top_substituents ts  
CROSS JOIN distribution_stats ds
CROSS JOIN pathogen_breakdown pb
GROUP BY os.total_dual_substituents, os.avg_tn_per_substituent, os.avg_tp_per_substituent,
         os.avg_compounds_per_substituent, os.total_tn_compounds, os.total_tp_compounds,
         os.total_compounds_analyzed, os.overall_avg_attribution, os.avg_inactivity_rate_percent,
         os.min_substituent_attribution, os.max_substituent_attribution, ds.tn_count_q25,
         ds.tn_count_median, ds.tn_count_q75, ds.attribution_q25, ds.attribution_median,
         ds.attribution_q75, ts.substituents_with_20plus_tn, ts.substituents_with_40plus_tn,
         ts.substituents_with_100plus_tn, ts.substituents_50plus_inactivity_rate,
         ts.substituents_70plus_inactivity_rate, ts.substituents_high_negative_attribution;
