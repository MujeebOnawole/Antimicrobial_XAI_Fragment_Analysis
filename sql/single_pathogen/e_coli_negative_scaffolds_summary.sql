-- Summary Statistics for E. coli Specific Negative Scaffolds (Using Reliable Compounds)

WITH e_coli_fragments AS (
    -- Get all negative scaffolds for E. coli (from reliable compounds only)
    SELECT DISTINCT cf.fragment_id
    FROM compound_fragments cf
    JOIN fragments f ON cf.fragment_id = f.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE cf.pathogen_id = 'e_coli'
      AND f.fragment_type = 'scaffold'
      AND cf.attribution_score <= -0.1  -- Negative threshold
      AND rc.has_mismatch = 0  -- Zero mismatch requirement
      AND rc.reliability_level = 'RELIABLE'  -- Only reliable compounds
),

other_pathogen_fragments AS (
    -- Get scaffolds present in other pathogens (from reliable compounds only)
    SELECT DISTINCT cf.fragment_id
    FROM compound_fragments cf
    JOIN fragments f ON cf.fragment_id = f.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE cf.pathogen_id IN ('s_aureus', 'c_albicans')
      AND f.fragment_type = 'scaffold'
      AND rc.has_mismatch = 0  -- Zero mismatch requirement
      AND rc.reliability_level = 'RELIABLE'  -- Only reliable compounds
),

e_coli_specific_negative_scaffolds AS (
    -- E. coli specific negative scaffolds (not in other pathogens)
    SELECT fragment_id
    FROM e_coli_fragments
    WHERE fragment_id NOT IN (SELECT fragment_id FROM other_pathogen_fragments)
),

scaffold_stats AS (
    -- Calculate statistics for each scaffold (from reliable compounds only)
    SELECT 
        ecns.fragment_id,
        f.smiles as fragment_smiles,
        COUNT(*) as total_compounds,
        COUNT(CASE WHEN rc.target = 1 THEN 1 END) as tp_count,
        COUNT(CASE WHEN rc.target = 0 THEN 1 END) as tn_count,
        AVG(cf.attribution_score) as avg_attribution,
        MIN(cf.attribution_score) as min_attribution,
        ROUND(COUNT(CASE WHEN rc.target = 0 THEN 1 END)::numeric / COUNT(*)::numeric * 100, 2) as inactivity_rate_percent
    FROM e_coli_specific_negative_scaffolds ecns
    JOIN fragments f ON ecns.fragment_id = f.fragment_id
    JOIN compound_fragments cf ON ecns.fragment_id = cf.fragment_id AND cf.pathogen_id = 'e_coli'
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE rc.has_mismatch = 0  -- Ensure all compounds are reliable
      AND rc.reliability_level = 'RELIABLE'
    GROUP BY ecns.fragment_id, f.smiles
    HAVING COUNT(CASE WHEN rc.target = 0 THEN 1 END) >= 5  -- At least 5 TN cases
       AND AVG(cf.attribution_score) <= -0.1  -- Ensure scaffold is genuinely negative
),

overall_stats AS (
    SELECT 
        COUNT(*) as total_specific_scaffolds,
        ROUND(AVG(tn_count)::numeric, 1) as avg_tn_per_scaffold,
        ROUND(AVG(tp_count)::numeric, 1) as avg_tp_per_scaffold,
        ROUND(AVG(total_compounds)::numeric, 1) as avg_compounds_per_scaffold,
        SUM(tn_count) as total_tn_compounds,
        SUM(tp_count) as total_tp_compounds,
        SUM(total_compounds) as total_compounds_analyzed,
        ROUND(AVG(avg_attribution)::numeric, 4) as overall_avg_attribution,
        ROUND(AVG(inactivity_rate_percent)::numeric, 1) as avg_inactivity_rate_percent,
        ROUND(MAX(avg_attribution)::numeric, 4) as max_scaffold_attribution,
        ROUND(MIN(avg_attribution)::numeric, 4) as min_scaffold_attribution
    FROM scaffold_stats
),

top_scaffolds AS (
    SELECT 
        COUNT(CASE WHEN tn_count >= 10 THEN 1 END) as scaffolds_with_10plus_tn,
        COUNT(CASE WHEN tn_count >= 20 THEN 1 END) as scaffolds_with_20plus_tn,
        COUNT(CASE WHEN tn_count >= 50 THEN 1 END) as scaffolds_with_50plus_tn,
        COUNT(CASE WHEN inactivity_rate_percent >= 50 THEN 1 END) as scaffolds_50plus_inactivity_rate,
        COUNT(CASE WHEN inactivity_rate_percent >= 70 THEN 1 END) as scaffolds_70plus_inactivity_rate,
        COUNT(CASE WHEN avg_attribution <= -0.2 THEN 1 END) as scaffolds_high_negative_attribution
    FROM scaffold_stats
),

distribution_stats AS (
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY tn_count) as tn_count_q25,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tn_count) as tn_count_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY tn_count) as tn_count_q75,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY avg_attribution) as attribution_q25,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_attribution) as attribution_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY avg_attribution) as attribution_q75
    FROM scaffold_stats
)

-- Final summary report
SELECT 
    'E. coli Specific Negative Scaffolds Summary (Reliable Compounds Only)' as analysis_type,
    os.total_specific_scaffolds,
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
    ts.scaffolds_with_10plus_tn,
    ts.scaffolds_with_20plus_tn,
    ts.scaffolds_with_50plus_tn,
    ts.scaffolds_50plus_inactivity_rate,
    ts.scaffolds_70plus_inactivity_rate,
    ts.scaffolds_high_negative_attribution,
    
    -- Quality percentages
    ROUND((ts.scaffolds_with_10plus_tn::numeric / os.total_specific_scaffolds * 100), 1) as pct_scaffolds_10plus_tn,
    ROUND((ts.scaffolds_50plus_inactivity_rate::numeric / os.total_specific_scaffolds * 100), 1) as pct_scaffolds_50plus_inactivity,
    ROUND((ts.scaffolds_high_negative_attribution::numeric / os.total_specific_scaffolds * 100), 1) as pct_scaffolds_high_negative_attribution

FROM overall_stats os
CROSS JOIN top_scaffolds ts  
CROSS JOIN distribution_stats ds;