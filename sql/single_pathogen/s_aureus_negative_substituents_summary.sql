-- Summary Statistics for S. aureus Specific Negative Substituents (Using Reliable Compounds)

WITH s_aureus_fragments AS (
    -- Get all negative substituents for S. aureus (from reliable compounds only)
    SELECT DISTINCT cf.fragment_id
    FROM compound_fragments cf
    JOIN fragments f ON cf.fragment_id = f.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE cf.pathogen_id = 's_aureus'
      AND f.fragment_type = 'substituent'
      AND cf.attribution_score <= -0.1  -- Negative threshold
      AND rc.has_mismatch = 0  -- Zero mismatch requirement
      AND rc.reliability_level = 'RELIABLE'  -- Only reliable compounds
),

other_pathogen_fragments AS (
    -- Get substituents present in other pathogens (from reliable compounds only)
    SELECT DISTINCT cf.fragment_id
    FROM compound_fragments cf
    JOIN fragments f ON cf.fragment_id = f.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE cf.pathogen_id IN ('c_albicans', 'e_coli')
      AND f.fragment_type = 'substituent'
      AND rc.has_mismatch = 0  -- Zero mismatch requirement
      AND rc.reliability_level = 'RELIABLE'  -- Only reliable compounds
),

s_aureus_specific_negative_substituents AS (
    -- S. aureus specific negative substituents (not in other pathogens)
    SELECT fragment_id
    FROM s_aureus_fragments
    WHERE fragment_id NOT IN (SELECT fragment_id FROM other_pathogen_fragments)
),

substituent_stats AS (
    -- Calculate statistics for each substituent (from reliable compounds only)
    SELECT 
        sans.fragment_id,
        f.smiles as fragment_smiles,
        COUNT(*) as total_compounds,
        COUNT(CASE WHEN rc.target = 1 THEN 1 END) as tp_count,
        COUNT(CASE WHEN rc.target = 0 THEN 1 END) as tn_count,
        AVG(cf.attribution_score) as avg_attribution,
        MIN(cf.attribution_score) as min_attribution,
        ROUND(COUNT(CASE WHEN rc.target = 0 THEN 1 END)::numeric / COUNT(*)::numeric * 100, 2) as inactivity_rate_percent
    FROM s_aureus_specific_negative_substituents sans
    JOIN fragments f ON sans.fragment_id = f.fragment_id
    JOIN compound_fragments cf ON sans.fragment_id = cf.fragment_id AND cf.pathogen_id = 's_aureus'
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE rc.has_mismatch = 0  -- Ensure all compounds are reliable
      AND rc.reliability_level = 'RELIABLE'
    GROUP BY sans.fragment_id, f.smiles
    HAVING COUNT(CASE WHEN rc.target = 0 THEN 1 END) >= 5  -- At least 5 TN cases
       AND AVG(cf.attribution_score) <= -0.1  -- Ensure substituent is genuinely negative
),

overall_stats AS (
    SELECT 
        COUNT(*) as total_specific_substituents,
        ROUND(AVG(tn_count)::numeric, 1) as avg_tn_per_substituent,
        ROUND(AVG(tp_count)::numeric, 1) as avg_tp_per_substituent,
        ROUND(AVG(total_compounds)::numeric, 1) as avg_compounds_per_substituent,
        SUM(tn_count) as total_tn_compounds,
        SUM(tp_count) as total_tp_compounds,
        SUM(total_compounds) as total_compounds_analyzed,
        ROUND(AVG(avg_attribution)::numeric, 4) as overall_avg_attribution,
        ROUND(AVG(inactivity_rate_percent)::numeric, 1) as avg_inactivity_rate_percent,
        ROUND(MAX(avg_attribution)::numeric, 4) as max_substituent_attribution,
        ROUND(MIN(avg_attribution)::numeric, 4) as min_substituent_attribution
    FROM substituent_stats
),

top_substituents AS (
    SELECT 
        COUNT(CASE WHEN tn_count >= 10 THEN 1 END) as substituents_with_10plus_tn,
        COUNT(CASE WHEN tn_count >= 20 THEN 1 END) as substituents_with_20plus_tn,
        COUNT(CASE WHEN tn_count >= 50 THEN 1 END) as substituents_with_50plus_tn,
        COUNT(CASE WHEN inactivity_rate_percent >= 50 THEN 1 END) as substituents_50plus_inactivity_rate,
        COUNT(CASE WHEN inactivity_rate_percent >= 70 THEN 1 END) as substituents_70plus_inactivity_rate,
        COUNT(CASE WHEN avg_attribution <= -0.2 THEN 1 END) as substituents_high_negative_attribution
    FROM substituent_stats
),

distribution_stats AS (
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY tn_count) as tn_count_q25,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tn_count) as tn_count_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY tn_count) as tn_count_q75,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY avg_attribution) as attribution_q25,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_attribution) as attribution_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY avg_attribution) as attribution_q75
    FROM substituent_stats
)

-- Final summary report
SELECT 
    'S. aureus Specific Negative Substituents Summary (Reliable Compounds Only)' as analysis_type,
    os.total_specific_substituents,
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
    ts.substituents_with_10plus_tn,
    ts.substituents_with_20plus_tn,
    ts.substituents_with_50plus_tn,
    ts.substituents_50plus_inactivity_rate,
    ts.substituents_70plus_inactivity_rate,
    ts.substituents_high_negative_attribution,
    
    -- Quality percentages
    ROUND((ts.substituents_with_10plus_tn::numeric / os.total_specific_substituents * 100), 1) as pct_substituents_10plus_tn,
    ROUND((ts.substituents_50plus_inactivity_rate::numeric / os.total_specific_substituents * 100), 1) as pct_substituents_50plus_inactivity,
    ROUND((ts.substituents_high_negative_attribution::numeric / os.total_specific_substituents * 100), 1) as pct_substituents_high_negative_attribution

FROM overall_stats os
CROSS JOIN top_substituents ts  
CROSS JOIN distribution_stats ds;