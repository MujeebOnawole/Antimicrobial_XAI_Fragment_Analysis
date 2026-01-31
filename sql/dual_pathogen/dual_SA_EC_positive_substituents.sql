-- ====================================================================================
-- QUERY 4: Dual S. aureus + E. coli Positive Substituents
-- ====================================================================================

-- Dual Pathogen Positive Substituents Analysis: S. aureus + E. coli (Using Reliable Compounds)
-- Substituents present in S. aureus AND E. coli but NOT in C. albicans

WITH target_pathogen_positive_substituents AS (
    SELECT cf.fragment_id
    FROM compound_fragments cf
    JOIN fragments f ON cf.fragment_id = f.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE f.fragment_type = 'substituent'
      AND cf.attribution_score >= 0.1
      AND rc.has_mismatch = 0
      AND rc.reliability_level = 'RELIABLE'
      AND cf.pathogen_id IN ('s_aureus', 'e_coli')
    GROUP BY cf.fragment_id
    HAVING COUNT(DISTINCT cf.pathogen_id) = 2
),

c_albicans_substituents AS (
    SELECT DISTINCT cf.fragment_id
    FROM compound_fragments cf
    JOIN fragments f ON cf.fragment_id = f.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE f.fragment_type = 'substituent'
      AND cf.pathogen_id = 'c_albicans'
      AND rc.has_mismatch = 0
      AND rc.reliability_level = 'RELIABLE'
),

dual_specific_positive_substituents AS (
    SELECT fragment_id
    FROM target_pathogen_positive_substituents
    WHERE fragment_id NOT IN (SELECT fragment_id FROM c_albicans_substituents)
),

substituent_stats AS (
    SELECT 
        dsps.fragment_id,
        f.smiles as fragment_smiles,
        cf.pathogen_id,
        COUNT(*) as total_compounds,
        COUNT(CASE WHEN rc.target = 1 THEN 1 END) as tp_count,
        COUNT(CASE WHEN rc.target = 0 THEN 1 END) as tn_count,
        AVG(cf.attribution_score) as avg_attribution,
        MAX(cf.attribution_score) as max_attribution,
        ROUND(COUNT(CASE WHEN rc.target = 1 THEN 1 END)::numeric / COUNT(*)::numeric * 100, 2) as activity_rate_percent
    FROM dual_specific_positive_substituents dsps
    JOIN fragments f ON dsps.fragment_id = f.fragment_id
    JOIN compound_fragments cf ON dsps.fragment_id = cf.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE rc.has_mismatch = 0
      AND rc.reliability_level = 'RELIABLE'
      AND cf.attribution_score >= 0.1
      AND cf.pathogen_id IN ('s_aureus', 'e_coli')
    GROUP BY dsps.fragment_id, f.smiles, cf.pathogen_id
    HAVING COUNT(CASE WHEN rc.target = 1 THEN 1 END) >= 3
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
        MAX(max_attribution) as overall_max_attribution,
        AVG(activity_rate_percent) as avg_activity_rate_percent,
        ROW_NUMBER() OVER (ORDER BY SUM(tp_count) DESC, AVG(avg_attribution) DESC) as rank
    FROM substituent_stats
    GROUP BY fragment_id, fragment_smiles
    HAVING COUNT(DISTINCT pathogen_id) = 2
)

SELECT 
    'POSITIVE SUBSTITUENTS' as query_type,
    ass.rank,
    ass.fragment_id,
    ass.fragment_smiles,
    'S. aureus + E. coli (NOT C. albicans)' as pathogen_combination,
    ass.pathogen_count,
    ass.total_compounds_both_pathogens,
    ass.total_tp_count,
    ass.total_tn_count,
    ROUND(ass.overall_avg_attribution::numeric, 4) as overall_avg_attribution,
    ROUND(ass.overall_max_attribution::numeric, 4) as overall_max_attribution,
    ROUND(ass.avg_activity_rate_percent::numeric, 2) as avg_activity_rate_percent,
    
    STRING_AGG(
        ss.pathogen_id || ': ' || ss.total_compounds || ' compounds (' || 
        ss.tp_count || ' TP, ' || ss.tn_count || ' TN) | Avg Attr: ' || 
        ROUND(ss.avg_attribution::numeric, 3) || ' | Activity: ' || 
        ss.activity_rate_percent || '%',
        ' || '
        ORDER BY ss.pathogen_id
    ) as pathogen_breakdown
FROM aggregated_substituent_stats ass
LEFT JOIN substituent_stats ss ON ass.fragment_id = ss.fragment_id
GROUP BY ass.rank, ass.fragment_id, ass.fragment_smiles, ass.pathogen_count,
         ass.total_compounds_both_pathogens, ass.total_tp_count, ass.total_tn_count,
         ass.overall_avg_attribution, ass.overall_max_attribution, ass.avg_activity_rate_percent
ORDER BY ass.rank;