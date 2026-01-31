-- Multipathogen Positive Scaffolds Analysis (Using Reliable Compounds - Complete Version)
-- Scaffolds present in ALL THREE pathogens with positive attribution

WITH all_pathogen_positive_scaffolds AS (
    -- Get scaffolds that appear with positive attribution in ALL three pathogens
    SELECT cf.fragment_id
    FROM compound_fragments cf
    JOIN fragments f ON cf.fragment_id = f.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE f.fragment_type = 'scaffold'
      AND cf.attribution_score >= 0.1  -- Positive threshold
      AND rc.has_mismatch = 0  -- Zero mismatch requirement
      AND rc.reliability_level = 'RELIABLE'  -- Only reliable compounds
    GROUP BY cf.fragment_id
    HAVING COUNT(DISTINCT cf.pathogen_id) = 3  -- Present in all three pathogens
),

scaffold_stats AS (
    -- Calculate statistics for each scaffold across all pathogens
    SELECT 
        apps.fragment_id,
        f.smiles as fragment_smiles,
        cf.pathogen_id,
        COUNT(*) as total_compounds,
        COUNT(CASE WHEN rc.target = 1 THEN 1 END) as tp_count,
        COUNT(CASE WHEN rc.target = 0 THEN 1 END) as tn_count,
        -- For reliable compounds, we don't expect FP/FN since they're all correct predictions
        0 as fp_count,
        0 as fn_count,
        AVG(cf.attribution_score) as avg_attribution,
        MAX(cf.attribution_score) as max_attribution,
        ROUND(COUNT(CASE WHEN rc.target = 1 THEN 1 END)::numeric / COUNT(*)::numeric * 100, 2) as activity_rate_percent
    FROM all_pathogen_positive_scaffolds apps
    JOIN fragments f ON apps.fragment_id = f.fragment_id
    JOIN compound_fragments cf ON apps.fragment_id = cf.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE rc.has_mismatch = 0  -- Ensure all compounds are reliable
      AND rc.reliability_level = 'RELIABLE'
      AND cf.attribution_score >= 0.1  -- Only positive attributions
    GROUP BY apps.fragment_id, f.smiles, cf.pathogen_id
    HAVING COUNT(CASE WHEN rc.target = 1 THEN 1 END) >= 3  -- At least 3 TP cases per pathogen
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
        SUM(fp_count) as total_fp_count,
        SUM(fn_count) as total_fn_count,
        AVG(avg_attribution) as overall_avg_attribution,
        MAX(max_attribution) as overall_max_attribution,
        AVG(activity_rate_percent) as avg_activity_rate_percent,
        ROW_NUMBER() OVER (ORDER BY SUM(tp_count) DESC, AVG(avg_attribution) DESC) as rank
    FROM scaffold_stats
    GROUP BY fragment_id, fragment_smiles
    HAVING COUNT(DISTINCT pathogen_id) = 3  -- Ensure present in all three pathogens
),

-- Examples from each pathogen
pathogen_examples AS (
    SELECT DISTINCT
        ass.fragment_id,
        cf.pathogen_id,
        rc.compound_id,
        cp.smiles as compound_smiles,
        rc.target,
        rc.ensemble_prediction,
        cf.attribution_score,
        cp.molecular_weight,
        cp.logp,
        ROW_NUMBER() OVER (PARTITION BY ass.fragment_id, cf.pathogen_id ORDER BY cf.attribution_score DESC) as rank
    FROM aggregated_scaffold_stats ass
    JOIN compound_fragments cf ON ass.fragment_id = cf.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    JOIN compound_properties cp ON rc.compound_id = cp.compound_id
    WHERE rc.has_mismatch = 0  -- Only reliable compounds
      AND rc.reliability_level = 'RELIABLE'
      AND cf.attribution_score >= 0.1  -- Only positive attributions
),

consistency_check AS (
    SELECT 
        cf.fragment_id,
        cf.pathogen_id,
        COUNT(CASE WHEN cf.attribution_score >= 0.1 THEN 1 END) as positive_appearances,
        COUNT(*) as total_appearances,
        MIN(cf.attribution_score) as min_attribution_ever,
        MAX(cf.attribution_score) as max_attribution_ever
    FROM compound_fragments cf
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE cf.fragment_id IN (SELECT fragment_id FROM all_pathogen_positive_scaffolds)
      AND rc.has_mismatch = 0  -- Only reliable compounds
      AND rc.reliability_level = 'RELIABLE'
    GROUP BY cf.fragment_id, cf.pathogen_id
)

-- Final results: Complete multipathogen positive scaffolds
SELECT 
    ass.rank,
    ass.fragment_id,
    ass.fragment_smiles,
    ass.pathogen_count,
    ass.total_compounds_all_pathogens,
    ass.total_tp_count,
    ass.total_tn_count,
    ass.total_fp_count,
    ass.total_fn_count,
    ROUND(ass.overall_avg_attribution::numeric, 4) as overall_avg_attribution,
    ROUND(ass.overall_max_attribution::numeric, 4) as overall_max_attribution,
    ROUND(ass.avg_activity_rate_percent::numeric, 2) as avg_activity_rate_percent,
    
    -- Per-pathogen breakdown
    STRING_AGG(
        ss.pathogen_id || ': ' || ss.total_compounds || ' compounds (' || 
        ss.tp_count || ' TP, ' || ss.tn_count || ' TN) | Avg Attr: ' || 
        ROUND(ss.avg_attribution::numeric, 3) || ' | Activity: ' || 
        ss.activity_rate_percent || '%',
        ' || '
        ORDER BY ss.pathogen_id
    ) as pathogen_breakdown,
    
    -- Example compounds from each pathogen
    STRING_AGG(
        CASE WHEN pe.rank = 1 THEN
            pe.pathogen_id || ' Example: ' || pe.compound_id || 
            ' | SMILES: ' || pe.compound_smiles || 
            ' | Target: ' || pe.target ||
            ' | Attribution: ' || ROUND(pe.attribution_score::numeric, 3) ||
            ' | Prediction: ' || ROUND(pe.ensemble_prediction::numeric, 3) ||
            ' | MW: ' || ROUND(pe.molecular_weight::numeric, 1)
        END,
        ' || '
        ORDER BY pe.pathogen_id
    ) as pathogen_examples,
    
    -- Consistency across pathogens
    STRING_AGG(
        cc.pathogen_id || ': ' || ROUND((cc.positive_appearances::numeric / cc.total_appearances::numeric * 100), 1) || 
        '% positive (' || cc.positive_appearances || '/' || cc.total_appearances || ') | Range: ' ||
        ROUND(cc.min_attribution_ever::numeric, 3) || ' to ' || ROUND(cc.max_attribution_ever::numeric, 3),
        ' || '
        ORDER BY cc.pathogen_id
    ) as consistency_breakdown

FROM aggregated_scaffold_stats ass
LEFT JOIN scaffold_stats ss ON ass.fragment_id = ss.fragment_id
LEFT JOIN pathogen_examples pe ON ass.fragment_id = pe.fragment_id
LEFT JOIN consistency_check cc ON ass.fragment_id = cc.fragment_id AND ss.pathogen_id = cc.pathogen_id
GROUP BY ass.rank, ass.fragment_id, ass.fragment_smiles, ass.pathogen_count,
         ass.total_compounds_all_pathogens, ass.total_tp_count, ass.total_tn_count,
         ass.total_fp_count, ass.total_fn_count, ass.overall_avg_attribution,
         ass.overall_max_attribution, ass.avg_activity_rate_percent
ORDER BY ass.rank;