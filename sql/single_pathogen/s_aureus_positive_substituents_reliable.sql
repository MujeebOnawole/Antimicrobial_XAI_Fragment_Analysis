-- S. aureus Specific Positive Substituents Analysis (Using Reliable Compounds - Complete Version)

WITH s_aureus_fragments AS (
    -- Get all positive substituents for S. aureus (from reliable compounds only)
    SELECT DISTINCT cf.fragment_id
    FROM compound_fragments cf
    JOIN fragments f ON cf.fragment_id = f.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE cf.pathogen_id = 's_aureus'
      AND f.fragment_type = 'substituent'
      AND cf.attribution_score >= 0.1  -- Positive threshold
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

s_aureus_specific_positive_substituents AS (
    -- S. aureus specific positive substituents (not in other pathogens)
    SELECT fragment_id
    FROM s_aureus_fragments
    WHERE fragment_id NOT IN (SELECT fragment_id FROM other_pathogen_fragments)
),

substituent_stats AS (
    -- Calculate statistics for each substituent (from reliable compounds only)
    SELECT 
        saps.fragment_id,
        f.smiles as fragment_smiles,
        COUNT(*) as total_compounds,
        COUNT(CASE WHEN rc.target = 1 THEN 1 END) as tp_count,
        COUNT(CASE WHEN rc.target = 0 THEN 1 END) as tn_count,
        -- For reliable compounds, we don't expect FP/FN since they're all correct predictions
        0 as fp_count,
        0 as fn_count,
        AVG(cf.attribution_score) as avg_attribution,
        MAX(cf.attribution_score) as max_attribution,
        ROUND(COUNT(CASE WHEN rc.target = 1 THEN 1 END)::numeric / COUNT(*)::numeric * 100, 2) as activity_rate_percent
    FROM s_aureus_specific_positive_substituents saps
    JOIN fragments f ON saps.fragment_id = f.fragment_id
    JOIN compound_fragments cf ON saps.fragment_id = cf.fragment_id AND cf.pathogen_id = 's_aureus'
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE rc.has_mismatch = 0  -- Ensure all compounds are reliable
      AND rc.reliability_level = 'RELIABLE'
    GROUP BY saps.fragment_id, f.smiles
    HAVING COUNT(CASE WHEN rc.target = 1 THEN 1 END) >= 5  -- At least 5 TP cases
       AND AVG(cf.attribution_score) >= 0.1  -- Ensure substituent is genuinely positive
),

all_substituents AS (
    SELECT *,
           ROW_NUMBER() OVER (ORDER BY tp_count DESC, avg_attribution DESC) as rank
    FROM substituent_stats
),

-- Highest attribution examples (should be active)
highest_attribution_examples AS (
    SELECT DISTINCT
        als.fragment_id,
        rc.compound_id,
        cp.smiles as compound_smiles,
        rc.target,
        rc.ensemble_prediction,
        cf.attribution_score,
        cp.molecular_weight,
        cp.logp,
        ROW_NUMBER() OVER (PARTITION BY als.fragment_id ORDER BY cf.attribution_score DESC) as rank
    FROM all_substituents als
    JOIN compound_fragments cf ON als.fragment_id = cf.fragment_id AND cf.pathogen_id = 's_aureus'
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    JOIN compound_properties cp ON rc.compound_id = cp.compound_id
    WHERE rc.has_mismatch = 0  -- Only reliable compounds
      AND rc.reliability_level = 'RELIABLE'
),

-- Lowest attribution examples (could be active or inactive)
lowest_attribution_examples AS (
    SELECT DISTINCT
        als.fragment_id,
        rc.compound_id,
        cp.smiles as compound_smiles,
        rc.target,
        rc.ensemble_prediction,
        cf.attribution_score,
        cp.molecular_weight,
        cp.logp,
        ROW_NUMBER() OVER (PARTITION BY als.fragment_id ORDER BY cf.attribution_score ASC) as rank
    FROM all_substituents als
    JOIN compound_fragments cf ON als.fragment_id = cf.fragment_id AND cf.pathogen_id = 's_aureus'
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    JOIN compound_properties cp ON rc.compound_id = cp.compound_id
    WHERE rc.has_mismatch = 0  -- Only reliable compounds
      AND rc.reliability_level = 'RELIABLE'
),

-- Inactive examples with positive substituent attribution (TN with positive attribution)
inactive_with_positive_attribution AS (
    SELECT DISTINCT
        als.fragment_id,
        rc.compound_id,
        cp.smiles as compound_smiles,
        cf.attribution_score,
        rc.ensemble_prediction,
        cp.molecular_weight,
        ROW_NUMBER() OVER (PARTITION BY als.fragment_id ORDER BY cf.attribution_score DESC) as rank
    FROM all_substituents als
    JOIN compound_fragments cf ON als.fragment_id = cf.fragment_id AND cf.pathogen_id = 's_aureus'
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    JOIN compound_properties cp ON rc.compound_id = cp.compound_id
    WHERE rc.has_mismatch = 0  -- Only reliable compounds
      AND rc.reliability_level = 'RELIABLE'
      AND rc.target = 0  -- Inactive (TN)
      AND cf.attribution_score > 0  -- But positive substituent attribution
),

-- Active examples with negative substituent attribution (TP with negative attribution)
active_with_negative_attribution AS (
    SELECT DISTINCT
        als.fragment_id,
        rc.compound_id,
        cp.smiles as compound_smiles,
        cf.attribution_score,
        rc.ensemble_prediction,
        cp.molecular_weight,
        ROW_NUMBER() OVER (PARTITION BY als.fragment_id ORDER BY cf.attribution_score ASC) as rank
    FROM all_substituents als
    JOIN compound_fragments cf ON als.fragment_id = cf.fragment_id AND cf.pathogen_id = 's_aureus'
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    JOIN compound_properties cp ON rc.compound_id = cp.compound_id
    WHERE rc.has_mismatch = 0  -- Only reliable compounds
      AND rc.reliability_level = 'RELIABLE'
      AND rc.target = 1  -- Active (TP)
      AND cf.attribution_score < 0  -- But negative substituent attribution
),

consistency_check AS (
    SELECT 
        cf.fragment_id,
        COUNT(CASE WHEN cf.attribution_score >= 0.1 THEN 1 END) as positive_appearances,
        COUNT(*) as total_appearances,
        MIN(cf.attribution_score) as min_attribution_ever,
        MAX(cf.attribution_score) as max_attribution_ever
    FROM compound_fragments cf
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE cf.pathogen_id = 's_aureus'
      AND cf.fragment_id IN (SELECT fragment_id FROM s_aureus_specific_positive_substituents)
      AND rc.has_mismatch = 0  -- Only reliable compounds
      AND rc.reliability_level = 'RELIABLE'
    GROUP BY cf.fragment_id
)

-- Final results: Complete S. aureus-specific positive substituents with all examples
SELECT 
    als.rank,
    als.fragment_id,
    als.fragment_smiles,
    als.total_compounds,
    als.tp_count,
    als.tn_count,
    als.fp_count,
    als.fn_count,
    ROUND(als.avg_attribution::numeric, 4) as avg_attribution,
    ROUND(als.max_attribution::numeric, 4) as max_attribution,
    als.activity_rate_percent,
    
    -- Consistency metrics
    cc.positive_appearances,
    cc.total_appearances,
    ROUND((cc.positive_appearances::numeric / cc.total_appearances::numeric * 100), 1) as positive_consistency_percent,
    ROUND(cc.min_attribution_ever::numeric, 3) as min_attribution_ever,
    ROUND(cc.max_attribution_ever::numeric, 3) as max_attribution_ever,
    
    -- Highest attribution example (should be active but verify)
    (SELECT CASE WHEN hae.target = 1 THEN 'HIGHEST ATTR (ACTIVE): ' ELSE 'HIGHEST ATTR (INACTIVE): ' END ||
            hae.compound_id || ' | SMILES: ' || hae.compound_smiles || 
            ' | Attribution: ' || ROUND(hae.attribution_score::numeric, 3) ||
            ' | Prediction: ' || ROUND(hae.ensemble_prediction::numeric, 3) ||
            ' | MW: ' || ROUND(hae.molecular_weight::numeric, 1) ||
            ' | LogP: ' || ROUND(hae.logp::numeric, 2)
     FROM highest_attribution_examples hae 
     WHERE hae.fragment_id = als.fragment_id AND hae.rank = 1 
     LIMIT 1) as highest_attribution_example,
    
    -- Lowest attribution example (could be active or inactive)
    (SELECT CASE WHEN lae.target = 1 THEN 'LOWEST ATTR (ACTIVE): ' ELSE 'LOWEST ATTR (INACTIVE): ' END ||
            lae.compound_id || ' | SMILES: ' || lae.compound_smiles || 
            ' | Attribution: ' || ROUND(lae.attribution_score::numeric, 3) ||
            ' | Prediction: ' || ROUND(lae.ensemble_prediction::numeric, 3) ||
            ' | MW: ' || ROUND(lae.molecular_weight::numeric, 1) ||
            ' | LogP: ' || ROUND(lae.logp::numeric, 2)
     FROM lowest_attribution_examples lae 
     WHERE lae.fragment_id = als.fragment_id AND lae.rank = 1 
     LIMIT 1) as lowest_attribution_example,
    
    -- Inactive with positive attribution (interesting contradiction)
    (SELECT 'INACTIVE w/ POS ATTR: ' || iwpa.compound_id || 
            ' | SMILES: ' || iwpa.compound_smiles ||
            ' | Attribution: ' || ROUND(iwpa.attribution_score::numeric, 3) || 
            ' | Prediction: ' || ROUND(iwpa.ensemble_prediction::numeric, 3) ||
            ' | MW: ' || ROUND(iwpa.molecular_weight::numeric, 1)
     FROM inactive_with_positive_attribution iwpa 
     WHERE iwpa.fragment_id = als.fragment_id AND iwpa.rank = 1 
     LIMIT 1) as inactive_positive_attribution_example,
     
    -- Active with negative attribution (interesting contradiction)
    (SELECT 'ACTIVE w/ NEG ATTR: ' || awna.compound_id || 
            ' | SMILES: ' || awna.compound_smiles ||
            ' | Attribution: ' || ROUND(awna.attribution_score::numeric, 3) || 
            ' | Prediction: ' || ROUND(awna.ensemble_prediction::numeric, 3) ||
            ' | MW: ' || ROUND(awna.molecular_weight::numeric, 1)
     FROM active_with_negative_attribution awna 
     WHERE awna.fragment_id = als.fragment_id AND awna.rank = 1 
     LIMIT 1) as active_negative_attribution_example

FROM all_substituents als
LEFT JOIN highest_attribution_examples hae ON als.fragment_id = hae.fragment_id AND hae.rank = 1
LEFT JOIN lowest_attribution_examples lae ON als.fragment_id = lae.fragment_id AND lae.rank = 1
LEFT JOIN inactive_with_positive_attribution iwpa ON als.fragment_id = iwpa.fragment_id AND iwpa.rank = 1
LEFT JOIN active_with_negative_attribution awna ON als.fragment_id = awna.fragment_id AND awna.rank = 1
LEFT JOIN consistency_check cc ON als.fragment_id = cc.fragment_id
GROUP BY als.rank, als.fragment_id, als.fragment_smiles, als.total_compounds, 
         als.tp_count, als.tn_count, als.fp_count, als.fn_count, 
         als.avg_attribution, als.max_attribution, als.activity_rate_percent,
         cc.positive_appearances, cc.total_appearances, cc.min_attribution_ever, cc.max_attribution_ever,
         hae.compound_id, hae.compound_smiles, hae.target, hae.attribution_score, hae.ensemble_prediction, hae.molecular_weight, hae.logp,
         lae.compound_id, lae.compound_smiles, lae.target, lae.attribution_score, lae.ensemble_prediction, lae.molecular_weight, lae.logp,
         iwpa.compound_id, iwpa.compound_smiles, iwpa.attribution_score, iwpa.ensemble_prediction, iwpa.molecular_weight,
         awna.compound_id, awna.compound_smiles, awna.attribution_score, awna.ensemble_prediction, awna.molecular_weight
ORDER BY als.rank;