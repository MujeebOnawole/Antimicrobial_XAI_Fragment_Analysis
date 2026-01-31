-- C. albicans Specific Negative Substituents Analysis (Using Reliable Compounds - Complete Version)

WITH c_albicans_fragments AS (
    -- Get all negative substituents for C. albicans (from reliable compounds only)
    SELECT DISTINCT cf.fragment_id
    FROM compound_fragments cf
    JOIN fragments f ON cf.fragment_id = f.fragment_id
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE cf.pathogen_id = 'c_albicans'
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
    WHERE cf.pathogen_id IN ('s_aureus', 'e_coli')
      AND f.fragment_type = 'substituent'
      AND rc.has_mismatch = 0  -- Zero mismatch requirement
      AND rc.reliability_level = 'RELIABLE'  -- Only reliable compounds
),

c_albicans_specific_negative_substituents AS (
    -- C. albicans specific negative substituents (not in other pathogens)
    SELECT fragment_id
    FROM c_albicans_fragments
    WHERE fragment_id NOT IN (SELECT fragment_id FROM other_pathogen_fragments)
),

substituent_stats AS (
    -- Calculate statistics for each substituent (from reliable compounds only)
    SELECT 
        cans.fragment_id,
        f.smiles as fragment_smiles,
        COUNT(*) as total_compounds,
        COUNT(CASE WHEN rc.target = 1 THEN 1 END) as tp_count,
        COUNT(CASE WHEN rc.target = 0 THEN 1 END) as tn_count,
        -- For reliable compounds, we don't expect FP/FN since they're all correct predictions
        0 as fp_count,
        0 as fn_count,
        AVG(cf.attribution_score) as avg_attribution,
        MIN(cf.attribution_score) as min_attribution,
        ROUND(COUNT(CASE WHEN rc.target = 0 THEN 1 END)::numeric / COUNT(*)::numeric * 100, 2) as inactivity_rate_percent
    FROM c_albicans_specific_negative_substituents cans
    JOIN fragments f ON cans.fragment_id = f.fragment_id
    JOIN compound_fragments cf ON cans.fragment_id = cf.fragment_id AND cf.pathogen_id = 'c_albicans'
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE rc.has_mismatch = 0  -- Ensure all compounds are reliable
      AND rc.reliability_level = 'RELIABLE'
    GROUP BY cans.fragment_id, f.smiles
    HAVING COUNT(CASE WHEN rc.target = 0 THEN 1 END) >= 5  -- At least 5 TN cases
       AND AVG(cf.attribution_score) <= -0.1  -- Ensure substituent is genuinely negative
),

all_substituents AS (
    SELECT *,
           ROW_NUMBER() OVER (ORDER BY tn_count DESC, avg_attribution ASC) as rank
    FROM substituent_stats
),

-- Example compounds (all from reliable compounds)
example_compounds AS (
    SELECT DISTINCT
        als.fragment_id,
        als.fragment_smiles,
        rc.compound_id,
        cp.smiles as compound_smiles,
        rc.target,
        rc.ensemble_prediction,
        cf.attribution_score,
        cp.molecular_weight,
        cp.logp,
        ROW_NUMBER() OVER (PARTITION BY als.fragment_id ORDER BY cf.attribution_score ASC, rc.ensemble_prediction ASC) as example_rank
    FROM all_substituents als
    JOIN compound_fragments cf ON als.fragment_id = cf.fragment_id AND cf.pathogen_id = 'c_albicans'
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    JOIN compound_properties cp ON rc.compound_id = cp.compound_id
    WHERE rc.has_mismatch = 0  -- Only reliable compounds
      AND rc.reliability_level = 'RELIABLE'
      AND rc.target = 0  -- True Negatives only
),

-- Lowest attribution examples (should be inactive)
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
    JOIN compound_fragments cf ON als.fragment_id = cf.fragment_id AND cf.pathogen_id = 'c_albicans'
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    JOIN compound_properties cp ON rc.compound_id = cp.compound_id
    WHERE rc.has_mismatch = 0  -- Only reliable compounds
      AND rc.reliability_level = 'RELIABLE'
),

-- Highest attribution examples (could be active or inactive)
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
    JOIN compound_fragments cf ON als.fragment_id = cf.fragment_id AND cf.pathogen_id = 'c_albicans'
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    JOIN compound_properties cp ON rc.compound_id = cp.compound_id
    WHERE rc.has_mismatch = 0  -- Only reliable compounds
      AND rc.reliability_level = 'RELIABLE'
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
    JOIN compound_fragments cf ON als.fragment_id = cf.fragment_id AND cf.pathogen_id = 'c_albicans'
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    JOIN compound_properties cp ON rc.compound_id = cp.compound_id
    WHERE rc.has_mismatch = 0  -- Only reliable compounds
      AND rc.reliability_level = 'RELIABLE'
      AND rc.target = 1  -- Active (TP)
      AND cf.attribution_score < 0  -- With negative substituent attribution
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
    JOIN compound_fragments cf ON als.fragment_id = cf.fragment_id AND cf.pathogen_id = 'c_albicans'
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    JOIN compound_properties cp ON rc.compound_id = cp.compound_id
    WHERE rc.has_mismatch = 0  -- Only reliable compounds
      AND rc.reliability_level = 'RELIABLE'
      AND rc.target = 0  -- Inactive (TN)
      AND cf.attribution_score > 0  -- But positive substituent attribution
),

consistency_check AS (
    SELECT 
        cf.fragment_id,
        COUNT(CASE WHEN cf.attribution_score <= -0.1 THEN 1 END) as negative_appearances,
        COUNT(*) as total_appearances,
        MIN(cf.attribution_score) as min_attribution_ever,
        MAX(cf.attribution_score) as max_attribution_ever
    FROM compound_fragments cf
    JOIN reliable_compounds rc ON cf.compound_id = rc.compound_id AND cf.pathogen_id = rc.pathogen_id
    WHERE cf.pathogen_id = 'c_albicans'
      AND cf.fragment_id IN (SELECT fragment_id FROM c_albicans_specific_negative_substituents)
      AND rc.has_mismatch = 0  -- Only reliable compounds
      AND rc.reliability_level = 'RELIABLE'
    GROUP BY cf.fragment_id
)

-- Final results: Complete C. albicans-specific negative substituents
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
    ROUND(als.min_attribution::numeric, 4) as min_attribution,
    als.inactivity_rate_percent,
    
    -- Consistency metrics
    cc.negative_appearances,
    cc.total_appearances,
    ROUND((cc.negative_appearances::numeric / cc.total_appearances::numeric * 100), 1) as negative_consistency_percent,
    ROUND(cc.min_attribution_ever::numeric, 3) as min_attribution_ever,
    ROUND(cc.max_attribution_ever::numeric, 3) as max_attribution_ever,
    
    -- Example compounds (up to 3 per substituent, all guaranteed reliable)
    STRING_AGG(
        CASE WHEN ec.example_rank <= 3 THEN
            'Compound: ' || ec.compound_id || 
            ' | SMILES: ' || ec.compound_smiles || 
            ' | Attribution: ' || ROUND(ec.attribution_score::numeric, 3) ||
            ' | Prediction: ' || ROUND(ec.ensemble_prediction::numeric, 3) ||
            ' | MW: ' || ROUND(ec.molecular_weight::numeric, 1) ||
            ' | LogP: ' || ROUND(ec.logp::numeric, 2)
        END,
        ' || '
    ) as example_tn_compounds,
    
    -- Lowest attribution example (should be inactive but verify)
    (SELECT CASE WHEN lae.target = 0 THEN 'LOWEST ATTR (INACTIVE): ' ELSE 'LOWEST ATTR (ACTIVE): ' END ||
            lae.compound_id || ' | SMILES: ' || lae.compound_smiles || 
            ' | Attribution: ' || ROUND(lae.attribution_score::numeric, 3) ||
            ' | Prediction: ' || ROUND(lae.ensemble_prediction::numeric, 3) ||
            ' | MW: ' || ROUND(lae.molecular_weight::numeric, 1) ||
            ' | LogP: ' || ROUND(lae.logp::numeric, 2)
     FROM lowest_attribution_examples lae 
     WHERE lae.fragment_id = als.fragment_id AND lae.rank = 1 
     LIMIT 1) as lowest_attribution_example,
    
    -- Highest attribution example (could be active or inactive)
    (SELECT CASE WHEN hae.target = 0 THEN 'HIGHEST ATTR (INACTIVE): ' ELSE 'HIGHEST ATTR (ACTIVE): ' END ||
            hae.compound_id || ' | SMILES: ' || hae.compound_smiles || 
            ' | Attribution: ' || ROUND(hae.attribution_score::numeric, 3) ||
            ' | Prediction: ' || ROUND(hae.ensemble_prediction::numeric, 3) ||
            ' | MW: ' || ROUND(hae.molecular_weight::numeric, 1) ||
            ' | LogP: ' || ROUND(hae.logp::numeric, 2)
     FROM highest_attribution_examples hae 
     WHERE hae.fragment_id = als.fragment_id AND hae.rank = 1 
     LIMIT 1) as highest_attribution_example,
    
    -- Active with negative attribution (expected case for negative substituents)
    (SELECT 'ACTIVE w/ NEG ATTR: ' || awna.compound_id || 
            ' | SMILES: ' || awna.compound_smiles ||
            ' | Attribution: ' || ROUND(awna.attribution_score::numeric, 3) || 
            ' | Prediction: ' || ROUND(awna.ensemble_prediction::numeric, 3) ||
            ' | MW: ' || ROUND(awna.molecular_weight::numeric, 1)
     FROM active_with_negative_attribution awna 
     WHERE awna.fragment_id = als.fragment_id AND awna.rank = 1 
     LIMIT 1) as active_negative_attribution_example,
     
    -- Inactive with positive attribution (interesting contradiction)
    (SELECT 'INACTIVE w/ POS ATTR: ' || iwpa.compound_id || 
            ' | SMILES: ' || iwpa.compound_smiles ||
            ' | Attribution: ' || ROUND(iwpa.attribution_score::numeric, 3) || 
            ' | Prediction: ' || ROUND(iwpa.ensemble_prediction::numeric, 3) ||
            ' | MW: ' || ROUND(iwpa.molecular_weight::numeric, 1)
     FROM inactive_with_positive_attribution iwpa 
     WHERE iwpa.fragment_id = als.fragment_id AND iwpa.rank = 1 
     LIMIT 1) as inactive_positive_attribution_example

FROM all_substituents als
LEFT JOIN example_compounds ec ON als.fragment_id = ec.fragment_id
LEFT JOIN lowest_attribution_examples lae ON als.fragment_id = lae.fragment_id AND lae.rank = 1
LEFT JOIN highest_attribution_examples hae ON als.fragment_id = hae.fragment_id AND hae.rank = 1
LEFT JOIN active_with_negative_attribution awna ON als.fragment_id = awna.fragment_id AND awna.rank = 1
LEFT JOIN inactive_with_positive_attribution iwpa ON als.fragment_id = iwpa.fragment_id AND iwpa.rank = 1
LEFT JOIN consistency_check cc ON als.fragment_id = cc.fragment_id
GROUP BY als.rank, als.fragment_id, als.fragment_smiles, als.total_compounds, 
         als.tp_count, als.tn_count, als.fp_count, als.fn_count, 
         als.avg_attribution, als.min_attribution, als.inactivity_rate_percent,
         cc.negative_appearances, cc.total_appearances, cc.min_attribution_ever, cc.max_attribution_ever,
         lae.compound_id, lae.compound_smiles, lae.target, lae.attribution_score, lae.ensemble_prediction, lae.molecular_weight, lae.logp,
         hae.compound_id, hae.compound_smiles, hae.target, hae.attribution_score, hae.ensemble_prediction, hae.molecular_weight, hae.logp,
         awna.compound_id, awna.compound_smiles, awna.attribution_score, awna.ensemble_prediction, awna.molecular_weight,
         iwpa.compound_id, iwpa.compound_smiles, iwpa.attribution_score, iwpa.ensemble_prediction, iwpa.molecular_weight
ORDER BY als.rank;