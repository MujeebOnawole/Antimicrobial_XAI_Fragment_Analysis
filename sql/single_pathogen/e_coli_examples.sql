-- ============================================================================
-- FIND E. COLI ACTIVE EXEMPLAR FOR SELECT-G⁻ VISUALIZATION
-- Priority: HBD 3-5, TPSA 25-50, LogP 1.3-3.4, MW 196-317
-- ============================================================================

WITH e_coli_active AS (
    -- Get E. coli active compounds from reliable data
    SELECT 
        c.compound_id,
        c.smiles,
        c.target,
        c.ensemble_prediction,
        c.prediction_class
    FROM compounds c
    WHERE c.pathogen_id = 'e_coli'
      AND c.target = 1
      -- Only from reliable_compounds if needed
      AND EXISTS (
          SELECT 1 
          FROM reliable_compounds rc 
          WHERE rc.compound_id = c.compound_id 
            AND rc.pathogen_id = 'e_coli'
            AND rc.target = 1
            AND rc.reliability_level = 'RELIABLE'
            AND rc.has_mismatch = 0
      )
),

candidates AS (
    -- Join with properties and score based on criteria
    SELECT 
        ea.compound_id,
        ea.smiles,
        ea.target,
        ROUND(ea.ensemble_prediction::numeric, 4) as ensemble_prediction,
        ea.prediction_class,
        
        -- Properties
        cp.molecular_weight as mw,
        cp.logp,
        cp.tpsa,
        cp.hbd_count as hbd,
        cp.hba_count as hba,
        
        -- Criteria matching scores (1 = perfect match, 0 = out of range)
        CASE 
            WHEN cp.hbd_count BETWEEN 3 AND 5 THEN 1 
            ELSE 0 
        END as hbd_match,
        
        CASE 
            WHEN cp.tpsa BETWEEN 25 AND 50 THEN 1 
            ELSE 0 
        END as tpsa_match,
        
        CASE 
            WHEN cp.logp BETWEEN 1.3 AND 3.4 THEN 1 
            ELSE 0 
        END as logp_match,
        
        CASE 
            WHEN cp.molecular_weight BETWEEN 196 AND 317 THEN 1 
            ELSE 0 
        END as mw_match,
        
        -- Distance from ideal TPSA (for tie-breaking)
        ABS(cp.tpsa - 37.5) as tpsa_distance_from_center,
        
        -- Distance from TPSA boundary (for fallback)
        LEAST(ABS(cp.tpsa - 25), ABS(cp.tpsa - 50)) as tpsa_distance_from_boundary
        
    FROM e_coli_active ea
    JOIN compound_properties cp ON ea.compound_id = cp.compound_id
    WHERE cp.hbd_count IS NOT NULL
      AND cp.tpsa IS NOT NULL
      AND cp.logp IS NOT NULL
      AND cp.molecular_weight IS NOT NULL
),

scored_candidates AS (
    -- Calculate composite score
    SELECT 
        *,
        -- Total criteria matched (0-4)
        (hbd_match + tpsa_match + logp_match + mw_match) as criteria_matched,
        
        -- Composite score for ranking
        (hbd_match * 1000) +           -- HBD is critical (priority 1)
        (logp_match * 100) +           -- LogP priority 2
        (tpsa_match * 10) +            -- TPSA priority 3
        (mw_match * 1) +               -- MW lowest priority
        (ensemble_prediction * 0.1)    -- Tie-breaker: higher prediction better
        as composite_score
        
    FROM candidates
)

-- Final output with ranking
SELECT 
    CASE 
        WHEN criteria_matched = 4 THEN 'PERFECT_MATCH'
        WHEN hbd_match = 1 AND logp_match = 1 THEN 'HBD_LOGP_MATCH'
        WHEN hbd_match = 1 THEN 'HBD_ONLY_MATCH'
        ELSE 'OTHER'
    END as match_type,
    compound_id,
    smiles,
    ensemble_prediction,
    ROUND(mw::numeric, 2) as mw,
    ROUND(logp::numeric, 2) as logp,
    ROUND(tpsa::numeric, 2) as tpsa,
    hbd,
    hba,
    criteria_matched,
    CASE WHEN hbd_match = 1 THEN 'Yes' ELSE 'No' END as hbd_match,
    CASE WHEN tpsa_match = 1 THEN 'Yes' ELSE 'No' END as tpsa_match,
    CASE WHEN logp_match = 1 THEN 'Yes' ELSE 'No' END as logp_match,
    CASE WHEN mw_match = 1 THEN 'Yes' ELSE 'No' END as mw_match,
    ROUND(composite_score::numeric, 2) as score
FROM scored_candidates
ORDER BY 
    criteria_matched DESC,
    composite_score DESC,
    ensemble_prediction DESC
LIMIT 50;