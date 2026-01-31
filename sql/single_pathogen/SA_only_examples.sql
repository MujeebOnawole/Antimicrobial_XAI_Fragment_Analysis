-- S. aureus-specific compounds: Active SA, Inactive EC, Inactive CA
-- Based on scaffold query template structure

WITH sa_specific_compounds AS (
    -- Get compounds that are Active SA, Inactive EC, Inactive CA
    SELECT DISTINCT rc_sa.compound_id
    FROM reliable_compounds rc_sa
    JOIN reliable_compounds rc_ec 
        ON rc_sa.compound_id = rc_ec.compound_id 
        AND rc_ec.pathogen_id = 'e_coli'
    JOIN reliable_compounds rc_ca 
        ON rc_sa.compound_id = rc_ca.compound_id 
        AND rc_ca.pathogen_id = 'c_albicans'
    WHERE rc_sa.pathogen_id = 's_aureus'
      AND rc_sa.target = 1  -- Active SA
      AND rc_ec.target = 0  -- Inactive EC
      AND rc_ca.target = 0  -- Inactive CA
      AND rc_sa.has_mismatch = 0 AND rc_sa.reliability_level = 'RELIABLE'
      AND rc_ec.has_mismatch = 0 AND rc_ec.reliability_level = 'RELIABLE'
      AND rc_ca.has_mismatch = 0 AND rc_ca.reliability_level = 'RELIABLE'
),

compound_stats AS (
    -- Get compound properties and predictions
    SELECT 
        sac.compound_id,
        cp.smiles,
        cp.molecular_weight,
        cp.logp,
        cp.tpsa,
        cp.hbd_count,
        cp.hba_count,
        cp.total_rings,
        cp.aromatic_rings,
        cp.hetero_atoms,
        cp.hetero_cycles,
        rc_sa.ensemble_prediction as sa_prediction,
        rc_ec.ensemble_prediction as ec_prediction,
        rc_ca.ensemble_prediction as ca_prediction
    FROM sa_specific_compounds sac
    JOIN compound_properties cp ON sac.compound_id = cp.compound_id
    JOIN reliable_compounds rc_sa ON sac.compound_id = rc_sa.compound_id AND rc_sa.pathogen_id = 's_aureus'
    JOIN reliable_compounds rc_ec ON sac.compound_id = rc_ec.compound_id AND rc_ec.pathogen_id = 'e_coli'
    JOIN reliable_compounds rc_ca ON sac.compound_id = rc_ca.compound_id AND rc_ca.pathogen_id = 'c_albicans'
    WHERE 
        -- Optional: Apply SELECT-G⁺ filters (comment out if too restrictive)
        cp.logp BETWEEN 1.9 AND 4.1
        AND cp.tpsa BETWEEN 17 AND 43
        AND cp.hbd_count BETWEEN 0 AND 1
        AND cp.molecular_weight BETWEEN 206 AND 325
),

compound_attribution AS (
    -- Calculate average SA attribution
    SELECT 
        cs.*,
        AVG(cf.attribution_score) as avg_sa_attribution,
        COUNT(cf.fragment_id) as num_fragments,
        MAX(cf.attribution_score) as max_sa_attribution
    FROM compound_stats cs
    LEFT JOIN compound_fragments cf 
        ON cs.compound_id = cf.compound_id 
        AND cf.pathogen_id = 's_aureus'
        AND cf.attribution_score >= 0.1  -- Positive attribution
    GROUP BY cs.compound_id, cs.smiles, cs.molecular_weight, cs.logp, cs.tpsa,
             cs.hbd_count, cs.hba_count, cs.total_rings, cs.aromatic_rings,
             cs.hetero_atoms, cs.hetero_cycles, cs.sa_prediction, 
             cs.ec_prediction, cs.ca_prediction
),

ranked_compounds AS (
    SELECT 
        *,
        (
            COALESCE(avg_sa_attribution, 0) * 0.5 +
            (CASE WHEN aromatic_rings >= 2 THEN 0.2 ELSE 0 END) +
            (CASE WHEN logp BETWEEN 2.5 AND 3.5 THEN 0.2 ELSE 0 END) +
            (CASE WHEN hetero_atoms >= 2 THEN 0.1 ELSE 0 END)
        ) as exemplar_score,
        ROW_NUMBER() OVER (ORDER BY 
            COALESCE(avg_sa_attribution, 0) DESC, 
            sa_prediction DESC
        ) as rank
    FROM compound_attribution
)

SELECT 
    rank,
    compound_id,
    smiles,
    ROUND(molecular_weight::numeric, 1) as mw,
    ROUND(logp::numeric, 2) as logp,
    ROUND(tpsa::numeric, 1) as tpsa,
    hbd_count as hbd,
    hba_count as hba,
    total_rings as rings,
    aromatic_rings as ar_rings,
    hetero_atoms,
    hetero_cycles,
    ROUND(sa_prediction::numeric, 3) as sa_pred,
    ROUND(ec_prediction::numeric, 3) as ec_pred,
    ROUND(ca_prediction::numeric, 3) as ca_pred,
    ROUND(COALESCE(avg_sa_attribution, 0)::numeric, 3) as avg_attr,
    ROUND(COALESCE(max_sa_attribution, 0)::numeric, 3) as max_attr,
    num_fragments,
    ROUND(exemplar_score::numeric, 3) as score,
    'S. aureus-specific (AII)' as pattern
FROM ranked_compounds
ORDER BY rank
LIMIT 30;