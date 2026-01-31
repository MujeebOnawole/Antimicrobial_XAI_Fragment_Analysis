-- E. coli-specific compounds: Inactive SA, Inactive CA, Active EC

WITH ec_specific_compounds AS (
    SELECT DISTINCT rc_ec.compound_id
    FROM reliable_compounds rc_ec
    JOIN reliable_compounds rc_sa 
        ON rc_ec.compound_id = rc_sa.compound_id 
        AND rc_sa.pathogen_id = 's_aureus'
    JOIN reliable_compounds rc_ca 
        ON rc_ec.compound_id = rc_ca.compound_id 
        AND rc_ca.pathogen_id = 'c_albicans'
    WHERE rc_ec.pathogen_id = 'e_coli'
      AND rc_ec.target = 1  -- Active EC
      AND rc_sa.target = 0  -- Inactive SA
      AND rc_ca.target = 0  -- Inactive CA
      AND rc_ec.has_mismatch = 0 AND rc_ec.reliability_level = 'RELIABLE'
      AND rc_sa.has_mismatch = 0 AND rc_sa.reliability_level = 'RELIABLE'
      AND rc_ca.has_mismatch = 0 AND rc_ca.reliability_level = 'RELIABLE'
),

compound_stats AS (
    SELECT 
        ecc.compound_id,
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
        rc_ec.ensemble_prediction as ec_prediction,
        rc_sa.ensemble_prediction as sa_prediction,
        rc_ca.ensemble_prediction as ca_prediction
    FROM ec_specific_compounds ecc
    JOIN compound_properties cp ON ecc.compound_id = cp.compound_id
    JOIN reliable_compounds rc_ec ON ecc.compound_id = rc_ec.compound_id AND rc_ec.pathogen_id = 'e_coli'
    JOIN reliable_compounds rc_sa ON ecc.compound_id = rc_sa.compound_id AND rc_sa.pathogen_id = 's_aureus'
    JOIN reliable_compounds rc_ca ON ecc.compound_id = rc_ca.compound_id AND rc_ca.pathogen_id = 'c_albicans'
    -- NO property filters initially to see what exists
),

compound_attribution AS (
    SELECT 
        cs.*,
        AVG(cf.attribution_score) as avg_ec_attribution,
        COUNT(cf.fragment_id) as num_fragments,
        MAX(cf.attribution_score) as max_ec_attribution
    FROM compound_stats cs
    LEFT JOIN compound_fragments cf 
        ON cs.compound_id = cf.compound_id 
        AND cf.pathogen_id = 'e_coli'
        AND cf.attribution_score >= 0.1
    GROUP BY cs.compound_id, cs.smiles, cs.molecular_weight, cs.logp, cs.tpsa,
             cs.hbd_count, cs.hba_count, cs.total_rings, cs.aromatic_rings,
             cs.hetero_atoms, cs.hetero_cycles, cs.ec_prediction, 
             cs.sa_prediction, cs.ca_prediction
),

ranked_compounds AS (
    SELECT 
        *,
        (
            COALESCE(avg_ec_attribution, 0) * 0.5 +
            (CASE WHEN hetero_cycles >= 2 THEN 0.25 ELSE 0 END) +
            (CASE WHEN hbd_count >= 3 THEN 0.25 ELSE 0 END)
        ) as exemplar_score,
        ROW_NUMBER() OVER (ORDER BY 
            COALESCE(avg_ec_attribution, 0) DESC, 
            ec_prediction DESC
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
    ROUND(ec_prediction::numeric, 3) as ec_pred,
    ROUND(sa_prediction::numeric, 3) as sa_pred,
    ROUND(ca_prediction::numeric, 3) as ca_pred,
    ROUND(COALESCE(avg_ec_attribution, 0)::numeric, 3) as avg_attr,
    ROUND(COALESCE(max_ec_attribution, 0)::numeric, 3) as max_attr,
    num_fragments,
    ROUND(exemplar_score::numeric, 3) as score,
    'E. coli-specific (IIA)' as pattern,
    -- Check SELECT-G⁻ compliance
    CASE 
        WHEN logp BETWEEN 1.3 AND 3.4
         AND tpsa BETWEEN 25 AND 50
         AND hbd_count BETWEEN 3 AND 5
         AND molecular_weight BETWEEN 196 AND 317
        THEN '✓ SELECT-G⁻'
        ELSE 'Outside windows'
    END as select_compliance
FROM ranked_compounds
ORDER BY rank
LIMIT 30;