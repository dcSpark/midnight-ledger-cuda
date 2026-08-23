//! Full immutable proving-key and SRS transport validation for Metal schema v2.

use std::mem::size_of_val;

use sha2::{Digest, Sha256};

use super::{MetalKeyContextInput, key_error, validate_metal_key_context_input};
use crate::{Bls12_381Fr, Bls12_381G1Affine, MetalError, NttDomainSpec, generated};

/// Version of the complete immutable proving-key and SRS Metal transport.
pub const METAL_KEY_CONTEXT_SCHEMA_VERSION_V2: u32 = 2;

const NEUTRAL_DOMAIN_SCHEMA_VERSION: u32 = 1;
const NEUTRAL_PREPARED_KEY_SCHEMA_VERSION: u32 = 1;
const NEUTRAL_PREPARED_SRS_SCHEMA_VERSION: u32 = 1;

const NEUTRAL_DOMAIN_FINGERPRINT_DOMAIN: &[u8] =
    b"midnight-gpu-prover/resident-domain-manifest/v1\0";
const NEUTRAL_PREPARED_KEY_FINGERPRINT_DOMAIN: &[u8] =
    b"midnight-gpu-prover/prepared-resident-proving-key/v1\0";
const NEUTRAL_SRS_COEFFICIENT_FINGERPRINT_DOMAIN: &[u8] =
    b"midnight-gpu-prover/resident-srs-coefficient/v1\0";
const NEUTRAL_SRS_LAGRANGE_FINGERPRINT_DOMAIN: &[u8] =
    b"midnight-gpu-prover/resident-srs-lagrange/v1\0";
const NEUTRAL_PREPARED_SRS_FINGERPRINT_DOMAIN: &[u8] =
    b"midnight-gpu-prover/prepared-resident-srs/v1\0";
const METAL_KEY_CONTEXT_V2_FINGERPRINT_DOMAIN: &[u8] =
    b"midnight-metal-prover/full-key-context/v2\0";

const COLUMN_KIND_FIXED: u32 = 0;
const COLUMN_KIND_ADVICE: u32 = 1;
const COLUMN_KIND_INSTANCE: u32 = 2;

const TWO_ADIC_ROOT: Bls12_381Fr = Bls12_381Fr {
    limbs: [
        0xb9b5_8d8c_5f0e_466a,
        0x5b1b_4c80_1819_d7ec,
        0x0af5_3ae3_52a3_1e64,
        0x5bf3_adda_19e9_b27b,
    ],
};
const TWO_ADIC_ROOT_INV: Bls12_381Fr = Bls12_381Fr {
    limbs: [
        0x4256_481a_dcf3_219a,
        0x45f3_7b7f_96b6_cad3,
        0xf9c3_f1d7_5f7a_3b27,
        0x2d2f_c049_658a_fd43,
    ],
};
const TWO_INV: Bls12_381Fr = Bls12_381Fr {
    limbs: [
        0x0000_0000_ffff_ffff,
        0xac42_5bfd_0001_a401,
        0xccc6_27f7_f65e_27fa,
        0x0c12_58ac_d662_82b7,
    ],
};
const ZETA: Bls12_381Fr = Bls12_381Fr {
    limbs: [
        0x92d9_090b_0930_11d2,
        0xfc9c_bd71_9d6a_a073,
        0xc1f1_4ef0_cd65_a1a6,
        0x017f_6d35_e72f_cdeb,
    ],
};
const DELTA: Bls12_381Fr = Bls12_381Fr {
    limbs: [
        0x70e3_10d3_d146_f96a,
        0x4b64_c089_19e2_99e6,
        0x51e1_1418_6a8b_970d,
        0x6185_d066_27c0_67cb,
    ],
};

/// Dense permutation-column reference in original protocol insertion order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct MetalResidentColumnRef {
    /// Numeric column kind: zero fixed, one advice, or two instance.
    pub kind: u32,
    /// Dense index in the corresponding graph-family compact map.
    pub index: u32,
}

/// Exact neutral domain identity and every field value needed by Metal.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MetalResidentDomainIdentity<'a> {
    /// Neutral resident-domain schema version.
    pub neutral_schema_version: u32,
    /// Neutral resident-domain SHA-256.
    pub neutral_fingerprint: [u8; 32],
    /// Base-domain element count.
    pub n: u32,
    /// Base-two logarithm of `n`.
    pub k: u32,
    /// Extended-domain element count.
    pub n_ext: u32,
    /// Base-two logarithm of `n_ext`.
    pub extended_k: u32,
    /// Maximum constraint-system degree.
    pub constraint_system_degree: u32,
    /// Number of quotient pieces.
    pub quotient_piece_count: u32,
    /// Number of proof blinding rows.
    pub blinding_factors: u32,
    /// Number of usable base-domain rows.
    pub usable_rows: u32,
    /// Base-row rotation multiplier in the extended domain.
    pub rotation_scale: i32,
    /// Rotation of the first inactive base-domain row.
    pub last_rotation: i32,
    /// Number of permutation columns combined per product chunk.
    pub permutation_chunk_len: u32,
    /// Exact canonical base-domain NTT parameters.
    pub base_ntt: NttDomainSpec,
    /// Exact canonical extended-domain NTT parameters.
    pub extended_ntt: NttDomainSpec,
    /// Barycentric weight at the first base-domain point.
    pub barycentric_weight: Bls12_381Fr,
    /// BLS12-381 scalar-field permutation delta.
    pub delta: Bls12_381Fr,
    /// Repeating inverses of the vanishing polynomial on the extended coset.
    pub vanishing_inverses: &'a [Bls12_381Fr],
}

/// Complete borrowed immutable proving-key and SRS transport for Metal v2.
#[derive(Clone, Copy, Debug)]
pub struct MetalKeyContextInputV2<'a> {
    /// Existing schema-v1 graph, compact fixed-coset, and selector transport.
    pub graph_fixed: MetalKeyContextInput<'a>,
    /// Exact neutral polynomial-domain identity.
    pub domain: MetalResidentDomainIdentity<'a>,
    /// Neutral prepared-proving-key schema version.
    pub prepared_key_schema_version: u32,
    /// Neutral prepared-proving-key SHA-256.
    pub prepared_key_fingerprint: [u8; 32],
    /// Neutral prepared-SRS schema version.
    pub prepared_srs_schema_version: u32,
    /// Base-domain length claimed by the prepared SRS.
    pub prepared_srs_n: u32,
    /// Neutral prepared-SRS SHA-256.
    pub prepared_srs_fingerprint: [u8; 32],
    /// Neutral coefficient-basis SRS prefix digest.
    pub coefficient_srs_digest: [u8; 32],
    /// Neutral Lagrange-basis SRS prefix digest.
    pub lagrange_srs_digest: [u8; 32],
    /// Permutation columns in original protocol insertion order.
    pub permutation_columns: &'a [MetalResidentColumnRef],
    /// Trash-selector indices in trash-argument order and the compact fixed map.
    pub trash_selector_fixed_indices: &'a [u32],
    /// Compact fixed columns over the base Lagrange domain.
    pub fixed_lagrange: &'a [Bls12_381Fr],
    /// Permutation columns over the extended coset in insertion order.
    pub permutation_cosets: &'a [Bls12_381Fr],
    /// Permutation columns over the base Lagrange domain in insertion order.
    pub permutation_lagrange: &'a [Bls12_381Fr],
    /// All logical fixed coefficients followed by permutation coefficients.
    pub coefficient_columns: &'a [Bls12_381Fr],
    /// Number of logical fixed columns at the start of `coefficient_columns`.
    pub fixed_coefficient_count: u32,
    /// Number of permutation columns at the end of `coefficient_columns`.
    pub permutation_coefficient_count: u32,
    /// Exact coefficient-basis G1 SRS prefix.
    pub coefficient_bases: &'a [Bls12_381G1Affine],
    /// Exact Lagrange-basis G1 SRS prefix.
    pub lagrange_bases: &'a [Bls12_381G1Affine],
}

#[allow(dead_code)]
pub(crate) struct ValidatedMetalKeyContextInputV2<'a> {
    pub(crate) input: MetalKeyContextInputV2<'a>,
    pub(crate) content_fingerprint: [u8; 32],
    pub(crate) immutable_bytes: usize,
}

#[allow(dead_code)]
pub(crate) fn validate_metal_key_context_input_v2(
    input: MetalKeyContextInputV2<'_>,
) -> Result<ValidatedMetalKeyContextInputV2<'_>, MetalError> {
    let validated = prevalidate_metal_key_context_input_v2(input)?;
    validate_prevalidated_metal_key_context_srs_subgroups_v2(&validated)?;
    Ok(validated)
}

/// Performs every cheap structural, reduction, and fingerprint check.
///
/// A later cache owner may use the returned authoritative content fingerprint
/// for lookup, then call `validate_prevalidated_metal_key_context_srs_subgroups_v2`
/// only before importing a cache miss.
#[allow(dead_code)]
pub(crate) fn prevalidate_metal_key_context_input_v2(
    input: MetalKeyContextInputV2<'_>,
) -> Result<ValidatedMetalKeyContextInputV2<'_>, MetalError> {
    let validated_v1 = validate_metal_key_context_input(input.graph_fixed)?;
    validate_domain(input.domain, &input.graph_fixed)?;
    validate_key_shapes_and_columns(&input)?;
    validate_srs_shapes_and_coordinates(&input)?;
    validate_neutral_fingerprints(&input)?;

    let immutable_bytes = v2_immutable_bytes(&input, validated_v1.immutable_bytes)?;
    let content_fingerprint = key_content_fingerprint_v2(&input, validated_v1.content_fingerprint);
    Ok(ValidatedMetalKeyContextInputV2 {
        input,
        content_fingerprint,
        immutable_bytes,
    })
}

/// Revalidates every SRS point with BLST before a prevalidated cache miss upload.
#[allow(dead_code)]
pub(crate) fn validate_prevalidated_metal_key_context_srs_subgroups_v2(
    validated: &ValidatedMetalKeyContextInputV2<'_>,
) -> Result<(), MetalError> {
    validate_srs_subgroups(&validated.input)
}

fn validate_domain(
    domain: MetalResidentDomainIdentity<'_>,
    graph_fixed: &MetalKeyContextInput<'_>,
) -> Result<(), MetalError> {
    if domain.neutral_schema_version != NEUTRAL_DOMAIN_SCHEMA_VERSION {
        return Err(key_error(format!(
            "unsupported neutral domain schema version {}",
            domain.neutral_schema_version
        )));
    }
    if domain.n != graph_fixed.n || domain.n_ext != graph_fixed.n_ext {
        return Err(key_error(
            "neutral domain sizes do not match the nested graph/fixed key input",
        ));
    }
    if 1u32.checked_shl(domain.k) != Some(domain.n)
        || 1u32.checked_shl(domain.extended_k) != Some(domain.n_ext)
    {
        return Err(key_error(
            "neutral domain lengths do not equal their base-two logarithms",
        ));
    }
    if domain.extended_k < domain.k || domain.constraint_system_degree < 3 {
        return Err(key_error(
            "neutral domain has an invalid extension or constraint-system degree",
        ));
    }

    let expected_quotient_pieces = domain.constraint_system_degree - 1;
    let expected_chunk_len = domain.constraint_system_degree - 2;
    if domain.quotient_piece_count != expected_quotient_pieces
        || domain.permutation_chunk_len != expected_chunk_len
    {
        return Err(key_error(
            "neutral domain quotient or permutation metadata is inconsistent",
        ));
    }
    let inactive_rows = domain
        .blinding_factors
        .checked_add(1)
        .ok_or_else(|| key_error("neutral domain inactive-row count overflows u32"))?;
    let expected_usable_rows = domain
        .n
        .checked_sub(inactive_rows)
        .filter(|rows| *rows != 0)
        .ok_or_else(|| key_error("neutral domain has no usable rows"))?;
    let expected_last_rotation = -i32::try_from(inactive_rows)
        .map_err(|_| key_error("neutral domain last rotation exceeds i32"))?;
    if domain.usable_rows != expected_usable_rows || domain.last_rotation != expected_last_rotation
    {
        return Err(key_error(
            "neutral domain usable-row or last-rotation metadata is inconsistent",
        ));
    }

    let extension = domain
        .n_ext
        .checked_div(domain.n)
        .ok_or_else(|| key_error("neutral domain base size is zero"))?;
    let expected_extension = expected_quotient_pieces
        .checked_next_power_of_two()
        .ok_or_else(|| key_error("neutral quotient extension exceeds u32"))?;
    let expected_rotation_scale = i32::try_from(extension)
        .map_err(|_| key_error("neutral domain rotation scale exceeds i32"))?;
    if extension != expected_extension || domain.rotation_scale != expected_rotation_scale {
        return Err(key_error(
            "neutral domain extension or rotation-scale metadata is inconsistent",
        ));
    }
    if domain.vanishing_inverses.len() != extension as usize {
        return Err(key_error(format!(
            "neutral domain has {} vanishing inverses, expected {extension}",
            domain.vanishing_inverses.len()
        )));
    }

    validate_reduced_fr(
        "base NTT parameter",
        &[
            domain.base_ntt.root,
            domain.base_ntt.inverse_root,
            domain.base_ntt.inverse_size,
            domain.base_ntt.coset_generator,
            domain.base_ntt.inverse_coset_generator,
        ],
    )?;
    validate_reduced_fr(
        "extended NTT parameter",
        &[
            domain.extended_ntt.root,
            domain.extended_ntt.inverse_root,
            domain.extended_ntt.inverse_size,
            domain.extended_ntt.coset_generator,
            domain.extended_ntt.inverse_coset_generator,
        ],
    )?;
    validate_reduced_fr(
        "neutral domain scalar",
        &[domain.barycentric_weight, domain.delta],
    )?;
    validate_reduced_fr("vanishing inverse", domain.vanishing_inverses)?;

    if domain.base_ntt != canonical_ntt_domain(domain.k)?
        || domain.extended_ntt != canonical_ntt_domain(domain.extended_k)?
    {
        return Err(key_error(
            "neutral domain NTT parameters are not the canonical BLS12-381 domains",
        ));
    }
    if domain.barycentric_weight != domain.base_ntt.inverse_size {
        return Err(key_error(
            "neutral domain barycentric weight differs from the base inverse size",
        ));
    }
    if domain.delta != DELTA {
        return Err(key_error(
            "neutral domain permutation delta is not the BLS12-381 constant",
        ));
    }
    validate_vanishing_inverses(domain)?;

    let actual_fingerprint = neutral_domain_fingerprint(domain);
    if actual_fingerprint != domain.neutral_fingerprint {
        return Err(key_error(
            "neutral domain fingerprint does not match the transported domain fields",
        ));
    }
    Ok(())
}

fn validate_key_shapes_and_columns(input: &MetalKeyContextInputV2<'_>) -> Result<(), MetalError> {
    if input.prepared_key_schema_version != NEUTRAL_PREPARED_KEY_SCHEMA_VERSION {
        return Err(key_error(format!(
            "unsupported neutral prepared-key schema version {}",
            input.prepared_key_schema_version
        )));
    }
    let n = input.domain.n as usize;
    let n_ext = input.domain.n_ext as usize;
    let permutation_count = input.permutation_columns.len();
    for (label, len) in [
        ("permutation columns", permutation_count),
        (
            "trash selector fixed indices",
            input.trash_selector_fixed_indices.len(),
        ),
        ("compact fixed Lagrange columns", input.fixed_lagrange.len()),
        ("permutation cosets", input.permutation_cosets.len()),
        (
            "permutation Lagrange columns",
            input.permutation_lagrange.len(),
        ),
        ("coefficient columns", input.coefficient_columns.len()),
    ] {
        checked_u32_len_v2(label, len)?;
    }
    exact_column_extent(
        "compact fixed Lagrange columns",
        input.fixed_lagrange.len(),
        input.graph_fixed.fixed_compact_to_logical.len(),
        n,
    )?;
    exact_column_extent(
        "permutation cosets",
        input.permutation_cosets.len(),
        permutation_count,
        n_ext,
    )?;
    exact_column_extent(
        "permutation Lagrange columns",
        input.permutation_lagrange.len(),
        permutation_count,
        n,
    )?;

    let permutation_count_u32 = u32::try_from(permutation_count)
        .map_err(|_| key_error("permutation column count exceeds u32"))?;
    if input.permutation_coefficient_count != permutation_count_u32 {
        return Err(key_error(
            "permutation coefficient count differs from the permutation descriptor count",
        ));
    }
    let coefficient_count = input
        .fixed_coefficient_count
        .checked_add(input.permutation_coefficient_count)
        .ok_or_else(|| key_error("coefficient column count overflows u32"))?;
    exact_column_extent(
        "coefficient columns",
        input.coefficient_columns.len(),
        coefficient_count as usize,
        n,
    )?;

    for (compact_index, logical_index) in input
        .graph_fixed
        .fixed_compact_to_logical
        .iter()
        .copied()
        .enumerate()
    {
        if logical_index >= input.fixed_coefficient_count {
            return Err(key_error(format!(
                "compact fixed column {compact_index} names logical column {logical_index}, outside {} coefficient columns",
                input.fixed_coefficient_count
            )));
        }
    }
    for (position, column) in input.permutation_columns.iter().copied().enumerate() {
        let bound = match column.kind {
            COLUMN_KIND_FIXED => input.graph_fixed.fixed_compact_to_logical.len(),
            COLUMN_KIND_ADVICE => input.graph_fixed.advice_compact_to_logical.len(),
            COLUMN_KIND_INSTANCE => input.graph_fixed.instance_compact_to_logical.len(),
            kind => {
                return Err(key_error(format!(
                    "permutation column {position} has unknown kind {kind}"
                )));
            }
        };
        if column.index as usize >= bound {
            return Err(key_error(format!(
                "permutation column {position} index {} exceeds compact bound {bound}",
                column.index
            )));
        }
    }
    if input.trash_selector_fixed_indices.len() != input.graph_fixed.sections.trash as usize {
        return Err(key_error(
            "trash-selector count differs from the graph-family trash section",
        ));
    }
    let fixed_bound = input.graph_fixed.fixed_compact_to_logical.len();
    for (position, index) in input
        .trash_selector_fixed_indices
        .iter()
        .copied()
        .enumerate()
    {
        if index as usize >= fixed_bound {
            return Err(key_error(format!(
                "trash selector {position} index {index} exceeds compact fixed bound {fixed_bound}"
            )));
        }
    }

    validate_reduced_fr("compact fixed Lagrange", input.fixed_lagrange)?;
    validate_reduced_fr("permutation coset", input.permutation_cosets)?;
    validate_reduced_fr("permutation Lagrange", input.permutation_lagrange)?;
    validate_reduced_fr("coefficient column", input.coefficient_columns)?;
    Ok(())
}

fn validate_srs_shapes_and_coordinates(
    input: &MetalKeyContextInputV2<'_>,
) -> Result<(), MetalError> {
    if input.prepared_srs_schema_version != NEUTRAL_PREPARED_SRS_SCHEMA_VERSION {
        return Err(key_error(format!(
            "unsupported neutral prepared-SRS schema version {}",
            input.prepared_srs_schema_version
        )));
    }
    if input.prepared_srs_n != input.domain.n {
        return Err(key_error(format!(
            "prepared SRS length {} differs from proving-key domain length {}",
            input.prepared_srs_n, input.domain.n
        )));
    }
    let n = input.prepared_srs_n as usize;
    if input.coefficient_bases.len() != n || input.lagrange_bases.len() != n {
        return Err(key_error(format!(
            "prepared SRS contains {} coefficient and {} Lagrange bases, expected {n} of each",
            input.coefficient_bases.len(),
            input.lagrange_bases.len()
        )));
    }
    checked_u32_len_v2("coefficient SRS", input.coefficient_bases.len())?;
    checked_u32_len_v2("Lagrange SRS", input.lagrange_bases.len())?;
    validate_reduced_g1_points("coefficient SRS", input.coefficient_bases)?;
    validate_reduced_g1_points("Lagrange SRS", input.lagrange_bases)?;
    Ok(())
}

fn validate_srs_subgroups(input: &MetalKeyContextInputV2<'_>) -> Result<(), MetalError> {
    validate_g1_subgroups("coefficient SRS", input.coefficient_bases)?;
    validate_g1_subgroups("Lagrange SRS", input.lagrange_bases)
}

fn validate_neutral_fingerprints(input: &MetalKeyContextInputV2<'_>) -> Result<(), MetalError> {
    let actual_key_fingerprint = neutral_prepared_key_fingerprint(input);
    if actual_key_fingerprint != input.prepared_key_fingerprint {
        return Err(key_error(
            "neutral prepared-key fingerprint does not match the transported fields",
        ));
    }

    let coefficient_digest = neutral_srs_prefix_fingerprint(
        NEUTRAL_SRS_COEFFICIENT_FINGERPRINT_DOMAIN,
        input.prepared_srs_schema_version,
        input.prepared_srs_n,
        input.coefficient_bases,
    );
    if coefficient_digest != input.coefficient_srs_digest {
        return Err(key_error(
            "neutral coefficient SRS digest does not match the transported bases",
        ));
    }
    let lagrange_digest = neutral_srs_prefix_fingerprint(
        NEUTRAL_SRS_LAGRANGE_FINGERPRINT_DOMAIN,
        input.prepared_srs_schema_version,
        input.prepared_srs_n,
        input.lagrange_bases,
    );
    if lagrange_digest != input.lagrange_srs_digest {
        return Err(key_error(
            "neutral Lagrange SRS digest does not match the transported bases",
        ));
    }
    let actual_srs_fingerprint = neutral_prepared_srs_fingerprint(
        input.prepared_srs_schema_version,
        input.prepared_srs_n,
        coefficient_digest,
        lagrange_digest,
    );
    if actual_srs_fingerprint != input.prepared_srs_fingerprint {
        return Err(key_error(
            "neutral prepared-SRS fingerprint does not match the transported fields",
        ));
    }
    Ok(())
}

fn validate_reduced_fr(label: &str, values: &[Bls12_381Fr]) -> Result<(), MetalError> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| greater_or_equal(value.limbs, generated::FR_MODULUS))
    {
        Err(key_error(format!(
            "{label} element {index} is not a reduced BLS12-381 Fr value"
        )))
    } else {
        Ok(())
    }
}

fn validate_reduced_g1_points(label: &str, points: &[Bls12_381G1Affine]) -> Result<(), MetalError> {
    for (index, point) in points.iter().copied().enumerate() {
        for (coordinate, limbs) in [("x", point.x), ("y", point.y)] {
            if greater_or_equal(limbs, generated::FP_MODULUS) {
                return Err(key_error(format!(
                    "{label} point {index} {coordinate}-coordinate is not reduced"
                )));
            }
        }
        if point.x == [0; 6] && point.y == [0; 6] {
            return Err(key_error(format!(
                "{label} point {index} is the affine identity; prepared KZG SRS bases must be nonzero"
            )));
        }
    }
    Ok(())
}

fn validate_g1_subgroups(label: &str, points: &[Bls12_381G1Affine]) -> Result<(), MetalError> {
    for (index, point) in points.iter().copied().enumerate() {
        debug_assert!(point.x != [0; 6] || point.y != [0; 6]);
        let affine = blst::blst_p1_affine {
            x: blst::blst_fp { l: point.x },
            y: blst::blst_fp { l: point.y },
        };
        // SAFETY: both coordinates were fieldwise copied into initialized BLST
        // values after their six limbs were checked to be reduced.
        if !unsafe { blst::blst_p1_affine_on_curve(&affine) } {
            return Err(key_error(format!(
                "{label} point {index} is not on the BLS12-381 curve"
            )));
        }
        // SAFETY: the initialized affine point passed BLST's on-curve check.
        if !unsafe { blst::blst_p1_affine_in_g1(&affine) } {
            return Err(key_error(format!(
                "{label} point {index} is not in the BLS12-381 G1 subgroup"
            )));
        }
    }
    Ok(())
}

fn canonical_ntt_domain(log_size: u32) -> Result<NttDomainSpec, MetalError> {
    if log_size > 32 {
        return Err(key_error(
            "BLS12-381 NTT domain exceeds scalar-field two-adicity",
        ));
    }
    let mut root = TWO_ADIC_ROOT;
    let mut inverse_root = TWO_ADIC_ROOT_INV;
    for _ in log_size..32 {
        root = fr_mul(root, root);
        inverse_root = fr_mul(inverse_root, inverse_root);
    }
    let mut inverse_size = fr_one();
    for _ in 0..log_size {
        inverse_size = fr_mul(inverse_size, TWO_INV);
    }
    Ok(NttDomainSpec {
        log_size,
        root,
        inverse_root,
        inverse_size,
        coset_generator: ZETA,
        inverse_coset_generator: fr_mul(ZETA, ZETA),
    })
}

fn validate_vanishing_inverses(domain: MetalResidentDomainIdentity<'_>) -> Result<(), MetalError> {
    let mut current = fr_pow(domain.extended_ntt.coset_generator, u64::from(domain.n));
    let step = fr_pow(domain.extended_ntt.root, u64::from(domain.n));
    for (index, supplied) in domain.vanishing_inverses.iter().copied().enumerate() {
        let denominator = fr_sub(current, fr_one());
        if denominator == Bls12_381Fr::default() {
            return Err(key_error(format!("vanishing denominator {index} is zero")));
        }
        let expected = fr_inverse(denominator);
        if supplied != expected {
            return Err(key_error(format!(
                "vanishing inverse {index} does not match the exact extended coset"
            )));
        }
        current = fr_mul(current, step);
    }
    Ok(())
}

fn fr_one() -> Bls12_381Fr {
    Bls12_381Fr {
        limbs: generated::FR_R,
    }
}

fn fr_mul(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
    let mut output = blst::blst_fr::default();
    let lhs = blst::blst_fr { l: lhs.limbs };
    let rhs = blst::blst_fr { l: rhs.limbs };
    // SAFETY: every BLST value is initialized, aligned, and the mutable output
    // is disjoint from both immutable inputs.
    unsafe { blst::blst_fr_mul(&mut output, &lhs, &rhs) };
    Bls12_381Fr { limbs: output.l }
}

fn fr_sub(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
    let mut output = blst::blst_fr::default();
    let lhs = blst::blst_fr { l: lhs.limbs };
    let rhs = blst::blst_fr { l: rhs.limbs };
    // SAFETY: every BLST value is initialized, aligned, and the mutable output
    // is disjoint from both immutable inputs.
    unsafe { blst::blst_fr_sub(&mut output, &lhs, &rhs) };
    Bls12_381Fr { limbs: output.l }
}

fn fr_inverse(value: Bls12_381Fr) -> Bls12_381Fr {
    let mut output = blst::blst_fr::default();
    let value = blst::blst_fr { l: value.limbs };
    // SAFETY: the BLST input and disjoint mutable output are initialized and aligned.
    unsafe { blst::blst_fr_inverse(&mut output, &value) };
    Bls12_381Fr { limbs: output.l }
}

fn fr_pow(mut base: Bls12_381Fr, mut exponent: u64) -> Bls12_381Fr {
    let mut accumulator = fr_one();
    while exponent != 0 {
        if exponent & 1 == 1 {
            accumulator = fr_mul(accumulator, base);
        }
        base = fr_mul(base, base);
        exponent >>= 1;
    }
    accumulator
}

fn exact_column_extent(
    label: &str,
    actual: usize,
    columns: usize,
    stride: usize,
) -> Result<(), MetalError> {
    let expected = columns
        .checked_mul(stride)
        .ok_or_else(|| key_error(format!("{label} extent overflows usize")))?;
    if actual == expected {
        Ok(())
    } else {
        Err(key_error(format!(
            "{label} contain {actual} elements, expected {expected}"
        )))
    }
}

fn checked_u32_len_v2(label: &str, len: usize) -> Result<u32, MetalError> {
    u32::try_from(len).map_err(|_| key_error(format!("{label} length {len} exceeds u32")))
}

fn v2_immutable_bytes(
    input: &MetalKeyContextInputV2<'_>,
    v1_bytes: usize,
) -> Result<usize, MetalError> {
    [
        v1_bytes,
        size_of_val(input.domain.vanishing_inverses),
        size_of_val(input.permutation_columns),
        size_of_val(input.trash_selector_fixed_indices),
        size_of_val(input.fixed_lagrange),
        size_of_val(input.permutation_cosets),
        size_of_val(input.permutation_lagrange),
        size_of_val(input.coefficient_columns),
        size_of_val(input.coefficient_bases),
        size_of_val(input.lagrange_bases),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or_else(|| key_error("v2 immutable key-context byte count overflows usize"))
}

fn neutral_domain_fingerprint(domain: MetalResidentDomainIdentity<'_>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(NEUTRAL_DOMAIN_FINGERPRINT_DOMAIN);
    for value in [
        domain.neutral_schema_version,
        domain.n,
        domain.k,
        domain.n_ext,
        domain.extended_k,
        domain.constraint_system_degree,
        domain.quotient_piece_count,
        domain.blinding_factors,
        domain.usable_rows,
    ] {
        hash_u32(&mut hasher, value);
    }
    hash_i32(&mut hasher, domain.rotation_scale);
    hash_i32(&mut hasher, domain.last_rotation);
    hash_u32(&mut hasher, domain.permutation_chunk_len);
    for value in [
        domain.base_ntt.root,
        domain.base_ntt.inverse_root,
        domain.extended_ntt.root,
        domain.extended_ntt.inverse_root,
        domain.base_ntt.inverse_size,
        domain.extended_ntt.inverse_size,
        domain.base_ntt.coset_generator,
        domain.base_ntt.inverse_coset_generator,
        domain.barycentric_weight,
    ] {
        hash_fr(&mut hasher, value);
    }
    hash_fr_slice(&mut hasher, domain.vanishing_inverses);
    hash_fr(&mut hasher, domain.delta);
    hasher.finalize().into()
}

fn neutral_prepared_key_fingerprint(input: &MetalKeyContextInputV2<'_>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(NEUTRAL_PREPARED_KEY_FINGERPRINT_DOMAIN);
    hash_u32(&mut hasher, input.prepared_key_schema_version);
    hasher.update(input.domain.neutral_fingerprint);
    hash_u32(
        &mut hasher,
        input.graph_fixed.graph_family_fingerprint_version,
    );
    hasher.update(input.graph_fixed.graph_family_fingerprint);
    for count in [
        input.graph_fixed.sections.custom,
        input.graph_fixed.sections.lookups,
        input.graph_fixed.sections.lookup_inputs,
        input.graph_fixed.sections.lookup_tables,
        input.graph_fixed.sections.trash,
    ] {
        hash_u32(&mut hasher, count);
    }
    hash_len(&mut hasher, input.permutation_columns.len());
    for column in input.permutation_columns {
        hash_u32(&mut hasher, column.kind);
        hash_u32(&mut hasher, column.index);
    }
    hash_u32_slice(&mut hasher, input.trash_selector_fixed_indices);
    hash_fr_slice(&mut hasher, input.graph_fixed.fixed_cosets);
    hash_fr_slice(&mut hasher, input.fixed_lagrange);
    hash_fr_slice(&mut hasher, input.permutation_cosets);
    hash_fr_slice(&mut hasher, input.permutation_lagrange);
    hash_u32(&mut hasher, input.fixed_coefficient_count);
    hash_u32(&mut hasher, input.permutation_coefficient_count);
    hash_fr_slice(&mut hasher, input.coefficient_columns);
    hash_fr_slice(&mut hasher, input.graph_fixed.lagrange_selectors);
    hasher.finalize().into()
}

fn neutral_srs_prefix_fingerprint(
    fingerprint_domain: &[u8],
    schema_version: u32,
    n: u32,
    points: &[Bls12_381G1Affine],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(fingerprint_domain);
    hash_u32(&mut hasher, schema_version);
    hash_u32(&mut hasher, n);
    hash_len(&mut hasher, points.len());
    for point in points {
        hash_g1(&mut hasher, *point);
    }
    hasher.finalize().into()
}

fn neutral_prepared_srs_fingerprint(
    schema_version: u32,
    n: u32,
    coefficient_digest: [u8; 32],
    lagrange_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(NEUTRAL_PREPARED_SRS_FINGERPRINT_DOMAIN);
    hash_u32(&mut hasher, schema_version);
    hash_u32(&mut hasher, n);
    hasher.update(coefficient_digest);
    hasher.update(lagrange_digest);
    hasher.finalize().into()
}

fn key_content_fingerprint_v2(
    input: &MetalKeyContextInputV2<'_>,
    validated_v1_fingerprint: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(METAL_KEY_CONTEXT_V2_FINGERPRINT_DOMAIN);
    hash_u32(&mut hasher, METAL_KEY_CONTEXT_SCHEMA_VERSION_V2);

    // This digest was recomputed by the schema-v1 validator and covers every
    // nested graph/fixed field and array, including the neutral graph digest.
    hasher.update(validated_v1_fingerprint);
    hasher.update(input.graph_fixed.graph_family_fingerprint);

    // These neutral identities were independently recomputed from all domain,
    // proving-key, and SRS arrays above. Hashing their verified digests avoids
    // a second O(key + SRS) traversal while still binding every transported
    // field transitively and domain-separating the Metal cache identity.
    hash_u32(&mut hasher, input.domain.neutral_schema_version);
    hasher.update(input.domain.neutral_fingerprint);
    hash_u32(&mut hasher, input.prepared_key_schema_version);
    hasher.update(input.prepared_key_fingerprint);
    hash_u32(&mut hasher, input.prepared_srs_schema_version);
    hash_u32(&mut hasher, input.prepared_srs_n);
    hasher.update(input.prepared_srs_fingerprint);
    hasher.update(input.coefficient_srs_digest);
    hasher.update(input.lagrange_srs_digest);
    hasher.finalize().into()
}

fn hash_fr_slice(hasher: &mut Sha256, values: &[Bls12_381Fr]) {
    hash_len(hasher, values.len());
    for value in values {
        hash_fr(hasher, *value);
    }
}

fn hash_u32_slice(hasher: &mut Sha256, values: &[u32]) {
    hash_len(hasher, values.len());
    for value in values {
        hash_u32(hasher, *value);
    }
}

fn hash_g1(hasher: &mut Sha256, point: Bls12_381G1Affine) {
    for limb in point.x.into_iter().chain(point.y) {
        hasher.update(limb.to_le_bytes());
    }
}

fn hash_fr(hasher: &mut Sha256, value: Bls12_381Fr) {
    for limb in value.limbs {
        hasher.update(limb.to_le_bytes());
    }
}

fn hash_len(hasher: &mut Sha256, len: usize) {
    hasher.update((len as u64).to_le_bytes());
}

fn hash_u32(hasher: &mut Sha256, value: u32) {
    hasher.update(value.to_le_bytes());
}

fn hash_i32(hasher: &mut Sha256, value: i32) {
    hasher.update(value.to_le_bytes());
}

fn greater_or_equal<const N: usize>(lhs: [u64; N], rhs: [u64; N]) -> bool {
    for index in (0..N).rev() {
        if lhs[index] != rhs[index] {
            return lhs[index] > rhs[index];
        }
    }
    true
}

#[cfg(test)]
pub(crate) mod tests {
    use std::mem::{align_of, size_of, size_of_val};

    use midnight_gpu_prover::{
        PreparedResidentProvingKey, PreparedResidentProvingKeyInput, PreparedResidentSrs, RawFr,
        RawG1Affine, ResidentColumnKind, ResidentColumnRef, ResidentDomainManifest,
        ResidentDomainManifestInput,
        expression::{
            CompiledGraph, CompiledProtocolGraphFamily, GraphColumnRequirements, GraphOp,
            GraphOpcode, GraphSourceCounts, GraphValueKind, GraphValueRef,
        },
    };

    use super::*;
    use crate::key_context::{OwnedMetalKeyContextInput, neutral_graph_family_fingerprint};
    use crate::{MetalGraphDescriptor, MetalGraphOperation, MetalGraphValueRef};

    #[derive(Clone)]
    pub(crate) struct OwnedDomain {
        pub(crate) neutral_schema_version: u32,
        pub(crate) neutral_fingerprint: [u8; 32],
        pub(crate) n: u32,
        pub(crate) k: u32,
        pub(crate) n_ext: u32,
        pub(crate) extended_k: u32,
        pub(crate) constraint_system_degree: u32,
        pub(crate) quotient_piece_count: u32,
        pub(crate) blinding_factors: u32,
        pub(crate) usable_rows: u32,
        pub(crate) rotation_scale: i32,
        pub(crate) last_rotation: i32,
        pub(crate) permutation_chunk_len: u32,
        pub(crate) base_ntt: NttDomainSpec,
        pub(crate) extended_ntt: NttDomainSpec,
        pub(crate) barycentric_weight: Bls12_381Fr,
        pub(crate) delta: Bls12_381Fr,
        pub(crate) vanishing_inverses: Vec<Bls12_381Fr>,
    }

    impl OwnedDomain {
        fn representative() -> Self {
            let n = 2;
            let n_ext = 4;
            let base_ntt = canonical_ntt_domain(1).expect("base NTT domain");
            let extended_ntt = canonical_ntt_domain(2).expect("extended NTT domain");
            let vanishing_inverses = exact_vanishing_inverses(n, n_ext, extended_ntt);
            Self {
                neutral_schema_version: 1,
                neutral_fingerprint: [0; 32],
                n,
                k: 1,
                n_ext,
                extended_k: 2,
                constraint_system_degree: 3,
                quotient_piece_count: 2,
                blinding_factors: 0,
                usable_rows: 1,
                rotation_scale: 2,
                last_rotation: -1,
                permutation_chunk_len: 1,
                base_ntt,
                extended_ntt,
                barycentric_weight: base_ntt.inverse_size,
                delta: DELTA,
                vanishing_inverses,
            }
        }

        fn input(&self) -> MetalResidentDomainIdentity<'_> {
            MetalResidentDomainIdentity {
                neutral_schema_version: self.neutral_schema_version,
                neutral_fingerprint: self.neutral_fingerprint,
                n: self.n,
                k: self.k,
                n_ext: self.n_ext,
                extended_k: self.extended_k,
                constraint_system_degree: self.constraint_system_degree,
                quotient_piece_count: self.quotient_piece_count,
                blinding_factors: self.blinding_factors,
                usable_rows: self.usable_rows,
                rotation_scale: self.rotation_scale,
                last_rotation: self.last_rotation,
                permutation_chunk_len: self.permutation_chunk_len,
                base_ntt: self.base_ntt,
                extended_ntt: self.extended_ntt,
                barycentric_weight: self.barycentric_weight,
                delta: self.delta,
                vanishing_inverses: &self.vanishing_inverses,
            }
        }

        fn neutral_manifest(&self) -> ResidentDomainManifest {
            ResidentDomainManifest::try_new(ResidentDomainManifestInput {
                k: self.k,
                extended_k: self.extended_k,
                constraint_system_degree: self.constraint_system_degree,
                blinding_factors: self.blinding_factors,
                omega: neutral_fr(self.base_ntt.root),
                omega_inv: neutral_fr(self.base_ntt.inverse_root),
                extended_omega: neutral_fr(self.extended_ntt.root),
                extended_omega_inv: neutral_fr(self.extended_ntt.inverse_root),
                n_inv: neutral_fr(self.base_ntt.inverse_size),
                n_ext_inv: neutral_fr(self.extended_ntt.inverse_size),
                coset_generator: neutral_fr(self.base_ntt.coset_generator),
                inverse_coset_generator: neutral_fr(self.base_ntt.inverse_coset_generator),
                barycentric_weight: neutral_fr(self.barycentric_weight),
                vanishing_inverses: self
                    .vanishing_inverses
                    .iter()
                    .copied()
                    .map(neutral_fr)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                delta: neutral_fr(self.delta),
            })
            .expect("representative neutral domain")
        }
    }

    pub(crate) struct OwnedV2Input {
        pub(crate) graph_fixed: OwnedMetalKeyContextInput,
        pub(crate) domain: OwnedDomain,
        pub(crate) prepared_key_schema_version: u32,
        pub(crate) prepared_key_fingerprint: [u8; 32],
        pub(crate) prepared_srs_schema_version: u32,
        pub(crate) prepared_srs_n: u32,
        pub(crate) prepared_srs_fingerprint: [u8; 32],
        pub(crate) coefficient_srs_digest: [u8; 32],
        pub(crate) lagrange_srs_digest: [u8; 32],
        pub(crate) permutation_columns: Vec<MetalResidentColumnRef>,
        pub(crate) trash_selector_fixed_indices: Vec<u32>,
        pub(crate) fixed_lagrange: Vec<Bls12_381Fr>,
        pub(crate) permutation_cosets: Vec<Bls12_381Fr>,
        pub(crate) permutation_lagrange: Vec<Bls12_381Fr>,
        pub(crate) coefficient_columns: Vec<Bls12_381Fr>,
        pub(crate) fixed_coefficient_count: u32,
        pub(crate) permutation_coefficient_count: u32,
        pub(crate) coefficient_bases: Vec<Bls12_381G1Affine>,
        pub(crate) lagrange_bases: Vec<Bls12_381G1Affine>,
    }

    impl OwnedV2Input {
        pub(crate) fn representative() -> Self {
            let one = fr_one();
            let mut graph_fixed = OwnedMetalKeyContextInput::representative();
            graph_fixed.sections.lookups = 1;
            graph_fixed.sections.lookup_inputs = 1;
            graph_fixed.sections.lookup_tables = 1;
            graph_fixed.sections.trash = 1;
            // V2's representative family exercises every protocol section in
            // the required custom -> lookup -> lookup-input -> lookup-table ->
            // trash packed order. Each graph intentionally returns one.
            for offset in 1..=4 {
                graph_fixed.descriptors.push(MetalGraphDescriptor {
                    constants_offset: offset,
                    constants_count: 1,
                    rotations_offset: offset,
                    rotations_count: 1,
                    operations_offset: offset,
                    operations_count: 1,
                    scratch_capacity: 8,
                    ..MetalGraphDescriptor::default()
                });
                graph_fixed.constants.push(one);
                graph_fixed.rotations.push(0);
                graph_fixed.operations.push(MetalGraphOperation {
                    opcode: super::super::GRAPH_OP_STORE,
                    target: 0,
                    lhs: MetalGraphValueRef {
                        kind: super::super::GRAPH_VALUE_CONSTANT,
                        a: 0,
                        b: 0,
                    },
                    ..MetalGraphOperation::default()
                });
            }
            let mut fixture = Self {
                graph_fixed,
                domain: OwnedDomain::representative(),
                prepared_key_schema_version: 1,
                prepared_key_fingerprint: [0; 32],
                prepared_srs_schema_version: 1,
                prepared_srs_n: 2,
                prepared_srs_fingerprint: [0; 32],
                coefficient_srs_digest: [0; 32],
                lagrange_srs_digest: [0; 32],
                permutation_columns: vec![
                    MetalResidentColumnRef {
                        kind: COLUMN_KIND_FIXED,
                        index: 0,
                    },
                    MetalResidentColumnRef {
                        kind: COLUMN_KIND_ADVICE,
                        index: 0,
                    },
                    MetalResidentColumnRef {
                        kind: COLUMN_KIND_INSTANCE,
                        index: 0,
                    },
                ],
                trash_selector_fixed_indices: vec![0],
                fixed_lagrange: vec![one; 2],
                permutation_cosets: vec![one; 12],
                permutation_lagrange: vec![one; 6],
                coefficient_columns: vec![one; 12],
                fixed_coefficient_count: 3,
                permutation_coefficient_count: 3,
                coefficient_bases: vec![g1_multiple(1), g1_multiple(2)],
                lagrange_bases: vec![g1_multiple(3), g1_multiple(4)],
            };
            fixture.refresh_neutral_fingerprints();
            fixture
        }

        pub(crate) fn input(&self) -> MetalKeyContextInputV2<'_> {
            MetalKeyContextInputV2 {
                graph_fixed: self.graph_fixed.input(),
                domain: self.domain.input(),
                prepared_key_schema_version: self.prepared_key_schema_version,
                prepared_key_fingerprint: self.prepared_key_fingerprint,
                prepared_srs_schema_version: self.prepared_srs_schema_version,
                prepared_srs_n: self.prepared_srs_n,
                prepared_srs_fingerprint: self.prepared_srs_fingerprint,
                coefficient_srs_digest: self.coefficient_srs_digest,
                lagrange_srs_digest: self.lagrange_srs_digest,
                permutation_columns: &self.permutation_columns,
                trash_selector_fixed_indices: &self.trash_selector_fixed_indices,
                fixed_lagrange: &self.fixed_lagrange,
                permutation_cosets: &self.permutation_cosets,
                permutation_lagrange: &self.permutation_lagrange,
                coefficient_columns: &self.coefficient_columns,
                fixed_coefficient_count: self.fixed_coefficient_count,
                permutation_coefficient_count: self.permutation_coefficient_count,
                coefficient_bases: &self.coefficient_bases,
                lagrange_bases: &self.lagrange_bases,
            }
        }

        pub(crate) fn refresh_neutral_fingerprints(&mut self) {
            let domain = self.domain.neutral_manifest();
            self.domain.neutral_fingerprint = domain.fingerprint();

            let graphs = representative_neutral_graphs();
            let graph_input = self.graph_fixed.input();
            assert_eq!(
                graphs.packed().fingerprint(),
                graph_input.graph_family_fingerprint,
                "Metal v1 graph bytes must remain neutral-byte compatible"
            );
            let permutation_columns = self
                .permutation_columns
                .iter()
                .map(|column| {
                    let kind = match column.kind {
                        COLUMN_KIND_FIXED => ResidentColumnKind::Fixed,
                        COLUMN_KIND_ADVICE => ResidentColumnKind::Advice,
                        COLUMN_KIND_INSTANCE => ResidentColumnKind::Instance,
                        kind => panic!("cannot refresh unknown column kind {kind}"),
                    };
                    ResidentColumnRef::new(kind, column.index)
                })
                .collect::<Vec<_>>();
            let prepared_key =
                PreparedResidentProvingKey::try_new(PreparedResidentProvingKeyInput {
                    domain,
                    graphs,
                    permutation_columns: permutation_columns.into_boxed_slice(),
                    trash_selector_fixed_indices: self
                        .trash_selector_fixed_indices
                        .clone()
                        .into_boxed_slice(),
                    fixed_cosets_compact: self
                        .graph_fixed
                        .fixed_cosets
                        .iter()
                        .copied()
                        .map(neutral_fr)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                    fixed_lagrange_compact: self
                        .fixed_lagrange
                        .iter()
                        .copied()
                        .map(neutral_fr)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                    permutation_cosets: self
                        .permutation_cosets
                        .iter()
                        .copied()
                        .map(neutral_fr)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                    permutation_lagrange: self
                        .permutation_lagrange
                        .iter()
                        .copied()
                        .map(neutral_fr)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                    coefficient_columns: self
                        .coefficient_columns
                        .iter()
                        .copied()
                        .map(neutral_fr)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                    extended_selectors: self
                        .graph_fixed
                        .selectors
                        .iter()
                        .copied()
                        .map(neutral_fr)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                })
                .expect("representative neutral proving key");
            assert_eq!(
                prepared_key.fixed_coefficient_count(),
                self.fixed_coefficient_count
            );
            assert_eq!(
                prepared_key.permutation_coefficient_count(),
                self.permutation_coefficient_count
            );
            self.prepared_key_schema_version = prepared_key.schema_version();
            self.prepared_key_fingerprint = prepared_key.fingerprint();

            let prepared_srs = PreparedResidentSrs::try_new(
                self.prepared_srs_n,
                self.coefficient_bases
                    .iter()
                    .copied()
                    .map(neutral_g1)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                self.lagrange_bases
                    .iter()
                    .copied()
                    .map(neutral_g1)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
            .expect("representative neutral SRS");
            self.prepared_srs_schema_version = prepared_srs.schema_version();
            self.prepared_srs_fingerprint = prepared_srs.fingerprint();
            self.coefficient_srs_digest = prepared_srs.coefficient_digest();
            self.lagrange_srs_digest = prepared_srs.lagrange_digest();
        }
    }

    #[test]
    fn v2_layout_and_cross_crate_neutral_fingerprints_are_exact() {
        assert_eq!(METAL_KEY_CONTEXT_SCHEMA_VERSION_V2, 2);
        assert_eq!(size_of::<MetalResidentColumnRef>(), 8);
        assert_eq!(align_of::<MetalResidentColumnRef>(), 4);

        let fixture = OwnedV2Input::representative();
        let input = fixture.input();
        assert_eq!(
            neutral_domain_fingerprint(input.domain),
            input.domain.neutral_fingerprint
        );
        assert_eq!(
            neutral_prepared_key_fingerprint(&input),
            input.prepared_key_fingerprint
        );
        assert_eq!(
            neutral_srs_prefix_fingerprint(
                NEUTRAL_SRS_COEFFICIENT_FINGERPRINT_DOMAIN,
                input.prepared_srs_schema_version,
                input.prepared_srs_n,
                input.coefficient_bases,
            ),
            input.coefficient_srs_digest
        );
        assert_eq!(
            neutral_srs_prefix_fingerprint(
                NEUTRAL_SRS_LAGRANGE_FINGERPRINT_DOMAIN,
                input.prepared_srs_schema_version,
                input.prepared_srs_n,
                input.lagrange_bases,
            ),
            input.lagrange_srs_digest
        );
        let validated = validate_metal_key_context_input_v2(input).expect("valid v2 transport");
        assert_ne!(validated.content_fingerprint, [0; 32]);
        assert_eq!(
            validated.immutable_bytes,
            super::super::immutable_bytes(&input.graph_fixed).unwrap()
                + size_of_val(input.domain.vanishing_inverses)
                + size_of_val(input.permutation_columns)
                + size_of_val(input.trash_selector_fixed_indices)
                + size_of_val(input.fixed_lagrange)
                + size_of_val(input.permutation_cosets)
                + size_of_val(input.permutation_lagrange)
                + size_of_val(input.coefficient_columns)
                + size_of_val(input.coefficient_bases)
                + size_of_val(input.lagrange_bases)
        );
    }

    #[test]
    fn every_large_content_family_changes_the_v2_identity() {
        let baseline_fixture = OwnedV2Input::representative();
        let baseline = validate_metal_key_context_input_v2(baseline_fixture.input())
            .unwrap()
            .content_fingerprint;
        let two = fr_from_u64(2);

        let mut changed = OwnedV2Input::representative();
        changed.graph_fixed.fixed_cosets[0] = two;
        changed.refresh_neutral_fingerprints();
        assert_content_changed(baseline, &changed);

        let mut changed = OwnedV2Input::representative();
        changed.fixed_lagrange[0] = two;
        changed.refresh_neutral_fingerprints();
        assert_content_changed(baseline, &changed);

        let mut changed = OwnedV2Input::representative();
        changed.permutation_cosets[0] = two;
        changed.refresh_neutral_fingerprints();
        assert_content_changed(baseline, &changed);

        let mut changed = OwnedV2Input::representative();
        changed.permutation_lagrange[0] = two;
        changed.refresh_neutral_fingerprints();
        assert_content_changed(baseline, &changed);

        let mut changed = OwnedV2Input::representative();
        changed.coefficient_columns[0] = two;
        changed.refresh_neutral_fingerprints();
        assert_content_changed(baseline, &changed);

        let mut changed = OwnedV2Input::representative();
        changed.permutation_columns.swap(0, 1);
        changed.refresh_neutral_fingerprints();
        assert_content_changed(baseline, &changed);

        let mut changed = OwnedV2Input::representative();
        changed.fixed_coefficient_count = 4;
        changed.coefficient_columns.extend([fr_one(); 2]);
        changed.refresh_neutral_fingerprints();
        assert_content_changed(baseline, &changed);

        let mut changed = OwnedV2Input::representative();
        changed.coefficient_bases[0] = g1_multiple(5);
        changed.refresh_neutral_fingerprints();
        assert_content_changed(baseline, &changed);

        let mut changed = OwnedV2Input::representative();
        changed.lagrange_bases[0] = g1_multiple(6);
        changed.refresh_neutral_fingerprints();
        assert_content_changed(baseline, &changed);
    }

    #[test]
    fn malformed_domain_shapes_columns_and_fingerprints_are_rejected() {
        let mut wrong_domain = OwnedV2Input::representative();
        wrong_domain.domain.rotation_scale = 3;
        assert_key_error(validate_metal_key_context_input_v2(wrong_domain.input()));

        let mut wrong_ntt = OwnedV2Input::representative();
        wrong_ntt.domain.base_ntt.root = fr_one();
        assert_key_error(validate_metal_key_context_input_v2(wrong_ntt.input()));

        let mut wrong_vanishing = OwnedV2Input::representative();
        wrong_vanishing.domain.vanishing_inverses[0] = fr_one();
        assert_key_error(validate_metal_key_context_input_v2(wrong_vanishing.input()));

        let mut bad_shape = OwnedV2Input::representative();
        bad_shape.fixed_lagrange.pop();
        assert_key_error(validate_metal_key_context_input_v2(bad_shape.input()));

        let mut bad_column = OwnedV2Input::representative();
        bad_column.permutation_columns[0].index = 1;
        assert_key_error(validate_metal_key_context_input_v2(bad_column.input()));

        let mut bad_kind = OwnedV2Input::representative();
        bad_kind.permutation_columns[0].kind = 9;
        assert_key_error(validate_metal_key_context_input_v2(bad_kind.input()));

        let mut unreduced = OwnedV2Input::representative();
        unreduced.coefficient_columns[0] = Bls12_381Fr {
            limbs: generated::FR_MODULUS,
        };
        assert_key_error(validate_metal_key_context_input_v2(unreduced.input()));

        let mut wrong_pk_fingerprint = OwnedV2Input::representative();
        wrong_pk_fingerprint.prepared_key_fingerprint[0] ^= 1;
        assert_key_error(validate_metal_key_context_input_v2(
            wrong_pk_fingerprint.input(),
        ));

        let mut wrong_srs_fingerprint = OwnedV2Input::representative();
        wrong_srs_fingerprint.prepared_srs_fingerprint[0] ^= 1;
        assert_key_error(validate_metal_key_context_input_v2(
            wrong_srs_fingerprint.input(),
        ));

        let mut mismatched_srs_n = OwnedV2Input::representative();
        mismatched_srs_n.prepared_srs_n = 4;
        assert_key_error(validate_metal_key_context_input_v2(
            mismatched_srs_n.input(),
        ));
    }

    #[test]
    fn g1_validation_rejects_unreduced_off_curve_and_non_subgroup_points() {
        let mut unreduced = OwnedV2Input::representative();
        unreduced.coefficient_bases[0].x = generated::FP_MODULUS;
        assert_key_error(validate_metal_key_context_input_v2(unreduced.input()));

        let mut off_curve = OwnedV2Input::representative();
        off_curve.coefficient_bases[0].y = [0; 6];
        off_curve.refresh_neutral_fingerprints();
        assert_key_error(validate_metal_key_context_input_v2(off_curve.input()));

        let mut non_subgroup = OwnedV2Input::representative();
        non_subgroup.coefficient_bases[0] = authoritative_non_subgroup_affine();
        non_subgroup.refresh_neutral_fingerprints();
        let prevalidated = prevalidate_metal_key_context_input_v2(non_subgroup.input()).unwrap();
        assert_key_error(validate_prevalidated_metal_key_context_srs_subgroups_v2(
            &prevalidated,
        ));
        assert_key_error(validate_metal_key_context_input_v2(non_subgroup.input()));
    }

    #[test]
    fn zero_zero_identity_is_rejected_for_both_kzg_srs_bases() {
        let mut coefficient_identity = OwnedV2Input::representative();
        coefficient_identity.coefficient_bases[0] = Bls12_381G1Affine::default();
        assert_key_error(validate_metal_key_context_input_v2(
            coefficient_identity.input(),
        ));

        let mut lagrange_identity = OwnedV2Input::representative();
        lagrange_identity.lagrange_bases[1] = Bls12_381G1Affine::default();
        assert_key_error(validate_metal_key_context_input_v2(
            lagrange_identity.input(),
        ));
    }

    #[test]
    fn lagrange_srs_only_change_is_bound_independently() {
        let baseline_fixture = OwnedV2Input::representative();
        let baseline_input = baseline_fixture.input();
        let baseline = validate_metal_key_context_input_v2(baseline_input)
            .unwrap()
            .content_fingerprint;

        let mut lagrange_only = OwnedV2Input::representative();
        lagrange_only.lagrange_bases[0] = g1_multiple(7);
        lagrange_only.refresh_neutral_fingerprints();
        assert_eq!(
            lagrange_only.coefficient_srs_digest,
            baseline_input.coefficient_srs_digest
        );
        assert_ne!(
            lagrange_only.lagrange_srs_digest,
            baseline_input.lagrange_srs_digest
        );
        let changed = validate_metal_key_context_input_v2(lagrange_only.input())
            .unwrap()
            .content_fingerprint;
        assert_ne!(changed, baseline);
    }

    fn assert_content_changed(baseline: [u8; 32], fixture: &OwnedV2Input) {
        let changed = validate_metal_key_context_input_v2(fixture.input())
            .expect("changed content remains valid")
            .content_fingerprint;
        assert_ne!(changed, baseline);
    }

    fn assert_key_error<T>(result: Result<T, MetalError>) {
        assert!(matches!(result, Err(MetalError::KeyContext(_))));
    }

    fn exact_vanishing_inverses(
        n: u32,
        n_ext: u32,
        extended_ntt: NttDomainSpec,
    ) -> Vec<Bls12_381Fr> {
        let mut current = fr_pow(extended_ntt.coset_generator, u64::from(n));
        let step = fr_pow(extended_ntt.root, u64::from(n));
        let mut values = Vec::with_capacity((n_ext / n) as usize);
        for _ in 0..n_ext / n {
            values.push(fr_inverse(fr_sub(current, fr_one())));
            current = fr_mul(current, step);
        }
        values
    }

    fn representative_neutral_graphs() -> CompiledProtocolGraphFamily {
        let graph = || {
            CompiledGraph::try_new(
                vec![neutral_fr(fr_one())].into_boxed_slice(),
                vec![0].into_boxed_slice(),
                vec![GraphOp::new(
                    GraphOpcode::Store,
                    0,
                    0,
                    0,
                    GraphValueRef::new(GraphValueKind::Constant, 0, 0),
                    GraphValueRef::default(),
                )]
                .into_boxed_slice(),
                Vec::new().into_boxed_slice(),
                GraphSourceCounts::default(),
            )
            .expect("representative neutral graph")
        };
        let fixed = [2];
        let advice = [7];
        let instance = [11];
        CompiledProtocolGraphFamily::try_new_with_required_columns(
            Some(graph()),
            vec![graph()].into_boxed_slice(),
            vec![graph()].into_boxed_slice(),
            vec![graph()].into_boxed_slice(),
            vec![graph()].into_boxed_slice(),
            GraphColumnRequirements::new(&fixed, &advice, &instance),
        )
        .expect("representative neutral graph family")
    }

    fn neutral_fr(value: Bls12_381Fr) -> RawFr {
        RawFr { limbs: value.limbs }
    }

    fn neutral_g1(value: Bls12_381G1Affine) -> RawG1Affine {
        RawG1Affine {
            x: value.x,
            y: value.y,
        }
    }

    fn fr_from_u64(value: u64) -> Bls12_381Fr {
        let canonical = [value, 0, 0, 0];
        let mut output = blst::blst_fr::default();
        // SAFETY: BLST receives four initialized canonical little-endian limbs.
        unsafe { blst::blst_fr_from_uint64(&mut output, canonical.as_ptr()) };
        Bls12_381Fr { limbs: output.l }
    }

    fn g1_multiple(scalar: u64) -> Bls12_381G1Affine {
        let scalar = scalar.to_le_bytes();
        let mut projective = blst::blst_p1::default();
        // SAFETY: BLST's process-lifetime generator and initialized scalar bytes
        // remain valid, while `projective` is a distinct mutable output.
        unsafe {
            blst::blst_p1_mult(
                &mut projective,
                blst::blst_p1_generator(),
                scalar.as_ptr(),
                u64::BITS as usize,
            );
        }
        let mut affine = blst::blst_p1_affine::default();
        // SAFETY: initialized projective input and affine output are disjoint.
        unsafe { blst::blst_p1_to_affine(&mut affine, &projective) };
        Bls12_381G1Affine {
            x: affine.x.l,
            y: affine.y.l,
        }
    }

    fn fp_from_u64(value: u64) -> blst::blst_fp {
        let canonical = [value, 0, 0, 0, 0, 0];
        let mut output = blst::blst_fp::default();
        // SAFETY: BLST receives six initialized canonical little-endian limbs.
        unsafe { blst::blst_fp_from_uint64(&mut output, canonical.as_ptr()) };
        output
    }

    fn authoritative_non_subgroup_affine() -> Bls12_381G1Affine {
        let four = fp_from_u64(4);
        for x_value in 1..=u64::MAX {
            let x = fp_from_u64(x_value);
            let mut x_squared = blst::blst_fp::default();
            let mut x_cubed = blst::blst_fp::default();
            let mut rhs = blst::blst_fp::default();
            let mut y = blst::blst_fp::default();
            // SAFETY: every input is initialized and mutable outputs are valid.
            let has_square_root = unsafe {
                blst::blst_fp_sqr(&mut x_squared, &x);
                blst::blst_fp_mul(&mut x_cubed, &x_squared, &x);
                blst::blst_fp_add(&mut rhs, &x_cubed, &four);
                blst::blst_fp_sqrt(&mut y, &rhs)
            };
            if has_square_root {
                let point = blst::blst_p1_affine { x, y };
                // SAFETY: `point` contains initialized, reduced Fp coordinates.
                let (on_curve, in_g1) = unsafe {
                    (
                        blst::blst_p1_affine_on_curve(&point),
                        blst::blst_p1_affine_in_g1(&point),
                    )
                };
                if on_curve && !in_g1 {
                    return Bls12_381G1Affine {
                        x: point.x.l,
                        y: point.y.l,
                    };
                }
            }
        }
        unreachable!("BLS12-381 has on-curve points outside the prime subgroup")
    }

    #[test]
    fn v1_graph_fingerprint_bytes_remain_stable() {
        let fixture = OwnedMetalKeyContextInput::representative();
        let input = fixture.input();
        assert_eq!(
            neutral_graph_family_fingerprint(&input),
            input.graph_family_fingerprint
        );
        // This exact v1 byte serialization remains pinned while v2 evolves additively.
        assert_eq!(
            input.graph_family_fingerprint,
            [
                0xdd, 0x21, 0x25, 0x18, 0xdd, 0x5c, 0x22, 0x32, 0xa4, 0x74, 0x18, 0x40, 0x46, 0x23,
                0x21, 0xc0, 0xd4, 0x40, 0x84, 0xe8, 0x3c, 0x8e, 0xe1, 0xff, 0x16, 0x62, 0x34, 0x52,
                0x4c, 0xa0, 0xb3, 0x88,
            ]
        );
    }
}
