//! Backend-neutral immutable proving-key and SRS preparation.
//!
//! These types own only public, reusable proof material. They deliberately do
//! not contain a device, queue, raw backend handle, proof-secret allocation, or
//! execution policy. Constructors validate every neutral shape and compute
//! canonical fingerprints before a concrete backend may import the data.

use sha2::{Digest, Sha256};

use crate::{
    RawFr,
    expression::{CompiledProtocolGraphFamily, GRAPH_FAMILY_FINGERPRINT_VERSION},
};

const DOMAIN_MANIFEST_SCHEMA_VERSION: u32 = 1;
const PREPARED_KEY_SCHEMA_VERSION: u32 = 1;
const PREPARED_SRS_SCHEMA_VERSION: u32 = 1;

const DOMAIN_MANIFEST_FINGERPRINT_DOMAIN: &[u8] =
    b"midnight-gpu-prover/resident-domain-manifest/v1\0";
const PREPARED_KEY_FINGERPRINT_DOMAIN: &[u8] =
    b"midnight-gpu-prover/prepared-resident-proving-key/v1\0";
const SRS_COEFFICIENT_FINGERPRINT_DOMAIN: &[u8] =
    b"midnight-gpu-prover/resident-srs-coefficient/v1\0";
const SRS_LAGRANGE_FINGERPRINT_DOMAIN: &[u8] = b"midnight-gpu-prover/resident-srs-lagrange/v1\0";
const PREPARED_SRS_FINGERPRINT_DOMAIN: &[u8] = b"midnight-gpu-prover/prepared-resident-srs/v1\0";

const BLS12_381_FR_MODULUS: [u64; 4] = [
    0xffff_ffff_0000_0001,
    0x53bd_a402_fffe_5bfe,
    0x3339_d808_09a1_d805,
    0x73ed_a753_299d_7d48,
];

const BLS12_381_FP_MODULUS: [u64; 6] = [
    0xb9fe_ffff_ffff_aaab,
    0x1eab_fffe_b153_ffff,
    0x6730_d2a0_f6b0_f624,
    0x6477_4b84_f385_12bf,
    0x4b1b_a7b6_434b_acd7,
    0x1a01_11ea_397f_e69a,
];

/// Failure while constructing backend-neutral immutable proof material.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedKeyError {
    /// Domain metadata is internally inconsistent or unsupported by the ABI.
    InvalidDomain(&'static str),
    /// An array length cannot be represented by the neutral fixed-width ABI.
    LengthExceedsAbi {
        /// Logical array name.
        array: &'static str,
        /// Host element count that does not fit the ABI.
        len: usize,
    },
    /// An array does not have the exact extent implied by its domain and count.
    LengthMismatch {
        /// Logical array name.
        array: &'static str,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// Multiplying a column count by its domain stride overflowed `usize`.
    LengthOverflow {
        /// Logical array whose expected extent overflowed.
        array: &'static str,
    },
    /// A permutation descriptor refers outside its compact column map.
    ColumnIndexOutOfBounds {
        /// Descriptor position in protocol order.
        position: usize,
        /// Kind of compact polynomial column.
        kind: ResidentColumnKind,
        /// Supplied dense column index.
        index: u32,
        /// Exclusive dense-column bound.
        bound: u32,
    },
    /// A compact fixed-column map names no corresponding logical coefficient column.
    LogicalFixedColumnOutOfBounds {
        /// Dense fixed-column position in the compact map.
        compact_index: usize,
        /// Original logical fixed-column index.
        logical_index: u32,
        /// Number of logical fixed coefficient columns.
        fixed_column_count: u32,
    },
    /// A trash selector refers outside the compact fixed-column map.
    TrashSelectorOutOfBounds {
        /// Trash argument position in protocol order.
        position: usize,
        /// Supplied dense fixed-column index.
        index: u32,
        /// Exclusive dense fixed-column bound.
        bound: u32,
    },
    /// A scalar-field value is not reduced modulo the BLS12-381 scalar modulus.
    UnreducedScalar {
        /// Logical array containing the value.
        array: &'static str,
        /// Element position within the array.
        index: usize,
    },
    /// An affine coordinate is not reduced modulo the BLS12-381 base-field modulus.
    UnreducedBaseCoordinate {
        /// Logical SRS array containing the point.
        array: &'static str,
        /// Point position within the array.
        point: usize,
        /// Coordinate name, either `x` or `y`.
        coordinate: &'static str,
    },
    /// An SRS prefix contains the affine identity, which is not a usable MSM base.
    IdentitySrsPoint {
        /// Logical SRS array containing the point.
        array: &'static str,
        /// Point position within the array.
        index: usize,
    },
}

impl std::fmt::Display for PreparedKeyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "invalid backend-neutral resident key preparation: {self:?}"
        )
    }
}

impl std::error::Error for PreparedKeyError {}

/// Polynomial column kind stored in a resident permutation descriptor.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResidentColumnKind {
    /// Fixed proving-key column.
    Fixed = 0,
    /// Advice proof column.
    Advice = 1,
    /// Public-instance proof column.
    Instance = 2,
}

/// Dense resident column reference retained in permutation protocol order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ResidentColumnRef {
    kind: ResidentColumnKind,
    index: u32,
}

impl ResidentColumnRef {
    /// Creates a reference to a dense column in the key-wide compact map.
    pub const fn new(kind: ResidentColumnKind, index: u32) -> Self {
        Self { kind, index }
    }

    /// Polynomial column kind.
    pub const fn kind(self) -> ResidentColumnKind {
        self.kind
    }

    /// Dense index within the compact map for [`Self::kind`].
    pub const fn index(self) -> u32 {
        self.index
    }
}

/// Raw BLS12-381 G1 affine point used by resident backend imports.
///
/// Coordinates contain little-endian reduced Montgomery limbs. The affine
/// identity is encoded with both coordinates zero. A proving-system producer
/// must normalize and subgroup-check points before constructing an SRS; a
/// concrete backend must repeat curve validation before device upload.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct RawG1Affine {
    /// Affine x-coordinate in reduced Montgomery form.
    pub x: [u64; 6],
    /// Affine y-coordinate in reduced Montgomery form.
    pub y: [u64; 6],
}

/// Non-derived inputs needed to construct an exact polynomial-domain manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentDomainManifestInput {
    /// Base-two logarithm of the base polynomial domain.
    pub k: u32,
    /// Base-two logarithm of the extended polynomial domain.
    pub extended_k: u32,
    /// Maximum constraint-system degree used to size the quotient.
    pub constraint_system_degree: u32,
    /// Number of proof blinding rows.
    pub blinding_factors: u32,
    /// Primitive root of the base evaluation domain.
    pub omega: RawFr,
    /// Multiplicative inverse of `omega`.
    pub omega_inv: RawFr,
    /// Primitive root of the extended evaluation domain.
    pub extended_omega: RawFr,
    /// Multiplicative inverse of `extended_omega`.
    pub extended_omega_inv: RawFr,
    /// Multiplicative inverse of the base-domain length.
    pub n_inv: RawFr,
    /// Multiplicative inverse of the extended-domain length.
    pub n_ext_inv: RawFr,
    /// Nontrivial order-three coset generator.
    pub coset_generator: RawFr,
    /// Multiplicative inverse of `coset_generator`.
    pub inverse_coset_generator: RawFr,
    /// Barycentric weight at the first base-domain point.
    pub barycentric_weight: RawFr,
    /// Repeating inverses of the vanishing polynomial over the extended coset.
    pub vanishing_inverses: Box<[RawFr]>,
    /// BLS12-381 scalar-field permutation delta constant.
    pub delta: RawFr,
}

/// Exact immutable parameters for a base and extended polynomial domain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentDomainManifest {
    n: u32,
    k: u32,
    n_ext: u32,
    extended_k: u32,
    constraint_system_degree: u32,
    quotient_piece_count: u32,
    blinding_factors: u32,
    usable_rows: u32,
    rotation_scale: i32,
    last_rotation: i32,
    permutation_chunk_len: u32,
    omega: RawFr,
    omega_inv: RawFr,
    extended_omega: RawFr,
    extended_omega_inv: RawFr,
    n_inv: RawFr,
    n_ext_inv: RawFr,
    coset_generator: RawFr,
    inverse_coset_generator: RawFr,
    barycentric_weight: RawFr,
    vanishing_inverses: Box<[RawFr]>,
    delta: RawFr,
    fingerprint: [u8; 32],
}

impl ResidentDomainManifest {
    /// Validates inputs, derives all shape metadata, and fingerprints the domain.
    pub fn try_new(input: ResidentDomainManifestInput) -> Result<Self, PreparedKeyError> {
        let n = 1u32
            .checked_shl(input.k)
            .ok_or(PreparedKeyError::InvalidDomain(
                "base-domain length does not fit u32",
            ))?;
        let n_ext = 1u32
            .checked_shl(input.extended_k)
            .ok_or(PreparedKeyError::InvalidDomain(
                "extended-domain length does not fit u32",
            ))?;
        if input.extended_k < input.k {
            return Err(PreparedKeyError::InvalidDomain(
                "extended domain is smaller than the base domain",
            ));
        }
        if input.constraint_system_degree < 3 {
            return Err(PreparedKeyError::InvalidDomain(
                "constraint-system degree is below three",
            ));
        }

        let quotient_piece_count = input.constraint_system_degree - 1;
        let permutation_chunk_len = input.constraint_system_degree - 2;
        let inactive_rows =
            input
                .blinding_factors
                .checked_add(1)
                .ok_or(PreparedKeyError::InvalidDomain(
                    "blinding row count overflows u32",
                ))?;
        let usable_rows = n
            .checked_sub(inactive_rows)
            .filter(|rows| *rows != 0)
            .ok_or(PreparedKeyError::InvalidDomain(
                "blinding rows leave no usable base-domain row",
            ))?;
        let extension_log = input.extended_k - input.k;
        let extension = 1u32
            .checked_shl(extension_log)
            .ok_or(PreparedKeyError::InvalidDomain(
                "domain extension factor does not fit u32",
            ))?;
        let required_extension = quotient_piece_count.checked_next_power_of_two().ok_or(
            PreparedKeyError::InvalidDomain("quotient extension factor does not fit u32"),
        )?;
        if extension != required_extension {
            return Err(PreparedKeyError::InvalidDomain(
                "extended domain is not the minimal quotient domain",
            ));
        }
        let rotation_scale = i32::try_from(extension).map_err(|_| {
            PreparedKeyError::InvalidDomain("domain rotation scale does not fit i32")
        })?;
        let last_rotation = -i32::try_from(inactive_rows).map_err(|_| {
            PreparedKeyError::InvalidDomain("last active-row rotation does not fit i32")
        })?;

        checked_abi_len("vanishing inverses", input.vanishing_inverses.len())?;
        exact_len(
            "vanishing inverses",
            input.vanishing_inverses.len(),
            extension as usize,
        )?;

        let scalar_fields = [
            ("omega", input.omega),
            ("omega inverse", input.omega_inv),
            ("extended omega", input.extended_omega),
            ("extended omega inverse", input.extended_omega_inv),
            ("base inverse size", input.n_inv),
            ("extended inverse size", input.n_ext_inv),
            ("coset generator", input.coset_generator),
            ("inverse coset generator", input.inverse_coset_generator),
            ("barycentric weight", input.barycentric_weight),
            ("permutation delta", input.delta),
        ];
        for (array, value) in scalar_fields {
            validate_fr(array, std::slice::from_ref(&value))?;
            if value.limbs == [0; 4] {
                return Err(PreparedKeyError::InvalidDomain(
                    "domain field parameters must be nonzero",
                ));
            }
        }
        validate_fr("vanishing inverses", &input.vanishing_inverses)?;
        if input
            .vanishing_inverses
            .iter()
            .any(|value| value.limbs == [0; 4])
        {
            return Err(PreparedKeyError::InvalidDomain(
                "vanishing inverses must be nonzero",
            ));
        }
        if input.barycentric_weight != input.n_inv {
            return Err(PreparedKeyError::InvalidDomain(
                "first-point barycentric weight differs from the base inverse size",
            ));
        }

        let mut manifest = Self {
            n,
            k: input.k,
            n_ext,
            extended_k: input.extended_k,
            constraint_system_degree: input.constraint_system_degree,
            quotient_piece_count,
            blinding_factors: input.blinding_factors,
            usable_rows,
            rotation_scale,
            last_rotation,
            permutation_chunk_len,
            omega: input.omega,
            omega_inv: input.omega_inv,
            extended_omega: input.extended_omega,
            extended_omega_inv: input.extended_omega_inv,
            n_inv: input.n_inv,
            n_ext_inv: input.n_ext_inv,
            coset_generator: input.coset_generator,
            inverse_coset_generator: input.inverse_coset_generator,
            barycentric_weight: input.barycentric_weight,
            vanishing_inverses: input.vanishing_inverses,
            delta: input.delta,
            fingerprint: [0; 32],
        };
        manifest.fingerprint = domain_fingerprint(&manifest);
        Ok(manifest)
    }

    /// Base-domain element count.
    pub const fn n(&self) -> u32 {
        self.n
    }

    /// Domain-manifest schema version included in the content fingerprint.
    pub const fn schema_version(&self) -> u32 {
        DOMAIN_MANIFEST_SCHEMA_VERSION
    }

    /// Base-two logarithm of the base-domain element count.
    pub const fn k(&self) -> u32 {
        self.k
    }

    /// Extended-domain element count.
    pub const fn n_ext(&self) -> u32 {
        self.n_ext
    }

    /// Base-two logarithm of the extended-domain element count.
    pub const fn extended_k(&self) -> u32 {
        self.extended_k
    }

    /// Maximum constraint-system degree.
    pub const fn constraint_system_degree(&self) -> u32 {
        self.constraint_system_degree
    }

    /// Number of degree-`n - 1` quotient pieces.
    pub const fn quotient_piece_count(&self) -> u32 {
        self.quotient_piece_count
    }

    /// Number of proof blinding rows.
    pub const fn blinding_factors(&self) -> u32 {
        self.blinding_factors
    }

    /// Number of base-domain rows usable by the circuit.
    pub const fn usable_rows(&self) -> u32 {
        self.usable_rows
    }

    /// Base-row rotation multiplier in the extended domain.
    pub const fn rotation_scale(&self) -> i32 {
        self.rotation_scale
    }

    /// Rotation of the first inactive base-domain row.
    pub const fn last_rotation(&self) -> i32 {
        self.last_rotation
    }

    /// Number of permutation columns combined in one product chunk.
    pub const fn permutation_chunk_len(&self) -> u32 {
        self.permutation_chunk_len
    }

    /// Primitive root of the base evaluation domain.
    pub const fn omega(&self) -> RawFr {
        self.omega
    }

    /// Multiplicative inverse of the base-domain root.
    pub const fn omega_inv(&self) -> RawFr {
        self.omega_inv
    }

    /// Primitive root of the extended evaluation domain.
    pub const fn extended_omega(&self) -> RawFr {
        self.extended_omega
    }

    /// Multiplicative inverse of the extended-domain root.
    pub const fn extended_omega_inv(&self) -> RawFr {
        self.extended_omega_inv
    }

    /// Multiplicative inverse of the base-domain length.
    pub const fn n_inv(&self) -> RawFr {
        self.n_inv
    }

    /// Multiplicative inverse of the extended-domain length.
    pub const fn n_ext_inv(&self) -> RawFr {
        self.n_ext_inv
    }

    /// Nontrivial order-three coset generator.
    pub const fn coset_generator(&self) -> RawFr {
        self.coset_generator
    }

    /// Multiplicative inverse of the coset generator.
    pub const fn inverse_coset_generator(&self) -> RawFr {
        self.inverse_coset_generator
    }

    /// Barycentric weight at the first base-domain point.
    pub const fn barycentric_weight(&self) -> RawFr {
        self.barycentric_weight
    }

    /// Repeating inverses of the vanishing polynomial over the extended coset.
    pub fn vanishing_inverses(&self) -> &[RawFr] {
        &self.vanishing_inverses
    }

    /// BLS12-381 scalar-field permutation delta constant.
    pub const fn delta(&self) -> RawFr {
        self.delta
    }

    /// SHA-256 of the canonical versioned domain serialization.
    pub const fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }
}

/// Owned inputs for one backend-neutral resident proving-key preparation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedResidentProvingKeyInput {
    /// Exact base and extended polynomial-domain manifest.
    pub domain: ResidentDomainManifest,
    /// Protocol-classified and canonically compacted expression graphs.
    pub graphs: CompiledProtocolGraphFamily,
    /// Permutation columns in original protocol order with dense indices.
    pub permutation_columns: Box<[ResidentColumnRef]>,
    /// Trash selector indices in trash-argument order and the compact fixed map.
    pub trash_selector_fixed_indices: Box<[u32]>,
    /// Compact fixed columns over the extended coset, column-major.
    pub fixed_cosets_compact: Box<[RawFr]>,
    /// Compact fixed columns over the base Lagrange domain, column-major.
    pub fixed_lagrange_compact: Box<[RawFr]>,
    /// Permutation-key columns over the extended coset, in permutation order.
    pub permutation_cosets: Box<[RawFr]>,
    /// Permutation-key columns over the base Lagrange domain, in permutation order.
    pub permutation_lagrange: Box<[RawFr]>,
    /// All logical fixed coefficient columns followed by all permutation coefficients.
    pub coefficient_columns: Box<[RawFr]>,
    /// Packed extended-domain selectors `l0 | l_last | l_active`.
    pub extended_selectors: Box<[RawFr]>,
}

/// Complete backend-neutral immutable proving-key preparation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedResidentProvingKey {
    domain: ResidentDomainManifest,
    graphs: CompiledProtocolGraphFamily,
    permutation_columns: Box<[ResidentColumnRef]>,
    trash_selector_fixed_indices: Box<[u32]>,
    fixed_cosets_compact: Box<[RawFr]>,
    fixed_lagrange_compact: Box<[RawFr]>,
    permutation_cosets: Box<[RawFr]>,
    permutation_lagrange: Box<[RawFr]>,
    coefficient_columns: Box<[RawFr]>,
    fixed_coefficient_count: u32,
    permutation_coefficient_count: u32,
    extended_selectors: Box<[RawFr]>,
    fingerprint: [u8; 32],
}

impl PreparedResidentProvingKey {
    /// Validates every array and computes a canonical content fingerprint.
    pub fn try_new(input: PreparedResidentProvingKeyInput) -> Result<Self, PreparedKeyError> {
        let packed = input.graphs.packed();
        let fixed_compact_count = packed.fixed_compact_to_logical().len();
        let advice_compact_count = packed.advice_compact_to_logical().len();
        let instance_compact_count = packed.instance_compact_to_logical().len();
        let permutation_count = input.permutation_columns.len();
        checked_abi_len("permutation columns", permutation_count)?;
        checked_abi_len(
            "trash selector fixed indices",
            input.trash_selector_fixed_indices.len(),
        )?;

        let n = input.domain.n as usize;
        let n_ext = input.domain.n_ext as usize;
        exact_column_extent(
            "compact fixed cosets",
            input.fixed_cosets_compact.len(),
            fixed_compact_count,
            n_ext,
        )?;
        exact_column_extent(
            "compact fixed Lagrange columns",
            input.fixed_lagrange_compact.len(),
            fixed_compact_count,
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
        exact_column_extent(
            "extended selectors",
            input.extended_selectors.len(),
            3,
            n_ext,
        )?;

        checked_abi_len("coefficient columns", input.coefficient_columns.len())?;
        if !input.coefficient_columns.len().is_multiple_of(n) {
            return Err(PreparedKeyError::LengthMismatch {
                array: "coefficient columns",
                expected: input.coefficient_columns.len().div_ceil(n) * n,
                actual: input.coefficient_columns.len(),
            });
        }
        let total_coefficient_count = input.coefficient_columns.len() / n;
        let fixed_coefficient_count = total_coefficient_count
            .checked_sub(permutation_count)
            .ok_or(PreparedKeyError::LengthMismatch {
                array: "coefficient columns",
                expected: permutation_count.checked_mul(n).ok_or(
                    PreparedKeyError::LengthOverflow {
                        array: "coefficient columns",
                    },
                )?,
                actual: input.coefficient_columns.len(),
            })?;
        let fixed_coefficient_count = u32::try_from(fixed_coefficient_count).map_err(|_| {
            PreparedKeyError::LengthExceedsAbi {
                array: "fixed coefficient column count",
                len: fixed_coefficient_count,
            }
        })?;
        let permutation_coefficient_count =
            u32::try_from(permutation_count).map_err(|_| PreparedKeyError::LengthExceedsAbi {
                array: "permutation coefficient column count",
                len: permutation_count,
            })?;

        for (compact_index, logical_index) in packed
            .fixed_compact_to_logical()
            .iter()
            .copied()
            .enumerate()
        {
            if logical_index >= fixed_coefficient_count {
                return Err(PreparedKeyError::LogicalFixedColumnOutOfBounds {
                    compact_index,
                    logical_index,
                    fixed_column_count: fixed_coefficient_count,
                });
            }
        }

        let bounds = [
            (ResidentColumnKind::Fixed, fixed_compact_count),
            (ResidentColumnKind::Advice, advice_compact_count),
            (ResidentColumnKind::Instance, instance_compact_count),
        ];
        for (position, column) in input.permutation_columns.iter().copied().enumerate() {
            let bound = bounds
                .iter()
                .find_map(|(kind, bound)| (*kind == column.kind).then_some(*bound))
                .expect("every ResidentColumnKind has a bound");
            if column.index as usize >= bound {
                return Err(PreparedKeyError::ColumnIndexOutOfBounds {
                    position,
                    kind: column.kind,
                    index: column.index,
                    bound: u32::try_from(bound).expect("graph-family map length fits u32"),
                });
            }
        }

        let fixed_bound =
            u32::try_from(fixed_compact_count).expect("graph-family fixed map length fits u32");
        for (position, index) in input
            .trash_selector_fixed_indices
            .iter()
            .copied()
            .enumerate()
        {
            if index >= fixed_bound {
                return Err(PreparedKeyError::TrashSelectorOutOfBounds {
                    position,
                    index,
                    bound: fixed_bound,
                });
            }
        }
        exact_len(
            "trash selector fixed indices",
            input.trash_selector_fixed_indices.len(),
            input.graphs.sections().trash() as usize,
        )?;

        validate_fr("graph constants", packed.constants())?;
        for (array, values) in [
            ("compact fixed cosets", input.fixed_cosets_compact.as_ref()),
            (
                "compact fixed Lagrange columns",
                input.fixed_lagrange_compact.as_ref(),
            ),
            ("permutation cosets", input.permutation_cosets.as_ref()),
            (
                "permutation Lagrange columns",
                input.permutation_lagrange.as_ref(),
            ),
            ("coefficient columns", input.coefficient_columns.as_ref()),
            ("extended selectors", input.extended_selectors.as_ref()),
        ] {
            validate_fr(array, values)?;
        }

        let mut prepared = Self {
            domain: input.domain,
            graphs: input.graphs,
            permutation_columns: input.permutation_columns,
            trash_selector_fixed_indices: input.trash_selector_fixed_indices,
            fixed_cosets_compact: input.fixed_cosets_compact,
            fixed_lagrange_compact: input.fixed_lagrange_compact,
            permutation_cosets: input.permutation_cosets,
            permutation_lagrange: input.permutation_lagrange,
            coefficient_columns: input.coefficient_columns,
            fixed_coefficient_count,
            permutation_coefficient_count,
            extended_selectors: input.extended_selectors,
            fingerprint: [0; 32],
        };
        prepared.fingerprint = prepared_key_fingerprint(&prepared);
        Ok(prepared)
    }

    /// Prepared-key schema version included in the content fingerprint.
    pub const fn schema_version(&self) -> u32 {
        PREPARED_KEY_SCHEMA_VERSION
    }

    /// Exact polynomial-domain manifest.
    pub const fn domain(&self) -> &ResidentDomainManifest {
        &self.domain
    }

    /// Protocol-classified compact expression graph family.
    pub const fn graphs(&self) -> &CompiledProtocolGraphFamily {
        &self.graphs
    }

    /// Permutation descriptors in original protocol order.
    pub fn permutation_columns(&self) -> &[ResidentColumnRef] {
        &self.permutation_columns
    }

    /// Trash selectors in trash-argument order and the compact fixed map.
    pub fn trash_selector_fixed_indices(&self) -> &[u32] {
        &self.trash_selector_fixed_indices
    }

    /// Compact fixed extended-coset columns, column-major with stride `n_ext`.
    pub fn fixed_cosets_compact(&self) -> &[RawFr] {
        &self.fixed_cosets_compact
    }

    /// Compact fixed base-Lagrange columns, column-major with stride `n`.
    pub fn fixed_lagrange_compact(&self) -> &[RawFr] {
        &self.fixed_lagrange_compact
    }

    /// Permutation extended-coset columns in original permutation order.
    pub fn permutation_cosets(&self) -> &[RawFr] {
        &self.permutation_cosets
    }

    /// Permutation base-Lagrange columns in original permutation order.
    pub fn permutation_lagrange(&self) -> &[RawFr] {
        &self.permutation_lagrange
    }

    /// All logical fixed coefficients followed by all permutation coefficients.
    pub fn coefficient_columns(&self) -> &[RawFr] {
        &self.coefficient_columns
    }

    /// Number of logical fixed columns at the start of `coefficient_columns`.
    pub const fn fixed_coefficient_count(&self) -> u32 {
        self.fixed_coefficient_count
    }

    /// Number of permutation columns at the end of `coefficient_columns`.
    pub const fn permutation_coefficient_count(&self) -> u32 {
        self.permutation_coefficient_count
    }

    /// Packed extended-domain `l0 | l_last | l_active` selectors.
    pub fn extended_selectors(&self) -> &[RawFr] {
        &self.extended_selectors
    }

    /// SHA-256 of every versioned logical prepared-key field.
    pub const fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }
}

/// Validated coefficient and Lagrange SRS prefixes for one base-domain length.
///
/// This neutral layer verifies exact lengths, rejects identity bases, and
/// verifies reduced coordinate limbs. It intentionally has no curve
/// dependency; the producer and concrete backend remain responsible for
/// independent on-curve and subgroup validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedResidentSrs {
    n: u32,
    coefficient_bases: Box<[RawG1Affine]>,
    lagrange_bases: Box<[RawG1Affine]>,
    coefficient_digest: [u8; 32],
    lagrange_digest: [u8; 32],
    fingerprint: [u8; 32],
}

impl PreparedResidentSrs {
    /// Validates two exact SRS prefixes and computes their canonical digests.
    pub fn try_new(
        n: u32,
        coefficient_bases: Box<[RawG1Affine]>,
        lagrange_bases: Box<[RawG1Affine]>,
    ) -> Result<Self, PreparedKeyError> {
        if !n.is_power_of_two() {
            return Err(PreparedKeyError::InvalidDomain(
                "SRS prefix length must be a nonzero power of two",
            ));
        }
        let n = n as usize;
        checked_abi_len("coefficient SRS", coefficient_bases.len())?;
        checked_abi_len("Lagrange SRS", lagrange_bases.len())?;
        exact_len("coefficient SRS", coefficient_bases.len(), n)?;
        exact_len("Lagrange SRS", lagrange_bases.len(), n)?;
        validate_g1("coefficient SRS", &coefficient_bases)?;
        validate_g1("Lagrange SRS", &lagrange_bases)?;

        let coefficient_digest = srs_prefix_fingerprint(
            SRS_COEFFICIENT_FINGERPRINT_DOMAIN,
            n as u32,
            &coefficient_bases,
        );
        let lagrange_digest =
            srs_prefix_fingerprint(SRS_LAGRANGE_FINGERPRINT_DOMAIN, n as u32, &lagrange_bases);
        let mut hasher = Sha256::new();
        hasher.update(PREPARED_SRS_FINGERPRINT_DOMAIN);
        hash_u32(&mut hasher, PREPARED_SRS_SCHEMA_VERSION);
        hash_u32(&mut hasher, n as u32);
        hasher.update(coefficient_digest);
        hasher.update(lagrange_digest);

        Ok(Self {
            n: n as u32,
            coefficient_bases,
            lagrange_bases,
            coefficient_digest,
            lagrange_digest,
            fingerprint: hasher.finalize().into(),
        })
    }

    /// Prepared-SRS schema version included in every digest.
    pub const fn schema_version(&self) -> u32 {
        PREPARED_SRS_SCHEMA_VERSION
    }

    /// Base-domain length covered by both prefixes.
    pub const fn n(&self) -> u32 {
        self.n
    }

    /// Normalized coefficient-basis G1 prefix `g[..n]`.
    pub fn coefficient_bases(&self) -> &[RawG1Affine] {
        &self.coefficient_bases
    }

    /// Normalized Lagrange-basis G1 prefix `g_lagrange[..n]`.
    pub fn lagrange_bases(&self) -> &[RawG1Affine] {
        &self.lagrange_bases
    }

    /// Domain-separated digest of the coefficient-basis prefix.
    pub const fn coefficient_digest(&self) -> [u8; 32] {
        self.coefficient_digest
    }

    /// Domain-separated digest of the Lagrange-basis prefix.
    pub const fn lagrange_digest(&self) -> [u8; 32] {
        self.lagrange_digest
    }

    /// Combined domain-separated digest of both prefixes and their length.
    pub const fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }
}

fn checked_abi_len(array: &'static str, len: usize) -> Result<(), PreparedKeyError> {
    u32::try_from(len)
        .map(|_| ())
        .map_err(|_| PreparedKeyError::LengthExceedsAbi { array, len })
}

fn exact_len(array: &'static str, actual: usize, expected: usize) -> Result<(), PreparedKeyError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PreparedKeyError::LengthMismatch {
            array,
            expected,
            actual,
        })
    }
}

fn exact_column_extent(
    array: &'static str,
    actual: usize,
    columns: usize,
    stride: usize,
) -> Result<(), PreparedKeyError> {
    let expected = columns
        .checked_mul(stride)
        .ok_or(PreparedKeyError::LengthOverflow { array })?;
    checked_abi_len(array, actual)?;
    exact_len(array, actual, expected)
}

fn validate_fr(array: &'static str, values: &[RawFr]) -> Result<(), PreparedKeyError> {
    for (index, value) in values.iter().enumerate() {
        if greater_or_equal(value.limbs, BLS12_381_FR_MODULUS) {
            return Err(PreparedKeyError::UnreducedScalar { array, index });
        }
    }
    Ok(())
}

fn validate_g1(array: &'static str, points: &[RawG1Affine]) -> Result<(), PreparedKeyError> {
    for (index, value) in points.iter().enumerate() {
        if value.x == [0; 6] && value.y == [0; 6] {
            return Err(PreparedKeyError::IdentitySrsPoint { array, index });
        }
        for (coordinate, limbs) in [("x", value.x), ("y", value.y)] {
            if greater_or_equal(limbs, BLS12_381_FP_MODULUS) {
                return Err(PreparedKeyError::UnreducedBaseCoordinate {
                    array,
                    point: index,
                    coordinate,
                });
            }
        }
    }
    Ok(())
}

fn greater_or_equal<const N: usize>(lhs: [u64; N], rhs: [u64; N]) -> bool {
    for index in (0..N).rev() {
        if lhs[index] != rhs[index] {
            return lhs[index] > rhs[index];
        }
    }
    true
}

fn domain_fingerprint(domain: &ResidentDomainManifest) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_MANIFEST_FINGERPRINT_DOMAIN);
    for value in [
        DOMAIN_MANIFEST_SCHEMA_VERSION,
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
        domain.omega,
        domain.omega_inv,
        domain.extended_omega,
        domain.extended_omega_inv,
        domain.n_inv,
        domain.n_ext_inv,
        domain.coset_generator,
        domain.inverse_coset_generator,
        domain.barycentric_weight,
    ] {
        hash_fr(&mut hasher, value);
    }
    hash_fr_slice(&mut hasher, &domain.vanishing_inverses);
    hash_fr(&mut hasher, domain.delta);
    hasher.finalize().into()
}

fn prepared_key_fingerprint(key: &PreparedResidentProvingKey) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(PREPARED_KEY_FINGERPRINT_DOMAIN);
    hash_u32(&mut hasher, PREPARED_KEY_SCHEMA_VERSION);
    hasher.update(key.domain.fingerprint);
    hash_u32(&mut hasher, GRAPH_FAMILY_FINGERPRINT_VERSION);
    hasher.update(key.graphs.packed().fingerprint());
    let sections = key.graphs.sections();
    for count in [
        sections.custom(),
        sections.lookups(),
        sections.lookup_inputs(),
        sections.lookup_tables(),
        sections.trash(),
    ] {
        hash_u32(&mut hasher, count);
    }
    hash_len(&mut hasher, key.permutation_columns.len());
    for column in &key.permutation_columns {
        hash_u32(&mut hasher, column.kind as u32);
        hash_u32(&mut hasher, column.index);
    }
    hash_u32_slice(&mut hasher, &key.trash_selector_fixed_indices);
    hash_fr_slice(&mut hasher, &key.fixed_cosets_compact);
    hash_fr_slice(&mut hasher, &key.fixed_lagrange_compact);
    hash_fr_slice(&mut hasher, &key.permutation_cosets);
    hash_fr_slice(&mut hasher, &key.permutation_lagrange);
    hash_u32(&mut hasher, key.fixed_coefficient_count);
    hash_u32(&mut hasher, key.permutation_coefficient_count);
    hash_fr_slice(&mut hasher, &key.coefficient_columns);
    hash_fr_slice(&mut hasher, &key.extended_selectors);
    hasher.finalize().into()
}

fn srs_prefix_fingerprint(domain: &[u8], n: u32, points: &[RawG1Affine]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hash_u32(&mut hasher, PREPARED_SRS_SCHEMA_VERSION);
    hash_u32(&mut hasher, n);
    hash_len(&mut hasher, points.len());
    for point in points {
        for limb in point.x.into_iter().chain(point.y) {
            hasher.update(limb.to_le_bytes());
        }
    }
    hasher.finalize().into()
}

fn hash_fr_slice(hasher: &mut Sha256, values: &[RawFr]) {
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

fn hash_fr(hasher: &mut Sha256, value: RawFr) {
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

#[cfg(test)]
mod tests {
    use crate::expression::{
        CompiledGraph, GraphColumnRequirements, GraphOp, GraphOpcode, GraphSourceCounts,
        GraphValueKind, GraphValueRef,
    };

    use super::*;

    fn fr(value: u64) -> RawFr {
        RawFr {
            limbs: [value, 0, 0, 0],
        }
    }

    fn representative_domain() -> ResidentDomainManifest {
        ResidentDomainManifest::try_new(ResidentDomainManifestInput {
            k: 1,
            extended_k: 2,
            constraint_system_degree: 3,
            blinding_factors: 0,
            omega: fr(2),
            omega_inv: fr(3),
            extended_omega: fr(4),
            extended_omega_inv: fr(5),
            n_inv: fr(6),
            n_ext_inv: fr(7),
            coset_generator: fr(8),
            inverse_coset_generator: fr(9),
            barycentric_weight: fr(6),
            vanishing_inverses: vec![fr(11), fr(12)].into_boxed_slice(),
            delta: fr(13),
        })
        .expect("construct representative domain")
    }

    fn representative_graphs() -> CompiledProtocolGraphFamily {
        let graph = CompiledGraph::try_new(
            vec![fr(17)].into_boxed_slice(),
            Vec::new().into_boxed_slice(),
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
        .expect("compile representative graph");
        let required_fixed = [2];
        let required_advice = [7];
        let required_instance = [11];
        CompiledProtocolGraphFamily::try_new_with_required_columns(
            Some(graph),
            Vec::new().into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            GraphColumnRequirements::new(&required_fixed, &required_advice, &required_instance),
        )
        .expect("pack representative graph family")
    }

    fn representative_key_input() -> PreparedResidentProvingKeyInput {
        let permutation_columns = [
            ResidentColumnRef::new(ResidentColumnKind::Fixed, 0),
            ResidentColumnRef::new(ResidentColumnKind::Advice, 0),
            ResidentColumnRef::new(ResidentColumnKind::Instance, 0),
        ];
        PreparedResidentProvingKeyInput {
            domain: representative_domain(),
            graphs: representative_graphs(),
            permutation_columns: permutation_columns.into(),
            trash_selector_fixed_indices: Box::new([]),
            fixed_cosets_compact: vec![fr(20); 4].into_boxed_slice(),
            fixed_lagrange_compact: vec![fr(21); 2].into_boxed_slice(),
            permutation_cosets: vec![fr(22); 12].into_boxed_slice(),
            permutation_lagrange: vec![fr(23); 6].into_boxed_slice(),
            // Three logical fixed columns followed by three permutation columns.
            coefficient_columns: vec![fr(24); 12].into_boxed_slice(),
            extended_selectors: vec![fr(25); 12].into_boxed_slice(),
        }
    }

    #[test]
    fn domain_constructor_derives_protocol_shape_and_fingerprints_content() {
        let domain = representative_domain();
        assert_eq!(domain.n(), 2);
        assert_eq!(domain.schema_version(), 1);
        assert_eq!(domain.n_ext(), 4);
        assert_eq!(domain.quotient_piece_count(), 2);
        assert_eq!(domain.usable_rows(), 1);
        assert_eq!(domain.rotation_scale(), 2);
        assert_eq!(domain.last_rotation(), -1);
        assert_eq!(domain.permutation_chunk_len(), 1);
        assert_eq!(domain.vanishing_inverses().len(), 2);

        let mut changed = ResidentDomainManifestInput {
            k: 1,
            extended_k: 2,
            constraint_system_degree: 3,
            blinding_factors: 0,
            omega: fr(2),
            omega_inv: fr(3),
            extended_omega: fr(4),
            extended_omega_inv: fr(5),
            n_inv: fr(6),
            n_ext_inv: fr(7),
            coset_generator: fr(8),
            inverse_coset_generator: fr(9),
            barycentric_weight: fr(6),
            vanishing_inverses: vec![fr(11), fr(12)].into_boxed_slice(),
            delta: fr(13),
        };
        changed.delta = fr(14);
        let changed = ResidentDomainManifest::try_new(changed).expect("changed valid domain");
        assert_ne!(domain.fingerprint(), changed.fingerprint());
    }

    #[test]
    fn domain_constructor_rejects_inconsistent_extension_arrays() {
        let mut input = ResidentDomainManifestInput {
            k: 1,
            extended_k: 2,
            constraint_system_degree: 3,
            blinding_factors: 0,
            omega: fr(2),
            omega_inv: fr(3),
            extended_omega: fr(4),
            extended_omega_inv: fr(5),
            n_inv: fr(6),
            n_ext_inv: fr(7),
            coset_generator: fr(8),
            inverse_coset_generator: fr(9),
            barycentric_weight: fr(6),
            vanishing_inverses: vec![fr(11), fr(12)].into_boxed_slice(),
            delta: fr(13),
        };
        input.vanishing_inverses = vec![fr(11)].into_boxed_slice();
        assert_eq!(
            ResidentDomainManifest::try_new(input),
            Err(PreparedKeyError::LengthMismatch {
                array: "vanishing inverses",
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn domain_constructor_rejects_nonminimal_extension_and_barycentric_mismatch() {
        let input = ResidentDomainManifestInput {
            k: 1,
            extended_k: 3,
            constraint_system_degree: 3,
            blinding_factors: 0,
            omega: fr(2),
            omega_inv: fr(3),
            extended_omega: fr(4),
            extended_omega_inv: fr(5),
            n_inv: fr(6),
            n_ext_inv: fr(7),
            coset_generator: fr(8),
            inverse_coset_generator: fr(9),
            barycentric_weight: fr(6),
            vanishing_inverses: vec![fr(11); 4].into_boxed_slice(),
            delta: fr(13),
        };
        assert_eq!(
            ResidentDomainManifest::try_new(input),
            Err(PreparedKeyError::InvalidDomain(
                "extended domain is not the minimal quotient domain"
            ))
        );

        let input = ResidentDomainManifestInput {
            k: 1,
            extended_k: 2,
            constraint_system_degree: 3,
            blinding_factors: 0,
            omega: fr(2),
            omega_inv: fr(3),
            extended_omega: fr(4),
            extended_omega_inv: fr(5),
            n_inv: fr(6),
            n_ext_inv: fr(7),
            coset_generator: fr(8),
            inverse_coset_generator: fr(9),
            barycentric_weight: fr(10),
            vanishing_inverses: vec![fr(11), fr(12)].into_boxed_slice(),
            delta: fr(13),
        };
        assert_eq!(
            ResidentDomainManifest::try_new(input),
            Err(PreparedKeyError::InvalidDomain(
                "first-point barycentric weight differs from the base inverse size"
            ))
        );
    }

    #[test]
    fn prepared_key_validates_exact_orders_and_fingerprint_sensitivity() {
        let key = PreparedResidentProvingKey::try_new(representative_key_input())
            .expect("construct representative prepared key");
        assert_eq!(key.schema_version(), 1);
        assert_eq!(key.fixed_coefficient_count(), 3);
        assert_eq!(key.permutation_coefficient_count(), 3);
        assert_eq!(key.permutation_columns().len(), 3);
        assert_eq!(key.fixed_cosets_compact().len(), 4);
        assert_eq!(key.fixed_lagrange_compact().len(), 2);
        assert_eq!(key.permutation_cosets().len(), 12);
        assert_eq!(key.permutation_lagrange().len(), 6);
        assert_eq!(key.coefficient_columns().len(), 12);
        assert_eq!(key.extended_selectors().len(), 12);

        let mut changed = representative_key_input();
        changed.coefficient_columns[0] = fr(26);
        let changed = PreparedResidentProvingKey::try_new(changed)
            .expect("construct content-changed prepared key");
        assert_ne!(key.fingerprint(), changed.fingerprint());
        assert_eq!(key, key.clone());
    }

    #[test]
    fn prepared_key_rejects_dense_column_and_logical_fixed_confusion() {
        let mut bad_dense = representative_key_input();
        bad_dense.permutation_columns[1] = ResidentColumnRef::new(ResidentColumnKind::Advice, 1);
        assert!(matches!(
            PreparedResidentProvingKey::try_new(bad_dense),
            Err(PreparedKeyError::ColumnIndexOutOfBounds {
                position: 1,
                kind: ResidentColumnKind::Advice,
                index: 1,
                bound: 1,
            })
        ));

        let mut short_coefficients = representative_key_input();
        // Two fixed columns cannot cover required logical fixed column 2.
        short_coefficients.coefficient_columns = vec![fr(24); 10].into_boxed_slice();
        assert!(matches!(
            PreparedResidentProvingKey::try_new(short_coefficients),
            Err(PreparedKeyError::LogicalFixedColumnOutOfBounds {
                logical_index: 2,
                fixed_column_count: 2,
                ..
            })
        ));
    }

    #[test]
    fn prepared_key_rejects_unreduced_scalar_before_fingerprinting() {
        let mut input = representative_key_input();
        input.extended_selectors[3] = RawFr {
            limbs: BLS12_381_FR_MODULUS,
        };
        assert_eq!(
            PreparedResidentProvingKey::try_new(input),
            Err(PreparedKeyError::UnreducedScalar {
                array: "extended selectors",
                index: 3,
            })
        );
    }

    fn raw_point(value: u64) -> RawG1Affine {
        RawG1Affine {
            x: [value, 0, 0, 0, 0, 0],
            y: [value + 1, 0, 0, 0, 0, 0],
        }
    }

    #[test]
    fn prepared_srs_separates_prefix_digests_and_validates_shape() {
        let coefficient = vec![raw_point(1), raw_point(3)].into_boxed_slice();
        let lagrange = vec![raw_point(5), raw_point(7)].into_boxed_slice();
        let srs = PreparedResidentSrs::try_new(2, coefficient.clone(), lagrange.clone())
            .expect("construct representative SRS");
        assert_eq!(srs.n(), 2);
        assert_eq!(srs.coefficient_bases(), coefficient.as_ref());
        assert_eq!(srs.lagrange_bases(), lagrange.as_ref());
        assert_ne!(srs.coefficient_digest(), srs.lagrange_digest());

        let mut changed_coefficient = Vec::from(coefficient);
        changed_coefficient[0].x[0] += 1;
        let changed =
            PreparedResidentSrs::try_new(2, changed_coefficient.into_boxed_slice(), lagrange)
                .expect("construct changed SRS");
        assert_ne!(srs.coefficient_digest(), changed.coefficient_digest());
        assert_eq!(srs.lagrange_digest(), changed.lagrange_digest());
        assert_ne!(srs.fingerprint(), changed.fingerprint());

        assert!(matches!(
            PreparedResidentSrs::try_new(
                2,
                vec![raw_point(1)].into_boxed_slice(),
                vec![raw_point(2), raw_point(3)].into_boxed_slice(),
            ),
            Err(PreparedKeyError::LengthMismatch {
                array: "coefficient SRS",
                expected: 2,
                actual: 1,
            })
        ));
    }

    #[test]
    fn prepared_srs_rejects_unreduced_coordinate() {
        let invalid = RawG1Affine {
            x: BLS12_381_FP_MODULUS,
            y: [1, 0, 0, 0, 0, 0],
        };
        assert_eq!(
            PreparedResidentSrs::try_new(
                1,
                vec![invalid].into_boxed_slice(),
                vec![raw_point(1)].into_boxed_slice(),
            ),
            Err(PreparedKeyError::UnreducedBaseCoordinate {
                array: "coefficient SRS",
                point: 0,
                coordinate: "x",
            })
        );
    }

    #[test]
    fn prepared_srs_rejects_identity_in_coefficient_bases() {
        assert_eq!(
            PreparedResidentSrs::try_new(
                1,
                vec![RawG1Affine::default()].into_boxed_slice(),
                vec![raw_point(1)].into_boxed_slice(),
            ),
            Err(PreparedKeyError::IdentitySrsPoint {
                array: "coefficient SRS",
                index: 0,
            })
        );
    }

    #[test]
    fn prepared_srs_rejects_identity_in_lagrange_bases() {
        assert_eq!(
            PreparedResidentSrs::try_new(
                2,
                vec![raw_point(1), raw_point(3)].into_boxed_slice(),
                vec![raw_point(5), RawG1Affine::default()].into_boxed_slice(),
            ),
            Err(PreparedKeyError::IdentitySrsPoint {
                array: "Lagrange SRS",
                index: 1,
            })
        );
    }
}
