//! Checked, pointer-free expression graph ABI shared by resident backends.
//!
//! [`CompiledGraph`] owns the arrays copied into a backend key context. Its
//! constructor validates the packed representation before a backend can create
//! buffers or commands. Logical expression targets are intentionally absent
//! from this ABI: the proving-system compiler validates them before compacting
//! them into reusable physical scratch slots.
//!
//! [`CompiledGraphFamily`] is the graph-family component of an immutable key
//! identity. Its fingerprint deliberately excludes the domain sizes, fixed
//! column contents, metallib identity, and Metal device identity; a backend key
//! context must bind those separate identity fields alongside this component.

use crate::RawFr;
use sha2::{Digest, Sha256};

/// Version of the canonical [`CompiledGraphFamily`] fingerprint serialization.
pub const GRAPH_FAMILY_FINGERPRINT_VERSION: u32 = 1;

const GRAPH_FAMILY_FINGERPRINT_DOMAIN: &[u8] = b"midnight-gpu-prover/compiled-graph-family\0";

/// Maximum number of simultaneously live field elements in one graph.
pub const MAX_GRAPH_SCRATCH_SLOTS: u32 = 256;

/// Source tag stored in [`GraphValueRef::kind`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GraphValueKind {
    /// Entry in the graph's constant array.
    Constant = 0,
    /// Previously written physical scratch slot.
    Intermediate = 1,
    /// Fixed polynomial column and rotation.
    Fixed = 2,
    /// Advice polynomial column and rotation.
    Advice = 3,
    /// Instance polynomial column and rotation.
    Instance = 4,
    /// Entry in the proof challenge array.
    Challenge = 5,
    /// Permutation challenge beta.
    Beta = 6,
    /// Permutation challenge gamma.
    Gamma = 7,
    /// Lookup compression challenge theta.
    Theta = 8,
    /// Trash argument challenge.
    TrashChallenge = 9,
    /// Quotient aggregation challenge y.
    Y = 10,
    /// Numerator value captured before graph evaluation.
    PreviousValue = 11,
}

impl TryFrom<u32> for GraphValueKind {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Constant),
            1 => Ok(Self::Intermediate),
            2 => Ok(Self::Fixed),
            3 => Ok(Self::Advice),
            4 => Ok(Self::Instance),
            5 => Ok(Self::Challenge),
            6 => Ok(Self::Beta),
            7 => Ok(Self::Gamma),
            8 => Ok(Self::Theta),
            9 => Ok(Self::TrashChallenge),
            10 => Ok(Self::Y),
            11 => Ok(Self::PreviousValue),
            _ => Err(()),
        }
    }
}

/// Operation tag stored in [`GraphOp::opcode`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GraphOpcode {
    /// `lhs + rhs`.
    Add = 0,
    /// `lhs - rhs`.
    Sub = 1,
    /// `lhs * rhs`.
    Mul = 2,
    /// `lhs * lhs`.
    Square = 3,
    /// `lhs + lhs`.
    Double = 4,
    /// `-lhs`.
    Negate = 5,
    /// Repeatedly evaluate `value = value * rhs + part`, starting at `lhs`.
    Horner = 6,
    /// Copy `lhs`.
    Store = 7,
}

impl TryFrom<u32> for GraphOpcode {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Add),
            1 => Ok(Self::Sub),
            2 => Ok(Self::Mul),
            3 => Ok(Self::Square),
            4 => Ok(Self::Double),
            5 => Ok(Self::Negate),
            6 => Ok(Self::Horner),
            7 => Ok(Self::Store),
            _ => Err(()),
        }
    }
}

/// Pointer-free source operand consumed by resident expression kernels.
///
/// For constants, intermediates, and challenges, `a` is the source index and
/// `b` is zero. For polynomial sources, `a` is the logical column before
/// family compaction and the dense family column afterward; `b` is the
/// graph-local rotation index. Challenge singleton tags require both fields to
/// be zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct GraphValueRef {
    /// Numeric [`GraphValueKind`] tag.
    pub kind: u32,
    /// Constant, scratch, column, or challenge index, depending on `kind`.
    pub a: u32,
    /// Rotation index for polynomial sources; zero for every other source.
    pub b: u32,
}

impl GraphValueRef {
    /// Creates a source reference with a known tag.
    pub const fn new(kind: GraphValueKind, a: u32, b: u32) -> Self {
        Self {
            kind: kind as u32,
            a,
            b,
        }
    }
}

/// Pointer-free operation consumed by resident expression kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct GraphOp {
    /// Numeric [`GraphOpcode`] tag.
    pub opcode: u32,
    /// Physical scratch slot receiving the result.
    pub target: u32,
    /// First entry in [`CompiledGraph::horner_parts`].
    pub parts_offset: u32,
    /// Number of Horner parts; zero for non-Horner operations.
    pub parts_len: u32,
    /// First operand, or Horner starting value.
    pub lhs: GraphValueRef,
    /// Second operand or Horner factor; zeroed for unary operations.
    pub rhs: GraphValueRef,
}

impl GraphOp {
    /// Creates an operation with a known opcode.
    pub const fn new(
        opcode: GraphOpcode,
        target: u32,
        parts_offset: u32,
        parts_len: u32,
        lhs: GraphValueRef,
        rhs: GraphValueRef,
    ) -> Self {
        Self {
            opcode: opcode as u32,
            target,
            parts_offset,
            parts_len,
            lhs,
            rhs,
        }
    }
}

/// Counts for source arrays supplied separately when a graph is evaluated.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct GraphSourceCounts {
    fixed_columns: u32,
    advice_columns: u32,
    instance_columns: u32,
    challenges: u32,
}

impl GraphSourceCounts {
    /// Creates the column-ID and challenge bounds for a graph.
    ///
    /// Before family compaction, the column fields bound original logical IDs.
    /// A [`CompiledGraphFamily`] replaces those sparse IDs with family-wide
    /// dense indices.
    pub const fn new(
        fixed_columns: u32,
        advice_columns: u32,
        instance_columns: u32,
        challenges: u32,
    ) -> Self {
        Self {
            fixed_columns,
            advice_columns,
            instance_columns,
            challenges,
        }
    }

    /// Exclusive fixed-column ID bound.
    pub const fn fixed_columns(self) -> u32 {
        self.fixed_columns
    }

    /// Exclusive advice-column ID bound.
    pub const fn advice_columns(self) -> u32 {
        self.advice_columns
    }

    /// Exclusive instance-column ID bound.
    pub const fn instance_columns(self) -> u32 {
        self.instance_columns
    }

    /// Number of proof challenges.
    pub const fn challenges(self) -> u32 {
        self.challenges
    }
}

/// Logical polynomial columns required by key-wide work outside graph operands.
///
/// Permutation columns and trash selectors can be consumed by resident proof
/// phases without appearing in an expression graph. Supplying them here makes
/// the family compaction maps and fingerprint cover those protocol inputs too.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct GraphColumnRequirements<'a> {
    fixed: &'a [u32],
    advice: &'a [u32],
    instance: &'a [u32],
}

impl<'a> GraphColumnRequirements<'a> {
    /// Creates borrowed logical-column requirements.
    ///
    /// Inputs may be sparse, unsorted, and repeated. Family construction sorts
    /// and deduplicates each column kind deterministically.
    pub const fn new(fixed: &'a [u32], advice: &'a [u32], instance: &'a [u32]) -> Self {
        Self {
            fixed,
            advice,
            instance,
        }
    }

    /// Required logical fixed-column IDs.
    pub const fn fixed(self) -> &'a [u32] {
        self.fixed
    }

    /// Required logical advice-column IDs.
    pub const fn advice(self) -> &'a [u32] {
        self.advice
    }

    /// Required logical instance-column IDs.
    pub const fn instance(self) -> &'a [u32] {
        self.instance
    }
}

/// Protocol-category ranges within a packed [`CompiledGraphFamily`].
///
/// Descriptors remain in protocol order: custom gates, lookup identities,
/// lookup-input compressors, lookup-table compressors, then trash identities.
/// A backend key context validates that these counts exactly cover the family
/// and that the three lookup counts agree.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct GraphFamilySections {
    custom: u32,
    lookups: u32,
    lookup_inputs: u32,
    lookup_tables: u32,
    trash: u32,
}

impl GraphFamilySections {
    /// Creates exact protocol-category counts for a packed family.
    pub const fn new(
        custom: u32,
        lookups: u32,
        lookup_inputs: u32,
        lookup_tables: u32,
        trash: u32,
    ) -> Self {
        Self {
            custom,
            lookups,
            lookup_inputs,
            lookup_tables,
            trash,
        }
    }

    /// Number of custom-gate graphs; a valid Metal family accepts zero or one.
    pub const fn custom(self) -> u32 {
        self.custom
    }

    /// Number of lookup identity graphs.
    pub const fn lookups(self) -> u32 {
        self.lookups
    }

    /// Number of lookup-input compression graphs.
    pub const fn lookup_inputs(self) -> u32 {
        self.lookup_inputs
    }

    /// Number of lookup-table compression graphs.
    pub const fn lookup_tables(self) -> u32 {
        self.lookup_tables
    }

    /// Number of trash-argument graphs.
    pub const fn trash(self) -> u32 {
        self.trash
    }

    /// Checked total number of descriptors covered by all categories.
    pub const fn checked_total(self) -> Option<u32> {
        let Some(total) = self.custom.checked_add(self.lookups) else {
            return None;
        };
        let Some(total) = total.checked_add(self.lookup_inputs) else {
            return None;
        };
        let Some(total) = total.checked_add(self.lookup_tables) else {
            return None;
        };
        total.checked_add(self.trash)
    }
}

/// Metal pipeline class selected from the graph's physical scratch footprint.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ScratchClass {
    /// Up to 8 live field elements.
    Slots8 = 8,
    /// Up to 16 live field elements.
    Slots16 = 16,
    /// Up to 32 live field elements.
    Slots32 = 32,
    /// Up to 64 live field elements.
    Slots64 = 64,
    /// Up to 128 live field elements.
    Slots128 = 128,
    /// Up to 256 live field elements.
    Slots256 = 256,
}

impl ScratchClass {
    /// Selects the smallest pipeline that can contain `required_slots`.
    pub fn for_required(required_slots: u32) -> Result<Self, GraphValidationError> {
        match required_slots {
            0..=8 => Ok(Self::Slots8),
            9..=16 => Ok(Self::Slots16),
            17..=32 => Ok(Self::Slots32),
            33..=64 => Ok(Self::Slots64),
            65..=128 => Ok(Self::Slots128),
            129..=256 => Ok(Self::Slots256),
            _ => Err(GraphValidationError::ScratchCapacityExceeded {
                required: u64::from(required_slots),
            }),
        }
    }

    /// Number of field elements reserved by this class.
    pub const fn capacity(self) -> u32 {
        self as u32
    }
}

/// Failure while validating a packed expression graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphValidationError {
    /// An owned array is too long for the fixed-width backend ABI.
    ArrayLengthExceedsAbi {
        /// Name of the offending array.
        array: &'static str,
        /// Host element count that cannot be represented as `u32`.
        len: usize,
    },
    /// An operation contains an unknown numeric opcode.
    UnknownOpcode {
        /// Operation index.
        operation: u32,
        /// Unrecognized opcode.
        opcode: u32,
    },
    /// A used operand contains an unknown numeric source tag.
    UnknownValueKind {
        /// Operation index.
        operation: u32,
        /// Unrecognized source tag.
        kind: u32,
    },
    /// Fields that must be zero for the source kind were nonzero.
    NonCanonicalValue {
        /// Operation index.
        operation: u32,
        /// Source tag.
        kind: u32,
        /// First source payload.
        a: u32,
        /// Second source payload.
        b: u32,
    },
    /// A constant, column, or challenge index exceeds its corresponding array.
    SourceIndexOutOfBounds {
        /// Operation index.
        operation: u32,
        /// Source tag.
        kind: u32,
        /// Requested source index.
        index: u32,
        /// Exclusive source bound.
        limit: u32,
    },
    /// A polynomial source references a missing rotation.
    RotationIndexOutOfBounds {
        /// Operation index.
        operation: u32,
        /// Requested rotation index.
        index: u32,
        /// Exclusive rotation bound.
        limit: u32,
    },
    /// An intermediate source is read before its physical slot is first written.
    IntermediateReadBeforeWrite {
        /// Operation index.
        operation: u32,
        /// Physical scratch slot.
        slot: u32,
    },
    /// A non-Horner operation contains a Horner range.
    UnexpectedHornerRange {
        /// Operation index.
        operation: u32,
        /// Supplied offset.
        offset: u32,
        /// Supplied length.
        len: u32,
    },
    /// Horner ranges are not a contiguous, ordered packing of the parts array.
    NonContiguousHornerParts {
        /// Operation index.
        operation: u32,
        /// Required next offset.
        expected_offset: u32,
        /// Supplied offset.
        actual_offset: u32,
    },
    /// Adding a Horner range's offset and length overflowed `u32`.
    HornerRangeOverflow {
        /// Operation index.
        operation: u32,
        /// Supplied offset.
        offset: u32,
        /// Supplied length.
        len: u32,
    },
    /// A Horner range exceeds the packed parts array.
    HornerRangeOutOfBounds {
        /// Operation index.
        operation: u32,
        /// Exclusive end of the supplied range.
        end: u32,
        /// Number of available parts.
        parts_len: u32,
    },
    /// Parts remain after all operation ranges have been consumed.
    UnreferencedHornerParts {
        /// First unreferenced part.
        first: u32,
        /// Number of available parts.
        parts_len: u32,
    },
    /// A unary or store operation has a nonzero unused operand.
    NonCanonicalUnusedOperand {
        /// Operation index.
        operation: u32,
    },
    /// The packed physical scratch footprint exceeds the supported maximum.
    ScratchCapacityExceeded {
        /// Minimum number of slots required by the offending index.
        required: u64,
    },
}

impl std::fmt::Display for GraphValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid resident expression graph: {self:?}")
    }
}

impl std::error::Error for GraphValidationError {}

/// Validated owned arrays for one compiled expression graph.
///
/// The proving-system compiler must validate logical targets and perform the
/// liveness allocation before calling [`CompiledGraph::try_new`]. This type
/// validates the resulting physical representation, including all source and
/// range bounds, but cannot reconstruct logical identities erased by packing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledGraph {
    constants: Box<[RawFr]>,
    rotations: Box<[i32]>,
    operations: Box<[GraphOp]>,
    horner_parts: Box<[GraphValueRef]>,
    source_counts: GraphSourceCounts,
    required_scratch_slots: u32,
    scratch_class: ScratchClass,
}

impl CompiledGraph {
    /// Validates and takes ownership of a physically packed graph.
    pub fn try_new(
        constants: Box<[RawFr]>,
        rotations: Box<[i32]>,
        operations: Box<[GraphOp]>,
        horner_parts: Box<[GraphValueRef]>,
        source_counts: GraphSourceCounts,
    ) -> Result<Self, GraphValidationError> {
        let constants_len = checked_abi_len("constants", constants.len())?;
        let rotations_len = checked_abi_len("rotations", rotations.len())?;
        let operations_len = checked_abi_len("operations", operations.len())?;
        let horner_parts_len = checked_abi_len("horner_parts", horner_parts.len())?;

        let mut initialized = [false; MAX_GRAPH_SCRATCH_SLOTS as usize];
        let mut required_scratch_slots = 0u32;
        let mut next_part = 0u32;

        for (operation_index, operation) in (0..operations_len).zip(operations.iter()) {
            let opcode = GraphOpcode::try_from(operation.opcode).map_err(|()| {
                GraphValidationError::UnknownOpcode {
                    operation: operation_index,
                    opcode: operation.opcode,
                }
            })?;
            require_scratch_slot(operation.target, &mut required_scratch_slots)?;

            match opcode {
                GraphOpcode::Add | GraphOpcode::Sub | GraphOpcode::Mul => {
                    require_no_parts(operation_index, operation)?;
                    validate_source(
                        operation.lhs,
                        operation_index,
                        constants_len,
                        rotations_len,
                        source_counts,
                        &initialized,
                        &mut required_scratch_slots,
                    )?;
                    validate_source(
                        operation.rhs,
                        operation_index,
                        constants_len,
                        rotations_len,
                        source_counts,
                        &initialized,
                        &mut required_scratch_slots,
                    )?;
                }
                GraphOpcode::Square
                | GraphOpcode::Double
                | GraphOpcode::Negate
                | GraphOpcode::Store => {
                    require_no_parts(operation_index, operation)?;
                    require_unused_operand_zero(operation_index, operation.rhs)?;
                    validate_source(
                        operation.lhs,
                        operation_index,
                        constants_len,
                        rotations_len,
                        source_counts,
                        &initialized,
                        &mut required_scratch_slots,
                    )?;
                }
                GraphOpcode::Horner => {
                    let end = operation
                        .parts_offset
                        .checked_add(operation.parts_len)
                        .ok_or(GraphValidationError::HornerRangeOverflow {
                            operation: operation_index,
                            offset: operation.parts_offset,
                            len: operation.parts_len,
                        })?;
                    if operation.parts_offset != next_part {
                        return Err(GraphValidationError::NonContiguousHornerParts {
                            operation: operation_index,
                            expected_offset: next_part,
                            actual_offset: operation.parts_offset,
                        });
                    }
                    if end > horner_parts_len {
                        return Err(GraphValidationError::HornerRangeOutOfBounds {
                            operation: operation_index,
                            end,
                            parts_len: horner_parts_len,
                        });
                    }
                    validate_source(
                        operation.lhs,
                        operation_index,
                        constants_len,
                        rotations_len,
                        source_counts,
                        &initialized,
                        &mut required_scratch_slots,
                    )?;
                    validate_source(
                        operation.rhs,
                        operation_index,
                        constants_len,
                        rotations_len,
                        source_counts,
                        &initialized,
                        &mut required_scratch_slots,
                    )?;
                    for part in &horner_parts[operation.parts_offset as usize..end as usize] {
                        validate_source(
                            *part,
                            operation_index,
                            constants_len,
                            rotations_len,
                            source_counts,
                            &initialized,
                            &mut required_scratch_slots,
                        )?;
                    }
                    next_part = end;
                }
            }

            initialized[operation.target as usize] = true;
        }

        if next_part != horner_parts_len {
            return Err(GraphValidationError::UnreferencedHornerParts {
                first: next_part,
                parts_len: horner_parts_len,
            });
        }

        let scratch_class = ScratchClass::for_required(required_scratch_slots)?;
        Ok(Self {
            constants,
            rotations,
            operations,
            horner_parts,
            source_counts,
            required_scratch_slots,
            scratch_class,
        })
    }

    /// Constants in Montgomery form.
    pub fn constants(&self) -> &[RawFr] {
        &self.constants
    }

    /// Signed rotations referenced by polynomial sources.
    pub fn rotations(&self) -> &[i32] {
        &self.rotations
    }

    /// Operations in evaluation order.
    pub fn operations(&self) -> &[GraphOp] {
        &self.operations
    }

    /// Contiguously packed Horner parts.
    pub fn horner_parts(&self) -> &[GraphValueRef] {
        &self.horner_parts
    }

    /// Bounds for dynamic source arrays.
    pub const fn source_counts(&self) -> GraphSourceCounts {
        self.source_counts
    }

    /// Number of physical slots addressed by the packed graph.
    pub const fn required_scratch_slots(&self) -> u32 {
        self.required_scratch_slots
    }

    /// Smallest scratch pipeline that can evaluate this graph.
    pub const fn scratch_class(&self) -> ScratchClass {
        self.scratch_class
    }

    /// Whether this graph has no operations and therefore evaluates to zero.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Physical slot containing the graph result, if the graph is nonempty.
    pub fn result_slot(&self) -> Option<u32> {
        self.operations.last().map(|operation| operation.target)
    }
}

/// Fixed-width location of one graph in a [`CompiledGraphFamily`].
///
/// Constant and rotation source indices, and every operation's Horner-part
/// offset, remain relative to this graph. The packed offsets below select the
/// corresponding graph-local slice from each concatenated family array.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CompiledGraphDescriptor {
    constants_offset: u32,
    constants_count: u32,
    rotations_offset: u32,
    rotations_count: u32,
    operations_offset: u32,
    operations_count: u32,
    parts_offset: u32,
    parts_count: u32,
    scratch_class: ScratchClass,
}

impl CompiledGraphDescriptor {
    /// First constant in the family's packed constant array.
    pub const fn constants_offset(self) -> u32 {
        self.constants_offset
    }

    /// Number of graph-local constants.
    pub const fn constants_count(self) -> u32 {
        self.constants_count
    }

    /// First rotation in the family's packed rotation array.
    pub const fn rotations_offset(self) -> u32 {
        self.rotations_offset
    }

    /// Number of graph-local rotations.
    pub const fn rotations_count(self) -> u32 {
        self.rotations_count
    }

    /// First operation in the family's packed operation array.
    pub const fn operations_offset(self) -> u32 {
        self.operations_offset
    }

    /// Number of operations in this graph.
    pub const fn operations_count(self) -> u32 {
        self.operations_count
    }

    /// First Horner part in the family's packed part array.
    pub const fn parts_offset(self) -> u32 {
        self.parts_offset
    }

    /// Number of graph-local Horner parts.
    pub const fn parts_count(self) -> u32 {
        self.parts_count
    }

    /// Metal scratch pipeline class required by this graph.
    pub const fn scratch_class(self) -> ScratchClass {
        self.scratch_class
    }
}

/// Failure while compacting and packing a family of validated graphs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphFamilyValidationError {
    /// A concatenated array is too long for its fixed-width backend offset.
    ArrayLengthExceedsAbi {
        /// Name of the offending packed array.
        array: &'static str,
        /// Host element count that cannot be represented as `u32`.
        len: usize,
    },
    /// A polynomial source was absent from the compaction map built from it.
    MissingCompactionEntry {
        /// Polynomial source kind.
        kind: GraphValueKind,
        /// Original logical column ID.
        logical_column: u32,
    },
    /// The three protocol lookup graph groups have different lengths.
    LookupSectionCountMismatch {
        /// Number of lookup-identity graphs.
        lookups: usize,
        /// Number of lookup-input compression graphs.
        lookup_inputs: usize,
        /// Number of lookup-table compression graphs.
        lookup_tables: usize,
    },
}

impl std::fmt::Display for GraphFamilyValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "invalid resident expression graph family: {self:?}"
        )
    }
}

impl std::error::Error for GraphFamilyValidationError {}

/// Canonically compacted and packed expression graphs in protocol order.
///
/// Logical fixed, advice, and instance column IDs are deduplicated key-wide,
/// sorted ascending within their source kind, and replaced by dense indices in
/// every operation and Horner part. Constants, rotations, and parts remain
/// graph-relative and are located through [`CompiledGraphDescriptor`].
///
/// The SHA-256 fingerprint covers a versioned, fieldwise, little-endian
/// serialization of this complete graph-family component. It intentionally is
/// not a complete backend key-context identity: callers must additionally bind
/// `n`, `n_ext`, fixed column contents, graph/backend ABI identity, metallib
/// identity, and device registry identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledGraphFamily {
    descriptors: Box<[CompiledGraphDescriptor]>,
    constants: Box<[RawFr]>,
    rotations: Box<[i32]>,
    operations: Box<[GraphOp]>,
    horner_parts: Box<[GraphValueRef]>,
    fixed_compact_to_logical: Box<[u32]>,
    advice_compact_to_logical: Box<[u32]>,
    instance_compact_to_logical: Box<[u32]>,
    challenge_bound: u32,
    fingerprint: [u8; 32],
}

impl CompiledGraphFamily {
    /// Compacts and packs validated graphs without changing protocol order.
    pub fn try_new(graphs: Box<[CompiledGraph]>) -> Result<Self, GraphFamilyValidationError> {
        Self::try_new_with_required_columns(graphs, GraphColumnRequirements::default())
    }

    /// Compacts graphs together with columns consumed by non-graph key phases.
    ///
    /// Required columns participate in the sorted compact maps and canonical
    /// fingerprint even when no expression operand references them.
    pub fn try_new_with_required_columns(
        graphs: Box<[CompiledGraph]>,
        required: GraphColumnRequirements<'_>,
    ) -> Result<Self, GraphFamilyValidationError> {
        family_abi_len("graphs", graphs.len())?;

        let mut fixed_columns = required.fixed().iter().copied().collect();
        let mut advice_columns = required.advice().iter().copied().collect();
        let mut instance_columns = required.instance().iter().copied().collect();
        let mut challenge_bound = 0u32;

        for graph in &graphs {
            challenge_bound = challenge_bound.max(graph.source_counts.challenges);
            for operation in &graph.operations {
                note_polynomial_column(
                    operation.lhs,
                    &mut fixed_columns,
                    &mut advice_columns,
                    &mut instance_columns,
                );
                note_polynomial_column(
                    operation.rhs,
                    &mut fixed_columns,
                    &mut advice_columns,
                    &mut instance_columns,
                );
            }
            for part in &graph.horner_parts {
                note_polynomial_column(
                    *part,
                    &mut fixed_columns,
                    &mut advice_columns,
                    &mut instance_columns,
                );
            }
        }

        let fixed_compact_to_logical = fixed_columns.into_iter().collect::<Vec<_>>();
        let advice_compact_to_logical = advice_columns.into_iter().collect::<Vec<_>>();
        let instance_compact_to_logical = instance_columns.into_iter().collect::<Vec<_>>();
        family_abi_len("fixed_compact_to_logical", fixed_compact_to_logical.len())?;
        family_abi_len("advice_compact_to_logical", advice_compact_to_logical.len())?;
        family_abi_len(
            "instance_compact_to_logical",
            instance_compact_to_logical.len(),
        )?;

        let mut descriptors = Vec::with_capacity(graphs.len());
        let mut constants = Vec::new();
        let mut rotations = Vec::new();
        let mut operations = Vec::new();
        let mut horner_parts = Vec::new();

        for graph in Vec::from(graphs) {
            let CompiledGraph {
                constants: graph_constants,
                rotations: graph_rotations,
                operations: graph_operations,
                horner_parts: graph_parts,
                source_counts: _,
                required_scratch_slots: _,
                scratch_class,
            } = graph;

            let (constants_offset, constants_count) =
                family_packed_range("constants", constants.len(), graph_constants.len())?;
            let (rotations_offset, rotations_count) =
                family_packed_range("rotations", rotations.len(), graph_rotations.len())?;
            let (operations_offset, operations_count) =
                family_packed_range("operations", operations.len(), graph_operations.len())?;
            let (parts_offset, parts_count) =
                family_packed_range("horner_parts", horner_parts.len(), graph_parts.len())?;

            let mut graph_operations = Vec::from(graph_operations);
            for operation in &mut graph_operations {
                remap_polynomial_column(
                    &mut operation.lhs,
                    &fixed_compact_to_logical,
                    &advice_compact_to_logical,
                    &instance_compact_to_logical,
                )?;
                remap_polynomial_column(
                    &mut operation.rhs,
                    &fixed_compact_to_logical,
                    &advice_compact_to_logical,
                    &instance_compact_to_logical,
                )?;
            }
            let mut graph_parts = Vec::from(graph_parts);
            for part in &mut graph_parts {
                remap_polynomial_column(
                    part,
                    &fixed_compact_to_logical,
                    &advice_compact_to_logical,
                    &instance_compact_to_logical,
                )?;
            }

            descriptors.push(CompiledGraphDescriptor {
                constants_offset,
                constants_count,
                rotations_offset,
                rotations_count,
                operations_offset,
                operations_count,
                parts_offset,
                parts_count,
                scratch_class,
            });
            constants.extend(Vec::from(graph_constants));
            rotations.extend(Vec::from(graph_rotations));
            operations.extend(graph_operations);
            horner_parts.extend(graph_parts);
        }

        let fingerprint = graph_family_fingerprint(
            &descriptors,
            &constants,
            &rotations,
            &operations,
            &horner_parts,
            &fixed_compact_to_logical,
            &advice_compact_to_logical,
            &instance_compact_to_logical,
            challenge_bound,
        );

        Ok(Self {
            descriptors: descriptors.into_boxed_slice(),
            constants: constants.into_boxed_slice(),
            rotations: rotations.into_boxed_slice(),
            operations: operations.into_boxed_slice(),
            horner_parts: horner_parts.into_boxed_slice(),
            fixed_compact_to_logical: fixed_compact_to_logical.into_boxed_slice(),
            advice_compact_to_logical: advice_compact_to_logical.into_boxed_slice(),
            instance_compact_to_logical: instance_compact_to_logical.into_boxed_slice(),
            challenge_bound,
            fingerprint,
        })
    }

    /// Per-graph packed ranges in protocol order.
    pub fn descriptors(&self) -> &[CompiledGraphDescriptor] {
        &self.descriptors
    }

    /// All graph-local constants concatenated in protocol order.
    pub fn constants(&self) -> &[RawFr] {
        &self.constants
    }

    /// All graph-local rotations concatenated in protocol order.
    pub fn rotations(&self) -> &[i32] {
        &self.rotations
    }

    /// All operations concatenated in protocol order.
    pub fn operations(&self) -> &[GraphOp] {
        &self.operations
    }

    /// All graph-local Horner parts concatenated in protocol order.
    pub fn horner_parts(&self) -> &[GraphValueRef] {
        &self.horner_parts
    }

    /// Dense fixed-column index to original logical column ID.
    pub fn fixed_compact_to_logical(&self) -> &[u32] {
        &self.fixed_compact_to_logical
    }

    /// Dense advice-column index to original logical column ID.
    pub fn advice_compact_to_logical(&self) -> &[u32] {
        &self.advice_compact_to_logical
    }

    /// Dense instance-column index to original logical column ID.
    pub fn instance_compact_to_logical(&self) -> &[u32] {
        &self.instance_compact_to_logical
    }

    /// Exclusive bound for the proof challenge array across all graphs.
    pub const fn challenge_bound(&self) -> u32 {
        self.challenge_bound
    }

    /// SHA-256 of the versioned canonical graph-family serialization.
    pub const fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }

    /// Whether the family contains no graph descriptors.
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }
}

/// Protocol-classified expression family with derived, immutable boundaries.
///
/// This is the only graph-family type accepted by immutable proving-key import.
/// Callers supply graphs by protocol category, and construction derives both
/// descriptor order and section counts. Consequently a later key-context
/// caller cannot relabel an already packed descriptor array with independent
/// boundary metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledProtocolGraphFamily {
    packed: CompiledGraphFamily,
    sections: GraphFamilySections,
}

impl CompiledProtocolGraphFamily {
    /// Packs one protocol-classified graph family without extra column needs.
    pub fn try_new(
        custom: Option<CompiledGraph>,
        lookups: Box<[CompiledGraph]>,
        lookup_inputs: Box<[CompiledGraph]>,
        lookup_tables: Box<[CompiledGraph]>,
        trash: Box<[CompiledGraph]>,
    ) -> Result<Self, GraphFamilyValidationError> {
        Self::try_new_with_required_columns(
            custom,
            lookups,
            lookup_inputs,
            lookup_tables,
            trash,
            GraphColumnRequirements::default(),
        )
    }

    /// Packs classified graphs together with non-graph column requirements.
    pub fn try_new_with_required_columns(
        custom: Option<CompiledGraph>,
        lookups: Box<[CompiledGraph]>,
        lookup_inputs: Box<[CompiledGraph]>,
        lookup_tables: Box<[CompiledGraph]>,
        trash: Box<[CompiledGraph]>,
        required: GraphColumnRequirements<'_>,
    ) -> Result<Self, GraphFamilyValidationError> {
        if lookups.len() != lookup_inputs.len() || lookups.len() != lookup_tables.len() {
            return Err(GraphFamilyValidationError::LookupSectionCountMismatch {
                lookups: lookups.len(),
                lookup_inputs: lookup_inputs.len(),
                lookup_tables: lookup_tables.len(),
            });
        }

        let custom_count = u32::from(custom.is_some());
        let lookup_count = family_abi_len("lookup graphs", lookups.len())?;
        let trash_count = family_abi_len("trash graphs", trash.len())?;
        let total = usize::try_from(custom_count)
            .ok()
            .and_then(|count| count.checked_add(lookups.len()))
            .and_then(|count| count.checked_add(lookup_inputs.len()))
            .and_then(|count| count.checked_add(lookup_tables.len()))
            .and_then(|count| count.checked_add(trash.len()))
            .ok_or(GraphFamilyValidationError::ArrayLengthExceedsAbi {
                array: "graphs",
                len: usize::MAX,
            })?;
        family_abi_len("graphs", total)?;

        let mut graphs = Vec::with_capacity(total);
        graphs.extend(custom);
        graphs.extend(Vec::from(lookups));
        graphs.extend(Vec::from(lookup_inputs));
        graphs.extend(Vec::from(lookup_tables));
        graphs.extend(Vec::from(trash));

        let packed = CompiledGraphFamily::try_new_with_required_columns(
            graphs.into_boxed_slice(),
            required,
        )?;
        let sections = GraphFamilySections::new(
            custom_count,
            lookup_count,
            lookup_count,
            lookup_count,
            trash_count,
        );
        debug_assert_eq!(
            sections
                .checked_total()
                .and_then(|count| usize::try_from(count).ok()),
            Some(packed.descriptors().len())
        );
        Ok(Self { packed, sections })
    }

    /// Canonically compacted arrays in derived protocol order.
    pub const fn packed(&self) -> &CompiledGraphFamily {
        &self.packed
    }

    /// Derived descriptor counts for each protocol category.
    pub const fn sections(&self) -> GraphFamilySections {
        self.sections
    }
}

fn family_abi_len(array: &'static str, len: usize) -> Result<u32, GraphFamilyValidationError> {
    u32::try_from(len).map_err(|_| GraphFamilyValidationError::ArrayLengthExceedsAbi { array, len })
}

fn family_packed_range(
    array: &'static str,
    offset: usize,
    count: usize,
) -> Result<(u32, u32), GraphFamilyValidationError> {
    let len = offset.saturating_add(count);
    if u32::try_from(len).is_err() {
        return Err(GraphFamilyValidationError::ArrayLengthExceedsAbi { array, len });
    }
    let offset = family_abi_len(array, offset)?;
    let count = family_abi_len(array, count)?;
    Ok((offset, count))
}

fn note_polynomial_column(
    source: GraphValueRef,
    fixed: &mut std::collections::BTreeSet<u32>,
    advice: &mut std::collections::BTreeSet<u32>,
    instance: &mut std::collections::BTreeSet<u32>,
) {
    match GraphValueKind::try_from(source.kind) {
        Ok(GraphValueKind::Fixed) => {
            fixed.insert(source.a);
        }
        Ok(GraphValueKind::Advice) => {
            advice.insert(source.a);
        }
        Ok(GraphValueKind::Instance) => {
            instance.insert(source.a);
        }
        Ok(
            GraphValueKind::Constant
            | GraphValueKind::Intermediate
            | GraphValueKind::Challenge
            | GraphValueKind::Beta
            | GraphValueKind::Gamma
            | GraphValueKind::Theta
            | GraphValueKind::TrashChallenge
            | GraphValueKind::Y
            | GraphValueKind::PreviousValue,
        )
        | Err(()) => {}
    }
}

fn remap_polynomial_column(
    source: &mut GraphValueRef,
    fixed: &[u32],
    advice: &[u32],
    instance: &[u32],
) -> Result<(), GraphFamilyValidationError> {
    let (kind, map) = match GraphValueKind::try_from(source.kind) {
        Ok(GraphValueKind::Fixed) => (GraphValueKind::Fixed, fixed),
        Ok(GraphValueKind::Advice) => (GraphValueKind::Advice, advice),
        Ok(GraphValueKind::Instance) => (GraphValueKind::Instance, instance),
        _ => return Ok(()),
    };
    let logical_column = source.a;
    let compact = map.binary_search(&logical_column).map_err(|_| {
        GraphFamilyValidationError::MissingCompactionEntry {
            kind,
            logical_column,
        }
    })?;
    source.a =
        u32::try_from(compact).map_err(|_| GraphFamilyValidationError::ArrayLengthExceedsAbi {
            array: match kind {
                GraphValueKind::Fixed => "fixed_compact_to_logical",
                GraphValueKind::Advice => "advice_compact_to_logical",
                GraphValueKind::Instance => "instance_compact_to_logical",
                _ => unreachable!("only polynomial kinds enter compaction"),
            },
            len: map.len(),
        })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn graph_family_fingerprint(
    descriptors: &[CompiledGraphDescriptor],
    constants: &[RawFr],
    rotations: &[i32],
    operations: &[GraphOp],
    horner_parts: &[GraphValueRef],
    fixed_compact_to_logical: &[u32],
    advice_compact_to_logical: &[u32],
    instance_compact_to_logical: &[u32],
    challenge_bound: u32,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(GRAPH_FAMILY_FINGERPRINT_DOMAIN);
    fingerprint_u32(&mut hasher, GRAPH_FAMILY_FINGERPRINT_VERSION);

    fingerprint_u32_slice(&mut hasher, fixed_compact_to_logical);
    fingerprint_u32_slice(&mut hasher, advice_compact_to_logical);
    fingerprint_u32_slice(&mut hasher, instance_compact_to_logical);
    fingerprint_u32(&mut hasher, challenge_bound);

    fingerprint_len(&mut hasher, descriptors.len());
    for descriptor in descriptors {
        fingerprint_u32(&mut hasher, descriptor.constants_offset);
        fingerprint_u32(&mut hasher, descriptor.constants_count);
        fingerprint_u32(&mut hasher, descriptor.rotations_offset);
        fingerprint_u32(&mut hasher, descriptor.rotations_count);
        fingerprint_u32(&mut hasher, descriptor.operations_offset);
        fingerprint_u32(&mut hasher, descriptor.operations_count);
        fingerprint_u32(&mut hasher, descriptor.parts_offset);
        fingerprint_u32(&mut hasher, descriptor.parts_count);
        fingerprint_u32(&mut hasher, descriptor.scratch_class as u32);
    }

    fingerprint_len(&mut hasher, constants.len());
    for constant in constants {
        for limb in constant.limbs {
            hasher.update(limb.to_le_bytes());
        }
    }

    fingerprint_len(&mut hasher, rotations.len());
    for rotation in rotations {
        hasher.update(rotation.to_le_bytes());
    }

    fingerprint_len(&mut hasher, operations.len());
    for operation in operations {
        fingerprint_u32(&mut hasher, operation.opcode);
        fingerprint_u32(&mut hasher, operation.target);
        fingerprint_u32(&mut hasher, operation.parts_offset);
        fingerprint_u32(&mut hasher, operation.parts_len);
        fingerprint_value_ref(&mut hasher, operation.lhs);
        fingerprint_value_ref(&mut hasher, operation.rhs);
    }

    fingerprint_len(&mut hasher, horner_parts.len());
    for source in horner_parts {
        fingerprint_value_ref(&mut hasher, *source);
    }

    hasher.finalize().into()
}

fn fingerprint_len(hasher: &mut Sha256, len: usize) {
    let len = u32::try_from(len).expect("family constructor checked every serialized length");
    fingerprint_u32(hasher, len);
}

fn fingerprint_u32_slice(hasher: &mut Sha256, values: &[u32]) {
    fingerprint_len(hasher, values.len());
    for value in values {
        fingerprint_u32(hasher, *value);
    }
}

fn fingerprint_value_ref(hasher: &mut Sha256, source: GraphValueRef) {
    fingerprint_u32(hasher, source.kind);
    fingerprint_u32(hasher, source.a);
    fingerprint_u32(hasher, source.b);
}

fn fingerprint_u32(hasher: &mut Sha256, value: u32) {
    hasher.update(value.to_le_bytes());
}

fn checked_abi_len(array: &'static str, len: usize) -> Result<u32, GraphValidationError> {
    u32::try_from(len).map_err(|_| GraphValidationError::ArrayLengthExceedsAbi { array, len })
}

fn require_scratch_slot(
    slot: u32,
    required_scratch_slots: &mut u32,
) -> Result<(), GraphValidationError> {
    if slot >= MAX_GRAPH_SCRATCH_SLOTS {
        return Err(GraphValidationError::ScratchCapacityExceeded {
            required: u64::from(slot) + 1,
        });
    }
    *required_scratch_slots = (*required_scratch_slots).max(slot + 1);
    Ok(())
}

fn require_no_parts(operation_index: u32, operation: &GraphOp) -> Result<(), GraphValidationError> {
    if operation.parts_offset != 0 || operation.parts_len != 0 {
        return Err(GraphValidationError::UnexpectedHornerRange {
            operation: operation_index,
            offset: operation.parts_offset,
            len: operation.parts_len,
        });
    }
    Ok(())
}

fn require_unused_operand_zero(
    operation_index: u32,
    source: GraphValueRef,
) -> Result<(), GraphValidationError> {
    if source != GraphValueRef::default() {
        return Err(GraphValidationError::NonCanonicalUnusedOperand {
            operation: operation_index,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_source(
    source: GraphValueRef,
    operation_index: u32,
    constants_len: u32,
    rotations_len: u32,
    source_counts: GraphSourceCounts,
    initialized: &[bool; MAX_GRAPH_SCRATCH_SLOTS as usize],
    required_scratch_slots: &mut u32,
) -> Result<(), GraphValidationError> {
    let kind = GraphValueKind::try_from(source.kind).map_err(|()| {
        GraphValidationError::UnknownValueKind {
            operation: operation_index,
            kind: source.kind,
        }
    })?;

    match kind {
        GraphValueKind::Constant => {
            require_zero_b(source, operation_index)?;
            require_source_index(source, operation_index, constants_len)?;
        }
        GraphValueKind::Intermediate => {
            require_zero_b(source, operation_index)?;
            require_scratch_slot(source.a, required_scratch_slots)?;
            if !initialized[source.a as usize] {
                return Err(GraphValidationError::IntermediateReadBeforeWrite {
                    operation: operation_index,
                    slot: source.a,
                });
            }
        }
        GraphValueKind::Fixed => validate_polynomial_source(
            source,
            operation_index,
            source_counts.fixed_columns,
            rotations_len,
        )?,
        GraphValueKind::Advice => validate_polynomial_source(
            source,
            operation_index,
            source_counts.advice_columns,
            rotations_len,
        )?,
        GraphValueKind::Instance => validate_polynomial_source(
            source,
            operation_index,
            source_counts.instance_columns,
            rotations_len,
        )?,
        GraphValueKind::Challenge => {
            require_zero_b(source, operation_index)?;
            require_source_index(source, operation_index, source_counts.challenges)?;
        }
        GraphValueKind::Beta
        | GraphValueKind::Gamma
        | GraphValueKind::Theta
        | GraphValueKind::TrashChallenge
        | GraphValueKind::Y
        | GraphValueKind::PreviousValue => {
            if source.a != 0 || source.b != 0 {
                return Err(GraphValidationError::NonCanonicalValue {
                    operation: operation_index,
                    kind: source.kind,
                    a: source.a,
                    b: source.b,
                });
            }
        }
    }
    Ok(())
}

fn require_zero_b(source: GraphValueRef, operation_index: u32) -> Result<(), GraphValidationError> {
    if source.b != 0 {
        return Err(GraphValidationError::NonCanonicalValue {
            operation: operation_index,
            kind: source.kind,
            a: source.a,
            b: source.b,
        });
    }
    Ok(())
}

fn require_source_index(
    source: GraphValueRef,
    operation_index: u32,
    limit: u32,
) -> Result<(), GraphValidationError> {
    if source.a >= limit {
        return Err(GraphValidationError::SourceIndexOutOfBounds {
            operation: operation_index,
            kind: source.kind,
            index: source.a,
            limit,
        });
    }
    Ok(())
}

fn validate_polynomial_source(
    source: GraphValueRef,
    operation_index: u32,
    columns: u32,
    rotations: u32,
) -> Result<(), GraphValidationError> {
    require_source_index(source, operation_index, columns)?;
    if source.b >= rotations {
        return Err(GraphValidationError::RotationIndexOutOfBounds {
            operation: operation_index,
            index: source.b,
            limit: rotations,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    fn source(kind: GraphValueKind, a: u32, b: u32) -> GraphValueRef {
        GraphValueRef::new(kind, a, b)
    }

    fn store(target: u32, value: GraphValueRef) -> GraphOp {
        GraphOp::new(
            GraphOpcode::Store,
            target,
            0,
            0,
            value,
            GraphValueRef::default(),
        )
    }

    fn compile(
        operations: Vec<GraphOp>,
        parts: Vec<GraphValueRef>,
    ) -> Result<CompiledGraph, GraphValidationError> {
        CompiledGraph::try_new(
            vec![RawFr::default()].into_boxed_slice(),
            vec![-1].into_boxed_slice(),
            operations.into_boxed_slice(),
            parts.into_boxed_slice(),
            GraphSourceCounts::new(1, 1, 1, 1),
        )
    }

    fn raw_fr(first_limb: u64) -> RawFr {
        RawFr {
            limbs: [first_limb, first_limb + 1, first_limb + 2, first_limb + 3],
        }
    }

    fn sparse_family_graphs(first_constant: u64) -> Box<[CompiledGraph]> {
        let first = CompiledGraph::try_new(
            vec![raw_fr(first_constant)].into_boxed_slice(),
            vec![-3, 5].into_boxed_slice(),
            vec![
                store(0, source(GraphValueKind::Fixed, 9, 1)),
                GraphOp::new(
                    GraphOpcode::Add,
                    1,
                    0,
                    0,
                    source(GraphValueKind::Advice, 42, 0),
                    source(GraphValueKind::Fixed, 2, 0),
                ),
                GraphOp::new(
                    GraphOpcode::Horner,
                    2,
                    0,
                    2,
                    source(GraphValueKind::Intermediate, 1, 0),
                    source(GraphValueKind::Y, 0, 0),
                ),
            ]
            .into_boxed_slice(),
            vec![
                source(GraphValueKind::Instance, 100, 1),
                source(GraphValueKind::Challenge, 4, 0),
            ]
            .into_boxed_slice(),
            GraphSourceCounts::new(10, 43, 101, 5),
        )
        .expect("first sparse graph is valid");

        let second = CompiledGraph::try_new(
            vec![raw_fr(21)].into_boxed_slice(),
            vec![7].into_boxed_slice(),
            vec![
                store(0, source(GraphValueKind::Fixed, 2, 0)),
                GraphOp::new(
                    GraphOpcode::Add,
                    1,
                    0,
                    0,
                    source(GraphValueKind::Advice, 42, 0),
                    source(GraphValueKind::Instance, 3, 0),
                ),
                GraphOp::new(
                    GraphOpcode::Horner,
                    2,
                    0,
                    2,
                    source(GraphValueKind::Intermediate, 1, 0),
                    source(GraphValueKind::Y, 0, 0),
                ),
            ]
            .into_boxed_slice(),
            vec![
                source(GraphValueKind::Fixed, 9, 0),
                source(GraphValueKind::Challenge, 1, 0),
            ]
            .into_boxed_slice(),
            GraphSourceCounts::new(10, 43, 4, 2),
        )
        .expect("second sparse graph is valid");

        vec![first, second].into_boxed_slice()
    }

    #[test]
    fn abi_layouts_are_frozen() {
        assert_eq!(size_of::<GraphValueRef>(), 12);
        assert_eq!(align_of::<GraphValueRef>(), 4);
        assert_eq!(offset_of!(GraphValueRef, kind), 0);
        assert_eq!(offset_of!(GraphValueRef, a), 4);
        assert_eq!(offset_of!(GraphValueRef, b), 8);

        assert_eq!(size_of::<GraphOp>(), 40);
        assert_eq!(align_of::<GraphOp>(), 4);
        assert_eq!(offset_of!(GraphOp, opcode), 0);
        assert_eq!(offset_of!(GraphOp, target), 4);
        assert_eq!(offset_of!(GraphOp, parts_offset), 8);
        assert_eq!(offset_of!(GraphOp, parts_len), 12);
        assert_eq!(offset_of!(GraphOp, lhs), 16);
        assert_eq!(offset_of!(GraphOp, rhs), 28);

        assert_eq!(size_of::<CompiledGraphDescriptor>(), 36);
        assert_eq!(align_of::<CompiledGraphDescriptor>(), 4);
        assert_eq!(offset_of!(CompiledGraphDescriptor, constants_offset), 0);
        assert_eq!(offset_of!(CompiledGraphDescriptor, constants_count), 4);
        assert_eq!(offset_of!(CompiledGraphDescriptor, rotations_offset), 8);
        assert_eq!(offset_of!(CompiledGraphDescriptor, rotations_count), 12);
        assert_eq!(offset_of!(CompiledGraphDescriptor, operations_offset), 16);
        assert_eq!(offset_of!(CompiledGraphDescriptor, operations_count), 20);
        assert_eq!(offset_of!(CompiledGraphDescriptor, parts_offset), 24);
        assert_eq!(offset_of!(CompiledGraphDescriptor, parts_count), 28);
        assert_eq!(offset_of!(CompiledGraphDescriptor, scratch_class), 32);
    }

    #[test]
    fn wire_tags_match_the_cuda_oracle() {
        assert_eq!(GraphValueKind::Constant as u32, 0);
        assert_eq!(GraphValueKind::Intermediate as u32, 1);
        assert_eq!(GraphValueKind::Fixed as u32, 2);
        assert_eq!(GraphValueKind::Advice as u32, 3);
        assert_eq!(GraphValueKind::Instance as u32, 4);
        assert_eq!(GraphValueKind::Challenge as u32, 5);
        assert_eq!(GraphValueKind::Beta as u32, 6);
        assert_eq!(GraphValueKind::Gamma as u32, 7);
        assert_eq!(GraphValueKind::Theta as u32, 8);
        assert_eq!(GraphValueKind::TrashChallenge as u32, 9);
        assert_eq!(GraphValueKind::Y as u32, 10);
        assert_eq!(GraphValueKind::PreviousValue as u32, 11);

        assert_eq!(GraphOpcode::Add as u32, 0);
        assert_eq!(GraphOpcode::Sub as u32, 1);
        assert_eq!(GraphOpcode::Mul as u32, 2);
        assert_eq!(GraphOpcode::Square as u32, 3);
        assert_eq!(GraphOpcode::Double as u32, 4);
        assert_eq!(GraphOpcode::Negate as u32, 5);
        assert_eq!(GraphOpcode::Horner as u32, 6);
        assert_eq!(GraphOpcode::Store as u32, 7);
    }

    #[test]
    fn validates_and_owns_a_packed_graph() {
        let constant = source(GraphValueKind::Constant, 0, 0);
        let intermediate = source(GraphValueKind::Intermediate, 0, 0);
        let fixed = source(GraphValueKind::Fixed, 0, 0);
        let challenge = source(GraphValueKind::Challenge, 0, 0);
        let y = source(GraphValueKind::Y, 0, 0);
        let operations = vec![
            store(0, constant),
            GraphOp::new(GraphOpcode::Add, 0, 0, 0, intermediate, fixed),
            GraphOp::new(GraphOpcode::Horner, 1, 0, 1, intermediate, y),
        ];

        let graph = compile(operations, vec![challenge]).expect("valid packed graph");
        assert_eq!(graph.constants().len(), 1);
        assert_eq!(graph.rotations(), &[-1]);
        assert_eq!(graph.operations().len(), 3);
        assert_eq!(graph.horner_parts(), &[challenge]);
        assert_eq!(graph.source_counts(), GraphSourceCounts::new(1, 1, 1, 1));
        assert_eq!(graph.required_scratch_slots(), 2);
        assert_eq!(graph.scratch_class(), ScratchClass::Slots8);
        assert_eq!(graph.result_slot(), Some(1));
        assert!(!graph.is_empty());
    }

    #[test]
    fn empty_graph_is_valid_and_uses_smallest_pipeline() {
        let graph = compile(Vec::new(), Vec::new()).expect("empty graph is valid");
        assert!(graph.is_empty());
        assert_eq!(graph.result_slot(), None);
        assert_eq!(graph.required_scratch_slots(), 0);
        assert_eq!(graph.scratch_class(), ScratchClass::Slots8);
    }

    #[test]
    fn scratch_classes_cover_boundaries_and_reject_257() {
        for (required, expected) in [
            (0, ScratchClass::Slots8),
            (8, ScratchClass::Slots8),
            (9, ScratchClass::Slots16),
            (16, ScratchClass::Slots16),
            (17, ScratchClass::Slots32),
            (32, ScratchClass::Slots32),
            (33, ScratchClass::Slots64),
            (64, ScratchClass::Slots64),
            (65, ScratchClass::Slots128),
            (128, ScratchClass::Slots128),
            (129, ScratchClass::Slots256),
            (256, ScratchClass::Slots256),
        ] {
            assert_eq!(ScratchClass::for_required(required), Ok(expected));
            assert!(expected.capacity() >= required);
        }
        assert_eq!(
            ScratchClass::for_required(257),
            Err(GraphValidationError::ScratchCapacityExceeded { required: 257 })
        );
        assert_eq!(
            compile(
                vec![store(256, source(GraphValueKind::Constant, 0, 0))],
                Vec::new(),
            ),
            Err(GraphValidationError::ScratchCapacityExceeded { required: 257 })
        );
    }

    #[test]
    fn rejects_unknown_tags_and_noncanonical_payloads() {
        let mut operation = store(0, source(GraphValueKind::Constant, 0, 0));
        operation.opcode = 8;
        assert!(matches!(
            compile(vec![operation], Vec::new()),
            Err(GraphValidationError::UnknownOpcode { opcode: 8, .. })
        ));

        let malformed = GraphValueRef {
            kind: 12,
            a: 0,
            b: 0,
        };
        assert!(matches!(
            compile(vec![store(0, malformed)], Vec::new()),
            Err(GraphValidationError::UnknownValueKind { kind: 12, .. })
        ));
        assert!(matches!(
            compile(
                vec![store(0, source(GraphValueKind::Beta, 1, 0))],
                Vec::new(),
            ),
            Err(GraphValidationError::NonCanonicalValue { .. })
        ));
        assert!(matches!(
            compile(
                vec![store(0, source(GraphValueKind::Challenge, 0, 1))],
                Vec::new(),
            ),
            Err(GraphValidationError::NonCanonicalValue { .. })
        ));
    }

    #[test]
    fn rejects_source_rotation_and_forward_read_bounds() {
        assert!(matches!(
            compile(
                vec![store(0, source(GraphValueKind::Constant, 1, 0))],
                Vec::new(),
            ),
            Err(GraphValidationError::SourceIndexOutOfBounds { kind: 0, .. })
        ));
        assert!(matches!(
            compile(
                vec![store(0, source(GraphValueKind::Advice, 1, 0))],
                Vec::new(),
            ),
            Err(GraphValidationError::SourceIndexOutOfBounds { kind: 3, .. })
        ));
        assert!(matches!(
            compile(
                vec![store(0, source(GraphValueKind::Instance, 0, 1))],
                Vec::new(),
            ),
            Err(GraphValidationError::RotationIndexOutOfBounds { .. })
        ));
        assert_eq!(
            compile(
                vec![store(0, source(GraphValueKind::Intermediate, 0, 0))],
                Vec::new(),
            ),
            Err(GraphValidationError::IntermediateReadBeforeWrite {
                operation: 0,
                slot: 0,
            })
        );
    }

    #[test]
    fn enforces_exact_horner_packing_and_ranges() {
        let constant = source(GraphValueKind::Constant, 0, 0);
        let y = source(GraphValueKind::Y, 0, 0);
        let part = source(GraphValueKind::Beta, 0, 0);

        let non_horner_range = GraphOp::new(GraphOpcode::Add, 0, 0, 1, constant, constant);
        assert!(matches!(
            compile(vec![non_horner_range], vec![part]),
            Err(GraphValidationError::UnexpectedHornerRange { .. })
        ));

        let gap = GraphOp::new(GraphOpcode::Horner, 0, 1, 0, constant, y);
        assert!(matches!(
            compile(vec![gap], vec![part]),
            Err(GraphValidationError::NonContiguousHornerParts { .. })
        ));

        let overflow = GraphOp::new(GraphOpcode::Horner, 0, 0, u32::MAX, constant, y);
        assert!(matches!(
            compile(vec![overflow], Vec::new()),
            Err(GraphValidationError::HornerRangeOutOfBounds { .. })
        ));
        let overflow = GraphOp::new(GraphOpcode::Horner, 0, u32::MAX, 2, constant, y);
        assert!(matches!(
            compile(vec![overflow], Vec::new()),
            Err(GraphValidationError::HornerRangeOverflow { .. })
        ));

        assert_eq!(
            compile(Vec::new(), vec![part]),
            Err(GraphValidationError::UnreferencedHornerParts {
                first: 0,
                parts_len: 1,
            })
        );
    }

    #[test]
    fn rejects_nonzero_unused_operand() {
        let operation = GraphOp::new(
            GraphOpcode::Square,
            0,
            0,
            0,
            source(GraphValueKind::Constant, 0, 0),
            source(GraphValueKind::Y, 0, 0),
        );
        assert_eq!(
            compile(vec![operation], Vec::new()),
            Err(GraphValidationError::NonCanonicalUnusedOperand { operation: 0 })
        );
    }

    #[test]
    fn family_compacts_sparse_columns_key_wide_and_packs_exact_arrays() {
        let family =
            CompiledGraphFamily::try_new(sparse_family_graphs(11)).expect("sparse family is valid");

        assert_eq!(family.fixed_compact_to_logical(), &[2, 9]);
        assert_eq!(family.advice_compact_to_logical(), &[42]);
        assert_eq!(family.instance_compact_to_logical(), &[3, 100]);
        assert_eq!(family.challenge_bound(), 5);
        assert_eq!(family.constants(), &[raw_fr(11), raw_fr(21)]);
        assert_eq!(family.rotations(), &[-3, 5, 7]);
        assert_eq!(
            family.descriptors(),
            &[
                CompiledGraphDescriptor {
                    constants_offset: 0,
                    constants_count: 1,
                    rotations_offset: 0,
                    rotations_count: 2,
                    operations_offset: 0,
                    operations_count: 3,
                    parts_offset: 0,
                    parts_count: 2,
                    scratch_class: ScratchClass::Slots8,
                },
                CompiledGraphDescriptor {
                    constants_offset: 1,
                    constants_count: 1,
                    rotations_offset: 2,
                    rotations_count: 1,
                    operations_offset: 3,
                    operations_count: 3,
                    parts_offset: 2,
                    parts_count: 2,
                    scratch_class: ScratchClass::Slots8,
                },
            ],
        );

        let expected_operations = vec![
            store(0, source(GraphValueKind::Fixed, 1, 1)),
            GraphOp::new(
                GraphOpcode::Add,
                1,
                0,
                0,
                source(GraphValueKind::Advice, 0, 0),
                source(GraphValueKind::Fixed, 0, 0),
            ),
            GraphOp::new(
                GraphOpcode::Horner,
                2,
                0,
                2,
                source(GraphValueKind::Intermediate, 1, 0),
                source(GraphValueKind::Y, 0, 0),
            ),
            store(0, source(GraphValueKind::Fixed, 0, 0)),
            GraphOp::new(
                GraphOpcode::Add,
                1,
                0,
                0,
                source(GraphValueKind::Advice, 0, 0),
                source(GraphValueKind::Instance, 0, 0),
            ),
            GraphOp::new(
                GraphOpcode::Horner,
                2,
                0,
                2,
                source(GraphValueKind::Intermediate, 1, 0),
                source(GraphValueKind::Y, 0, 0),
            ),
        ];
        assert_eq!(family.operations(), expected_operations);
        assert_eq!(
            family.horner_parts(),
            &[
                source(GraphValueKind::Instance, 1, 1),
                source(GraphValueKind::Challenge, 4, 0),
                source(GraphValueKind::Fixed, 1, 0),
                source(GraphValueKind::Challenge, 1, 0),
            ],
        );

        // Horner offsets remain relative to each descriptor's part range.
        assert_eq!(family.operations()[2].parts_offset, 0);
        assert_eq!(family.operations()[5].parts_offset, 0);
        // Reused logical columns have one map entry and one dense ID.
        assert_eq!(family.operations()[0].lhs.a, family.horner_parts()[2].a);
        assert_eq!(family.operations()[1].lhs.a, family.operations()[4].lhs.a);
    }

    #[test]
    fn family_fingerprint_is_deterministic_ordered_and_content_sensitive() {
        assert_eq!(GRAPH_FAMILY_FINGERPRINT_VERSION, 1);
        let graphs = sparse_family_graphs(11);
        let first = CompiledGraphFamily::try_new(graphs.clone()).expect("valid family");
        let again = CompiledGraphFamily::try_new(graphs).expect("valid family");
        assert_eq!(first.fingerprint(), again.fingerprint());

        let mut reversed = Vec::from(sparse_family_graphs(11));
        reversed.reverse();
        let reversed = CompiledGraphFamily::try_new(reversed.into_boxed_slice())
            .expect("reversed family is valid");
        assert_ne!(first.fingerprint(), reversed.fingerprint());

        let changed = CompiledGraphFamily::try_new(sparse_family_graphs(12))
            .expect("content-changed family is valid");
        assert_ne!(first.fingerprint(), changed.fingerprint());

        assert_eq!(
            first.fingerprint(),
            [
                103, 254, 42, 134, 202, 4, 131, 253, 200, 135, 213, 74, 175, 136, 255, 81, 7, 188,
                57, 93, 141, 66, 213, 225, 113, 170, 189, 46, 152, 182, 78, 115,
            ],
        );
    }

    #[test]
    fn family_includes_required_non_graph_columns_in_maps_and_identity() {
        let fixed = [77, 1, 2, 77];
        let advice = [8, 8];
        let instance = [200, 3];
        let required = GraphColumnRequirements::new(&fixed, &advice, &instance);
        let family =
            CompiledGraphFamily::try_new_with_required_columns(sparse_family_graphs(11), required)
                .expect("required columns extend a valid family");

        assert_eq!(family.fixed_compact_to_logical(), &[1, 2, 9, 77]);
        assert_eq!(family.advice_compact_to_logical(), &[8, 42]);
        assert_eq!(family.instance_compact_to_logical(), &[3, 100, 200]);
        assert_eq!(family.operations()[0].lhs.a, 2);
        assert_eq!(family.operations()[1].lhs.a, 1);
        assert_eq!(family.operations()[1].rhs.a, 1);
        assert_eq!(family.horner_parts()[0].a, 1);

        let reordered_fixed = [2, 77, 1];
        let reordered_advice = [8];
        let reordered_instance = [3, 200, 3];
        let reordered = CompiledGraphFamily::try_new_with_required_columns(
            sparse_family_graphs(11),
            GraphColumnRequirements::new(&reordered_fixed, &reordered_advice, &reordered_instance),
        )
        .expect("requirement order and duplicates are canonicalized");
        assert_eq!(family, reordered);

        let graph_only =
            CompiledGraphFamily::try_new(sparse_family_graphs(11)).expect("valid graph family");
        assert_ne!(family.fingerprint(), graph_only.fingerprint());
    }

    #[test]
    fn protocol_family_derives_section_boundaries_and_rejects_lookup_mismatch() {
        let graph = |seed| Vec::from(sparse_family_graphs(seed)).remove(0);
        let family = CompiledProtocolGraphFamily::try_new(
            Some(graph(11)),
            vec![graph(12)].into_boxed_slice(),
            vec![graph(13)].into_boxed_slice(),
            vec![graph(14)].into_boxed_slice(),
            vec![graph(15), graph(16)].into_boxed_slice(),
        )
        .expect("classified protocol family is valid");

        assert_eq!(family.sections(), GraphFamilySections::new(1, 1, 1, 1, 2));
        assert_eq!(family.packed().descriptors().len(), 6);
        assert_eq!(
            family.packed().constants(),
            &[
                raw_fr(11),
                raw_fr(12),
                raw_fr(13),
                raw_fr(14),
                raw_fr(15),
                raw_fr(16)
            ]
        );

        assert!(matches!(
            CompiledProtocolGraphFamily::try_new(
                None,
                vec![graph(21)].into_boxed_slice(),
                Vec::new().into_boxed_slice(),
                vec![graph(22)].into_boxed_slice(),
                Vec::new().into_boxed_slice(),
            ),
            Err(GraphFamilyValidationError::LookupSectionCountMismatch {
                lookups: 1,
                lookup_inputs: 0,
                lookup_tables: 1,
            })
        ));
    }

    #[test]
    fn family_accepts_empty_family_and_empty_graphs() {
        let empty_family = CompiledGraphFamily::try_new(Vec::new().into_boxed_slice())
            .expect("empty family is valid");
        assert!(empty_family.is_empty());
        assert!(empty_family.descriptors().is_empty());
        assert!(empty_family.constants().is_empty());
        assert!(empty_family.rotations().is_empty());
        assert!(empty_family.operations().is_empty());
        assert!(empty_family.horner_parts().is_empty());
        assert!(empty_family.fixed_compact_to_logical().is_empty());
        assert!(empty_family.advice_compact_to_logical().is_empty());
        assert!(empty_family.instance_compact_to_logical().is_empty());
        assert_eq!(empty_family.challenge_bound(), 0);

        let graph = CompiledGraph::try_new(
            Vec::new().into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            GraphSourceCounts::default(),
        )
        .expect("empty graph is valid");
        let empty_graph_family = CompiledGraphFamily::try_new(vec![graph].into_boxed_slice())
            .expect("family containing an empty graph is valid");
        assert!(!empty_graph_family.is_empty());
        assert_eq!(
            empty_graph_family.descriptors(),
            &[CompiledGraphDescriptor {
                constants_offset: 0,
                constants_count: 0,
                rotations_offset: 0,
                rotations_count: 0,
                operations_offset: 0,
                operations_count: 0,
                parts_offset: 0,
                parts_count: 0,
                scratch_class: ScratchClass::Slots8,
            }],
        );
        assert_ne!(empty_family.fingerprint(), empty_graph_family.fingerprint());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn abi_length_conversion_rejects_host_only_lengths() {
        let len = usize::try_from(u64::from(u32::MAX) + 1).expect("64-bit test target");
        assert_eq!(
            checked_abi_len("test", len),
            Err(GraphValidationError::ArrayLengthExceedsAbi { array: "test", len })
        );
    }
}
