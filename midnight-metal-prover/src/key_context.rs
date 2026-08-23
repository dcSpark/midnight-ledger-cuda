// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Pointer-free immutable-key input and validation shared by Metal platforms.

use std::hash::{Hash, Hasher};
use std::mem::size_of_val;

use sha2::{Digest, Sha256};

use crate::{Bls12_381Fr, MetalError, NttDomainSpec, generated};

pub(crate) mod v2;
#[cfg(test)]
pub(crate) use v2::tests::OwnedV2Input;
pub use v2::{
    METAL_KEY_CONTEXT_SCHEMA_VERSION_V2, MetalKeyContextInputV2, MetalResidentColumnRef,
    MetalResidentDomainIdentity,
};

/// Version of the immutable graph/fixed Metal key-context content schema.
pub const METAL_KEY_CONTEXT_SCHEMA_VERSION: u32 = 1;

const GRAPH_FAMILY_FINGERPRINT_VERSION: u32 = 1;
const GRAPH_FAMILY_FINGERPRINT_DOMAIN: &[u8] = b"midnight-gpu-prover/compiled-graph-family\0";
const KEY_CONTENT_FINGERPRINT_DOMAIN: &[u8] = b"midnight-metal-prover/graph-fixed-key-context\0";

const GRAPH_VALUE_CONSTANT: u32 = 0;
const GRAPH_VALUE_INTERMEDIATE: u32 = 1;
const GRAPH_VALUE_FIXED: u32 = 2;
const GRAPH_VALUE_ADVICE: u32 = 3;
const GRAPH_VALUE_INSTANCE: u32 = 4;
const GRAPH_VALUE_CHALLENGE: u32 = 5;
const GRAPH_VALUE_BETA: u32 = 6;
const GRAPH_VALUE_GAMMA: u32 = 7;
const GRAPH_VALUE_THETA: u32 = 8;
const GRAPH_VALUE_TRASH_CHALLENGE: u32 = 9;
const GRAPH_VALUE_Y: u32 = 10;
const GRAPH_VALUE_PREVIOUS: u32 = 11;

const GRAPH_OP_ADD: u32 = 0;
const GRAPH_OP_SUB: u32 = 1;
const GRAPH_OP_MUL: u32 = 2;
const GRAPH_OP_SQUARE: u32 = 3;
const GRAPH_OP_DOUBLE: u32 = 4;
const GRAPH_OP_NEGATE: u32 = 5;
const GRAPH_OP_HORNER: u32 = 6;
const GRAPH_OP_STORE: u32 = 7;

/// Pointer-free source operand consumed by a Metal expression graph.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct MetalGraphValueRef {
    /// Numeric source tag.
    pub kind: u32,
    /// Constant, intermediate, compact column, or challenge index.
    pub a: u32,
    /// Graph-local rotation index for polynomial sources; zero otherwise.
    pub b: u32,
}

/// Pointer-free expression operation consumed by Metal.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct MetalGraphOperation {
    /// Numeric operation tag.
    pub opcode: u32,
    /// Physical scratch slot receiving the result.
    pub target: u32,
    /// First graph-local Horner part.
    pub parts_offset: u32,
    /// Number of graph-local Horner parts.
    pub parts_len: u32,
    /// First operand or Horner initial value.
    pub lhs: MetalGraphValueRef,
    /// Second operand or Horner factor.
    pub rhs: MetalGraphValueRef,
}

/// Fixed-width packed-array location for one expression graph.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct MetalGraphDescriptor {
    /// First graph constant in the packed family array.
    pub constants_offset: u32,
    /// Number of graph-local constants.
    pub constants_count: u32,
    /// First graph rotation in the packed family array.
    pub rotations_offset: u32,
    /// Number of graph-local rotations.
    pub rotations_count: u32,
    /// First operation in the packed family array.
    pub operations_offset: u32,
    /// Number of graph-local operations.
    pub operations_count: u32,
    /// First Horner part in the packed family array.
    pub parts_offset: u32,
    /// Number of graph-local Horner parts.
    pub parts_count: u32,
    /// Required private scratch capacity: 8, 16, 32, 64, 128, or 256.
    pub scratch_capacity: u32,
}

/// Protocol categories represented by a packed graph family.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct MetalGraphFamilySections {
    /// Number of custom-gate graphs; currently zero or one.
    pub custom: u32,
    /// Number of lookup identity graphs.
    pub lookups: u32,
    /// Number of lookup-input compression graphs.
    pub lookup_inputs: u32,
    /// Number of lookup-table compression graphs.
    pub lookup_tables: u32,
    /// Number of trash-argument graphs.
    pub trash: u32,
}

/// Low-level borrowed arrays imported into an immutable Metal key context.
///
/// Fixed cosets are column-major in `fixed_compact_to_logical` order and each
/// column contains exactly `n_ext` elements. `lagrange_selectors` is the packed
/// sequence `l0 | l_last | l_active`, with three `n_ext`-element columns.
/// Shape and content are validated here, but this transport type cannot prove
/// that a caller assigned each graph to the correct protocol category.
/// Production proof code must construct the classified family through
/// `midnight-gpu-prover` and use its opaque preparation function; this raw type
/// is intentionally not re-exported by that proof-facing facade.
#[derive(Clone, Copy, Debug)]
pub struct MetalKeyContextInput<'a> {
    /// Base polynomial-domain length.
    pub n: u32,
    /// Extended polynomial-domain length.
    pub n_ext: u32,
    /// Version used by the neutral graph-family SHA-256 serialization.
    pub graph_family_fingerprint_version: u32,
    /// Neutral graph-family SHA-256 supplied by the facade.
    pub graph_family_fingerprint: [u8; 32],
    /// Protocol category counts for `descriptors`.
    pub sections: MetalGraphFamilySections,
    /// Graph descriptors in protocol order.
    pub descriptors: &'a [MetalGraphDescriptor],
    /// Packed graph constants in reduced Montgomery form.
    pub constants: &'a [Bls12_381Fr],
    /// Packed graph-local signed rotations.
    pub rotations: &'a [i32],
    /// Packed graph operations.
    pub operations: &'a [MetalGraphOperation],
    /// Packed graph-local Horner sources.
    pub horner_parts: &'a [MetalGraphValueRef],
    /// Dense fixed-column index to original logical column ID.
    pub fixed_compact_to_logical: &'a [u32],
    /// Dense advice-column index to original logical column ID.
    pub advice_compact_to_logical: &'a [u32],
    /// Dense instance-column index to original logical column ID.
    pub instance_compact_to_logical: &'a [u32],
    /// Exclusive bound for proof challenges used by any graph.
    pub challenge_bound: u32,
    /// Packed fixed extended-coset columns in compact-map order.
    pub fixed_cosets: &'a [Bls12_381Fr],
    /// Packed `l0`, `l_last`, and `l_active` extended-domain selectors.
    pub lagrange_selectors: &'a [Bls12_381Fr],
}

/// Complete cache identity for one immutable Metal graph/fixed key context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalKeyContextIdentity {
    schema_version: u32,
    content_fingerprint: [u8; 32],
    graph_family_fingerprint: [u8; 32],
    n: u32,
    n_ext: u32,
    graph_abi_version: u64,
    graph_abi_fingerprint: u64,
    field_layout_fingerprint: u64,
    field_constants_fingerprint: u64,
    metallib_fingerprint: [u8; 32],
    device_registry_id: u64,
    v2: Option<MetalKeyContextIdentityV2>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct MetalKeyContextIdentityV2 {
    domain_fingerprint: [u8; 32],
    prepared_key_fingerprint: [u8; 32],
    prepared_srs_fingerprint: [u8; 32],
    base_ntt_domain: NttDomainSpec,
    extended_ntt_domain: NttDomainSpec,
    ntt_abi_version: u64,
    ntt_abi_fingerprint: u64,
}

impl Hash for MetalKeyContextIdentity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Keep the exact field sequence emitted by the former derived `Hash`
        // for schema v1. That identity is already used by the warm cache and
        // remains logically byte-for-byte stable while v2 extends it.
        self.schema_version.hash(state);
        self.content_fingerprint.hash(state);
        self.graph_family_fingerprint.hash(state);
        self.n.hash(state);
        self.n_ext.hash(state);
        self.graph_abi_version.hash(state);
        self.graph_abi_fingerprint.hash(state);
        self.field_layout_fingerprint.hash(state);
        self.field_constants_fingerprint.hash(state);
        self.metallib_fingerprint.hash(state);
        self.device_registry_id.hash(state);
        if self.schema_version == METAL_KEY_CONTEXT_SCHEMA_VERSION_V2 {
            self.v2.hash(state);
        }
    }
}

impl MetalKeyContextIdentity {
    /// Immutable key-context schema version.
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// SHA-256 of every logical graph/fixed input field.
    pub const fn content_fingerprint(&self) -> [u8; 32] {
        self.content_fingerprint
    }

    /// Validated neutral graph-family SHA-256 component.
    pub const fn graph_family_fingerprint(&self) -> [u8; 32] {
        self.graph_family_fingerprint
    }

    /// Base polynomial-domain length.
    pub const fn n(&self) -> u32 {
        self.n
    }

    /// Extended polynomial-domain length.
    pub const fn n_ext(&self) -> u32 {
        self.n_ext
    }

    /// Expression shader ABI version.
    pub const fn graph_abi_version(&self) -> u64 {
        self.graph_abi_version
    }

    /// Pinned expression shader ABI fingerprint.
    pub const fn graph_abi_fingerprint(&self) -> u64 {
        self.graph_abi_fingerprint
    }

    /// Pinned BLS12-381 host/shader layout fingerprint.
    pub const fn field_layout_fingerprint(&self) -> u64 {
        self.field_layout_fingerprint
    }

    /// Generated BLS12-381 constant fingerprint.
    pub const fn field_constants_fingerprint(&self) -> u64 {
        self.field_constants_fingerprint
    }

    /// SHA-256 of the exact embedded Metal library bytes.
    pub const fn metallib_fingerprint(&self) -> [u8; 32] {
        self.metallib_fingerprint
    }

    /// Stable Metal registry identity, or zero when the driver supplies none.
    pub const fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }

    /// Verified neutral resident-domain fingerprint for schema v2.
    ///
    /// Schema-v1 graph/fixed contexts return `None`.
    pub const fn domain_fingerprint(&self) -> Option<[u8; 32]> {
        match &self.v2 {
            Some(v2) => Some(v2.domain_fingerprint),
            None => None,
        }
    }

    /// Verified neutral prepared-proving-key fingerprint for schema v2.
    ///
    /// Schema-v1 graph/fixed contexts return `None`.
    pub const fn prepared_key_fingerprint(&self) -> Option<[u8; 32]> {
        match &self.v2 {
            Some(v2) => Some(v2.prepared_key_fingerprint),
            None => None,
        }
    }

    /// Verified neutral prepared-SRS fingerprint for schema v2.
    ///
    /// Schema-v1 graph/fixed contexts return `None`.
    pub const fn prepared_srs_fingerprint(&self) -> Option<[u8; 32]> {
        match &self.v2 {
            Some(v2) => Some(v2.prepared_srs_fingerprint),
            None => None,
        }
    }

    /// Exact validated base-domain NTT parameters for schema v2.
    ///
    /// Schema-v1 graph/fixed contexts return `None` because v1 did not carry
    /// an authoritative domain specification.
    pub const fn base_ntt_domain(&self) -> Option<NttDomainSpec> {
        match &self.v2 {
            Some(v2) => Some(v2.base_ntt_domain),
            None => None,
        }
    }

    /// Exact validated extended-domain NTT parameters for schema v2.
    ///
    /// Schema-v1 graph/fixed contexts return `None` because v1 did not carry
    /// an authoritative domain specification.
    pub const fn extended_ntt_domain(&self) -> Option<NttDomainSpec> {
        match &self.v2 {
            Some(v2) => Some(v2.extended_ntt_domain),
            None => None,
        }
    }

    /// Pinned schema-v2 NTT parameter ABI version.
    pub const fn ntt_abi_version(&self) -> Option<u64> {
        match &self.v2 {
            Some(v2) => Some(v2.ntt_abi_version),
            None => None,
        }
    }

    /// Pinned schema-v2 NTT parameter ABI fingerprint.
    pub const fn ntt_abi_fingerprint(&self) -> Option<u64> {
        match &self.v2 {
            Some(v2) => Some(v2.ntt_abi_fingerprint),
            None => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) const fn new(
        content_fingerprint: [u8; 32],
        graph_family_fingerprint: [u8; 32],
        n: u32,
        n_ext: u32,
        graph_abi_version: u64,
        graph_abi_fingerprint: u64,
        field_layout_fingerprint: u64,
        field_constants_fingerprint: u64,
        metallib_fingerprint: [u8; 32],
        device_registry_id: u64,
    ) -> Self {
        Self {
            schema_version: METAL_KEY_CONTEXT_SCHEMA_VERSION,
            content_fingerprint,
            graph_family_fingerprint,
            n,
            n_ext,
            graph_abi_version,
            graph_abi_fingerprint,
            field_layout_fingerprint,
            field_constants_fingerprint,
            metallib_fingerprint,
            device_registry_id,
            v2: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) const fn new_v2(
        content_fingerprint: [u8; 32],
        graph_family_fingerprint: [u8; 32],
        n: u32,
        n_ext: u32,
        graph_abi_version: u64,
        graph_abi_fingerprint: u64,
        field_layout_fingerprint: u64,
        field_constants_fingerprint: u64,
        metallib_fingerprint: [u8; 32],
        device_registry_id: u64,
        domain_fingerprint: [u8; 32],
        prepared_key_fingerprint: [u8; 32],
        prepared_srs_fingerprint: [u8; 32],
        base_ntt_domain: NttDomainSpec,
        extended_ntt_domain: NttDomainSpec,
        ntt_abi_version: u64,
        ntt_abi_fingerprint: u64,
    ) -> Self {
        Self {
            schema_version: METAL_KEY_CONTEXT_SCHEMA_VERSION_V2,
            content_fingerprint,
            graph_family_fingerprint,
            n,
            n_ext,
            graph_abi_version,
            graph_abi_fingerprint,
            field_layout_fingerprint,
            field_constants_fingerprint,
            metallib_fingerprint,
            device_registry_id,
            v2: Some(MetalKeyContextIdentityV2 {
                domain_fingerprint,
                prepared_key_fingerprint,
                prepared_srs_fingerprint,
                base_ntt_domain,
                extended_ntt_domain,
                ntt_abi_version,
                ntt_abi_fingerprint,
            }),
        }
    }
}

pub(crate) struct ValidatedMetalKeyContextInput<'a> {
    pub(crate) input: MetalKeyContextInput<'a>,
    pub(crate) content_fingerprint: [u8; 32],
    pub(crate) immutable_bytes: usize,
}

pub(crate) fn validate_metal_key_context_input(
    input: MetalKeyContextInput<'_>,
) -> Result<ValidatedMetalKeyContextInput<'_>, MetalError> {
    validate_domains(input.n, input.n_ext)?;
    validate_sections(input.sections, input.descriptors.len())?;
    validate_strict_map("fixed", input.fixed_compact_to_logical)?;
    validate_strict_map("advice", input.advice_compact_to_logical)?;
    validate_strict_map("instance", input.instance_compact_to_logical)?;
    validate_graph_family(&input)?;
    checked_u32_len("fixed cosets", input.fixed_cosets.len())?;
    checked_u32_len("Lagrange selectors", input.lagrange_selectors.len())?;

    validate_reduced("graph constant", input.constants)?;
    validate_reduced("fixed coset", input.fixed_cosets)?;
    validate_reduced("Lagrange selector", input.lagrange_selectors)?;

    let expected_fixed = input
        .fixed_compact_to_logical
        .len()
        .checked_mul(input.n_ext as usize)
        .ok_or_else(|| key_error("fixed-coset extent overflows usize"))?;
    if input.fixed_cosets.len() != expected_fixed {
        return Err(key_error(format!(
            "fixed cosets contain {} elements, expected {expected_fixed}",
            input.fixed_cosets.len()
        )));
    }
    let expected_selectors = 3usize
        .checked_mul(input.n_ext as usize)
        .ok_or_else(|| key_error("Lagrange-selector extent overflows usize"))?;
    if input.lagrange_selectors.len() != expected_selectors {
        return Err(key_error(format!(
            "Lagrange selectors contain {} elements, expected {expected_selectors}",
            input.lagrange_selectors.len()
        )));
    }

    if input.graph_family_fingerprint_version != GRAPH_FAMILY_FINGERPRINT_VERSION {
        return Err(key_error(format!(
            "unsupported neutral graph-family fingerprint version {}",
            input.graph_family_fingerprint_version
        )));
    }
    let actual_graph_fingerprint = neutral_graph_family_fingerprint(&input);
    if actual_graph_fingerprint != input.graph_family_fingerprint {
        return Err(key_error(
            "neutral graph-family fingerprint does not match the supplied arrays",
        ));
    }

    let immutable_bytes = immutable_bytes(&input)?;
    let content_fingerprint = key_content_fingerprint(&input);
    Ok(ValidatedMetalKeyContextInput {
        input,
        content_fingerprint,
        immutable_bytes,
    })
}

fn validate_domains(n: u32, n_ext: u32) -> Result<(), MetalError> {
    if !n.is_power_of_two() || !n_ext.is_power_of_two() {
        return Err(key_error("n and n_ext must be nonzero powers of two"));
    }
    if n_ext < n || !n_ext.is_multiple_of(n) {
        return Err(key_error(
            "n_ext must be an integer power-of-two extension of n",
        ));
    }
    let rotation_scale = n_ext / n;
    if i32::try_from(rotation_scale).is_err() {
        return Err(key_error("extended-domain rotation scale exceeds i32"));
    }
    Ok(())
}

fn validate_sections(
    sections: MetalGraphFamilySections,
    descriptor_count: usize,
) -> Result<(), MetalError> {
    if sections.custom > 1 {
        return Err(key_error(
            "a graph family may contain at most one custom graph",
        ));
    }
    if sections.lookups != sections.lookup_inputs || sections.lookups != sections.lookup_tables {
        return Err(key_error(
            "lookup, lookup-input, and lookup-table graph counts must match",
        ));
    }
    let total = [
        sections.custom,
        sections.lookups,
        sections.lookup_inputs,
        sections.lookup_tables,
        sections.trash,
    ]
    .into_iter()
    .try_fold(0u32, u32::checked_add)
    .ok_or_else(|| key_error("graph-family section count overflows u32"))?;
    if usize::try_from(total).ok() != Some(descriptor_count) {
        return Err(key_error(format!(
            "graph-family sections cover {total} descriptors, but {descriptor_count} were supplied"
        )));
    }
    Ok(())
}

fn validate_strict_map(label: &str, map: &[u32]) -> Result<(), MetalError> {
    if map.windows(2).any(|window| window[0] >= window[1]) {
        return Err(key_error(format!(
            "{label} compact-to-logical map is not strictly increasing"
        )));
    }
    checked_u32_len(label, map.len())?;
    Ok(())
}

fn validate_graph_family(input: &MetalKeyContextInput<'_>) -> Result<(), MetalError> {
    for (label, len) in [
        ("descriptors", input.descriptors.len()),
        ("constants", input.constants.len()),
        ("rotations", input.rotations.len()),
        ("operations", input.operations.len()),
        ("Horner parts", input.horner_parts.len()),
    ] {
        checked_u32_len(label, len)?;
    }

    let mut next_constants = 0usize;
    let mut next_rotations = 0usize;
    let mut next_operations = 0usize;
    let mut next_parts = 0usize;
    for (graph_index, descriptor) in input.descriptors.iter().copied().enumerate() {
        let constants = descriptor_window(
            graph_index,
            "constants",
            descriptor.constants_offset,
            descriptor.constants_count,
            next_constants,
            input.constants.len(),
        )?;
        next_constants = constants.end;
        let rotations = descriptor_window(
            graph_index,
            "rotations",
            descriptor.rotations_offset,
            descriptor.rotations_count,
            next_rotations,
            input.rotations.len(),
        )?;
        next_rotations = rotations.end;
        let operations = descriptor_window(
            graph_index,
            "operations",
            descriptor.operations_offset,
            descriptor.operations_count,
            next_operations,
            input.operations.len(),
        )?;
        next_operations = operations.end;
        let parts = descriptor_window(
            graph_index,
            "Horner parts",
            descriptor.parts_offset,
            descriptor.parts_count,
            next_parts,
            input.horner_parts.len(),
        )?;
        next_parts = parts.end;

        if !matches!(descriptor.scratch_capacity, 8 | 16 | 32 | 64 | 128 | 256) {
            return Err(key_error(format!(
                "graph {graph_index} has unsupported scratch capacity {}",
                descriptor.scratch_capacity
            )));
        }
        validate_graph(
            graph_index,
            descriptor,
            &input.operations[operations],
            &input.horner_parts[parts],
            input.fixed_compact_to_logical.len(),
            input.advice_compact_to_logical.len(),
            input.instance_compact_to_logical.len(),
            input.challenge_bound,
        )?;
    }

    for (label, consumed, available) in [
        ("constants", next_constants, input.constants.len()),
        ("rotations", next_rotations, input.rotations.len()),
        ("operations", next_operations, input.operations.len()),
        ("Horner parts", next_parts, input.horner_parts.len()),
    ] {
        if consumed != available {
            return Err(key_error(format!(
                "packed {label} contain unreferenced elements starting at {consumed}"
            )));
        }
    }
    Ok(())
}

fn descriptor_window(
    graph: usize,
    label: &str,
    offset: u32,
    count: u32,
    expected_offset: usize,
    available: usize,
) -> Result<std::ops::Range<usize>, MetalError> {
    let offset = offset as usize;
    let count = count as usize;
    if offset != expected_offset {
        return Err(key_error(format!(
            "graph {graph} {label} offset {offset} is not contiguous at {expected_offset}"
        )));
    }
    let end = offset
        .checked_add(count)
        .ok_or_else(|| key_error(format!("graph {graph} {label} window overflows usize")))?;
    if end > available {
        return Err(key_error(format!(
            "graph {graph} {label} window {offset}..{end} exceeds {available} elements"
        )));
    }
    Ok(offset..end)
}

#[allow(clippy::too_many_arguments)]
fn validate_graph(
    graph: usize,
    descriptor: MetalGraphDescriptor,
    operations: &[MetalGraphOperation],
    parts: &[MetalGraphValueRef],
    fixed_columns: usize,
    advice_columns: usize,
    instance_columns: usize,
    challenge_bound: u32,
) -> Result<(), MetalError> {
    let scratch_capacity = descriptor.scratch_capacity as usize;
    let required_scratch = operations
        .iter()
        .map(|operation| u64::from(operation.target) + 1)
        .max()
        .unwrap_or(0);
    let expected_scratch = match required_scratch {
        0..=8 => 8,
        9..=16 => 16,
        17..=32 => 32,
        33..=64 => 64,
        65..=128 => 128,
        129..=256 => 256,
        _ => {
            return Err(key_error(format!(
                "graph {graph} requires {required_scratch} scratch slots, exceeding 256"
            )));
        }
    };
    if descriptor.scratch_capacity != expected_scratch {
        return Err(key_error(format!(
            "graph {graph} uses scratch class {scratch_capacity}, expected smallest class {expected_scratch}"
        )));
    }
    let mut initialized = vec![false; scratch_capacity];
    let mut next_part = 0usize;

    for (operation_index, operation) in operations.iter().copied().enumerate() {
        if operation.target as usize >= scratch_capacity {
            return Err(key_error(format!(
                "graph {graph} operation {operation_index} target exceeds scratch capacity"
            )));
        }
        match operation.opcode {
            GRAPH_OP_ADD | GRAPH_OP_SUB | GRAPH_OP_MUL => {
                require_no_parts(graph, operation_index, operation)?;
                validate_value(
                    graph,
                    operation_index,
                    operation.lhs,
                    descriptor,
                    &initialized,
                    fixed_columns,
                    advice_columns,
                    instance_columns,
                    challenge_bound,
                )?;
                validate_value(
                    graph,
                    operation_index,
                    operation.rhs,
                    descriptor,
                    &initialized,
                    fixed_columns,
                    advice_columns,
                    instance_columns,
                    challenge_bound,
                )?;
            }
            GRAPH_OP_SQUARE | GRAPH_OP_DOUBLE | GRAPH_OP_NEGATE | GRAPH_OP_STORE => {
                require_no_parts(graph, operation_index, operation)?;
                if operation.rhs != MetalGraphValueRef::default() {
                    return Err(key_error(format!(
                        "graph {graph} operation {operation_index} has a noncanonical unused operand"
                    )));
                }
                validate_value(
                    graph,
                    operation_index,
                    operation.lhs,
                    descriptor,
                    &initialized,
                    fixed_columns,
                    advice_columns,
                    instance_columns,
                    challenge_bound,
                )?;
            }
            GRAPH_OP_HORNER => {
                if operation.parts_offset as usize != next_part {
                    return Err(key_error(format!(
                        "graph {graph} operation {operation_index} Horner parts are not contiguous"
                    )));
                }
                let end = (operation.parts_offset as usize)
                    .checked_add(operation.parts_len as usize)
                    .ok_or_else(|| {
                        key_error(format!(
                            "graph {graph} operation {operation_index} Horner range overflows"
                        ))
                    })?;
                if end > parts.len() {
                    return Err(key_error(format!(
                        "graph {graph} operation {operation_index} Horner range exceeds its graph"
                    )));
                }
                validate_value(
                    graph,
                    operation_index,
                    operation.lhs,
                    descriptor,
                    &initialized,
                    fixed_columns,
                    advice_columns,
                    instance_columns,
                    challenge_bound,
                )?;
                validate_value(
                    graph,
                    operation_index,
                    operation.rhs,
                    descriptor,
                    &initialized,
                    fixed_columns,
                    advice_columns,
                    instance_columns,
                    challenge_bound,
                )?;
                for part in &parts[next_part..end] {
                    validate_value(
                        graph,
                        operation_index,
                        *part,
                        descriptor,
                        &initialized,
                        fixed_columns,
                        advice_columns,
                        instance_columns,
                        challenge_bound,
                    )?;
                }
                next_part = end;
            }
            opcode => {
                return Err(key_error(format!(
                    "graph {graph} operation {operation_index} has unknown opcode {opcode}"
                )));
            }
        }
        initialized[operation.target as usize] = true;
    }
    if next_part != parts.len() {
        return Err(key_error(format!(
            "graph {graph} has unreferenced Horner parts starting at {next_part}"
        )));
    }
    Ok(())
}

fn require_no_parts(
    graph: usize,
    operation_index: usize,
    operation: MetalGraphOperation,
) -> Result<(), MetalError> {
    if operation.parts_offset != 0 || operation.parts_len != 0 {
        Err(key_error(format!(
            "graph {graph} operation {operation_index} has an unexpected Horner range"
        )))
    } else {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_value(
    graph: usize,
    operation: usize,
    value: MetalGraphValueRef,
    descriptor: MetalGraphDescriptor,
    initialized: &[bool],
    fixed_columns: usize,
    advice_columns: usize,
    instance_columns: usize,
    challenge_bound: u32,
) -> Result<(), MetalError> {
    let context = || format!("graph {graph} operation {operation} source");
    match value.kind {
        GRAPH_VALUE_CONSTANT => {
            require_zero_b(&context(), value)?;
            require_bound(
                &context(),
                "constant",
                value.a as usize,
                descriptor.constants_count as usize,
            )
        }
        GRAPH_VALUE_INTERMEDIATE => {
            require_zero_b(&context(), value)?;
            if initialized.get(value.a as usize) != Some(&true) {
                Err(key_error(format!(
                    "{} reads an uninitialized intermediate {}",
                    context(),
                    value.a
                )))
            } else {
                Ok(())
            }
        }
        GRAPH_VALUE_FIXED => {
            require_bound(&context(), "fixed column", value.a as usize, fixed_columns)?;
            require_bound(
                &context(),
                "rotation",
                value.b as usize,
                descriptor.rotations_count as usize,
            )
        }
        GRAPH_VALUE_ADVICE => {
            require_bound(
                &context(),
                "advice column",
                value.a as usize,
                advice_columns,
            )?;
            require_bound(
                &context(),
                "rotation",
                value.b as usize,
                descriptor.rotations_count as usize,
            )
        }
        GRAPH_VALUE_INSTANCE => {
            require_bound(
                &context(),
                "instance column",
                value.a as usize,
                instance_columns,
            )?;
            require_bound(
                &context(),
                "rotation",
                value.b as usize,
                descriptor.rotations_count as usize,
            )
        }
        GRAPH_VALUE_CHALLENGE => {
            require_zero_b(&context(), value)?;
            require_bound(
                &context(),
                "challenge",
                value.a as usize,
                challenge_bound as usize,
            )
        }
        GRAPH_VALUE_BETA
        | GRAPH_VALUE_GAMMA
        | GRAPH_VALUE_THETA
        | GRAPH_VALUE_TRASH_CHALLENGE
        | GRAPH_VALUE_Y
        | GRAPH_VALUE_PREVIOUS => {
            if value.a == 0 && value.b == 0 {
                Ok(())
            } else {
                Err(key_error(format!(
                    "{} has nonzero singleton payload ({}, {})",
                    context(),
                    value.a,
                    value.b
                )))
            }
        }
        kind => Err(key_error(format!(
            "{} has unknown source tag {kind}",
            context()
        ))),
    }
}

fn require_zero_b(context: &str, value: MetalGraphValueRef) -> Result<(), MetalError> {
    if value.b == 0 {
        Ok(())
    } else {
        Err(key_error(format!(
            "{context} has nonzero unused auxiliary field {}",
            value.b
        )))
    }
}

fn require_bound(context: &str, label: &str, index: usize, bound: usize) -> Result<(), MetalError> {
    if index < bound {
        Ok(())
    } else {
        Err(key_error(format!(
            "{context} {label} index {index} exceeds bound {bound}"
        )))
    }
}

fn checked_u32_len(label: &str, len: usize) -> Result<u32, MetalError> {
    u32::try_from(len).map_err(|_| key_error(format!("{label} length {len} exceeds u32")))
}

fn validate_reduced(label: &str, values: &[Bls12_381Fr]) -> Result<(), MetalError> {
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

fn greater_or_equal<const N: usize>(lhs: [u64; N], rhs: [u64; N]) -> bool {
    for index in (0..N).rev() {
        if lhs[index] != rhs[index] {
            return lhs[index] > rhs[index];
        }
    }
    true
}

fn immutable_bytes(input: &MetalKeyContextInput<'_>) -> Result<usize, MetalError> {
    [
        size_of_val(input.descriptors),
        size_of_val(input.constants),
        size_of_val(input.rotations),
        size_of_val(input.operations),
        size_of_val(input.horner_parts),
        size_of_val(input.fixed_compact_to_logical),
        size_of_val(input.advice_compact_to_logical),
        size_of_val(input.instance_compact_to_logical),
        size_of_val(input.fixed_cosets),
        size_of_val(input.lagrange_selectors),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or_else(|| key_error("immutable key-context byte count overflows usize"))
}

pub(crate) fn neutral_graph_family_fingerprint(input: &MetalKeyContextInput<'_>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(GRAPH_FAMILY_FINGERPRINT_DOMAIN);
    hash_u32(&mut hasher, input.graph_family_fingerprint_version);
    hash_u32_slice(&mut hasher, input.fixed_compact_to_logical);
    hash_u32_slice(&mut hasher, input.advice_compact_to_logical);
    hash_u32_slice(&mut hasher, input.instance_compact_to_logical);
    hash_u32(&mut hasher, input.challenge_bound);

    hash_len(&mut hasher, input.descriptors.len());
    for descriptor in input.descriptors {
        hash_descriptor(&mut hasher, *descriptor);
    }
    hash_len(&mut hasher, input.constants.len());
    for constant in input.constants {
        hash_fr(&mut hasher, *constant);
    }
    hash_len(&mut hasher, input.rotations.len());
    for rotation in input.rotations {
        hasher.update(rotation.to_le_bytes());
    }
    hash_len(&mut hasher, input.operations.len());
    for operation in input.operations {
        hash_operation(&mut hasher, *operation);
    }
    hash_len(&mut hasher, input.horner_parts.len());
    for part in input.horner_parts {
        hash_value(&mut hasher, *part);
    }
    hasher.finalize().into()
}

fn key_content_fingerprint(input: &MetalKeyContextInput<'_>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(KEY_CONTENT_FINGERPRINT_DOMAIN);
    hash_u32(&mut hasher, METAL_KEY_CONTEXT_SCHEMA_VERSION);
    hash_u32(&mut hasher, input.n);
    hash_u32(&mut hasher, input.n_ext);
    hash_u32(&mut hasher, input.graph_family_fingerprint_version);
    hasher.update(input.graph_family_fingerprint);
    for count in [
        input.sections.custom,
        input.sections.lookups,
        input.sections.lookup_inputs,
        input.sections.lookup_tables,
        input.sections.trash,
    ] {
        hash_u32(&mut hasher, count);
    }

    hash_len(&mut hasher, input.descriptors.len());
    for descriptor in input.descriptors {
        hash_descriptor(&mut hasher, *descriptor);
    }
    hash_len(&mut hasher, input.constants.len());
    for constant in input.constants {
        hash_fr(&mut hasher, *constant);
    }
    hash_len(&mut hasher, input.rotations.len());
    for rotation in input.rotations {
        hasher.update(rotation.to_le_bytes());
    }
    hash_len(&mut hasher, input.operations.len());
    for operation in input.operations {
        hash_operation(&mut hasher, *operation);
    }
    hash_len(&mut hasher, input.horner_parts.len());
    for part in input.horner_parts {
        hash_value(&mut hasher, *part);
    }
    hash_u32_slice(&mut hasher, input.fixed_compact_to_logical);
    hash_u32_slice(&mut hasher, input.advice_compact_to_logical);
    hash_u32_slice(&mut hasher, input.instance_compact_to_logical);
    hash_u32(&mut hasher, input.challenge_bound);
    hash_len(&mut hasher, input.fixed_cosets.len());
    for value in input.fixed_cosets {
        hash_fr(&mut hasher, *value);
    }
    hash_len(&mut hasher, input.lagrange_selectors.len());
    for value in input.lagrange_selectors {
        hash_fr(&mut hasher, *value);
    }
    hasher.finalize().into()
}

fn hash_descriptor(hasher: &mut Sha256, descriptor: MetalGraphDescriptor) {
    for value in [
        descriptor.constants_offset,
        descriptor.constants_count,
        descriptor.rotations_offset,
        descriptor.rotations_count,
        descriptor.operations_offset,
        descriptor.operations_count,
        descriptor.parts_offset,
        descriptor.parts_count,
        descriptor.scratch_capacity,
    ] {
        hash_u32(hasher, value);
    }
}

fn hash_operation(hasher: &mut Sha256, operation: MetalGraphOperation) {
    for value in [
        operation.opcode,
        operation.target,
        operation.parts_offset,
        operation.parts_len,
    ] {
        hash_u32(hasher, value);
    }
    hash_value(hasher, operation.lhs);
    hash_value(hasher, operation.rhs);
}

fn hash_value(hasher: &mut Sha256, value: MetalGraphValueRef) {
    hash_u32(hasher, value.kind);
    hash_u32(hasher, value.a);
    hash_u32(hasher, value.b);
}

fn hash_fr(hasher: &mut Sha256, value: Bls12_381Fr) {
    for limb in value.limbs {
        hasher.update(limb.to_le_bytes());
    }
}

fn hash_u32_slice(hasher: &mut Sha256, values: &[u32]) {
    hash_len(hasher, values.len());
    for value in values {
        hash_u32(hasher, *value);
    }
}

fn hash_len(hasher: &mut Sha256, len: usize) {
    hash_u32(
        hasher,
        u32::try_from(len).expect("validated key-context array length fits u32"),
    );
}

fn hash_u32(hasher: &mut Sha256, value: u32) {
    hasher.update(value.to_le_bytes());
}

fn key_error(reason: impl Into<String>) -> MetalError {
    MetalError::KeyContext(reason.into())
}

#[cfg(test)]
pub(crate) struct OwnedMetalKeyContextInput {
    pub(crate) n: u32,
    pub(crate) n_ext: u32,
    pub(crate) sections: MetalGraphFamilySections,
    pub(crate) descriptors: Vec<MetalGraphDescriptor>,
    pub(crate) constants: Vec<Bls12_381Fr>,
    pub(crate) rotations: Vec<i32>,
    pub(crate) operations: Vec<MetalGraphOperation>,
    pub(crate) horner_parts: Vec<MetalGraphValueRef>,
    pub(crate) fixed_map: Vec<u32>,
    pub(crate) advice_map: Vec<u32>,
    pub(crate) instance_map: Vec<u32>,
    pub(crate) challenge_bound: u32,
    pub(crate) fixed_cosets: Vec<Bls12_381Fr>,
    pub(crate) selectors: Vec<Bls12_381Fr>,
}

#[cfg(test)]
impl OwnedMetalKeyContextInput {
    pub(crate) fn representative() -> Self {
        let one = Bls12_381Fr {
            limbs: generated::FR_R,
        };
        Self {
            n: 2,
            n_ext: 4,
            sections: MetalGraphFamilySections {
                custom: 1,
                ..MetalGraphFamilySections::default()
            },
            descriptors: vec![MetalGraphDescriptor {
                constants_count: 1,
                rotations_count: 1,
                operations_count: 1,
                scratch_capacity: 8,
                ..MetalGraphDescriptor::default()
            }],
            constants: vec![one],
            rotations: vec![0],
            operations: vec![MetalGraphOperation {
                opcode: GRAPH_OP_STORE,
                target: 0,
                lhs: MetalGraphValueRef {
                    kind: GRAPH_VALUE_CONSTANT,
                    a: 0,
                    b: 0,
                },
                ..MetalGraphOperation::default()
            }],
            horner_parts: Vec::new(),
            fixed_map: vec![2],
            advice_map: vec![7],
            instance_map: vec![11],
            challenge_bound: 0,
            fixed_cosets: vec![one; 4],
            selectors: vec![one; 12],
        }
    }

    pub(crate) fn input(&self) -> MetalKeyContextInput<'_> {
        let mut input = MetalKeyContextInput {
            n: self.n,
            n_ext: self.n_ext,
            graph_family_fingerprint_version: GRAPH_FAMILY_FINGERPRINT_VERSION,
            graph_family_fingerprint: [0; 32],
            sections: self.sections,
            descriptors: &self.descriptors,
            constants: &self.constants,
            rotations: &self.rotations,
            operations: &self.operations,
            horner_parts: &self.horner_parts,
            fixed_compact_to_logical: &self.fixed_map,
            advice_compact_to_logical: &self.advice_map,
            instance_compact_to_logical: &self.instance_map,
            challenge_bound: self.challenge_bound,
            fixed_cosets: &self.fixed_cosets,
            lagrange_selectors: &self.selectors,
        };
        input.graph_family_fingerprint = neutral_graph_family_fingerprint(&input);
        input
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of, size_of};

    use super::*;

    #[test]
    fn public_graph_layouts_match_the_shader_abi() {
        assert_eq!(size_of::<MetalGraphValueRef>(), 12);
        assert_eq!(align_of::<MetalGraphValueRef>(), 4);
        assert_eq!(offset_of!(MetalGraphValueRef, kind), 0);
        assert_eq!(offset_of!(MetalGraphValueRef, a), 4);
        assert_eq!(offset_of!(MetalGraphValueRef, b), 8);

        assert_eq!(size_of::<MetalGraphOperation>(), 40);
        assert_eq!(align_of::<MetalGraphOperation>(), 4);
        assert_eq!(offset_of!(MetalGraphOperation, lhs), 16);
        assert_eq!(offset_of!(MetalGraphOperation, rhs), 28);

        assert_eq!(size_of::<MetalGraphDescriptor>(), 36);
        assert_eq!(align_of::<MetalGraphDescriptor>(), 4);
        assert_eq!(offset_of!(MetalGraphDescriptor, scratch_capacity), 32);
        assert_eq!(size_of::<MetalGraphFamilySections>(), 20);
    }

    #[test]
    fn representative_input_validates_and_binds_content_and_domain() {
        let fixture = OwnedMetalKeyContextInput::representative();
        let input = fixture.input();
        let validated = validate_metal_key_context_input(input).expect("representative key input");
        assert_ne!(validated.content_fingerprint, [0; 32]);
        assert_eq!(
            validated.immutable_bytes,
            size_of_val(input.descriptors)
                + size_of_val(input.constants)
                + size_of_val(input.rotations)
                + size_of_val(input.operations)
                + size_of_val(input.horner_parts)
                + size_of_val(input.fixed_compact_to_logical)
                + size_of_val(input.advice_compact_to_logical)
                + size_of_val(input.instance_compact_to_logical)
                + size_of_val(input.fixed_cosets)
                + size_of_val(input.lagrange_selectors)
        );

        let mut larger_domain = OwnedMetalKeyContextInput::representative();
        larger_domain.n_ext = 8;
        larger_domain.fixed_cosets.resize(8, fixture.constants[0]);
        larger_domain.selectors.resize(24, fixture.constants[0]);
        let larger =
            validate_metal_key_context_input(larger_domain.input()).expect("larger domain input");
        assert_ne!(validated.content_fingerprint, larger.content_fingerprint);

        let mut changed = OwnedMetalKeyContextInput::representative();
        changed.fixed_cosets[0].limbs[0] ^= 1;
        let changed = validate_metal_key_context_input(changed.input())
            .expect("changed fixed content remains reduced");
        assert_ne!(validated.content_fingerprint, changed.content_fingerprint);
    }

    #[test]
    fn malformed_inputs_fail_before_platform_work() {
        let mut noncontiguous = OwnedMetalKeyContextInput::representative();
        noncontiguous.descriptors[0].constants_offset = 1;
        assert!(matches!(
            validate_metal_key_context_input(noncontiguous.input()),
            Err(MetalError::KeyContext(_))
        ));

        let mut unsorted = OwnedMetalKeyContextInput::representative();
        unsorted.fixed_map = vec![2, 2];
        unsorted.fixed_cosets.resize(8, unsorted.constants[0]);
        assert!(matches!(
            validate_metal_key_context_input(unsorted.input()),
            Err(MetalError::KeyContext(_))
        ));

        let mut bad_domain = OwnedMetalKeyContextInput::representative();
        bad_domain.n_ext = 3;
        assert!(matches!(
            validate_metal_key_context_input(bad_domain.input()),
            Err(MetalError::KeyContext(_))
        ));

        let mut oversized_scratch = OwnedMetalKeyContextInput::representative();
        oversized_scratch.descriptors[0].scratch_capacity = 16;
        assert!(matches!(
            validate_metal_key_context_input(oversized_scratch.input()),
            Err(MetalError::KeyContext(_))
        ));

        let mut mismatched_lookups = OwnedMetalKeyContextInput::representative();
        mismatched_lookups.sections = MetalGraphFamilySections {
            custom: 0,
            lookups: 1,
            ..MetalGraphFamilySections::default()
        };
        assert!(matches!(
            validate_metal_key_context_input(mismatched_lookups.input()),
            Err(MetalError::KeyContext(_))
        ));

        let fixture = OwnedMetalKeyContextInput::representative();
        let mut wrong_fingerprint = fixture.input();
        wrong_fingerprint.graph_family_fingerprint[0] ^= 1;
        assert!(matches!(
            validate_metal_key_context_input(wrong_fingerprint),
            Err(MetalError::KeyContext(_))
        ));
    }
}
