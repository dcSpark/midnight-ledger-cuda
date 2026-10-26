// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

//! Proving-key-specialized Metal expression kernels.
//!
//! Graph structure, constants, rotations, and compact column identifiers are
//! immutable proving-key data. Compiling that public structure once removes
//! opcode/value-tag interpretation and repeated rotation resolution from every
//! extended-domain row. Challenges and polynomial values remain runtime buffer
//! inputs and are never embedded in generated source.

use super::*;
use std::fmt::Write as _;

const FIELD_CONSTANTS: &str = include_str!(concat!(env!("OUT_DIR"), "/bls12_381_constants.metal"));
const FIELD_SOURCE: &str = include_str!("../../shaders/field.metal");

const VALUE_CONSTANT: u32 = 0;
const VALUE_INTERMEDIATE: u32 = 1;
const VALUE_FIXED: u32 = 2;
const VALUE_ADVICE: u32 = 3;
const VALUE_INSTANCE: u32 = 4;
const VALUE_CHALLENGE: u32 = 5;
const VALUE_BETA: u32 = 6;
const VALUE_GAMMA: u32 = 7;
const VALUE_THETA: u32 = 8;
const VALUE_TRASH_CHALLENGE: u32 = 9;
const VALUE_Y: u32 = 10;
const VALUE_PREVIOUS: u32 = 11;

const OP_ADD: u32 = 0;
const OP_SUB: u32 = 1;
const OP_MUL: u32 = 2;
const OP_SQUARE: u32 = 3;
const OP_DOUBLE: u32 = 4;
const OP_NEGATE: u32 = 5;
const OP_HORNER: u32 = 6;
const OP_STORE: u32 = 7;

#[derive(Clone, Copy)]
enum PostKind {
    Store,
    Lookup,
    Trash,
}

pub(super) struct SpecializedGraphPipelines {
    pipelines: Box<[Option<Pipeline>]>,
}

impl SpecializedGraphPipelines {
    pub(super) fn disabled(graph_count: usize) -> Self {
        Self {
            pipelines: (0..graph_count).map(|_| None).collect(),
        }
    }

    pub(super) fn pipeline(&self, graph_index: usize) -> Option<&Pipeline> {
        self.pipelines.get(graph_index).and_then(Option::as_ref)
    }
}

impl MetalSession {
    pub(super) fn compile_specialized_graph_pipelines(
        &self,
        input: MetalKeyContextInput<'_>,
    ) -> Result<SpecializedGraphPipelines, MetalError> {
        let post_kinds = specialized_post_kinds(input.sections, input.descriptors.len())?;
        if post_kinds.iter().all(Option::is_none) {
            return Ok(SpecializedGraphPipelines::disabled(input.descriptors.len()));
        }

        let mut source = specialized_source_prelude();
        let mut names = vec![None; input.descriptors.len()];
        for (graph_index, post_kind) in post_kinds.iter().copied().enumerate() {
            let Some(post_kind) = post_kind else {
                continue;
            };
            let name = format!("midnight_bls12_381_graph_specialized_{graph_index}");
            emit_graph_kernel(&mut source, &name, graph_index, post_kind, input)?;
            names[graph_index] = Some(name);
        }

        let source = NSString::from_str(&source);
        let library = self
            .device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|error| {
                MetalError::KeyContext(format!(
                    "specialized expression Metal compilation failed: {error:?}"
                ))
            })?;

        let mut pipelines = Vec::with_capacity(names.len());
        for name in names {
            let Some(name) = name else {
                pipelines.push(None);
                continue;
            };
            let function = library
                .newFunctionWithName(&NSString::from_str(&name))
                .ok_or_else(|| {
                    MetalError::KeyContext(format!(
                        "specialized expression function {name} is missing"
                    ))
                })?;
            let pipeline = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| {
                    MetalError::KeyContext(format!(
                        "specialized expression pipeline {name} failed: {error:?}"
                    ))
                })?;
            pipelines.push(Some(pipeline));
        }
        Ok(SpecializedGraphPipelines {
            pipelines: pipelines.into_boxed_slice(),
        })
    }
}

fn specialized_post_kinds(
    sections: MetalGraphFamilySections,
    graph_count: usize,
) -> Result<Vec<Option<PostKind>>, MetalError> {
    let mut kinds = vec![None; graph_count];
    let custom = sections.custom as usize;
    let lookups = sections.lookups as usize;
    let trash = sections.trash as usize;
    if custom == 1 {
        kinds[0] = Some(PostKind::Store);
    }
    kinds[custom..custom + lookups].fill(Some(PostKind::Lookup));
    let trash_start = custom
        .checked_add(lookups.checked_mul(3).ok_or_else(|| {
            MetalError::KeyContext("specialized graph lookup range overflows".into())
        })?)
        .ok_or_else(|| MetalError::KeyContext("specialized graph range overflows".into()))?;
    let trash_end = trash_start
        .checked_add(trash)
        .ok_or_else(|| MetalError::KeyContext("specialized trash range overflows".into()))?;
    if trash_end > graph_count {
        return Err(MetalError::KeyContext(
            "specialized graph categories exceed the descriptor array".into(),
        ));
    }
    kinds[trash_start..trash_end].fill(Some(PostKind::Trash));
    Ok(kinds)
}

fn specialized_source_prelude() -> String {
    let field = FIELD_SOURCE
        .replace("#include <metal_stdlib>\n", "")
        .replace("#include \"bls12_381_constants.metal\"\n", "");
    let mut source = String::from(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         #define MIDNIGHT_SPECIALIZED_GRAPH_ABI 1\n",
    );
    source.push_str(FIELD_CONSTANTS);
    source.push_str(&field);
    source.push_str(
        r#"
struct MidnightSpecializedGraphParameters {
    uint len;
    uint stride;
    int rotation_scale;
    uint post_kind;
    uint constants_offset;
    uint constants_count;
    uint rotations_offset;
    uint rotations_count;
    uint operations_offset;
    uint operations_count;
    uint parts_offset;
    uint parts_count;
    uint fixed_columns;
    uint advice_offset;
    uint advice_columns;
    uint instance_offset;
    uint instance_columns;
    uint challenge_count;
    uint previous_mode;
    uint lookup_product_column;
    uint lookup_input_column;
    uint lookup_table_column;
    uint trash_column;
    uint trash_selector_fixed_column;
    Bls12_381Fr beta;
    Bls12_381Fr gamma;
    Bls12_381Fr theta;
    Bls12_381Fr trash_challenge;
    Bls12_381Fr y;
};

static_assert(sizeof(MidnightSpecializedGraphParameters) == 256);

inline __attribute__((always_inline))
Bls12_381Fr midnight_specialized_zero()
{
    Bls12_381Fr result;
#pragma unroll
    for (uint index = 0U; index < 4U; ++index) {
        result.limbs[index] = 0UL;
    }
    return result;
}

inline __attribute__((always_inline))
Bls12_381Fr midnight_specialized_one()
{
    Bls12_381Fr result;
#pragma unroll
    for (uint index = 0U; index < 4U; ++index) {
        result.limbs[index] = MIDNIGHT_BLS12_381_FR_R[index];
    }
    return result;
}

inline __attribute__((always_inline))
Bls12_381Fr midnight_specialized_accumulate(
    Bls12_381Fr accumulator,
    Bls12_381Fr y,
    Bls12_381Fr term)
{
    return fr_add(fr_montgomery_mul(accumulator, y), term);
}
"#,
    );
    source
}

fn emit_graph_kernel(
    source: &mut String,
    name: &str,
    graph_index: usize,
    post_kind: PostKind,
    input: MetalKeyContextInput<'_>,
) -> Result<(), MetalError> {
    let descriptor = input.descriptors[graph_index];
    let operations = descriptor_window(
        "operation",
        input.operations,
        descriptor.operations_offset,
        descriptor.operations_count,
    )?;
    let rotations = descriptor_window(
        "rotation",
        input.rotations,
        descriptor.rotations_offset,
        descriptor.rotations_count,
    )?;

    writeln!(
        source,
        r#"
kernel void {name}(
    device const Bls12_381Fr* constants [[buffer(0)]],
    device const Bls12_381Fr* fixed_values [[buffer(4)]],
    device const Bls12_381Fr* dynamic_values [[buffer(5)]],
    device const Bls12_381Fr* challenges [[buffer(6)]],
    device const Bls12_381Fr* l_values [[buffer(7)]],
    device Bls12_381Fr* numerator [[buffer(8)]],
    constant MidnightSpecializedGraphParameters& parameters [[buffer(9)]],
    uint row [[thread_position_in_grid]])
{{
    if (row >= parameters.len) {{
        return;
    }}
    const Bls12_381Fr zero = midnight_specialized_zero();"#,
    )
    .map_err(format_error)?;

    for (rotation_index, rotation) in rotations.iter().copied().enumerate() {
        if rotation == 0 {
            writeln!(source, "    const uint rotated_{rotation_index} = row;")
                .map_err(format_error)?;
        } else {
            writeln!(
                source,
                "    const uint rotated_{rotation_index} = (row + static_cast<uint>(static_cast<long>({rotation}) * static_cast<long>(parameters.rotation_scale))) & (parameters.len - 1U);"
            )
            .map_err(format_error)?;
        }
    }

    for slot in 0..descriptor.scratch_capacity {
        writeln!(source, "    Bls12_381Fr v{slot};").map_err(format_error)?;
    }

    for (operation_index, operation) in operations.iter().copied().enumerate() {
        emit_operation(source, operation_index, operation, descriptor, input)?;
    }
    let graph_result = operations.last().map_or_else(
        || "zero".to_owned(),
        |operation| format!("v{}", operation.target),
    );
    writeln!(
        source,
        "    const Bls12_381Fr graph_result = {graph_result};"
    )
    .map_err(format_error)?;
    emit_post(source, post_kind)?;
    source.push_str("}\n");
    Ok(())
}

fn emit_operation(
    source: &mut String,
    operation_index: usize,
    operation: MetalGraphOperation,
    descriptor: MetalGraphDescriptor,
    input: MetalKeyContextInput<'_>,
) -> Result<(), MetalError> {
    let lhs = value_expression(operation.lhs, descriptor, input)?;
    let target = operation.target;
    match operation.opcode {
        OP_ADD | OP_SUB | OP_MUL => {
            let rhs = value_expression(operation.rhs, descriptor, input)?;
            let function = match operation.opcode {
                OP_ADD => "fr_add",
                OP_SUB => "fr_sub",
                OP_MUL => "fr_montgomery_mul",
                _ => unreachable!(),
            };
            writeln!(
                source,
                "    {{ const Bls12_381Fr result_{operation_index} = {function}({lhs}, {rhs}); v{target} = result_{operation_index}; }}"
            )
            .map_err(format_error)?;
        }
        OP_SQUARE => {
            writeln!(
                source,
                "    {{ const Bls12_381Fr operand_{operation_index} = {lhs}; v{target} = fr_montgomery_mul(operand_{operation_index}, operand_{operation_index}); }}"
            )
            .map_err(format_error)?;
        }
        OP_DOUBLE => {
            writeln!(source, "    v{target} = fr_add({lhs}, {lhs});").map_err(format_error)?;
        }
        OP_NEGATE => {
            writeln!(source, "    v{target} = fr_sub(zero, {lhs});").map_err(format_error)?;
        }
        OP_STORE => {
            writeln!(source, "    v{target} = {lhs};").map_err(format_error)?;
        }
        OP_HORNER => {
            let factor = value_expression(operation.rhs, descriptor, input)?;
            let parts = descriptor_window(
                "Horner part",
                input.horner_parts,
                descriptor
                    .parts_offset
                    .checked_add(operation.parts_offset)
                    .ok_or_else(|| {
                        MetalError::KeyContext("specialized Horner offset overflows".into())
                    })?,
                operation.parts_len,
            )?;
            writeln!(source, "    {{").map_err(format_error)?;
            writeln!(
                source,
                "        const Bls12_381Fr factor_{operation_index} = {factor};"
            )
            .map_err(format_error)?;
            writeln!(
                source,
                "        Bls12_381Fr result_{operation_index} = {lhs};"
            )
            .map_err(format_error)?;
            for part in parts.iter().copied() {
                let part = value_expression(part, descriptor, input)?;
                writeln!(
                    source,
                    "        result_{operation_index} = fr_add(fr_montgomery_mul(result_{operation_index}, factor_{operation_index}), {part});"
                )
                .map_err(format_error)?;
            }
            writeln!(
                source,
                "        v{target} = result_{operation_index};\n    }}"
            )
            .map_err(format_error)?;
        }
        opcode => {
            return Err(MetalError::KeyContext(format!(
                "specialized graph contains unsupported opcode {opcode}"
            )));
        }
    }
    Ok(())
}

fn value_expression(
    value: MetalGraphValueRef,
    descriptor: MetalGraphDescriptor,
    input: MetalKeyContextInput<'_>,
) -> Result<String, MetalError> {
    let expression = match value.kind {
        VALUE_CONSTANT => format!("constants[{}U]", descriptor.constants_offset + value.a),
        VALUE_INTERMEDIATE => format!("v{}", value.a),
        VALUE_FIXED | VALUE_ADVICE | VALUE_INSTANCE => {
            if value.b >= descriptor.rotations_count {
                return Err(MetalError::KeyContext(
                    "specialized graph rotation escapes its descriptor".into(),
                ));
            }
            let column = match value.kind {
                VALUE_FIXED => format!("{}UL", value.a),
                VALUE_ADVICE => format!(
                    "static_cast<ulong>(parameters.advice_offset + {}U)",
                    value.a
                ),
                VALUE_INSTANCE => format!(
                    "static_cast<ulong>(parameters.instance_offset + {}U)",
                    value.a
                ),
                _ => unreachable!(),
            };
            let values = if value.kind == VALUE_FIXED {
                "fixed_values"
            } else {
                "dynamic_values"
            };
            format!(
                "{values}[{column} * static_cast<ulong>(parameters.stride) + rotated_{}]",
                value.b
            )
        }
        VALUE_CHALLENGE => format!("challenges[{}U]", value.a),
        VALUE_BETA => "parameters.beta".into(),
        VALUE_GAMMA => "parameters.gamma".into(),
        VALUE_THETA => "parameters.theta".into(),
        VALUE_TRASH_CHALLENGE => "parameters.trash_challenge".into(),
        VALUE_Y => "parameters.y".into(),
        VALUE_PREVIOUS => "zero".into(),
        kind => {
            return Err(MetalError::KeyContext(format!(
                "specialized graph contains unsupported value kind {kind}"
            )));
        }
    };
    // Keep this parameter in the signature so callers cannot accidentally
    // generate from a different packed family while reusing a descriptor.
    let _ = input;
    Ok(expression)
}

fn emit_post(source: &mut String, post_kind: PostKind) -> Result<(), MetalError> {
    match post_kind {
        PostKind::Store => source.push_str("    numerator[row] = graph_result;\n"),
        PostKind::Lookup => source.push_str(
            r#"    const Bls12_381Fr one = midnight_specialized_one();
    const Bls12_381Fr l0 = l_values[row];
    const Bls12_381Fr l_last = l_values[static_cast<ulong>(parameters.stride) + row];
    const Bls12_381Fr l_active = l_values[2UL * static_cast<ulong>(parameters.stride) + row];
    const uint next_row = (row + static_cast<uint>(parameters.rotation_scale)) & (parameters.len - 1U);
    const uint previous_row = (row - static_cast<uint>(parameters.rotation_scale)) & (parameters.len - 1U);
    const ulong product_offset = static_cast<ulong>(parameters.lookup_product_column) * parameters.stride;
    const ulong input_offset = static_cast<ulong>(parameters.lookup_input_column) * parameters.stride;
    const ulong table_offset = static_cast<ulong>(parameters.lookup_table_column) * parameters.stride;
    const Bls12_381Fr product = dynamic_values[product_offset + row];
    const Bls12_381Fr product_next = dynamic_values[product_offset + next_row];
    const Bls12_381Fr input = dynamic_values[input_offset + row];
    const Bls12_381Fr input_previous = dynamic_values[input_offset + previous_row];
    const Bls12_381Fr table = dynamic_values[table_offset + row];
    const Bls12_381Fr input_minus_table = fr_sub(input, table);
    Bls12_381Fr value = numerator[row];
    value = midnight_specialized_accumulate(value, parameters.y, fr_montgomery_mul(fr_sub(one, product), l0));
    value = midnight_specialized_accumulate(value, parameters.y, fr_montgomery_mul(fr_sub(fr_montgomery_mul(product, product), product), l_last));
    const Bls12_381Fr left = fr_montgomery_mul(fr_montgomery_mul(product_next, fr_add(input, parameters.beta)), fr_add(table, parameters.gamma));
    const Bls12_381Fr right = fr_montgomery_mul(product, graph_result);
    value = midnight_specialized_accumulate(value, parameters.y, fr_montgomery_mul(fr_sub(left, right), l_active));
    value = midnight_specialized_accumulate(value, parameters.y, fr_montgomery_mul(input_minus_table, l0));
    value = midnight_specialized_accumulate(value, parameters.y, fr_montgomery_mul(fr_montgomery_mul(input_minus_table, fr_sub(input, input_previous)), l_active));
    numerator[row] = value;
"#,
        ),
        PostKind::Trash => source.push_str(
            r#"    const Bls12_381Fr one = midnight_specialized_one();
    const Bls12_381Fr selector = fixed_values[static_cast<ulong>(parameters.trash_selector_fixed_column) * parameters.stride + row];
    const Bls12_381Fr trash = dynamic_values[static_cast<ulong>(parameters.trash_column) * parameters.stride + row];
    const Bls12_381Fr term = fr_sub(graph_result, fr_montgomery_mul(fr_sub(one, selector), trash));
    numerator[row] = midnight_specialized_accumulate(numerator[row], parameters.y, term);
"#,
        ),
    }
    Ok(())
}

fn descriptor_window<'a, T>(
    label: &'static str,
    values: &'a [T],
    offset: u32,
    count: u32,
) -> Result<&'a [T], MetalError> {
    let start = offset as usize;
    let end = start
        .checked_add(count as usize)
        .ok_or_else(|| MetalError::KeyContext(format!("specialized {label} range overflows")))?;
    values.get(start..end).ok_or_else(|| {
        MetalError::KeyContext(format!(
            "specialized {label} range exceeds the packed graph family"
        ))
    })
}

fn format_error(_: std::fmt::Error) -> MetalError {
    MetalError::KeyContext("specialized graph source formatting failed".into())
}
