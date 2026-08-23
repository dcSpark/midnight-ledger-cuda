// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Checked diagnostic dispatch for the resident graph-expression shaders.

use std::mem::{size_of, size_of_val};
use std::ops::Range;

use super::*;

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

const GRAPH_POST_STORE: u32 = 0;
const GRAPH_POST_LOOKUP: u32 = 1;
const GRAPH_POST_TRASH: u32 = 2;
#[derive(Clone, Copy, Default)]
struct GraphProbeParameters {
    len: usize,
    stride: usize,
    rotation_scale: i32,
    post_kind: u32,
    constants_offset: usize,
    constants_count: usize,
    rotations_offset: usize,
    rotations_count: usize,
    operations_offset: usize,
    operations_count: usize,
    parts_offset: usize,
    parts_count: usize,
    fixed_columns: usize,
    advice_offset: usize,
    advice_columns: usize,
    instance_offset: usize,
    instance_columns: usize,
    challenge_count: usize,
    previous_mode: u32,
    lookup_product_column: usize,
    lookup_input_column: usize,
    lookup_table_column: usize,
    trash_column: usize,
    trash_selector_fixed_column: usize,
    beta: Bls12_381Fr,
    gamma: Bls12_381Fr,
    theta: Bls12_381Fr,
    trash_challenge: Bls12_381Fr,
    y: Bls12_381Fr,
}

#[derive(Clone, Copy)]
struct GraphProbeRequest<'a> {
    constants: &'a [Bls12_381Fr],
    rotations: &'a [i32],
    operations: &'a [GraphOperationAbi],
    horner_parts: &'a [GraphValueRefAbi],
    fixed_values: &'a [Bls12_381Fr],
    dynamic_values: &'a [Bls12_381Fr],
    challenges: &'a [Bls12_381Fr],
    l_values: &'a [Bls12_381Fr],
    numerator: &'a [Bls12_381Fr],
    parameters: GraphProbeParameters,
    scratch_capacity: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GraphPipelineLimits {
    scratch_capacity: usize,
    thread_execution_width: usize,
    maximum_threads_per_threadgroup: usize,
    static_threadgroup_bytes: usize,
}

impl MetalSession {
    fn graph_evaluate_probe(
        &self,
        request: GraphProbeRequest<'_>,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        let parameters = validate_graph_probe_request(&request)?;
        let evaluator = self
            .graph
            .evaluator(request.scratch_capacity)
            .ok_or_else(|| graph_error("scratch capacity has no compiled evaluator"))?;
        let _execution = self.execution_guard();

        let constants = self.graph_probe_buffer(request.constants, "midnight.graph.constants")?;
        let rotations = self.graph_probe_buffer(request.rotations, "midnight.graph.rotations")?;
        let operations =
            self.graph_probe_buffer(request.operations, "midnight.graph.operations")?;
        let horner_parts =
            self.graph_probe_buffer(request.horner_parts, "midnight.graph.horner_parts")?;
        let fixed_values =
            self.graph_probe_buffer(request.fixed_values, "midnight.graph.fixed_values")?;
        let dynamic_values =
            self.graph_probe_buffer(request.dynamic_values, "midnight.graph.dynamic_values")?;
        let challenges =
            self.graph_probe_buffer(request.challenges, "midnight.graph.challenges")?;
        let l_values = self.graph_probe_buffer(request.l_values, "midnight.graph.l_values")?;
        let numerator = self.graph_probe_buffer(request.numerator, "midnight.graph.numerator")?;
        let parameter_buffer =
            self.buffer_from_slice(&[parameters], "midnight.graph.parameters")?;

        let command = self.command_buffer("midnight.graph.evaluate_probe")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&evaluator.pipeline);
        // SAFETY: validation proves every logical range and matrix extent is
        // inside its retained allocation, and numerator has no read alias.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&constants.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&rotations.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&operations.raw), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&horner_parts.raw), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&fixed_values.raw), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&dynamic_values.raw), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&challenges.raw), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&l_values.raw), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&numerator.raw), 0, 8);
            encoder.setBuffer_offset_atIndex(Some(&parameter_buffer.raw), 0, 9);
        }
        self.dispatch(&encoder, &evaluator.pipeline, request.parameters.len);
        encoder.endEncoding();

        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![
                constants.raw,
                rotations.raw,
                operations.raw,
                horner_parts.raw,
                fixed_values.raw,
                dynamic_values.raw,
                challenges.raw,
                l_values.raw,
                numerator.raw.clone(),
                parameter_buffer.raw,
            ],
        })?;
        self.read_values(&numerator, request.parameters.len)
    }

    fn graph_probe_buffer<T: MetalPod>(
        &self,
        values: &[T],
        label: &str,
    ) -> Result<TrackedBuffer, MetalError> {
        if values.is_empty() {
            self.buffer(size_of::<T>(), label)
        } else {
            self.buffer_from_slice(values, label)
        }
    }

    fn graph_pipeline_limits(&self) -> [GraphPipelineLimits; 6] {
        std::array::from_fn(|index| {
            let evaluator = &self.graph.evaluators[index];
            GraphPipelineLimits {
                scratch_capacity: evaluator.scratch_capacity,
                thread_execution_width: evaluator.pipeline.threadExecutionWidth(),
                maximum_threads_per_threadgroup: evaluator.pipeline.maxTotalThreadsPerThreadgroup(),
                static_threadgroup_bytes: evaluator.pipeline.staticThreadgroupMemoryLength(),
            }
        })
    }
}

fn graph_error(reason: impl Into<String>) -> MetalError {
    MetalError::GraphEvaluation(reason.into())
}

fn validate_graph_probe_request(
    request: &GraphProbeRequest<'_>,
) -> Result<GraphEvalParametersAbi, MetalError> {
    let parameters = request.parameters;
    if parameters.len == 0 {
        return Err(graph_error("len must be nonzero"));
    }
    if parameters.stride == 0 || parameters.len > parameters.stride {
        return Err(graph_error("len must not exceed a nonzero stride"));
    }
    if request.numerator.len() < parameters.len {
        return Err(graph_error("numerator does not contain len rows"));
    }
    if !GRAPH_SCRATCH_CLASSES.contains(&request.scratch_capacity) {
        return Err(graph_error("scratch capacity has no compiled evaluator"));
    }
    if parameters.post_kind > GRAPH_POST_TRASH {
        return Err(graph_error("post_kind is not store, lookup, or trash"));
    }
    if parameters.previous_mode > 1 {
        return Err(graph_error("previous_mode is not zero or one"));
    }
    if parameters.post_kind != GRAPH_POST_STORE && parameters.previous_mode != 1 {
        return Err(graph_error(
            "lookup and trash folds require the resident numerator accumulator",
        ));
    }

    validate_non_overlapping_numerator(request)?;
    validate_reduced_slice("constants", request.constants)?;
    validate_reduced_slice("fixed values", request.fixed_values)?;
    validate_reduced_slice("dynamic values", request.dynamic_values)?;
    validate_reduced_slice("challenges", request.challenges)?;
    validate_reduced_slice("Lagrange values", request.l_values)?;
    validate_reduced_slice("numerator", request.numerator)?;
    for (label, value) in [
        ("beta", parameters.beta),
        ("gamma", parameters.gamma),
        ("theta", parameters.theta),
        ("trash challenge", parameters.trash_challenge),
        ("y", parameters.y),
    ] {
        if fr_greater_or_equal_host(value.limbs, generated::FR_MODULUS) {
            return Err(graph_error(format!("{label} is not a reduced Fr value")));
        }
    }

    checked_window(
        "constants",
        request.constants.len(),
        parameters.constants_offset,
        parameters.constants_count,
    )?;
    checked_window(
        "rotations",
        request.rotations.len(),
        parameters.rotations_offset,
        parameters.rotations_count,
    )?;
    let operation_range = checked_window(
        "operations",
        request.operations.len(),
        parameters.operations_offset,
        parameters.operations_count,
    )?;
    checked_window(
        "Horner parts",
        request.horner_parts.len(),
        parameters.parts_offset,
        parameters.parts_count,
    )?;

    if !request.fixed_values.len().is_multiple_of(parameters.stride) {
        return Err(graph_error(
            "fixed values do not contain complete stride columns",
        ));
    }
    if !request
        .dynamic_values
        .len()
        .is_multiple_of(parameters.stride)
    {
        return Err(graph_error(
            "dynamic values do not contain complete stride columns",
        ));
    }
    let fixed_required = parameters
        .fixed_columns
        .checked_mul(parameters.stride)
        .ok_or_else(|| graph_error("fixed column extent overflows"))?;
    if fixed_required > request.fixed_values.len() {
        return Err(graph_error("fixed column extent exceeds its allocation"));
    }
    let dynamic_columns = request.dynamic_values.len() / parameters.stride;
    let advice_end = parameters
        .advice_offset
        .checked_add(parameters.advice_columns)
        .ok_or_else(|| graph_error("advice column range overflows"))?;
    let instance_end = parameters
        .instance_offset
        .checked_add(parameters.instance_columns)
        .ok_or_else(|| graph_error("instance column range overflows"))?;
    if advice_end > dynamic_columns || instance_end > dynamic_columns {
        return Err(graph_error(
            "advice or instance column range exceeds dynamic values",
        ));
    }
    if parameters.challenge_count > request.challenges.len() {
        return Err(graph_error("challenge count exceeds its allocation"));
    }

    if parameters.post_kind != GRAPH_POST_STORE {
        let lagrange_required = parameters
            .stride
            .checked_mul(3)
            .ok_or_else(|| graph_error("Lagrange value extent overflows"))?;
        if lagrange_required > request.l_values.len() {
            return Err(graph_error(
                "lookup or trash fold requires three stride-sized Lagrange columns",
            ));
        }
    }
    if parameters.post_kind == GRAPH_POST_LOOKUP {
        for (label, column) in [
            ("lookup product", parameters.lookup_product_column),
            ("lookup input", parameters.lookup_input_column),
            ("lookup table", parameters.lookup_table_column),
        ] {
            if column >= dynamic_columns {
                return Err(graph_error(format!(
                    "{label} column exceeds dynamic values"
                )));
            }
        }
    }
    if parameters.post_kind == GRAPH_POST_TRASH {
        if parameters.trash_column >= dynamic_columns {
            return Err(graph_error("trash column exceeds dynamic values"));
        }
        if parameters.trash_selector_fixed_column >= parameters.fixed_columns {
            return Err(graph_error("trash selector exceeds fixed values"));
        }
    }

    let mut initialized = vec![false; request.scratch_capacity];
    let mut next_horner_part = 0usize;
    for (relative_index, operation) in request.operations[operation_range].iter().enumerate() {
        if operation.opcode > GRAPH_OP_STORE {
            return Err(graph_error(format!(
                "operation {relative_index} has an unknown opcode"
            )));
        }
        let target = operation.target as usize;
        if target >= request.scratch_capacity {
            return Err(graph_error(format!(
                "operation {relative_index} target exceeds scratch capacity"
            )));
        }
        validate_graph_value(
            operation.lhs,
            request,
            &initialized,
            &format!("operation {relative_index} lhs"),
        )?;
        if matches!(operation.opcode, GRAPH_OP_ADD | GRAPH_OP_SUB | GRAPH_OP_MUL) {
            validate_graph_value(
                operation.rhs,
                request,
                &initialized,
                &format!("operation {relative_index} rhs"),
            )?;
        } else if operation.opcode == GRAPH_OP_HORNER {
            validate_graph_value(
                operation.rhs,
                request,
                &initialized,
                &format!("operation {relative_index} Horner factor"),
            )?;
            let relative_start = operation.parts_offset as usize;
            if relative_start != next_horner_part {
                return Err(graph_error(format!(
                    "operation {relative_index} Horner parts are not contiguous"
                )));
            }
            let relative_end = relative_start
                .checked_add(operation.parts_len as usize)
                .ok_or_else(|| graph_error("Horner part range overflows"))?;
            if relative_end > parameters.parts_count {
                return Err(graph_error(format!(
                    "operation {relative_index} Horner parts escape the graph part window"
                )));
            }
            let absolute_start = parameters
                .parts_offset
                .checked_add(relative_start)
                .ok_or_else(|| graph_error("absolute Horner part range overflows"))?;
            let absolute_end = parameters
                .parts_offset
                .checked_add(relative_end)
                .ok_or_else(|| graph_error("absolute Horner part range overflows"))?;
            for (part_index, part) in request.horner_parts[absolute_start..absolute_end]
                .iter()
                .enumerate()
            {
                validate_graph_value(
                    *part,
                    request,
                    &initialized,
                    &format!("operation {relative_index} Horner part {part_index}"),
                )?;
            }
            next_horner_part = relative_end;
        } else if operation.parts_offset != 0 || operation.parts_len != 0 {
            return Err(graph_error(format!(
                "operation {relative_index} has Horner metadata on a non-Horner opcode"
            )));
        } else if operation.rhs.kind != 0 || operation.rhs.a != 0 || operation.rhs.b != 0 {
            return Err(graph_error(format!(
                "operation {relative_index} has a noncanonical unused rhs"
            )));
        }
        initialized[target] = true;
    }
    if next_horner_part != parameters.parts_count {
        return Err(graph_error(
            "the graph part window contains unreferenced Horner parts",
        ));
    }

    Ok(GraphEvalParametersAbi {
        len: graph_u32("len", parameters.len)?,
        stride: graph_u32("stride", parameters.stride)?,
        rotation_scale: parameters.rotation_scale,
        post_kind: parameters.post_kind,
        constants_offset: graph_u32("constants offset", parameters.constants_offset)?,
        constants_count: graph_u32("constants count", parameters.constants_count)?,
        rotations_offset: graph_u32("rotations offset", parameters.rotations_offset)?,
        rotations_count: graph_u32("rotations count", parameters.rotations_count)?,
        operations_offset: graph_u32("operations offset", parameters.operations_offset)?,
        operations_count: graph_u32("operations count", parameters.operations_count)?,
        parts_offset: graph_u32("parts offset", parameters.parts_offset)?,
        parts_count: graph_u32("parts count", parameters.parts_count)?,
        fixed_columns: graph_u32("fixed columns", parameters.fixed_columns)?,
        advice_offset: graph_u32("advice offset", parameters.advice_offset)?,
        advice_columns: graph_u32("advice columns", parameters.advice_columns)?,
        instance_offset: graph_u32("instance offset", parameters.instance_offset)?,
        instance_columns: graph_u32("instance columns", parameters.instance_columns)?,
        challenge_count: graph_u32("challenge count", parameters.challenge_count)?,
        previous_mode: parameters.previous_mode,
        lookup_product_column: graph_u32(
            "lookup product column",
            parameters.lookup_product_column,
        )?,
        lookup_input_column: graph_u32("lookup input column", parameters.lookup_input_column)?,
        lookup_table_column: graph_u32("lookup table column", parameters.lookup_table_column)?,
        trash_column: graph_u32("trash column", parameters.trash_column)?,
        trash_selector_fixed_column: graph_u32(
            "trash selector fixed column",
            parameters.trash_selector_fixed_column,
        )?,
        beta: parameters.beta,
        gamma: parameters.gamma,
        theta: parameters.theta,
        trash_challenge: parameters.trash_challenge,
        y: parameters.y,
    })
}

fn checked_window(
    label: &str,
    available: usize,
    offset: usize,
    count: usize,
) -> Result<Range<usize>, MetalError> {
    let end = offset
        .checked_add(count)
        .ok_or_else(|| graph_error(format!("{label} offset plus count overflows")))?;
    if end > available {
        return Err(graph_error(format!(
            "{label} window {offset}..{end} exceeds {available} elements"
        )));
    }
    Ok(offset..end)
}

fn graph_u32(label: &str, value: usize) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| graph_error(format!("{label} exceeds the u32 shader ABI")))
}

fn validate_graph_value(
    value: GraphValueRefAbi,
    request: &GraphProbeRequest<'_>,
    initialized: &[bool],
    context: &str,
) -> Result<(), MetalError> {
    let parameters = request.parameters;
    let a = value.a as usize;
    let b = value.b as usize;
    match value.kind {
        GRAPH_VALUE_CONSTANT => {
            if a >= parameters.constants_count {
                return Err(graph_error(format!(
                    "{context} constant index exceeds the graph constant window"
                )));
            }
            require_zero_aux(value.b, context)
        }
        GRAPH_VALUE_INTERMEDIATE => {
            if a >= request.scratch_capacity || !initialized[a] {
                return Err(graph_error(format!(
                    "{context} reads an unavailable intermediate"
                )));
            }
            require_zero_aux(value.b, context)
        }
        GRAPH_VALUE_FIXED => {
            if a >= parameters.fixed_columns || b >= parameters.rotations_count {
                return Err(graph_error(format!(
                    "{context} fixed column or rotation index is out of range"
                )));
            }
            Ok(())
        }
        GRAPH_VALUE_ADVICE => {
            if a >= parameters.advice_columns || b >= parameters.rotations_count {
                return Err(graph_error(format!(
                    "{context} advice column or rotation index is out of range"
                )));
            }
            Ok(())
        }
        GRAPH_VALUE_INSTANCE => {
            if a >= parameters.instance_columns || b >= parameters.rotations_count {
                return Err(graph_error(format!(
                    "{context} instance column or rotation index is out of range"
                )));
            }
            Ok(())
        }
        GRAPH_VALUE_CHALLENGE => {
            if a >= parameters.challenge_count {
                return Err(graph_error(format!(
                    "{context} challenge index is out of range"
                )));
            }
            require_zero_aux(value.b, context)
        }
        GRAPH_VALUE_BETA
        | GRAPH_VALUE_GAMMA
        | GRAPH_VALUE_THETA
        | GRAPH_VALUE_TRASH_CHALLENGE
        | GRAPH_VALUE_Y
        | GRAPH_VALUE_PREVIOUS => {
            if value.a != 0 || value.b != 0 {
                Err(graph_error(format!(
                    "{context} scalar source has nonzero auxiliary fields"
                )))
            } else {
                Ok(())
            }
        }
        _ => Err(graph_error(format!("{context} has an unknown value tag"))),
    }
}

fn require_zero_aux(value: u32, context: &str) -> Result<(), MetalError> {
    if value == 0 {
        Ok(())
    } else {
        Err(graph_error(format!(
            "{context} has a nonzero unused auxiliary field"
        )))
    }
}

fn validate_reduced_slice(label: &str, values: &[Bls12_381Fr]) -> Result<(), MetalError> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| fr_greater_or_equal_host(value.limbs, generated::FR_MODULUS))
    {
        Err(graph_error(format!(
            "{label} element {index} is not a reduced Fr value"
        )))
    } else {
        Ok(())
    }
}

fn validate_non_overlapping_numerator(request: &GraphProbeRequest<'_>) -> Result<(), MetalError> {
    let numerator = byte_range(request.numerator)?;
    for (label, range) in [
        ("constants", byte_range(request.constants)?),
        ("rotations", byte_range(request.rotations)?),
        ("operations", byte_range(request.operations)?),
        ("Horner parts", byte_range(request.horner_parts)?),
        ("fixed values", byte_range(request.fixed_values)?),
        ("dynamic values", byte_range(request.dynamic_values)?),
        ("challenges", byte_range(request.challenges)?),
        ("Lagrange values", byte_range(request.l_values)?),
    ] {
        if ranges_overlap(numerator.as_ref(), range.as_ref()) {
            return Err(graph_error(format!(
                "numerator overlaps the {label} read allocation"
            )));
        }
    }
    Ok(())
}

fn byte_range<T>(values: &[T]) -> Result<Option<Range<usize>>, MetalError> {
    let bytes = size_of_val(values);
    if bytes == 0 {
        return Ok(None);
    }
    let start = values.as_ptr() as usize;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| graph_error("host allocation address range overflows"))?;
    Ok(Some(start..end))
}

fn ranges_overlap(lhs: Option<&Range<usize>>, rhs: Option<&Range<usize>>) -> bool {
    matches!((lhs, rhs), (Some(lhs), Some(rhs)) if lhs.start < rhs.end && rhs.start < lhs.end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_operation_source_and_rotation_extreme_matches_cpu_blst() {
        for rotation_scale in [1, 4] {
            let fixture = CompleteFixture::new(rotation_scale);
            let request = fixture.request();
            let expected = cpu_graph_evaluate(request);
            let session = MetalSession::new().expect("initialize Metal runtime");
            let actual = session
                .graph_evaluate_probe(request)
                .expect("evaluate complete graph probe");
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn empty_graph_and_empty_or_ordered_horner_match_cpu_blst() {
        let session = MetalSession::new().expect("initialize Metal runtime");

        let empty_graph = MinimalFixture::new();
        let mut request = empty_graph.request();
        request.parameters.operations_count = 0;
        request.operations = &[];
        assert_eq!(
            session
                .graph_evaluate_probe(request)
                .expect("evaluate empty graph"),
            vec![Bls12_381Fr::default()]
        );

        let lhs = fr(2);
        let factor = fr(3);
        let parts = [value_constant(2), value_constant(3), value_constant(4)];
        let constants = [lhs, factor, fr(5), fr(7), fr(11)];
        let operations = [GraphOperationAbi {
            opcode: GRAPH_OP_HORNER,
            target: 0,
            parts_offset: 0,
            parts_len: 0,
            lhs: value_constant(0),
            rhs: value_constant(1),
        }];
        let numerator = [fr(19)];
        let mut parameters = minimal_parameters();
        parameters.constants_count = constants.len();
        parameters.parts_count = 0;
        let empty_horner = GraphProbeRequest {
            constants: &constants,
            rotations: &[],
            operations: &operations,
            horner_parts: &parts,
            fixed_values: &[],
            dynamic_values: &[],
            challenges: &[],
            l_values: &[],
            numerator: &numerator,
            parameters,
            scratch_capacity: 8,
        };
        assert_eq!(
            session
                .graph_evaluate_probe(empty_horner)
                .expect("evaluate empty Horner sequence"),
            vec![lhs]
        );

        let operations = [GraphOperationAbi {
            parts_len: parts.len() as u32,
            ..operations[0]
        }];
        parameters.parts_count = parts.len();
        let ordered_horner = GraphProbeRequest {
            operations: &operations,
            parameters,
            ..empty_horner
        };
        let expected = blst_add(
            blst_mul(
                blst_add(
                    blst_mul(blst_add(blst_mul(lhs, factor), fr(5)), factor),
                    fr(7),
                ),
                factor,
            ),
            fr(11),
        );
        assert_eq!(
            session
                .graph_evaluate_probe(ordered_horner)
                .expect("evaluate ordered Horner sequence"),
            vec![expected]
        );
    }

    #[test]
    fn every_scratch_class_dispatches_and_reports_pipeline_limits() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let limits = session.graph_pipeline_limits();
        for (index, limit) in limits.iter().enumerate() {
            assert_eq!(limit.scratch_capacity, GRAPH_SCRATCH_CLASSES[index]);
            assert_ne!(limit.thread_execution_width, 0);
            assert!(limit.maximum_threads_per_threadgroup >= limit.thread_execution_width);
            eprintln!(
                "GRAPH_PIPELINE_LIMIT scratch={} execution_width={} max_threads={} static_threadgroup_bytes={}",
                limit.scratch_capacity,
                limit.thread_execution_width,
                limit.maximum_threads_per_threadgroup,
                limit.static_threadgroup_bytes
            );

            let fixture = MinimalFixture::new();
            let mut operation = fixture.operations[0];
            operation.target = (limit.scratch_capacity - 1) as u32;
            let operations = [operation];
            let request = GraphProbeRequest {
                operations: &operations,
                scratch_capacity: limit.scratch_capacity,
                ..fixture.request()
            };
            assert_eq!(
                session
                    .graph_evaluate_probe(request)
                    .expect("dispatch graph scratch class"),
                vec![fixture.parameters.y]
            );
        }
    }

    #[test]
    fn lookup_applies_the_five_protocol_folds_in_order() {
        let fixture = LookupFixture::new();
        let request = fixture.request();
        let expected = cpu_graph_evaluate(request);
        let session = MetalSession::new().expect("initialize Metal runtime");
        let actual = session
            .graph_evaluate_probe(request)
            .expect("evaluate lookup fold");
        assert_eq!(actual, expected);
    }

    #[test]
    fn previous_value_is_visible_only_to_custom_store_graphs() {
        let numerator = [fr(41)];
        let operations = [operation(
            GRAPH_OP_STORE,
            0,
            value(GRAPH_VALUE_PREVIOUS, 0, 0),
            value_constant(0),
        )];
        let mut parameters = minimal_parameters();
        parameters.previous_mode = 1;
        let request = GraphProbeRequest {
            constants: &[],
            rotations: &[],
            operations: &operations,
            horner_parts: &[],
            fixed_values: &[],
            dynamic_values: &[],
            challenges: &[],
            l_values: &[],
            numerator: &numerator,
            parameters,
            scratch_capacity: 8,
        };
        let session = MetalSession::new().expect("initialize Metal runtime");
        assert_eq!(
            session
                .graph_evaluate_probe(request)
                .expect("evaluate custom graph with previous value"),
            numerator
        );

        parameters.previous_mode = 0;
        assert_eq!(
            session
                .graph_evaluate_probe(GraphProbeRequest {
                    parameters,
                    ..request
                })
                .expect("evaluate custom graph without previous value"),
            vec![Bls12_381Fr::default()]
        );
    }

    #[test]
    fn trash_applies_selector_term_after_the_resident_accumulator() {
        let fixture = TrashFixture::new();
        let request = fixture.request();
        let expected = cpu_graph_evaluate(request);
        let session = MetalSession::new().expect("initialize Metal runtime");
        let actual = session
            .graph_evaluate_probe(request)
            .expect("evaluate trash fold");
        assert_eq!(actual, expected);
    }

    #[test]
    fn diagnostic_dispatch_has_one_honest_command_and_readback() {
        let fixture = MinimalFixture::new();
        let request = fixture.request();
        let session = MetalSession::new().expect("initialize Metal runtime");
        session.reset_activity();
        session
            .graph_evaluate_probe(request)
            .expect("evaluate minimal graph");
        let activity = session.activity();
        assert_eq!(activity.command_buffers, 1);
        assert_eq!(activity.dispatches, 1);
        assert_eq!(activity.host_waits, 1);
        assert_eq!(activity.allocations, 10);
        assert_eq!(activity.host_read_bytes, size_of::<Bls12_381Fr>() as u64);
        assert_eq!(
            activity.host_write_bytes,
            (size_of::<GraphOperationAbi>()
                + size_of::<Bls12_381Fr>()
                + size_of::<GraphEvalParametersAbi>()) as u64
        );
    }

    #[test]
    fn malformed_requests_fail_before_allocation_or_commands() {
        let session = MetalSession::new().expect("initialize Metal runtime");

        let fixture = MinimalFixture::new();
        let mut request = fixture.request();
        request.parameters.len = 0;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let mut request = fixture.request();
        request.parameters.stride = 0;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let mut request = fixture.request();
        request.parameters.operations_offset = usize::MAX;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let mut request = fixture.request();
        request.parameters.constants_offset = 1;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let mut request = fixture.request();
        request.parameters.rotations_offset = usize::MAX;
        request.parameters.rotations_count = 1;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let mut request = fixture.request();
        request.parameters.parts_offset = usize::MAX;
        request.parameters.parts_count = 1;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let mut request = fixture.request();
        request.parameters.fixed_columns = 1;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let mut request = fixture.request();
        request.parameters.advice_columns = 1;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let mut request = fixture.request();
        request.parameters.instance_columns = 1;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let mut request = fixture.request();
        request.parameters.challenge_count = 1;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let mut request = fixture.request();
        request.parameters.post_kind = 3;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let mut request = fixture.request();
        request.parameters.previous_mode = 2;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let mut request = fixture.request();
        request.scratch_capacity = 9;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let invalid_operation = GraphOperationAbi {
            target: 8,
            ..fixture.operations[0]
        };
        let operations = [invalid_operation];
        let request = GraphProbeRequest {
            operations: &operations,
            ..fixture.request()
        };
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let invalid_operation = GraphOperationAbi {
            rhs: value(GRAPH_VALUE_Y, 0, 0),
            ..fixture.operations[0]
        };
        let operations = [invalid_operation];
        let request = GraphProbeRequest {
            operations: &operations,
            ..fixture.request()
        };
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let parts = [value(GRAPH_VALUE_Y, 0, 0)];
        let mut request = fixture.request();
        request.horner_parts = &parts;
        request.parameters.parts_count = 1;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let invalid_operation = GraphOperationAbi {
            opcode: GRAPH_OP_HORNER,
            parts_offset: 1,
            parts_len: 0,
            rhs: value(GRAPH_VALUE_Y, 0, 0),
            ..fixture.operations[0]
        };
        let operations = [invalid_operation];
        let parts = [value(GRAPH_VALUE_Y, 0, 0)];
        let mut request = GraphProbeRequest {
            operations: &operations,
            horner_parts: &parts,
            ..fixture.request()
        };
        request.parameters.parts_count = 1;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let invalid_operation = GraphOperationAbi {
            lhs: GraphValueRefAbi {
                kind: GRAPH_VALUE_INTERMEDIATE,
                a: 0,
                b: 0,
            },
            ..fixture.operations[0]
        };
        let operations = [invalid_operation];
        let request = GraphProbeRequest {
            operations: &operations,
            ..fixture.request()
        };
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let invalid_operation = GraphOperationAbi {
            lhs: GraphValueRefAbi {
                kind: GRAPH_VALUE_FIXED,
                a: 0,
                b: 0,
            },
            ..fixture.operations[0]
        };
        let operations = [invalid_operation];
        let request = GraphProbeRequest {
            operations: &operations,
            ..fixture.request()
        };
        assert_pre_dispatch_rejection(&session, request);

        let fixture = MinimalFixture::new();
        let mut request = fixture.request();
        request.parameters.post_kind = GRAPH_POST_TRASH;
        request.parameters.previous_mode = 1;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = LookupFixture::new();
        let mut request = fixture.request();
        request.parameters.lookup_table_column = 3;
        assert_pre_dispatch_rejection(&session, request);

        let fixture = TrashFixture::new();
        let mut request = fixture.request();
        request.parameters.trash_selector_fixed_column = 1;
        assert_pre_dispatch_rejection(&session, request);

        let shared = [fr(1)];
        let operations = [GraphOperationAbi {
            opcode: GRAPH_OP_STORE,
            target: 0,
            parts_offset: 0,
            parts_len: 0,
            lhs: value_constant(0),
            rhs: value_constant(0),
        }];
        let mut parameters = minimal_parameters();
        parameters.constants_count = 1;
        let request = GraphProbeRequest {
            constants: &shared,
            rotations: &[],
            operations: &operations,
            horner_parts: &[],
            fixed_values: &[],
            dynamic_values: &[],
            challenges: &[],
            l_values: &[],
            numerator: &shared,
            parameters,
            scratch_capacity: 8,
        };
        assert_pre_dispatch_rejection(&session, request);
    }

    fn assert_pre_dispatch_rejection(session: &MetalSession, request: GraphProbeRequest<'_>) {
        session.reset_activity();
        assert!(matches!(
            session.graph_evaluate_probe(request),
            Err(MetalError::GraphEvaluation(_))
        ));
        let activity = session.activity();
        assert_eq!(activity.allocations, 0);
        assert_eq!(activity.command_buffers, 0);
        assert_eq!(activity.dispatches, 0);
        assert_eq!(activity.host_waits, 0);
        assert_eq!(activity.host_write_bytes, 0);
        assert_eq!(activity.host_read_bytes, 0);
    }

    struct MinimalFixture {
        operations: [GraphOperationAbi; 1],
        numerator: [Bls12_381Fr; 1],
        parameters: GraphProbeParameters,
    }

    impl MinimalFixture {
        fn new() -> Self {
            let parameters = minimal_parameters();
            Self {
                operations: [GraphOperationAbi {
                    opcode: GRAPH_OP_STORE,
                    target: 0,
                    parts_offset: 0,
                    parts_len: 0,
                    lhs: GraphValueRefAbi {
                        kind: GRAPH_VALUE_Y,
                        a: 0,
                        b: 0,
                    },
                    rhs: value_constant(0),
                }],
                numerator: [fr(17)],
                parameters,
            }
        }

        fn request(&self) -> GraphProbeRequest<'_> {
            GraphProbeRequest {
                constants: &[],
                rotations: &[],
                operations: &self.operations,
                horner_parts: &[],
                fixed_values: &[],
                dynamic_values: &[],
                challenges: &[],
                l_values: &[],
                numerator: &self.numerator,
                parameters: self.parameters,
                scratch_capacity: 8,
            }
        }
    }

    fn minimal_parameters() -> GraphProbeParameters {
        GraphProbeParameters {
            len: 1,
            stride: 1,
            operations_count: 1,
            post_kind: GRAPH_POST_STORE,
            y: fr(23),
            ..GraphProbeParameters::default()
        }
    }

    struct CompleteFixture {
        constants: Vec<Bls12_381Fr>,
        rotations: Vec<i32>,
        operations: Vec<GraphOperationAbi>,
        parts: Vec<GraphValueRefAbi>,
        fixed: Vec<Bls12_381Fr>,
        dynamic: Vec<Bls12_381Fr>,
        challenges: Vec<Bls12_381Fr>,
        numerator: Vec<Bls12_381Fr>,
        parameters: GraphProbeParameters,
    }

    impl CompleteFixture {
        fn new(rotation_scale: i32) -> Self {
            let len = 7;
            let constants = vec![fr(101), fr(2), fr(3), fr(5), fr(7), fr(11)];
            let rotations = vec![99, 0, 1, -1, i32::MIN, i32::MAX];
            let fixed = matrix(2, len, 13);
            let dynamic = matrix(2, len, 47);
            let challenges = vec![fr(17)];
            let numerator = sequence(len, 71);
            let operations = vec![
                GraphOperationAbi::default(),
                operation(
                    GRAPH_OP_ADD,
                    0,
                    value_constant(0),
                    value(GRAPH_VALUE_FIXED, 0, 0),
                ),
                operation(
                    GRAPH_OP_SUB,
                    1,
                    value(GRAPH_VALUE_ADVICE, 0, 1),
                    value(GRAPH_VALUE_INSTANCE, 0, 2),
                ),
                operation(
                    GRAPH_OP_MUL,
                    2,
                    value(GRAPH_VALUE_CHALLENGE, 0, 0),
                    value(GRAPH_VALUE_BETA, 0, 0),
                ),
                operation(
                    GRAPH_OP_SQUARE,
                    3,
                    value(GRAPH_VALUE_GAMMA, 0, 0),
                    value_constant(0),
                ),
                operation(
                    GRAPH_OP_DOUBLE,
                    4,
                    value(GRAPH_VALUE_THETA, 0, 0),
                    value_constant(0),
                ),
                operation(
                    GRAPH_OP_NEGATE,
                    5,
                    value(GRAPH_VALUE_TRASH_CHALLENGE, 0, 0),
                    value_constant(0),
                ),
                operation(
                    GRAPH_OP_ADD,
                    6,
                    value(GRAPH_VALUE_Y, 0, 0),
                    value(GRAPH_VALUE_PREVIOUS, 0, 0),
                ),
                operation(
                    GRAPH_OP_ADD,
                    7,
                    value(GRAPH_VALUE_FIXED, 1, 3),
                    value(GRAPH_VALUE_ADVICE, 0, 4),
                ),
                GraphOperationAbi {
                    opcode: GRAPH_OP_HORNER,
                    target: 8,
                    parts_offset: 0,
                    parts_len: 7,
                    lhs: value(GRAPH_VALUE_INTERMEDIATE, 0, 0),
                    rhs: value(GRAPH_VALUE_INTERMEDIATE, 1, 0),
                },
                operation(
                    GRAPH_OP_STORE,
                    0,
                    value(GRAPH_VALUE_INTERMEDIATE, 8, 0),
                    value_constant(0),
                ),
            ];
            let parts = vec![
                value_constant(0),
                value(GRAPH_VALUE_INTERMEDIATE, 2, 0),
                value(GRAPH_VALUE_INTERMEDIATE, 3, 0),
                value(GRAPH_VALUE_INTERMEDIATE, 4, 0),
                value(GRAPH_VALUE_INTERMEDIATE, 5, 0),
                value(GRAPH_VALUE_INTERMEDIATE, 6, 0),
                value(GRAPH_VALUE_INTERMEDIATE, 7, 0),
                value_constant(4),
            ];
            let parameters = GraphProbeParameters {
                len,
                stride: len,
                rotation_scale,
                post_kind: GRAPH_POST_STORE,
                constants_offset: 1,
                constants_count: constants.len() - 1,
                rotations_offset: 1,
                rotations_count: rotations.len() - 1,
                operations_offset: 1,
                operations_count: operations.len() - 1,
                parts_offset: 1,
                parts_count: parts.len() - 1,
                fixed_columns: 2,
                advice_offset: 0,
                advice_columns: 1,
                instance_offset: 1,
                instance_columns: 1,
                challenge_count: 1,
                previous_mode: 1,
                beta: fr(19),
                gamma: fr(23),
                theta: fr(29),
                trash_challenge: fr(31),
                y: fr(37),
                ..GraphProbeParameters::default()
            };
            Self {
                constants,
                rotations,
                operations,
                parts,
                fixed,
                dynamic,
                challenges,
                numerator,
                parameters,
            }
        }

        fn request(&self) -> GraphProbeRequest<'_> {
            GraphProbeRequest {
                constants: &self.constants,
                rotations: &self.rotations,
                operations: &self.operations,
                horner_parts: &self.parts,
                fixed_values: &self.fixed,
                dynamic_values: &self.dynamic,
                challenges: &self.challenges,
                l_values: &[],
                numerator: &self.numerator,
                parameters: self.parameters,
                scratch_capacity: 16,
            }
        }
    }

    struct LookupFixture {
        constants: [Bls12_381Fr; 1],
        operations: [GraphOperationAbi; 1],
        dynamic: Vec<Bls12_381Fr>,
        l_values: Vec<Bls12_381Fr>,
        numerator: Vec<Bls12_381Fr>,
        parameters: GraphProbeParameters,
    }

    impl LookupFixture {
        fn new() -> Self {
            let len = 7;
            Self {
                constants: [fr(13)],
                operations: [operation(
                    GRAPH_OP_ADD,
                    0,
                    value(GRAPH_VALUE_PREVIOUS, 0, 0),
                    value_constant(0),
                )],
                dynamic: matrix(3, len, 3),
                l_values: matrix(3, len, 41),
                numerator: sequence(len, 97),
                parameters: GraphProbeParameters {
                    len,
                    stride: len,
                    rotation_scale: 1,
                    post_kind: GRAPH_POST_LOOKUP,
                    constants_count: 1,
                    operations_count: 1,
                    previous_mode: 1,
                    lookup_product_column: 0,
                    lookup_input_column: 1,
                    lookup_table_column: 2,
                    beta: fr(17),
                    gamma: fr(19),
                    y: fr(23),
                    ..GraphProbeParameters::default()
                },
            }
        }

        fn request(&self) -> GraphProbeRequest<'_> {
            GraphProbeRequest {
                constants: &self.constants,
                rotations: &[],
                operations: &self.operations,
                horner_parts: &[],
                fixed_values: &[],
                dynamic_values: &self.dynamic,
                challenges: &[],
                l_values: &self.l_values,
                numerator: &self.numerator,
                parameters: self.parameters,
                scratch_capacity: 8,
            }
        }
    }

    struct TrashFixture {
        constants: [Bls12_381Fr; 1],
        operations: [GraphOperationAbi; 1],
        fixed: Vec<Bls12_381Fr>,
        dynamic: Vec<Bls12_381Fr>,
        l_values: Vec<Bls12_381Fr>,
        numerator: Vec<Bls12_381Fr>,
        parameters: GraphProbeParameters,
    }

    impl TrashFixture {
        fn new() -> Self {
            let len = 7;
            Self {
                constants: [fr(13)],
                operations: [operation(
                    GRAPH_OP_STORE,
                    0,
                    value_constant(0),
                    value_constant(0),
                )],
                fixed: matrix(1, len, 5),
                dynamic: matrix(1, len, 29),
                l_values: matrix(3, len, 43),
                numerator: sequence(len, 101),
                parameters: GraphProbeParameters {
                    len,
                    stride: len,
                    rotation_scale: 1,
                    post_kind: GRAPH_POST_TRASH,
                    constants_count: 1,
                    operations_count: 1,
                    fixed_columns: 1,
                    previous_mode: 1,
                    trash_column: 0,
                    trash_selector_fixed_column: 0,
                    y: fr(31),
                    ..GraphProbeParameters::default()
                },
            }
        }

        fn request(&self) -> GraphProbeRequest<'_> {
            GraphProbeRequest {
                constants: &self.constants,
                rotations: &[],
                operations: &self.operations,
                horner_parts: &[],
                fixed_values: &self.fixed,
                dynamic_values: &self.dynamic,
                challenges: &[],
                l_values: &self.l_values,
                numerator: &self.numerator,
                parameters: self.parameters,
                scratch_capacity: 8,
            }
        }
    }

    fn operation(
        opcode: u32,
        target: u32,
        lhs: GraphValueRefAbi,
        rhs: GraphValueRefAbi,
    ) -> GraphOperationAbi {
        GraphOperationAbi {
            opcode,
            target,
            parts_offset: 0,
            parts_len: 0,
            lhs,
            rhs,
        }
    }

    fn value(kind: u32, a: u32, b: u32) -> GraphValueRefAbi {
        GraphValueRefAbi { kind, a, b }
    }

    fn value_constant(index: u32) -> GraphValueRefAbi {
        value(GRAPH_VALUE_CONSTANT, index, 0)
    }

    fn matrix(columns: usize, stride: usize, seed: u64) -> Vec<Bls12_381Fr> {
        (0..columns * stride)
            .map(|index| fr(seed + index as u64 * 3 + 1))
            .collect()
    }

    fn sequence(len: usize, seed: u64) -> Vec<Bls12_381Fr> {
        (0..len).map(|index| fr(seed + index as u64)).collect()
    }

    fn cpu_graph_evaluate(request: GraphProbeRequest<'_>) -> Vec<Bls12_381Fr> {
        let p = request.parameters;
        let mut output = request.numerator[..p.len].to_vec();
        for (row, output_row) in output.iter_mut().enumerate() {
            let accumulator = if p.post_kind != GRAPH_POST_STORE || p.previous_mode != 0 {
                *output_row
            } else {
                Bls12_381Fr::default()
            };
            let graph_previous = if p.post_kind == GRAPH_POST_STORE && p.previous_mode != 0 {
                accumulator
            } else {
                Bls12_381Fr::default()
            };
            let graph_result = cpu_graph_operations(request, row, graph_previous);
            if p.post_kind == GRAPH_POST_STORE {
                *output_row = graph_result;
                continue;
            }
            let l0 = request.l_values[row];
            let l_last = request.l_values[p.stride + row];
            let l_active = request.l_values[2 * p.stride + row];
            if p.post_kind == GRAPH_POST_LOOKUP {
                let next = rotated_row(row, 1, p.rotation_scale, p.len);
                let previous = rotated_row(row, -1, p.rotation_scale, p.len);
                let product = dynamic(
                    request.dynamic_values,
                    p.lookup_product_column,
                    row,
                    p.stride,
                );
                let product_next = dynamic(
                    request.dynamic_values,
                    p.lookup_product_column,
                    next,
                    p.stride,
                );
                let input = dynamic(request.dynamic_values, p.lookup_input_column, row, p.stride);
                let input_previous = dynamic(
                    request.dynamic_values,
                    p.lookup_input_column,
                    previous,
                    p.stride,
                );
                let table = dynamic(request.dynamic_values, p.lookup_table_column, row, p.stride);
                let difference = blst_sub(input, table);
                let mut value = accumulator;
                value = accumulate(value, p.y, blst_mul(blst_sub(fr(1), product), l0));
                value = accumulate(
                    value,
                    p.y,
                    blst_mul(blst_sub(blst_mul(product, product), product), l_last),
                );
                let left = blst_mul(
                    blst_mul(product_next, blst_add(input, p.beta)),
                    blst_add(table, p.gamma),
                );
                let right = blst_mul(product, graph_result);
                value = accumulate(value, p.y, blst_mul(blst_sub(left, right), l_active));
                value = accumulate(value, p.y, blst_mul(difference, l0));
                value = accumulate(
                    value,
                    p.y,
                    blst_mul(
                        blst_mul(difference, blst_sub(input, input_previous)),
                        l_active,
                    ),
                );
                *output_row = value;
            } else {
                let selector = dynamic(
                    request.fixed_values,
                    p.trash_selector_fixed_column,
                    row,
                    p.stride,
                );
                let trash = dynamic(request.dynamic_values, p.trash_column, row, p.stride);
                let term = blst_sub(graph_result, blst_mul(blst_sub(fr(1), selector), trash));
                *output_row = accumulate(accumulator, p.y, term);
            }
        }
        output
    }

    fn cpu_graph_operations(
        request: GraphProbeRequest<'_>,
        row: usize,
        previous: Bls12_381Fr,
    ) -> Bls12_381Fr {
        let p = request.parameters;
        let mut scratch = vec![Bls12_381Fr::default(); request.scratch_capacity];
        let mut result = Bls12_381Fr::default();
        for operation in
            &request.operations[p.operations_offset..p.operations_offset + p.operations_count]
        {
            let lhs = cpu_value(operation.lhs, request, &scratch, previous, row);
            result = match operation.opcode {
                GRAPH_OP_ADD => blst_add(
                    lhs,
                    cpu_value(operation.rhs, request, &scratch, previous, row),
                ),
                GRAPH_OP_SUB => blst_sub(
                    lhs,
                    cpu_value(operation.rhs, request, &scratch, previous, row),
                ),
                GRAPH_OP_MUL => blst_mul(
                    lhs,
                    cpu_value(operation.rhs, request, &scratch, previous, row),
                ),
                GRAPH_OP_SQUARE => blst_mul(lhs, lhs),
                GRAPH_OP_DOUBLE => blst_add(lhs, lhs),
                GRAPH_OP_NEGATE => blst_sub(Bls12_381Fr::default(), lhs),
                GRAPH_OP_HORNER => {
                    let factor = cpu_value(operation.rhs, request, &scratch, previous, row);
                    let start = p.parts_offset + operation.parts_offset as usize;
                    let end = start + operation.parts_len as usize;
                    request.horner_parts[start..end]
                        .iter()
                        .fold(lhs, |value, part| {
                            blst_add(
                                blst_mul(value, factor),
                                cpu_value(*part, request, &scratch, previous, row),
                            )
                        })
                }
                GRAPH_OP_STORE => lhs,
                _ => unreachable!("validated graph opcode"),
            };
            scratch[operation.target as usize] = result;
        }
        result
    }

    fn cpu_value(
        value: GraphValueRefAbi,
        request: GraphProbeRequest<'_>,
        scratch: &[Bls12_381Fr],
        previous: Bls12_381Fr,
        row: usize,
    ) -> Bls12_381Fr {
        let p = request.parameters;
        match value.kind {
            GRAPH_VALUE_CONSTANT => request.constants[p.constants_offset + value.a as usize],
            GRAPH_VALUE_INTERMEDIATE => scratch[value.a as usize],
            GRAPH_VALUE_FIXED => {
                let rotation = request.rotations[p.rotations_offset + value.b as usize];
                dynamic(
                    request.fixed_values,
                    value.a as usize,
                    rotated_row(row, rotation, p.rotation_scale, p.len),
                    p.stride,
                )
            }
            GRAPH_VALUE_ADVICE => {
                let rotation = request.rotations[p.rotations_offset + value.b as usize];
                dynamic(
                    request.dynamic_values,
                    p.advice_offset + value.a as usize,
                    rotated_row(row, rotation, p.rotation_scale, p.len),
                    p.stride,
                )
            }
            GRAPH_VALUE_INSTANCE => {
                let rotation = request.rotations[p.rotations_offset + value.b as usize];
                dynamic(
                    request.dynamic_values,
                    p.instance_offset + value.a as usize,
                    rotated_row(row, rotation, p.rotation_scale, p.len),
                    p.stride,
                )
            }
            GRAPH_VALUE_CHALLENGE => request.challenges[value.a as usize],
            GRAPH_VALUE_BETA => p.beta,
            GRAPH_VALUE_GAMMA => p.gamma,
            GRAPH_VALUE_THETA => p.theta,
            GRAPH_VALUE_TRASH_CHALLENGE => p.trash_challenge,
            GRAPH_VALUE_Y => p.y,
            GRAPH_VALUE_PREVIOUS => previous,
            _ => unreachable!("validated graph value tag"),
        }
    }

    fn rotated_row(row: usize, rotation: i32, scale: i32, len: usize) -> usize {
        ((row as i64 + i64::from(rotation) * i64::from(scale)).rem_euclid(len as i64)) as usize
    }

    fn dynamic(values: &[Bls12_381Fr], column: usize, row: usize, stride: usize) -> Bls12_381Fr {
        values[column * stride + row]
    }

    fn accumulate(accumulator: Bls12_381Fr, y: Bls12_381Fr, term: Bls12_381Fr) -> Bls12_381Fr {
        blst_add(blst_mul(accumulator, y), term)
    }

    fn fr(value: u64) -> Bls12_381Fr {
        let mut output = blst::blst_fr::default();
        let canonical = [value, 0, 0, 0];
        // SAFETY: output and the complete four-limb canonical input remain live.
        unsafe {
            blst::blst_fr_from_uint64(&mut output, canonical.as_ptr());
        }
        Bls12_381Fr { limbs: output.l }
    }

    fn blst_add(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
        let mut output = blst::blst_fr::default();
        // SAFETY: all BLST field objects are initialized and non-overlapping.
        unsafe {
            blst::blst_fr_add(
                &mut output,
                &blst::blst_fr { l: lhs.limbs },
                &blst::blst_fr { l: rhs.limbs },
            );
        }
        Bls12_381Fr { limbs: output.l }
    }

    fn blst_sub(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
        let mut output = blst::blst_fr::default();
        // SAFETY: all BLST field objects are initialized and non-overlapping.
        unsafe {
            blst::blst_fr_sub(
                &mut output,
                &blst::blst_fr { l: lhs.limbs },
                &blst::blst_fr { l: rhs.limbs },
            );
        }
        Bls12_381Fr { limbs: output.l }
    }

    fn blst_mul(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
        let mut output = blst::blst_fr::default();
        // SAFETY: all BLST field objects are initialized and non-overlapping.
        unsafe {
            blst::blst_fr_mul(
                &mut output,
                &blst::blst_fr { l: lhs.limbs },
                &blst::blst_fr { l: rhs.limbs },
            );
        }
        Bls12_381Fr { limbs: output.l }
    }
}
