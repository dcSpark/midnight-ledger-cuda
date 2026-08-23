// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Fail-closed, proof-facing diagnostic resident quotient epoch.

use super::*;

const COEFFICIENT_BUFFER: usize = 0;
const COSET_BUFFER: usize = 1;
const AUXILIARY_BUFFER: usize = 2;
const DUMMY_BUFFER: usize = 3;
const QUOTIENT_NTT_TILE_COLUMNS: usize = 8;

const GRAPH_POST_STORE: u32 = 0;
const GRAPH_POST_LOOKUP: u32 = 1;
const GRAPH_POST_TRASH: u32 = 2;

#[cfg(feature = "metal-diagnostics")]
const QUOTIENT_EPOCH_PHASES: [&str; 4] = [
    "coset_forward_ntt",
    "numerator_evaluation",
    "inverse_ntt",
    "split_blind",
];

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

const _: [(); 8] = [(); size_of::<MetalResidentColumnRef>()];
const _: [(); 0] = [(); offset_of!(MetalResidentColumnRef, kind)];
const _: [(); 4] = [(); offset_of!(MetalResidentColumnRef, index)];

#[repr(C)]
#[derive(Clone, Copy)]
struct QuotientCopyParameters {
    len: u32,
    source_stride: u32,
    destination_stride: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QuotientPermutationParameters {
    len: u32,
    stride: u32,
    rotation_scale: i32,
    last_rotation: i32,
    fixed_columns: u32,
    advice_offset: u32,
    advice_columns: u32,
    instance_offset: u32,
    instance_columns: u32,
    permutation_offset: u32,
    permutation_sets: u32,
    permutation_column_count: u32,
    chunk_len: u32,
    reserved: u32,
    beta: Bls12_381Fr,
    gamma: Bls12_381Fr,
    y: Bls12_381Fr,
    root: Bls12_381Fr,
    coset_generator: Bls12_381Fr,
    delta: Bls12_381Fr,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QuotientMultiplyParameters {
    len: u32,
    period: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QuotientSplitParameters {
    piece_len: u32,
    piece_count: u32,
    source_len: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QuotientCombineParameters {
    piece_len: u32,
    piece_count: u32,
    reserved_0: u32,
    reserved_1: u32,
    splitting_factor: Bls12_381Fr,
}

const _: [(); 16] = [(); size_of::<QuotientCopyParameters>()];
const _: [(); 0] = [(); offset_of!(QuotientCopyParameters, len)];
const _: [(); 4] = [(); offset_of!(QuotientCopyParameters, source_stride)];
const _: [(); 8] = [(); offset_of!(QuotientCopyParameters, destination_stride)];
const _: [(); 12] = [(); offset_of!(QuotientCopyParameters, reserved)];
const _: [(); 248] = [(); size_of::<QuotientPermutationParameters>()];
const _: [(); 0] = [(); offset_of!(QuotientPermutationParameters, len)];
const _: [(); 52] = [(); offset_of!(QuotientPermutationParameters, reserved)];
const _: [(); 8] = [(); size_of::<QuotientMultiplyParameters>()];
const _: [(); 16] = [(); size_of::<QuotientSplitParameters>()];
const _: [(); 48] = [(); size_of::<QuotientCombineParameters>()];
const _: [(); 56] = [(); offset_of!(QuotientPermutationParameters, beta)];
const _: [(); 88] = [(); offset_of!(QuotientPermutationParameters, gamma)];
const _: [(); 120] = [(); offset_of!(QuotientPermutationParameters, y)];
const _: [(); 152] = [(); offset_of!(QuotientPermutationParameters, root)];
const _: [(); 184] = [(); offset_of!(QuotientPermutationParameters, coset_generator)];
const _: [(); 216] = [(); offset_of!(QuotientPermutationParameters, delta)];
const _: [(); 0] = [(); offset_of!(QuotientMultiplyParameters, len)];
const _: [(); 4] = [(); offset_of!(QuotientMultiplyParameters, period)];
const _: [(); 0] = [(); offset_of!(QuotientSplitParameters, piece_len)];
const _: [(); 4] = [(); offset_of!(QuotientSplitParameters, piece_count)];
const _: [(); 8] = [(); offset_of!(QuotientSplitParameters, source_len)];
const _: [(); 12] = [(); offset_of!(QuotientSplitParameters, reserved)];
const _: [(); 0] = [(); offset_of!(QuotientCombineParameters, piece_len)];
const _: [(); 4] = [(); offset_of!(QuotientCombineParameters, piece_count)];
const _: [(); 16] = [(); offset_of!(QuotientCombineParameters, splitting_factor)];

struct ValidatedQuotientEpoch<'a> {
    request: MetalQuotientEpochRequest<'a>,
    n: usize,
    n_ext: usize,
    quotient_piece_count: usize,
    lookup_count: usize,
    trash_count: usize,
    extended_domain: NttDomainSpec,
}

/// One owner for every request-secret allocation in the quotient epoch.
///
/// `buffers` contains exact D=C*n coefficients, exact E=C*n_ext transformed
/// columns, and a small challenge/blind block. The two resident slots hold one
/// bounded multi-column NTT tile and are later reused for numerator, inverse
/// NTT, and quotient splitting.
struct QuotientEpochOwner<'session> {
    session: &'session MetalSession,
    scratch: ResidentFrArena<'session>,
    buffers: Vec<SecretBufferLease>,
    ntt_tile_columns: usize,
}

impl<'session> QuotientEpochOwner<'session> {
    fn allocate(
        session: &'session MetalSession,
        validated: &ValidatedQuotientEpoch<'_>,
    ) -> Result<Self, MetalError> {
        let arena_id = session.next_resident_arena_id()?;
        let mut owner = Self {
            session,
            scratch: ResidentFrArena {
                session,
                arena_id,
                slots: Vec::with_capacity(2),
                write_generations: [0, 0],
                slot_states: [None, None],
            },
            buffers: Vec::with_capacity(4),
            ntt_tile_columns: validated
                .request
                .coefficient_columns
                .len()
                .min(QUOTIENT_NTT_TILE_COLUMNS),
        };

        let columns = validated.request.coefficient_columns.len();
        let coefficient_elements = columns
            .checked_mul(validated.n)
            .ok_or_else(|| quotient_error("coefficient arena element count overflows"))?;
        let coset_elements = columns
            .checked_mul(validated.n_ext)
            .ok_or_else(|| quotient_error("coset arena element count overflows"))?;
        let auxiliary_elements = validated
            .request
            .challenges
            .len()
            .checked_add(validated.request.quotient_blinds.len())
            .ok_or_else(|| quotient_error("challenge and blind count overflows"))?;

        owner.buffers.push(session.secret_buffer(
            fr_bytes(coefficient_elements)?,
            "midnight.quotient.coefficients",
        )?);
        owner
            .buffers
            .push(session.secret_buffer(fr_bytes(coset_elements)?, "midnight.quotient.cosets")?);
        owner.buffers.push(session.secret_buffer(
            fr_bytes(auxiliary_elements)?,
            "midnight.quotient.challenges_blinds",
        )?);
        owner.buffers.push(session.secret_buffer(
            size_of::<Bls12_381Fr>(),
            "midnight.quotient.nonaliasing_dummy",
        )?);
        let tile_elements = owner
            .ntt_tile_columns
            .checked_mul(validated.n_ext)
            .ok_or_else(|| quotient_error("quotient NTT tile element count overflows"))?;
        for label in ["midnight.quotient.tile_ping", "midnight.quotient.tile_pong"] {
            owner
                .scratch
                .slots
                .push(session.secret_buffer(fr_bytes(tile_elements)?, label)?);
        }

        for (column_index, column) in validated
            .request
            .coefficient_columns
            .iter()
            .copied()
            .enumerate()
        {
            // SAFETY: validation fixes every source at exactly n Fr elements;
            // the destination is the disjoint column_index*n range in D.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    column.as_ptr(),
                    owner.buffers[COEFFICIENT_BUFFER]
                        .raw
                        .contents()
                        .as_ptr()
                        .cast::<Bls12_381Fr>()
                        .add(column_index * validated.n),
                    validated.n,
                );
            }
        }
        let auxiliary = &owner.buffers[AUXILIARY_BUFFER];
        // SAFETY: the auxiliary allocation is exactly challenges followed by
        // quotient blinds and both source slices contain padding-free RawFr.
        unsafe {
            std::ptr::copy_nonoverlapping(
                validated.request.challenges.as_ptr(),
                auxiliary.raw.contents().as_ptr().cast::<Bls12_381Fr>(),
                validated.request.challenges.len(),
            );
            std::ptr::copy_nonoverlapping(
                validated.request.quotient_blinds.as_ptr(),
                auxiliary
                    .raw
                    .contents()
                    .as_ptr()
                    .cast::<Bls12_381Fr>()
                    .add(validated.request.challenges.len()),
                validated.request.quotient_blinds.len(),
            );
            owner.buffers[DUMMY_BUFFER]
                .raw
                .contents()
                .as_ptr()
                .cast::<Bls12_381Fr>()
                .write(Bls12_381Fr::default());
        }
        let written = coefficient_elements
            .checked_add(auxiliary_elements)
            .and_then(|elements| elements.checked_add(1))
            .and_then(|elements| elements.checked_mul(size_of::<Bls12_381Fr>()))
            .ok_or_else(|| quotient_error("quotient host-write byte count overflows"))?;
        session
            .activity
            .host_write_bytes
            .fetch_add(written as u64, Ordering::Relaxed);
        Ok(owner)
    }

    fn retained_resources(&self) -> Vec<Buffer> {
        self.buffers
            .iter()
            .map(|buffer| buffer.raw.clone())
            .chain(self.scratch.slots.iter().map(|buffer| buffer.raw.clone()))
            .collect()
    }

    fn finish<T>(mut self, operation: Result<T, MetalError>) -> Result<T, MetalError> {
        let cleanup = self.cleanup();
        match (operation, cleanup) {
            (result, Ok(())) => result,
            (Ok(_), Err(cleanup)) => Err(cleanup),
            (Err(operation), Err(cleanup)) => Err(MetalError::SessionPoisoned {
                reason: format!("{cleanup}; original quotient epoch error: {operation}"),
            }),
        }
    }

    fn cleanup(&mut self) -> Result<(), MetalError> {
        if self.buffers.is_empty() && self.scratch.slots.is_empty() {
            return Ok(());
        }
        let mut allocations = std::mem::take(&mut self.buffers);
        allocations.append(&mut self.scratch.slots);
        self.scratch.slot_states = [None, None];
        self.session.clear_and_recycle_secret_buffers(allocations)
    }
}

impl Drop for QuotientEpochOwner<'_> {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

struct EpochKeyResources {
    constants: Buffer,
    rotations: Buffer,
    operations: Buffer,
    horner_parts: Buffer,
    fixed_cosets: Buffer,
    selectors: Buffer,
    permutation_columns: Buffer,
    permutation_cosets: Buffer,
    extended_coset_points: Buffer,
    vanishing_inverses: Buffer,
}

struct CompletedQuotientEpoch<'a> {
    buffer: &'a Buffer,
    continuation_buffer: &'a Buffer,
    continuation_buffer_bytes: usize,
    elements: usize,
    bytes: usize,
    piece_len: usize,
    piece_count: usize,
}

struct CompletedQuotientEpochLocation {
    piece_slot: usize,
    elements: usize,
    bytes: usize,
    piece_len: usize,
    piece_count: usize,
}

impl MetalSession {
    /// Executes the diagnostic field-side quotient epoch and temporarily lends
    /// its exact blinded pieces to the existing CPU commitment implementation.
    ///
    /// The method never routes production proofs or claims pure-Metal backend
    /// readiness. It records one work command, waits once at the intentional
    /// CPU MSM boundary, invokes `with_cpu_msm`, then GPU-clears every secret
    /// allocation before returning. A callback panic is resumed only after the
    /// same cleanup path runs.
    pub fn with_diagnostic_quotient_epoch<T, F>(
        &self,
        context: &MetalKeyContext,
        request: MetalQuotientEpochRequest<'_>,
        with_cpu_msm: F,
    ) -> Result<T, MetalError>
    where
        F: for<'lease> FnOnce(MetalQuotientCpuMsmLease<'lease>) -> T,
    {
        self.with_quotient_epoch_output(context, request, |completed| {
            // SAFETY: the completed split kernel initialized this exact prefix.
            // The scoped callback finishes before owner cleanup can begin.
            let values = unsafe {
                std::slice::from_raw_parts(
                    completed.buffer.contents().as_ptr().cast::<Bls12_381Fr>(),
                    completed.elements,
                )
            };
            let resident_scalars = ResidentMontgomeryScalars::from_synchronized_mtl_buffer(
                completed.buffer.clone(),
                0,
                completed.elements,
            )
            .map_err(|error| MetalError::Msm(error.to_string()))?;
            self.activity
                .host_read_bytes
                .fetch_add(completed.bytes as u64, Ordering::Relaxed);
            self.activity.cpu_msm_leases.fetch_add(1, Ordering::Relaxed);
            self.activity
                .cpu_msm_lease_bytes
                .fetch_add(completed.bytes as u64, Ordering::Relaxed);
            Ok(with_cpu_msm(MetalQuotientCpuMsmLease::new(
                values,
                completed.piece_len,
                completed.piece_count,
                context.identity(),
                resident_scalars,
            )))
        })
    }

    /// Executes the quotient epoch and lends its completed pieces directly to Metal MSM.
    ///
    /// No quotient coefficient is mapped or read by the CPU. The callback may
    /// synchronously submit the opaque resident view to the standalone MSM
    /// worker; cleanup and erasure begin only after it returns.
    pub fn with_resident_quotient_epoch<T, F>(
        &self,
        context: &MetalKeyContext,
        request: MetalQuotientEpochRequest<'_>,
        with_msm: F,
    ) -> Result<T, MetalError>
    where
        F: for<'lease> FnOnce(MetalQuotientResidentMsmLease<'lease>) -> T,
    {
        self.with_quotient_epoch_output(context, request, |completed| {
            let scalars = ResidentMontgomeryScalars::from_synchronized_mtl_buffer(
                completed.buffer.clone(),
                0,
                completed.elements,
            )
            .map_err(|error| MetalError::Msm(error.to_string()))?;
            self.activity
                .resident_msm_leases
                .fetch_add(1, Ordering::Relaxed);
            self.activity
                .resident_msm_lease_bytes
                .fetch_add(completed.bytes as u64, Ordering::Relaxed);
            Ok(with_msm(MetalQuotientResidentMsmLease::new(
                scalars,
                QuotientContinuationBuffers {
                    pieces: completed.buffer.clone(),
                    scratch: completed.continuation_buffer.clone(),
                    scratch_bytes: completed.continuation_buffer_bytes,
                    session_id: self.session_id,
                },
                completed.piece_len,
                completed.piece_count,
                context.identity(),
            )))
        })
    }

    /// Combines resident quotient pieces and materializes one opening polynomial.
    ///
    /// The lease can exist only inside [`Self::with_resident_quotient_epoch`].
    /// The source pieces stay opaque; this transitional API reads only the
    /// resulting base-domain polynomial so an existing host multi-open
    /// implementation can continue exactly.
    pub fn materialize_bls12_381_quotient_combination(
        &self,
        lease: &MetalQuotientResidentMsmLease<'_>,
        splitting_factor: Bls12_381Fr,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        if lease.continuation.session_id != self.session_id
            || lease.scalars.device_registry_id() != self.device_registry_id
        {
            return Err(MetalError::ForeignKeyContext);
        }
        if lease.piece_len == 0 || lease.piece_count < 2 {
            return Err(quotient_error(
                "resident quotient continuation requires at least two nonempty pieces",
            ));
        }
        let output_bytes = fr_bytes(lease.piece_len)?;
        if output_bytes > lease.continuation.scratch_bytes {
            return Err(MetalError::BufferBounds {
                requested: output_bytes,
                available: lease.continuation.scratch_bytes,
            });
        }
        validate_quotient_reduced("quotient splitting factor", &[splitting_factor])?;
        let _execution = self.execution_guard();
        self.ensure_session_healthy()?;

        let parameters = QuotientCombineParameters {
            piece_len: u32_len(lease.piece_len, "quotient continuation piece length")?,
            piece_count: u32_len(lease.piece_count, "quotient continuation piece count")?,
            reserved_0: 0,
            reserved_1: 0,
            splitting_factor,
        };
        let command = self.command_buffer("midnight.quotient.combine_pieces")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.quotient.combine_pieces);
        // SAFETY: the callback-scoped lease pins both buffers until the epoch
        // owner clears them. Its validated dimensions cover every source piece
        // and the checked scratch prefix written by this dispatch.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&lease.continuation.pieces), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&lease.continuation.scratch), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::from(&parameters).cast::<c_void>(),
                size_of::<QuotientCombineParameters>(),
                2,
            );
        }
        self.dispatch(&encoder, &self.quotient.combine_pieces, lease.piece_len);
        encoder.endEncoding();
        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![
                lease.continuation.pieces.clone(),
                lease.continuation.scratch.clone(),
            ],
        })?;
        self.activity
            .materializations
            .fetch_add(1, Ordering::Relaxed);
        self.activity
            .whole_polynomial_materializations
            .fetch_add(1, Ordering::Relaxed);
        self.read_values_from_raw(
            &lease.continuation.scratch,
            lease.continuation.scratch_bytes,
            lease.piece_len,
        )
    }

    fn with_quotient_epoch_output<T, F>(
        &self,
        context: &MetalKeyContext,
        request: MetalQuotientEpochRequest<'_>,
        with_output: F,
    ) -> Result<T, MetalError>
    where
        F: for<'lease> FnOnce(CompletedQuotientEpoch<'lease>) -> Result<T, MetalError>,
    {
        let started = Instant::now();
        let materializations_before = self.activity.materializations.load(Ordering::Relaxed);
        let whole_materializations_before = self
            .activity
            .whole_polynomial_materializations
            .load(Ordering::Relaxed);
        let validated = match self.validate_quotient_epoch_request(context, request) {
            Ok(validated) => validated,
            Err(error) => {
                self.activity
                    .quotient_epoch_failures
                    .fetch_add(1, Ordering::Relaxed);
                self.activity
                    .quotient_epoch_failure_elapsed_ns
                    .fetch_add(elapsed_ns(started), Ordering::Relaxed);
                return Err(error);
            }
        };
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let (mut owner, completed) = {
                let _execution = self.execution_guard();
                self.ensure_session_healthy()?;
                let mut owner = QuotientEpochOwner::allocate(self, &validated)?;
                let completed = self.encode_quotient_epoch(context, &validated, &mut owner)?;
                (owner, completed)
            };

            // The quotient pieces are complete and pinned by `owner`. Release
            // the proof-runtime mutex while the callback waits on the
            // independent standalone MSM queue.
            let callback_outcome = catch_unwind(AssertUnwindSafe(|| {
                let completed = CompletedQuotientEpoch {
                    buffer: &owner.scratch.slots[completed.piece_slot].raw,
                    continuation_buffer: &owner.scratch.slots[1 - completed.piece_slot].raw,
                    continuation_buffer_bytes: owner.scratch.slots[1 - completed.piece_slot].bytes,
                    elements: completed.elements,
                    bytes: completed.bytes,
                    piece_len: completed.piece_len,
                    piece_count: completed.piece_count,
                };
                with_output(completed)
            }));
            let execution = self.execution_guard();
            match callback_outcome {
                Ok(operation) => owner.finish(operation),
                Err(payload) => {
                    let _ = owner.cleanup();
                    drop(execution);
                    resume_unwind(payload)
                }
            }
        }));
        self.record_quotient_materialization_delta(
            materializations_before,
            whole_materializations_before,
        );
        match outcome {
            Ok(Ok(value)) => {
                self.activity
                    .quotient_epochs
                    .fetch_add(1, Ordering::Relaxed);
                self.activity
                    .quotient_epoch_elapsed_ns
                    .fetch_add(elapsed_ns(started), Ordering::Relaxed);
                Ok(value)
            }
            Ok(Err(error)) => {
                self.activity
                    .quotient_epoch_failures
                    .fetch_add(1, Ordering::Relaxed);
                self.activity
                    .quotient_epoch_failure_elapsed_ns
                    .fetch_add(elapsed_ns(started), Ordering::Relaxed);
                Err(error)
            }
            Err(payload) => {
                self.activity
                    .quotient_epoch_failures
                    .fetch_add(1, Ordering::Relaxed);
                self.activity
                    .quotient_epoch_failure_elapsed_ns
                    .fetch_add(elapsed_ns(started), Ordering::Relaxed);
                resume_unwind(payload)
            }
        }
    }

    fn record_quotient_materialization_delta(
        &self,
        materializations_before: u64,
        whole_materializations_before: u64,
    ) {
        let materializations = self
            .activity
            .materializations
            .load(Ordering::Relaxed)
            .saturating_sub(materializations_before);
        let whole_materializations = self
            .activity
            .whole_polynomial_materializations
            .load(Ordering::Relaxed)
            .saturating_sub(whole_materializations_before);
        self.activity
            .quotient_materializations
            .fetch_add(materializations, Ordering::Relaxed);
        self.activity
            .quotient_whole_polynomial_materializations
            .fetch_add(whole_materializations, Ordering::Relaxed);
    }

    #[cfg(feature = "metal-diagnostics")]
    fn new_quotient_epoch_phase_samples(&self) -> Option<CounterSampleBuffer> {
        if std::env::var_os("MIDNIGHT_METAL_TRACE_QUOTIENT_PHASES").is_none()
            || !self
                .device
                .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary)
        {
            return None;
        }
        let counter_sets = self.device.counterSets()?;
        // SAFETY: this is an immutable Metal framework string constant.
        let timestamp_name = unsafe { MTLCommonCounterSetTimestamp };
        let mut timestamp_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>> = None;
        for index in 0..counter_sets.count() {
            let candidate = counter_sets.objectAtIndex(index);
            if &*candidate.name() == timestamp_name {
                timestamp_set = Some(candidate);
                break;
            }
        }
        let timestamp_set = timestamp_set?;
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(&timestamp_set));
        descriptor.setLabel(&NSString::from_str("midnight.quotient.epoch.phases"));
        descriptor.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: each profiled compute pass records one start/end pair.
        unsafe { descriptor.setSampleCount(QUOTIENT_EPOCH_PHASES.len() * 2) };
        self.device
            .newCounterSampleBufferWithDescriptor_error(&descriptor)
            .ok()
    }

    #[cfg(feature = "metal-diagnostics")]
    fn quotient_epoch_phase_encoder(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        samples: Option<&ProtocolObject<dyn MTLCounterSampleBuffer>>,
        phase: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, MetalError> {
        let Some(samples) = samples else {
            return command.computeCommandEncoder().ok_or(MetalError::Encoder);
        };
        let descriptor = MTLComputePassDescriptor::new();
        // SAFETY: Metal compute passes expose one counter attachment and the
        // sample buffer has one start/end pair for every named phase.
        let attachment = unsafe {
            descriptor
                .sampleBufferAttachments()
                .objectAtIndexedSubscript(0)
        };
        attachment.setSampleBuffer(Some(samples));
        unsafe {
            attachment.setStartOfEncoderSampleIndex(phase * 2);
            attachment.setEndOfEncoderSampleIndex(phase * 2 + 1);
        }
        command
            .computeCommandEncoderWithDescriptor(&descriptor)
            .ok_or(MetalError::Encoder)
    }

    #[cfg(feature = "metal-diagnostics")]
    fn advance_quotient_epoch_phase(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        encoder: &mut Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        samples: Option<&ProtocolObject<dyn MTLCounterSampleBuffer>>,
        next_phase: usize,
    ) -> Result<(), MetalError> {
        if samples.is_none() {
            return Ok(());
        }
        encoder.endEncoding();
        *encoder = self.quotient_epoch_phase_encoder(command, samples, next_phase)?;
        Ok(())
    }

    #[cfg(feature = "metal-diagnostics")]
    fn trace_quotient_epoch_phases(
        &self,
        samples: Option<&ProtocolObject<dyn MTLCounterSampleBuffer>>,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
    ) {
        if std::env::var_os("MIDNIGHT_METAL_TRACE_QUOTIENT_PHASES").is_none() {
            return;
        }
        let Some(samples) = samples else {
            eprintln!(
                "{{\"schema\":\"midnight-metal-prover/timing/quotient-phases-v1\",\"available\":false}}"
            );
            return;
        };
        // SAFETY: the command completed before this shared counter buffer is resolved.
        let Some(data) = (unsafe {
            samples.resolveCounterRange(NSRange::new(0, QUOTIENT_EPOCH_PHASES.len() * 2))
        }) else {
            return;
        };
        // SAFETY: NSData remains immutable while its bytes are copied below.
        let bytes = unsafe { data.as_bytes_unchecked() };
        let sample_count = QUOTIENT_EPOCH_PHASES.len() * 2;
        if bytes.len() < sample_count * size_of::<u64>() {
            return;
        }
        let timestamps = bytes
            .chunks_exact(size_of::<u64>())
            .take(sample_count)
            .map(|bytes| u64::from_ne_bytes(bytes.try_into().expect("counter timestamp width")))
            .collect::<Vec<_>>();
        if timestamps.iter().any(|timestamp| *timestamp == u64::MAX) {
            return;
        }
        let total_ticks = (0..QUOTIENT_EPOCH_PHASES.len())
            .map(|index| timestamps[index * 2 + 1].saturating_sub(timestamps[index * 2]))
            .sum::<u64>();
        if total_ticks == 0 {
            return;
        }
        // SAFETY: these read-only selectors are queried after synchronous completion.
        let gpu_start: f64 = unsafe { objc2::msg_send![command, GPUStartTime] };
        let gpu_end: f64 = unsafe { objc2::msg_send![command, GPUEndTime] };
        let gpu_ms = (gpu_start.is_finite()
            && gpu_end.is_finite()
            && gpu_end >= gpu_start
            && gpu_start > 0.0)
            .then(|| (gpu_end - gpu_start) * 1_000.0);
        let phases = QUOTIENT_EPOCH_PHASES
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let ticks = timestamps[index * 2 + 1].saturating_sub(timestamps[index * 2]);
                let calibrated_ms = gpu_ms.map(|total| total * ticks as f64 / total_ticks as f64);
                format!(
                    "\"{name}\":{{\"ticks\":{ticks},\"calibrated_ms\":{}}}",
                    calibrated_ms
                        .map_or_else(|| String::from("null"), |duration| format!("{duration:.6}")),
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "{{\"schema\":\"midnight-metal-prover/timing/quotient-phases-v1\",\"available\":true,\"tick_sum\":{total_ticks},\"gpu_ms\":{},\"phases\":{{{phases}}}}}",
            gpu_ms.map_or_else(|| String::from("null"), |duration| format!("{duration:.6}")),
        );
    }

    fn encode_quotient_epoch<'owner>(
        &self,
        context: &MetalKeyContext,
        validated: &ValidatedQuotientEpoch<'_>,
        owner: &'owner mut QuotientEpochOwner<'_>,
    ) -> Result<CompletedQuotientEpochLocation, MetalError> {
        #[cfg(feature = "metal-diagnostics")]
        if std::env::var_os("MIDNIGHT_METAL_TRACE_QUOTIENT_PHASES").is_some() {
            let graphs = context
                .inner
                .descriptors
                .iter()
                .map(|graph| {
                    format!(
                        "{{\"operations\":{},\"parts\":{},\"scratch\":{}}}",
                        graph.operations_count, graph.parts_count, graph.scratch_capacity,
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "{{\"schema\":\"midnight-metal-prover/timing/quotient-shape-v1\",\"n\":{},\"n_ext\":{},\"columns\":{},\"tile_columns\":{},\"graphs\":[{}]}}",
                validated.n,
                validated.n_ext,
                validated.request.coefficient_columns.len(),
                owner.ntt_tile_columns,
                graphs,
            );
        }
        let plan = self.ntt_plan(validated.extended_domain)?;
        let mut resources = owner.retained_resources();
        let key = self.bind_quotient_epoch_key_resources(
            context,
            &owner.buffers[DUMMY_BUFFER].raw,
            &mut resources,
        )?;
        let command = self.command_buffer("midnight.quotient.diagnostic_epoch")?;
        #[cfg(feature = "metal-diagnostics")]
        let phase_samples = self.new_quotient_epoch_phase_samples();
        #[cfg(feature = "metal-diagnostics")]
        let mut encoder =
            self.quotient_epoch_phase_encoder(&command, phase_samples.as_deref(), 0)?;
        #[cfg(not(feature = "metal-diagnostics"))]
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;

        for column_start in
            (0..validated.request.coefficient_columns.len()).step_by(owner.ntt_tile_columns)
        {
            let column_count = owner
                .ntt_tile_columns
                .min(validated.request.coefficient_columns.len() - column_start);
            let coset_coefficients = self.encode_quotient_coset_prepare(
                &encoder,
                owner,
                column_start,
                column_count,
                validated,
            )?;
            let evaluations = self.encode_ntt(
                &encoder,
                &mut resources,
                &mut owner.scratch,
                coset_coefficients,
                &plan,
                NttDirection::Forward,
                false,
            )?;
            for tile_column in 0..column_count {
                let evaluation = owner.scratch.columns(&evaluations, tile_column, 1)?;
                self.encode_quotient_copy_to_coset_arena(
                    &encoder,
                    owner,
                    &evaluation,
                    column_start + tile_column,
                    validated.n_ext,
                )?;
            }
        }
        #[cfg(feature = "metal-diagnostics")]
        self.advance_quotient_epoch_phase(&command, &mut encoder, phase_samples.as_deref(), 1)?;

        let mut numerator = publish_scratch_write(
            &mut owner.scratch,
            0,
            validated.n_ext,
            FrDomainTag::for_ntt(validated.extended_domain, FrBasis::ExtendedCoset),
        )?;
        if context.inner.sections.custom == 0 {
            self.encode_quotient_zero(&encoder, owner, &numerator, validated.n_ext)?;
        } else {
            self.encode_quotient_graph(
                &encoder,
                context,
                validated,
                owner,
                &key,
                &numerator,
                0,
                GRAPH_POST_STORE,
                0,
                0,
                0,
                0,
                0,
                0,
            )?;
        }

        self.encode_quotient_permutation(&encoder, context, validated, owner, &key, &numerator)?;

        let custom = context.inner.sections.custom as usize;
        for lookup in 0..validated.lookup_count {
            self.encode_quotient_graph(
                &encoder,
                context,
                validated,
                owner,
                &key,
                &numerator,
                custom + lookup,
                GRAPH_POST_LOOKUP,
                1,
                validated.request.lookup_product_offset as usize + lookup,
                validated.request.lookup_input_offset as usize + lookup,
                validated.request.lookup_table_offset as usize + lookup,
                0,
                0,
            )?;
        }
        let trash_graph_start = custom + 3 * validated.lookup_count;
        for trash in 0..validated.trash_count {
            self.encode_quotient_graph(
                &encoder,
                context,
                validated,
                owner,
                &key,
                &numerator,
                trash_graph_start + trash,
                GRAPH_POST_TRASH,
                1,
                0,
                0,
                0,
                validated.request.trash_offset as usize + trash,
                context.inner.trash_selector_fixed_indices[trash] as usize,
            )?;
        }

        self.encode_quotient_vanishing_multiply(&encoder, validated, owner, &key, &numerator)?;
        #[cfg(feature = "metal-diagnostics")]
        self.advance_quotient_epoch_phase(&command, &mut encoder, phase_samples.as_deref(), 2)?;
        numerator = self.encode_ntt(
            &encoder,
            &mut resources,
            &mut owner.scratch,
            numerator,
            &plan,
            NttDirection::Inverse,
            true,
        )?;
        #[cfg(feature = "metal-diagnostics")]
        self.advance_quotient_epoch_phase(&command, &mut encoder, phase_samples.as_deref(), 3)?;
        let piece_slot = 1 - numerator.slot;
        self.encode_quotient_split_blind(&encoder, validated, owner, &numerator, piece_slot)?;
        encoder.endEncoding();

        #[cfg(feature = "metal-diagnostics")]
        let timing_command = command.clone();
        let completion = self.wait_for_completion(self.submit(PendingCommand {
            command,
            _resources: resources,
        }));
        #[cfg(feature = "metal-diagnostics")]
        if completion.is_ok() {
            self.trace_quotient_epoch_phases(phase_samples.as_deref(), &timing_command);
        }
        completion?;

        let lease_elements = validated
            .quotient_piece_count
            .checked_mul(validated.n)
            .ok_or_else(|| quotient_error("CPU MSM lease element count overflows"))?;
        let lease_bytes = fr_bytes(lease_elements)?;
        Ok(CompletedQuotientEpochLocation {
            piece_slot,
            elements: lease_elements,
            bytes: lease_bytes,
            piece_len: validated.n,
            piece_count: validated.quotient_piece_count,
        })
    }

    fn validate_quotient_epoch_request<'a>(
        &self,
        context: &MetalKeyContext,
        request: MetalQuotientEpochRequest<'a>,
    ) -> Result<ValidatedQuotientEpoch<'a>, MetalError> {
        self.validate_key_context(context)?;
        let identity = context.identity();
        let base_domain = identity.base_ntt_domain().ok_or_else(|| {
            quotient_error("diagnostic quotient epochs require a schema-v2 key context")
        })?;
        let extended_domain = identity.extended_ntt_domain().ok_or_else(|| {
            quotient_error("diagnostic quotient epochs require an exact extended domain")
        })?;
        validate_ntt_domain_pair(base_domain, extended_domain)?;
        let n = identity.n() as usize;
        let n_ext = identity.n_ext() as usize;
        if n < 2
            || (1usize << base_domain.log_size) != n
            || (1usize << extended_domain.log_size) != n_ext
        {
            return Err(quotient_error(
                "key-context domain lengths are inconsistent",
            ));
        }
        let quotient_piece_count = context.inner.quotient_piece_count;
        if quotient_piece_count < 2
            || quotient_piece_count
                .checked_mul(n)
                .is_none_or(|elements| elements > n_ext)
        {
            return Err(quotient_error(
                "q*n does not fit the extended quotient arena",
            ));
        }
        if request.quotient_blinds.len() != quotient_piece_count - 1 {
            return Err(quotient_error("quotient blind count is not q-1"));
        }
        if request.total_columns as usize != request.coefficient_columns.len()
            || request.coefficient_columns.is_empty()
        {
            return Err(quotient_error(
                "total_columns does not name a nonempty borrowed column set",
            ));
        }
        for (index, column) in request.coefficient_columns.iter().copied().enumerate() {
            if column.len() != n {
                return Err(quotient_error(format!(
                    "coefficient column {index} has length {}, expected {n}",
                    column.len()
                )));
            }
            validate_quotient_reduced(&format!("coefficient column {index}"), column)?;
        }

        let lookup_count = context.inner.sections.lookups as usize;
        let trash_count = context.inner.sections.trash as usize;
        let permutation_sets = quotient_permutation_sets(
            context.inner.permutation_column_count,
            context.inner.permutation_chunk_len,
        )?;
        if request.advice_columns as usize != context.inner.advice_column_count
            || request.instance_columns as usize != context.inner.instance_column_count
            || request.permutation_sets as usize != permutation_sets
        {
            return Err(quotient_error(
                "dynamic advice, instance, or permutation shape differs from the key context",
            ));
        }

        let mut cursor = 0usize;
        require_offset("advice", request.advice_offset, cursor)?;
        cursor = checked_advance(cursor, context.inner.advice_column_count, "advice")?;
        require_offset("instance", request.instance_offset, cursor)?;
        cursor = checked_advance(cursor, context.inner.instance_column_count, "instance")?;
        require_offset("permutation", request.permutation_offset, cursor)?;
        cursor = checked_advance(cursor, permutation_sets, "permutation")?;
        require_offset("lookup product", request.lookup_product_offset, cursor)?;
        cursor = checked_advance(cursor, lookup_count, "lookup product")?;
        require_offset("lookup permuted input", request.lookup_input_offset, cursor)?;
        cursor = checked_advance(cursor, lookup_count, "lookup permuted input")?;
        require_offset("lookup permuted table", request.lookup_table_offset, cursor)?;
        cursor = checked_advance(cursor, lookup_count, "lookup permuted table")?;
        require_offset("trash", request.trash_offset, cursor)?;
        cursor = checked_advance(cursor, trash_count, "trash")?;
        if cursor != request.total_columns as usize {
            return Err(quotient_error(
                "canonical dynamic source ranges do not cover total_columns",
            ));
        }
        if request.challenges.len() < context.inner.challenge_bound as usize {
            return Err(quotient_error(
                "challenge slice is shorter than the compiled graph bound",
            ));
        }
        u32::try_from(request.challenges.len())
            .map_err(|_| quotient_error("challenge count exceeds the shader ABI"))?;
        validate_quotient_reduced("challenge", request.challenges)?;
        validate_quotient_reduced("quotient blind", request.quotient_blinds)?;
        validate_quotient_reduced(
            "quotient scalar",
            &[
                request.beta,
                request.gamma,
                request.theta,
                request.trash_challenge,
                request.y,
            ],
        )?;
        if context.inner.descriptors.len() != context.inner.graph_count
            || context.inner.permutation_columns.len() != context.inner.permutation_column_count
            || context.inner.trash_selector_fixed_indices.len() != trash_count
        {
            return Err(quotient_error(
                "key-context host metadata differs from its validated immutable arrays",
            ));
        }
        Ok(ValidatedQuotientEpoch {
            request,
            n,
            n_ext,
            quotient_piece_count,
            lookup_count,
            trash_count,
            extended_domain,
        })
    }

    fn bind_quotient_epoch_key_resources(
        &self,
        context: &MetalKeyContext,
        dummy: &Buffer,
        resources: &mut Vec<Buffer>,
    ) -> Result<EpochKeyResources, MetalError> {
        let descriptors = &context.inner.descriptors;
        let constants_required = descriptors.iter().any(|graph| graph.constants_count != 0);
        let rotations_required = descriptors.iter().any(|graph| graph.rotations_count != 0);
        let operations_required = descriptors.iter().any(|graph| graph.operations_count != 0);
        let parts_required = descriptors.iter().any(|graph| graph.parts_count != 0);
        let constants = self.bind_quotient_raw_key_buffer(
            context,
            context.inner.buffers.constants.as_ref(),
            ImmutableBufferSource::GraphConstants,
            constants_required,
            dummy,
            resources,
        )?;
        let rotations = self.bind_quotient_raw_key_buffer(
            context,
            context.inner.buffers.rotations.as_ref(),
            ImmutableBufferSource::GraphRotations,
            rotations_required,
            dummy,
            resources,
        )?;
        let operations = self.bind_quotient_raw_key_buffer(
            context,
            context.inner.buffers.operations.as_ref(),
            ImmutableBufferSource::GraphOperations,
            operations_required,
            dummy,
            resources,
        )?;
        let horner_parts = self.bind_quotient_raw_key_buffer(
            context,
            context.inner.buffers.horner_parts.as_ref(),
            ImmutableBufferSource::GraphHornerParts,
            parts_required,
            dummy,
            resources,
        )?;

        let fixed_cosets = if context.inner.fixed_column_count == 0 {
            dummy.clone()
        } else {
            let view = self.immutable_fr_view(context, MetalKeyFrSource::FixedCosetCompact)?;
            self.bind_immutable_fr_view(
                context,
                &view,
                MetalKeyFrSource::FixedCosetCompact,
                resources,
            )?;
            view.buffer.raw.clone()
        };
        let selectors = {
            let view = self.immutable_fr_view(context, MetalKeyFrSource::Selectors)?;
            self.bind_immutable_fr_view(context, &view, MetalKeyFrSource::Selectors, resources)?;
            view.buffer.raw.clone()
        };
        let (permutation_columns, permutation_cosets, extended_coset_points) =
            if context.inner.permutation_column_count == 0 {
                (dummy.clone(), dummy.clone(), dummy.clone())
            } else {
                let columns = self.bind_quotient_raw_key_buffer(
                    context,
                    context.inner.buffers.permutation_columns.as_ref(),
                    ImmutableBufferSource::PermutationColumns,
                    true,
                    dummy,
                    resources,
                )?;
                let cosets =
                    self.immutable_fr_view(context, MetalKeyFrSource::PermutationCosets)?;
                self.bind_immutable_fr_view(
                    context,
                    &cosets,
                    MetalKeyFrSource::PermutationCosets,
                    resources,
                )?;
                let points =
                    self.immutable_fr_view(context, MetalKeyFrSource::ExtendedCosetPoints)?;
                self.bind_immutable_fr_view(
                    context,
                    &points,
                    MetalKeyFrSource::ExtendedCosetPoints,
                    resources,
                )?;
                (
                    columns,
                    cosets.buffer.raw.clone(),
                    points.buffer.raw.clone(),
                )
            };
        let vanishing_inverses = {
            let view = self
                .immutable_periodic_fr_view(context, MetalKeyPeriodicFrSource::VanishingInverses)?;
            self.bind_immutable_periodic_fr_view(
                context,
                &view,
                MetalKeyPeriodicFrSource::VanishingInverses,
                resources,
            )?;
            view.buffer.raw.clone()
        };
        Ok(EpochKeyResources {
            constants,
            rotations,
            operations,
            horner_parts,
            fixed_cosets,
            selectors,
            permutation_columns,
            permutation_cosets,
            extended_coset_points,
            vanishing_inverses,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_quotient_graph(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        context: &MetalKeyContext,
        validated: &ValidatedQuotientEpoch<'_>,
        owner: &QuotientEpochOwner<'_>,
        key: &EpochKeyResources,
        numerator: &ResidentFrView,
        graph_index: usize,
        post_kind: u32,
        previous_mode: u32,
        lookup_product_column: usize,
        lookup_input_column: usize,
        lookup_table_column: usize,
        trash_column: usize,
        trash_selector_fixed_column: usize,
    ) -> Result<(), MetalError> {
        owner.scratch.validate(numerator)?;
        let descriptor = context
            .inner
            .descriptors
            .get(graph_index)
            .copied()
            .ok_or_else(|| quotient_error("protocol graph index escapes the key context"))?;
        let evaluator = self
            .graph
            .evaluator(descriptor.scratch_capacity as usize)
            .ok_or_else(|| quotient_error("protocol graph has no compiled scratch pipeline"))?;
        let pipeline = context
            .inner
            .specialized_graphs
            .pipeline(graph_index)
            .unwrap_or(&evaluator.pipeline);
        let parameters = GraphEvalParametersAbi {
            len: u32_len(validated.n_ext, "extended graph length")?,
            stride: u32_len(validated.n_ext, "extended graph stride")?,
            rotation_scale: context.inner.rotation_scale,
            post_kind,
            constants_offset: descriptor.constants_offset,
            constants_count: descriptor.constants_count,
            rotations_offset: descriptor.rotations_offset,
            rotations_count: descriptor.rotations_count,
            operations_offset: descriptor.operations_offset,
            operations_count: descriptor.operations_count,
            parts_offset: descriptor.parts_offset,
            parts_count: descriptor.parts_count,
            fixed_columns: u32_len(context.inner.fixed_column_count, "fixed column count")?,
            advice_offset: validated.request.advice_offset,
            advice_columns: validated.request.advice_columns,
            instance_offset: validated.request.instance_offset,
            instance_columns: validated.request.instance_columns,
            challenge_count: u32_len(validated.request.challenges.len(), "challenge count")?,
            previous_mode,
            lookup_product_column: u32_len(lookup_product_column, "lookup product column")?,
            lookup_input_column: u32_len(lookup_input_column, "lookup input column")?,
            lookup_table_column: u32_len(lookup_table_column, "lookup table column")?,
            trash_column: u32_len(trash_column, "trash column")?,
            trash_selector_fixed_column: u32_len(
                trash_selector_fixed_column,
                "trash selector fixed column",
            )?,
            beta: validated.request.beta,
            gamma: validated.request.gamma,
            theta: validated.request.theta,
            trash_challenge: validated.request.trash_challenge,
            y: validated.request.y,
        };

        encoder.setComputePipelineState(pipeline);
        // SAFETY: request/key validation proves every packed graph window,
        // column range, selector range, and numerator view. Every allocation is
        // retained by the one pending epoch command; setBytes copies parameters.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&key.constants), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&key.rotations), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&key.operations), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&key.horner_parts), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&key.fixed_cosets), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&owner.buffers[COSET_BUFFER].raw), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&owner.buffers[AUXILIARY_BUFFER].raw), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&key.selectors), 0, 7);
            encoder.setBuffer_offset_atIndex(
                Some(&owner.scratch.slots[numerator.slot].raw),
                numerator.byte_offset,
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::from(&parameters).cast::<c_void>(),
                size_of::<GraphEvalParametersAbi>(),
                9,
            );
        }
        self.dispatch(encoder, pipeline, validated.n_ext);
        Ok(())
    }

    fn encode_quotient_permutation(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        context: &MetalKeyContext,
        validated: &ValidatedQuotientEpoch<'_>,
        owner: &QuotientEpochOwner<'_>,
        key: &EpochKeyResources,
        numerator: &ResidentFrView,
    ) -> Result<(), MetalError> {
        if validated.request.permutation_sets == 0 {
            return Ok(());
        }
        owner.scratch.validate(numerator)?;
        let parameters = QuotientPermutationParameters {
            len: u32_len(validated.n_ext, "permutation length")?,
            stride: u32_len(validated.n_ext, "permutation stride")?,
            rotation_scale: context.inner.rotation_scale,
            last_rotation: context.inner.last_rotation,
            fixed_columns: u32_len(context.inner.fixed_column_count, "fixed column count")?,
            advice_offset: validated.request.advice_offset,
            advice_columns: validated.request.advice_columns,
            instance_offset: validated.request.instance_offset,
            instance_columns: validated.request.instance_columns,
            permutation_offset: validated.request.permutation_offset,
            permutation_sets: validated.request.permutation_sets,
            permutation_column_count: u32_len(
                context.inner.permutation_column_count,
                "permutation column count",
            )?,
            chunk_len: u32_len(
                context.inner.permutation_chunk_len,
                "permutation chunk length",
            )?,
            reserved: 0,
            beta: validated.request.beta,
            gamma: validated.request.gamma,
            y: validated.request.y,
            root: validated.extended_domain.root,
            coset_generator: validated.extended_domain.coset_generator,
            delta: context.inner.permutation_delta,
        };
        encoder.setComputePipelineState(&self.quotient.permutation);
        // SAFETY: schema-v2 binding proves the exact fixed, selector,
        // permutation-column, and permutation-coset shapes. Dynamic offsets
        // were checked against canonical C columns before allocation.
        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(&owner.scratch.slots[numerator.slot].raw),
                numerator.byte_offset,
                0,
            );
            encoder.setBuffer_offset_atIndex(Some(&key.fixed_cosets), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&owner.buffers[COSET_BUFFER].raw), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&key.selectors), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&key.permutation_columns), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&key.permutation_cosets), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&key.extended_coset_points), 0, 6);
            encoder.setBytes_length_atIndex(
                NonNull::from(&parameters).cast::<c_void>(),
                size_of::<QuotientPermutationParameters>(),
                7,
            );
        }
        self.dispatch(encoder, &self.quotient.permutation, validated.n_ext);
        Ok(())
    }

    fn encode_quotient_vanishing_multiply(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        validated: &ValidatedQuotientEpoch<'_>,
        owner: &QuotientEpochOwner<'_>,
        key: &EpochKeyResources,
        numerator: &ResidentFrView,
    ) -> Result<(), MetalError> {
        owner.scratch.validate(numerator)?;
        let parameters = QuotientMultiplyParameters {
            len: u32_len(validated.n_ext, "vanishing multiply length")?,
            period: u32_len(validated.n_ext / validated.n, "vanishing inverse period")?,
        };
        encoder.setComputePipelineState(&self.quotient.multiply_repeating);
        // SAFETY: the numerator is a complete n_ext view and the immutable
        // periodic table was bound through its distinct checked view type.
        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(&owner.scratch.slots[numerator.slot].raw),
                numerator.byte_offset,
                0,
            );
            encoder.setBuffer_offset_atIndex(Some(&key.vanishing_inverses), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::from(&parameters).cast::<c_void>(),
                size_of::<QuotientMultiplyParameters>(),
                2,
            );
        }
        self.dispatch(encoder, &self.quotient.multiply_repeating, validated.n_ext);
        Ok(())
    }

    fn encode_quotient_split_blind(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        validated: &ValidatedQuotientEpoch<'_>,
        owner: &mut QuotientEpochOwner<'_>,
        coefficients: &ResidentFrView,
        output_slot: usize,
    ) -> Result<(), MetalError> {
        owner.scratch.validate(coefficients)?;
        let source_len = validated
            .quotient_piece_count
            .checked_mul(validated.n - 1)
            .ok_or_else(|| quotient_error("quotient split source length overflows"))?;
        let output_len = validated
            .quotient_piece_count
            .checked_mul(validated.n)
            .ok_or_else(|| quotient_error("quotient split output length overflows"))?;
        if source_len > validated.n_ext || output_len > validated.n_ext {
            return Err(quotient_error("quotient split escapes its tiled scratch"));
        }
        invalidate_scratch_for_opaque_write(&mut owner.scratch, output_slot)?;
        let parameters = QuotientSplitParameters {
            piece_len: u32_len(validated.n, "quotient piece length")?,
            piece_count: u32_len(validated.quotient_piece_count, "quotient piece count")?,
            source_len: u32_len(source_len, "quotient split source length")?,
            reserved: 0,
        };
        let blind_offset = fr_bytes(validated.request.challenges.len())?;
        encoder.setComputePipelineState(&self.quotient.split_blind);
        // SAFETY: source and destination are distinct n_ext scratch buffers;
        // the q-1 blinds occupy the checked auxiliary suffix and q*n <= n_ext.
        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(&owner.scratch.slots[coefficients.slot].raw),
                coefficients.byte_offset,
                0,
            );
            encoder.setBuffer_offset_atIndex(
                Some(&owner.buffers[AUXILIARY_BUFFER].raw),
                blind_offset,
                1,
            );
            encoder.setBuffer_offset_atIndex(Some(&owner.scratch.slots[output_slot].raw), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::from(&parameters).cast::<c_void>(),
                size_of::<QuotientSplitParameters>(),
                3,
            );
        }
        self.dispatch(encoder, &self.quotient.split_blind, output_len);
        Ok(())
    }

    fn encode_quotient_coset_prepare(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        owner: &mut QuotientEpochOwner<'_>,
        column_start: usize,
        column_count: usize,
        validated: &ValidatedQuotientEpoch<'_>,
    ) -> Result<ResidentFrView, MetalError> {
        if column_count == 0 || column_count > owner.ntt_tile_columns {
            return Err(quotient_error("quotient NTT tile count is invalid"));
        }
        let output = publish_scratch_batch_write(
            &mut owner.scratch,
            0,
            validated.n_ext,
            column_count,
            FrDomainTag::for_ntt(validated.extended_domain, FrBasis::CosetCoefficient),
        )?;
        let parameters = NttCosetParameters {
            input_length: u32_len(validated.n, "coefficient tile length")?,
            output_length: u32_len(validated.n_ext, "coset tile length")?,
            source_stride: u32_len(validated.n, "coefficient tile stride")?,
            destination_stride: u32_len(validated.n_ext, "coset tile stride")?,
            batch_count: u32_len(column_count, "coefficient tile column count")?,
            reserved: 0,
            coset_generator: validated.extended_domain.coset_generator,
            inverse_coset_generator: validated.extended_domain.inverse_coset_generator,
        };
        let source_offset = fr_bytes(
            column_start
                .checked_mul(validated.n)
                .ok_or_else(|| quotient_error("coefficient tile offset overflows"))?,
        )?;
        encoder.setComputePipelineState(&self.ntt.coset_prepare);
        // SAFETY: D owns exactly C*n elements and the bounded tile owns
        // column_count*n_ext elements; the parameter block is copied now.
        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(&owner.buffers[COEFFICIENT_BUFFER].raw),
                source_offset,
                0,
            );
            encoder.setBuffer_offset_atIndex(
                Some(&owner.scratch.slots[output.slot].raw),
                output.byte_offset,
                1,
            );
            encoder.setBytes_length_atIndex(
                NonNull::from(&parameters).cast::<c_void>(),
                size_of::<NttCosetParameters>(),
                2,
            );
        }
        let threads = NTT_COSET_THREADS
            .min(self.ntt.coset_prepare.maxTotalThreadsPerThreadgroup())
            .max(1);
        self.dispatch_threadgroups(
            encoder,
            &self.ntt.coset_prepare,
            MTLSize {
                width: validated.n_ext.div_ceil(threads),
                height: column_count,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        )?;
        Ok(output)
    }

    fn encode_quotient_copy_to_coset_arena(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        owner: &QuotientEpochOwner<'_>,
        source: &ResidentFrView,
        column: usize,
        n_ext: usize,
    ) -> Result<(), MetalError> {
        owner.scratch.validate(source)?;
        let parameters = QuotientCopyParameters {
            len: u32_len(n_ext, "coset copy length")?,
            source_stride: 1,
            destination_stride: 1,
            reserved: 0,
        };
        let destination_offset = fr_bytes(
            column
                .checked_mul(n_ext)
                .ok_or_else(|| quotient_error("coset destination offset overflows"))?,
        )?;
        encoder.setComputePipelineState(&self.quotient.copy);
        // SAFETY: source is one complete n_ext tile and E owns C disjoint n_ext
        // columns; source and destination allocations do not alias.
        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(&owner.scratch.slots[source.slot].raw),
                source.byte_offset,
                0,
            );
            encoder.setBuffer_offset_atIndex(
                Some(&owner.buffers[COSET_BUFFER].raw),
                destination_offset,
                1,
            );
            encoder.setBytes_length_atIndex(
                NonNull::from(&parameters).cast::<c_void>(),
                size_of::<QuotientCopyParameters>(),
                2,
            );
        }
        self.dispatch(encoder, &self.quotient.copy, n_ext);
        Ok(())
    }

    fn encode_quotient_zero(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        owner: &QuotientEpochOwner<'_>,
        numerator: &ResidentFrView,
        n_ext: usize,
    ) -> Result<(), MetalError> {
        owner.scratch.validate(numerator)?;
        let words = n_ext
            .checked_mul(size_of::<Bls12_381Fr>() / size_of::<u64>())
            .ok_or_else(|| quotient_error("numerator clear word count overflows"))?;
        encoder.setComputePipelineState(&self.clear_words);
        // SAFETY: the numerator view starts at a u64-aligned offset and covers
        // exactly n_ext RawFr values in its secret scratch allocation.
        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(&owner.scratch.slots[numerator.slot].raw),
                numerator.byte_offset,
                0,
            );
        }
        self.dispatch(encoder, &self.clear_words, words);
        Ok(())
    }

    fn bind_quotient_raw_key_buffer(
        &self,
        context: &MetalKeyContext,
        buffer: Option<&ImmutableBuffer>,
        source: ImmutableBufferSource,
        required: bool,
        dummy: &Buffer,
        resources: &mut Vec<Buffer>,
    ) -> Result<Buffer, MetalError> {
        self.validate_key_context(context)?;
        let Some(buffer) = buffer else {
            return if required {
                Err(quotient_error("required immutable key buffer is absent"))
            } else {
                Ok(dummy.clone())
            };
        };
        if buffer.session_id != self.session_id
            || buffer.context_id != context.inner.context_id
            || buffer.allocation_id == 0
            || buffer.source != source
            || buffer.bytes == 0
        {
            return Err(quotient_error(
                "immutable key buffer identity differs from its quotient binding",
            ));
        }
        resources.push(buffer.raw.clone());
        Ok(buffer.raw.clone())
    }
}

fn publish_scratch_write(
    arena: &mut ResidentFrArena<'_>,
    slot: usize,
    len: usize,
    domain: FrDomainTag,
) -> Result<ResidentFrView, MetalError> {
    publish_scratch_batch_write(arena, slot, len, 1, domain)
}

fn publish_scratch_batch_write(
    arena: &mut ResidentFrArena<'_>,
    slot: usize,
    len: usize,
    batch_count: usize,
    domain: FrDomainTag,
) -> Result<ResidentFrView, MetalError> {
    let generation = arena.write_generations[slot]
        .checked_add(1)
        .ok_or_else(|| quotient_error("quotient scratch generation is exhausted"))?;
    arena.write_generations[slot] = generation;
    arena.slot_states[slot] = Some(ResidentFrSlotState::new(
        generation,
        0,
        len,
        len,
        batch_count,
        domain,
    ));
    arena.view_for_slot(slot, generation, 0, len, len, batch_count, domain)
}

fn invalidate_scratch_for_opaque_write(
    arena: &mut ResidentFrArena<'_>,
    slot: usize,
) -> Result<(), MetalError> {
    let generation = arena.write_generations[slot]
        .checked_add(1)
        .ok_or_else(|| quotient_error("quotient scratch generation is exhausted"))?;
    arena.write_generations[slot] = generation;
    // Blinded q*n pieces are intentionally not an NTT-domain ResidentFrView.
    // Removing the prior authoritative state prevents an earlier basis view
    // from validating after the split kernel overwrites this allocation.
    arena.slot_states[slot] = None;
    Ok(())
}

fn quotient_error(reason: impl Into<String>) -> MetalError {
    MetalError::QuotientEpoch(reason.into())
}

fn fr_bytes(elements: usize) -> Result<usize, MetalError> {
    elements
        .checked_mul(size_of::<Bls12_381Fr>())
        .ok_or_else(|| quotient_error("RawFr byte count overflows"))
}

fn u32_len(value: usize, label: &str) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| quotient_error(format!("{label} exceeds the u32 ABI")))
}

fn checked_advance(cursor: usize, count: usize, label: &str) -> Result<usize, MetalError> {
    cursor
        .checked_add(count)
        .ok_or_else(|| quotient_error(format!("{label} source range overflows")))
}

fn quotient_permutation_sets(columns: usize, chunk_len: usize) -> Result<usize, MetalError> {
    if columns == 0 {
        return Ok(0);
    }
    if chunk_len == 0 {
        return Err(quotient_error("permutation chunk length is zero"));
    }
    Ok(columns.div_ceil(chunk_len))
}

fn require_offset(label: &str, actual: u32, expected: usize) -> Result<(), MetalError> {
    if actual as usize == expected {
        Ok(())
    } else {
        Err(quotient_error(format!(
            "{label} offset {actual} is not canonical offset {expected}"
        )))
    }
}

fn validate_quotient_reduced(label: &str, values: &[Bls12_381Fr]) -> Result<(), MetalError> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| fr_greater_or_equal_host(value.limbs, generated::FR_MODULUS))
    {
        Err(quotient_error(format!(
            "{label} element {index} is not reduced"
        )))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key_context::OwnedV2Input;

    struct QuotientCase {
        fixture: OwnedV2Input,
        columns: Vec<Vec<Bls12_381Fr>>,
        column_values: Vec<Bls12_381Fr>,
        challenges: Vec<Bls12_381Fr>,
        blinds: Vec<Bls12_381Fr>,
        permutation_sets: usize,
        lookup_product_offset: usize,
        lookup_input_offset: usize,
        lookup_table_offset: usize,
        trash_offset: usize,
        beta: Bls12_381Fr,
        gamma: Bls12_381Fr,
        theta: Bls12_381Fr,
        trash_challenge: Bls12_381Fr,
        y: Bls12_381Fr,
    }

    impl QuotientCase {
        fn new(piece_count: usize) -> Self {
            assert!((2..=4).contains(&piece_count));
            let mut fixture = OwnedV2Input::representative();
            let n = fixture.domain.n as usize;
            let extension = piece_count.next_power_of_two();
            let n_ext = n * extension;
            let one = cpu_fr_one();

            fixture.graph_fixed.n_ext = n_ext as u32;
            fixture.graph_fixed.fixed_cosets =
                vec![one; fixture.graph_fixed.fixed_map.len() * n_ext];
            fixture.graph_fixed.selectors = vec![one; 3 * n_ext];
            fixture.domain.n_ext = n_ext as u32;
            fixture.domain.extended_k = n_ext.ilog2();
            fixture.domain.constraint_system_degree = (piece_count + 1) as u32;
            fixture.domain.quotient_piece_count = piece_count as u32;
            fixture.domain.rotation_scale = extension as i32;
            fixture.domain.permutation_chunk_len = (piece_count - 1) as u32;
            fixture.domain.extended_ntt =
                bls12_381_ntt_domain(n_ext.ilog2()).expect("canonical extended domain");
            fixture.domain.vanishing_inverses =
                cpu_vanishing_inverses(n, n_ext, fixture.domain.extended_ntt);
            fixture.permutation_cosets = vec![one; fixture.permutation_columns.len() * n_ext];
            fixture.refresh_neutral_fingerprints();

            let permutation_sets = fixture.permutation_columns.len().div_ceil(piece_count - 1);
            let lookup_product_offset = 2 + permutation_sets;
            let lookup_input_offset = lookup_product_offset + 1;
            let lookup_table_offset = lookup_input_offset + 1;
            let trash_offset = lookup_table_offset + 1;
            let total_columns = trash_offset + 1;
            let column_values = (0..total_columns)
                .map(|index| cpu_fr_from_u64(23 + 7 * index as u64))
                .collect::<Vec<_>>();
            let columns = column_values
                .iter()
                .copied()
                .map(|constant| vec![constant, Bls12_381Fr::default()])
                .collect();

            Self {
                fixture,
                columns,
                column_values,
                challenges: Vec::new(),
                blinds: (0..piece_count - 1)
                    .map(|index| cpu_fr_from_u64(101 + index as u64))
                    .collect(),
                permutation_sets,
                lookup_product_offset,
                lookup_input_offset,
                lookup_table_offset,
                trash_offset,
                beta: cpu_fr_from_u64(7),
                gamma: cpu_fr_from_u64(11),
                theta: cpu_fr_from_u64(13),
                trash_challenge: cpu_fr_from_u64(17),
                y: cpu_fr_from_u64(19),
            }
        }

        fn column_refs(&self) -> Vec<&[Bls12_381Fr]> {
            self.columns.iter().map(Vec::as_slice).collect()
        }

        fn request<'a>(
            &'a self,
            columns: &'a [&'a [Bls12_381Fr]],
        ) -> MetalQuotientEpochRequest<'a> {
            MetalQuotientEpochRequest {
                coefficient_columns: columns,
                total_columns: columns.len() as u32,
                advice_offset: 0,
                advice_columns: 1,
                instance_offset: 1,
                instance_columns: 1,
                permutation_offset: 2,
                permutation_sets: self.permutation_sets as u32,
                lookup_product_offset: self.lookup_product_offset as u32,
                lookup_input_offset: self.lookup_input_offset as u32,
                lookup_table_offset: self.lookup_table_offset as u32,
                trash_offset: self.trash_offset as u32,
                challenges: &self.challenges,
                beta: self.beta,
                gamma: self.gamma,
                theta: self.theta,
                trash_challenge: self.trash_challenge,
                y: self.y,
                quotient_blinds: &self.blinds,
            }
        }

        fn expected_blinded_pieces(&self) -> Vec<Bls12_381Fr> {
            let n = self.fixture.domain.n as usize;
            let n_ext = self.fixture.domain.n_ext as usize;
            let domain = self.fixture.domain.extended_ntt;
            let rotation_scale = self.fixture.domain.rotation_scale;
            let last_rotation = self.fixture.domain.last_rotation;
            let chunk_len = self.fixture.domain.permutation_chunk_len as usize;
            let one = cpu_fr_one();
            let fixed = &self.fixture.graph_fixed.fixed_cosets;
            let selectors = &self.fixture.graph_fixed.selectors;
            let permutation_cosets = &self.fixture.permutation_cosets;
            let dynamic = |column: usize, _row: usize| self.column_values[column];

            let mut numerator = Vec::with_capacity(n_ext);
            for row in 0..n_ext {
                // The custom graph in the authentic v2 fixture stores one.
                let mut value = one;
                let next_row = rotated_row(row, 1, rotation_scale, n_ext);
                let final_row = rotated_row(row, last_rotation, rotation_scale, n_ext);
                let l0 = selectors[row];
                let l_last = selectors[n_ext + row];
                let l_active = selectors[2 * n_ext + row];
                let first_product = dynamic(2, row);
                let last_product = dynamic(2 + self.permutation_sets - 1, row);
                value = cpu_accumulate(
                    value,
                    self.y,
                    cpu_fr_mul(cpu_fr_sub(one, first_product), l0),
                );
                value = cpu_accumulate(
                    value,
                    self.y,
                    cpu_fr_mul(
                        cpu_fr_sub(cpu_fr_mul(last_product, last_product), last_product),
                        l_last,
                    ),
                );
                for set in 1..self.permutation_sets {
                    let current = dynamic(2 + set, row);
                    let previous = dynamic(2 + set - 1, final_row);
                    value = cpu_accumulate(
                        value,
                        self.y,
                        cpu_fr_mul(cpu_fr_sub(current, previous), l0),
                    );
                }

                let point = cpu_fr_mul(domain.coset_generator, cpu_fr_pow(domain.root, row as u64));
                let mut current_delta = cpu_fr_mul(self.beta, point);
                for set in 0..self.permutation_sets {
                    let mut left = dynamic(2 + set, next_row);
                    let mut right = dynamic(2 + set, row);
                    let start = set * chunk_len;
                    let end = (start + chunk_len).min(self.fixture.permutation_columns.len());
                    for column_index in start..end {
                        let reference = self.fixture.permutation_columns[column_index];
                        let column = match reference.kind {
                            0 => fixed[reference.index as usize * n_ext + row],
                            1 => dynamic(reference.index as usize, row),
                            2 => dynamic(1 + reference.index as usize, row),
                            kind => panic!("unexpected permutation column kind {kind}"),
                        };
                        let sigma = permutation_cosets[column_index * n_ext + row];
                        left = cpu_fr_mul(
                            left,
                            cpu_fr_add(
                                cpu_fr_add(column, cpu_fr_mul(self.beta, sigma)),
                                self.gamma,
                            ),
                        );
                        right = cpu_fr_mul(
                            right,
                            cpu_fr_add(cpu_fr_add(column, current_delta), self.gamma),
                        );
                        current_delta = cpu_fr_mul(current_delta, self.fixture.domain.delta);
                    }
                    value = cpu_accumulate(
                        value,
                        self.y,
                        cpu_fr_mul(cpu_fr_sub(left, right), l_active),
                    );
                }

                // The fixture has one lookup graph, whose independent graph
                // result is one, followed by the exact five lookup folds.
                let previous_row = rotated_row(row, -1, rotation_scale, n_ext);
                let product = dynamic(self.lookup_product_offset, row);
                let product_next = dynamic(self.lookup_product_offset, next_row);
                let input = dynamic(self.lookup_input_offset, row);
                let input_previous = dynamic(self.lookup_input_offset, previous_row);
                let table = dynamic(self.lookup_table_offset, row);
                let input_minus_table = cpu_fr_sub(input, table);
                value = cpu_accumulate(value, self.y, cpu_fr_mul(cpu_fr_sub(one, product), l0));
                value = cpu_accumulate(
                    value,
                    self.y,
                    cpu_fr_mul(cpu_fr_sub(cpu_fr_mul(product, product), product), l_last),
                );
                let lookup_left = cpu_fr_mul(
                    cpu_fr_mul(product_next, cpu_fr_add(input, self.beta)),
                    cpu_fr_add(table, self.gamma),
                );
                let lookup_right = cpu_fr_mul(product, one);
                value = cpu_accumulate(
                    value,
                    self.y,
                    cpu_fr_mul(cpu_fr_sub(lookup_left, lookup_right), l_active),
                );
                value = cpu_accumulate(value, self.y, cpu_fr_mul(input_minus_table, l0));
                value = cpu_accumulate(
                    value,
                    self.y,
                    cpu_fr_mul(
                        cpu_fr_mul(input_minus_table, cpu_fr_sub(input, input_previous)),
                        l_active,
                    ),
                );

                // The trash graph independently evaluates to one.
                let trash_selector = fixed[row];
                let trash = dynamic(self.trash_offset, row);
                let trash_term =
                    cpu_fr_sub(one, cpu_fr_mul(cpu_fr_sub(one, trash_selector), trash));
                value = cpu_accumulate(value, self.y, trash_term);
                numerator.push(cpu_fr_mul(
                    value,
                    self.fixture.domain.vanishing_inverses[row % (n_ext / n)],
                ));
            }

            let coefficients = cpu_inverse_coset_ntt(&numerator, domain);
            let piece_count = self.fixture.domain.quotient_piece_count as usize;
            let mut pieces = Vec::with_capacity(piece_count * n);
            for piece in 0..piece_count {
                for index in 0..n - 1 {
                    let mut value = coefficients[piece * (n - 1) + index];
                    if index == 0 && piece != 0 {
                        value = cpu_fr_sub(value, self.blinds[piece - 1]);
                    }
                    pieces.push(value);
                }
                pieces.push(if piece + 1 < piece_count {
                    self.blinds[piece]
                } else {
                    Bls12_381Fr::default()
                });
            }
            pieces
        }
    }

    #[test]
    fn quotient_epoch_matches_independent_cpu_for_q2_q3_q4() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        for piece_count in 2..=4 {
            let case = QuotientCase::new(piece_count);
            let context = session
                .get_or_create_key_context_v2(case.fixture.input())
                .expect("build authentic v2 quotient context");
            let expected = case.expected_blinded_pieces();
            let columns = case.column_refs();
            session.reset_activity();

            session
                .with_diagnostic_quotient_epoch(&context, case.request(&columns), |lease| {
                    assert_eq!(lease.piece_len(), 2);
                    assert_eq!(lease.piece_count(), piece_count);
                    assert_eq!(lease.key_context_identity(), context.identity());
                    assert_eq!(lease.as_raw_fr(), expected.as_slice());
                })
                .expect("execute diagnostic quotient epoch");

            let activity = session.activity();
            let lease_bytes = (piece_count * 2 * size_of::<Bls12_381Fr>()) as u64;
            assert_eq!(activity.quotient_epochs, 1, "q={piece_count}");
            assert_eq!(activity.quotient_epoch_failures, 0, "q={piece_count}");
            assert_eq!(activity.cpu_msm_leases, 1, "q={piece_count}");
            assert_eq!(activity.cpu_msm_lease_bytes, lease_bytes, "q={piece_count}");
            assert_eq!(activity.host_read_bytes, lease_bytes, "q={piece_count}");
            assert_eq!(activity.materializations, 0, "q={piece_count}");
            assert_eq!(
                activity.whole_polynomial_materializations, 0,
                "q={piece_count}"
            );
            assert_eq!(activity.command_buffers - activity.clear_commands, 1);
            assert_eq!(activity.host_waits - activity.clear_waits, 1);
            assert_eq!(activity.clear_commands, 1);
            assert_eq!(activity.clear_waits, 1);
            assert_eq!(activity.clear_dispatches, 6);
            assert_eq!(activity.quarantined_secret_bytes, 0);
        }
    }

    #[test]
    fn resident_quotient_epoch_issues_no_cpu_scalar_view() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let case = QuotientCase::new(3);
        let context = session
            .get_or_create_key_context_v2(case.fixture.input())
            .expect("build authentic v2 quotient context");
        let columns = case.column_refs();
        session.reset_activity();

        session
            .with_resident_quotient_epoch(&context, case.request(&columns), |lease| {
                assert_eq!(lease.piece_len(), 2);
                assert_eq!(lease.piece_count(), 3);
                assert_eq!(lease.key_context_identity(), context.identity());
            })
            .expect("execute resident quotient epoch");

        let activity = session.activity();
        let lease_bytes = (3 * 2 * size_of::<Bls12_381Fr>()) as u64;
        assert_eq!(activity.quotient_epochs, 1);
        assert_eq!(activity.quotient_epoch_failures, 0);
        assert_eq!(activity.resident_msm_leases, 1);
        assert_eq!(activity.resident_msm_lease_bytes, lease_bytes);
        assert_eq!(activity.cpu_msm_leases, 0);
        assert_eq!(activity.cpu_msm_lease_bytes, 0);
        assert_eq!(activity.host_read_bytes, 0);
        assert_eq!(activity.materializations, 0);
        assert_eq!(activity.whole_polynomial_materializations, 0);
        assert_eq!(activity.quarantined_secret_bytes, 0);
    }

    #[test]
    fn resident_quotient_combination_matches_cpu_horner() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let case = QuotientCase::new(4);
        let context = session
            .get_or_create_key_context_v2(case.fixture.input())
            .expect("build authentic v2 quotient context");
        let columns = case.column_refs();
        let factor = cpu_fr_from_u64(29);
        let expected_pieces = case.expected_blinded_pieces();
        let expected = expected_pieces
            .chunks_exact(2)
            .rev()
            .map(<[Bls12_381Fr]>::to_vec)
            .reduce(|accumulator, piece| {
                accumulator
                    .into_iter()
                    .zip(piece)
                    .map(|(value, addend)| cpu_fr_add(cpu_fr_mul(value, factor), addend))
                    .collect()
            })
            .expect("the quotient has at least two pieces");
        session.reset_activity();

        let actual = session
            .with_resident_quotient_epoch(&context, case.request(&columns), |lease| {
                session.materialize_bls12_381_quotient_combination(&lease, factor)
            })
            .expect("execute resident quotient epoch")
            .expect("combine resident quotient pieces");
        assert_eq!(actual, expected);

        let activity = session.activity();
        assert_eq!(activity.quotient_epochs, 1);
        assert_eq!(activity.quotient_epoch_failures, 0);
        assert_eq!(activity.resident_msm_leases, 1);
        assert_eq!(activity.cpu_msm_leases, 0);
        assert_eq!(
            activity.host_read_bytes,
            (2 * size_of::<Bls12_381Fr>()) as u64
        );
        assert_eq!(activity.materializations, 1);
        assert_eq!(activity.whole_polynomial_materializations, 1);
        assert_eq!(activity.command_buffers - activity.clear_commands, 2);
        assert_eq!(activity.host_waits - activity.clear_waits, 2);
        assert_eq!(activity.quarantined_secret_bytes, 0);
    }

    #[test]
    fn quotient_epoch_preflight_rejects_before_allocation_or_command() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let case = QuotientCase::new(2);
        let context = session
            .get_or_create_key_context_v2(case.fixture.input())
            .expect("build authentic v2 quotient context");
        let columns = case.column_refs();
        let mut request = case.request(&columns);
        request.total_columns -= 1;
        session.reset_activity();

        let result = session.with_diagnostic_quotient_epoch(&context, request, |_| {
            panic!("preflight failure must not issue a CPU lease")
        });
        assert!(matches!(result, Err(MetalError::QuotientEpoch(_))));
        let activity = session.activity();
        assert_eq!(activity.quotient_epochs, 0);
        assert_eq!(activity.quotient_epoch_failures, 1);
        assert_eq!(activity.allocations, 0);
        assert_eq!(activity.command_buffers, 0);
        assert_eq!(activity.dispatches, 0);
        assert_eq!(activity.host_waits, 0);
        assert_eq!(activity.clear_commands, 0);
        assert_eq!(activity.cpu_msm_leases, 0);
        assert_eq!(activity.host_read_bytes, 0);
        assert_eq!(activity.materializations, 0);
    }

    #[test]
    fn zero_permutation_chunk_is_an_error_not_a_panic() {
        let outcome = catch_unwind(|| quotient_permutation_sets(1, 0));
        assert!(matches!(outcome, Ok(Err(MetalError::QuotientEpoch(_)))));
    }

    #[test]
    fn callback_result_is_opaque_and_cleanup_still_completes() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let case = QuotientCase::new(2);
        let context = session
            .get_or_create_key_context_v2(case.fixture.input())
            .expect("build authentic v2 quotient context");
        let columns = case.column_refs();
        session.reset_activity();

        let nested = session
            .with_diagnostic_quotient_epoch::<Result<(), &'static str>, _>(
                &context,
                case.request(&columns),
                |_| Err("CPU MSM rejected the lease"),
            )
            .expect("the generic callback result is opaque to the backend");
        assert_eq!(nested, Err("CPU MSM rejected the lease"));
        let activity = session.activity();
        assert_eq!(activity.quotient_epochs, 1);
        assert_eq!(activity.quotient_epoch_failures, 0);
        assert_eq!(activity.cpu_msm_leases, 1);
        assert_eq!(activity.clear_commands, 1);
        assert_eq!(activity.clear_waits, 1);
        assert_eq!(activity.clear_dispatches, 6);
        assert_eq!(activity.quarantined_secret_bytes, 0);
        assert_eq!(activity.materializations, 0);
    }

    #[test]
    fn callback_panic_cleans_before_resume_and_session_reuse() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let case = QuotientCase::new(2);
        let context = session
            .get_or_create_key_context_v2(case.fixture.input())
            .expect("build authentic v2 quotient context");
        session.reset_activity();

        let panic = catch_unwind(AssertUnwindSafe(|| {
            let columns = case.column_refs();
            let _ = session.with_diagnostic_quotient_epoch::<(), _>(
                &context,
                case.request(&columns),
                |_| panic!("CPU MSM callback panic"),
            );
        }));
        assert!(panic.is_err());
        let activity = session.activity();
        assert_eq!(activity.quotient_epochs, 0);
        assert_eq!(activity.quotient_epoch_failures, 1);
        assert_eq!(activity.cpu_msm_leases, 1);
        assert_eq!(activity.clear_commands, 1);
        assert_eq!(activity.clear_waits, 1);
        assert_eq!(activity.clear_dispatches, 6);
        assert_eq!(activity.quarantined_secret_bytes, 0);
        assert_eq!(activity.materializations, 0);

        session.reset_activity();
        let columns = case.column_refs();
        session
            .with_diagnostic_quotient_epoch(&context, case.request(&columns), |_| ())
            .expect("reuse session after callback panic cleanup");
        assert_eq!(session.activity().quotient_epochs, 1);
        assert_eq!(session.activity().quotient_epoch_failures, 0);
        assert_eq!(session.activity().quarantined_secret_bytes, 0);
    }

    #[test]
    fn clear_failure_quarantines_epoch_and_poison_session() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let case = QuotientCase::new(2);
        let context = session
            .get_or_create_key_context_v2(case.fixture.input())
            .expect("build authentic v2 quotient context");
        let columns = case.column_refs();
        session.reset_activity();
        session
            .inject_next_clear_failure
            .store(true, Ordering::SeqCst);

        let result =
            session.with_diagnostic_quotient_epoch(&context, case.request(&columns), |_| ());
        assert!(matches!(result, Err(MetalError::SessionPoisoned { .. })));
        let activity = session.activity();
        assert_eq!(activity.quotient_epochs, 0);
        assert_eq!(activity.quotient_epoch_failures, 1);
        assert_eq!(activity.cpu_msm_leases, 1);
        assert_eq!(activity.clear_commands, 1);
        assert_eq!(activity.clear_failures, 1);
        assert_eq!(activity.poison_events, 1);
        assert_ne!(activity.quarantined_secret_bytes, 0);
        assert_eq!(activity.materializations, 0);
    }

    fn cpu_vanishing_inverses(n: usize, n_ext: usize, domain: NttDomainSpec) -> Vec<Bls12_381Fr> {
        let mut current = cpu_fr_pow(domain.coset_generator, n as u64);
        let step = cpu_fr_pow(domain.root, n as u64);
        (0..n_ext / n)
            .map(|_| {
                let inverse = cpu_fr_inverse(cpu_fr_sub(current, cpu_fr_one()));
                current = cpu_fr_mul(current, step);
                inverse
            })
            .collect()
    }

    fn cpu_inverse_coset_ntt(
        evaluations: &[Bls12_381Fr],
        domain: NttDomainSpec,
    ) -> Vec<Bls12_381Fr> {
        (0..evaluations.len())
            .map(|coefficient| {
                let sum = evaluations.iter().copied().enumerate().fold(
                    Bls12_381Fr::default(),
                    |sum, (row, value)| {
                        let twiddle = cpu_fr_pow(domain.inverse_root, (row * coefficient) as u64);
                        cpu_fr_add(sum, cpu_fr_mul(value, twiddle))
                    },
                );
                cpu_fr_mul(
                    cpu_fr_mul(sum, domain.inverse_size),
                    cpu_fr_pow(domain.inverse_coset_generator, coefficient as u64),
                )
            })
            .collect()
    }

    fn rotated_row(row: usize, rotation: i32, scale: i32, len: usize) -> usize {
        (row as i64 + i64::from(rotation) * i64::from(scale)).rem_euclid(len as i64) as usize
    }

    fn cpu_accumulate(value: Bls12_381Fr, y: Bls12_381Fr, term: Bls12_381Fr) -> Bls12_381Fr {
        cpu_fr_add(cpu_fr_mul(value, y), term)
    }

    fn cpu_fr_one() -> Bls12_381Fr {
        Bls12_381Fr {
            limbs: generated::FR_R,
        }
    }

    fn cpu_fr_from_u64(value: u64) -> Bls12_381Fr {
        let canonical = [value, 0, 0, 0];
        let mut output = blst::blst_fr::default();
        // SAFETY: BLST receives four initialized canonical little-endian limbs.
        unsafe { blst::blst_fr_from_uint64(&mut output, canonical.as_ptr()) };
        Bls12_381Fr { limbs: output.l }
    }

    fn cpu_fr_add(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
        let mut output = blst::blst_fr::default();
        let lhs = blst::blst_fr { l: lhs.limbs };
        let rhs = blst::blst_fr { l: rhs.limbs };
        // SAFETY: initialized BLST operands and disjoint output.
        unsafe { blst::blst_fr_add(&mut output, &lhs, &rhs) };
        Bls12_381Fr { limbs: output.l }
    }

    fn cpu_fr_sub(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
        let mut output = blst::blst_fr::default();
        let lhs = blst::blst_fr { l: lhs.limbs };
        let rhs = blst::blst_fr { l: rhs.limbs };
        // SAFETY: initialized BLST operands and disjoint output.
        unsafe { blst::blst_fr_sub(&mut output, &lhs, &rhs) };
        Bls12_381Fr { limbs: output.l }
    }

    fn cpu_fr_mul(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
        let mut output = blst::blst_fr::default();
        let lhs = blst::blst_fr { l: lhs.limbs };
        let rhs = blst::blst_fr { l: rhs.limbs };
        // SAFETY: initialized BLST operands and disjoint output.
        unsafe { blst::blst_fr_mul(&mut output, &lhs, &rhs) };
        Bls12_381Fr { limbs: output.l }
    }

    fn cpu_fr_inverse(value: Bls12_381Fr) -> Bls12_381Fr {
        let mut output = blst::blst_fr::default();
        let value = blst::blst_fr { l: value.limbs };
        // SAFETY: initialized nonzero BLST operand and disjoint output.
        unsafe { blst::blst_fr_inverse(&mut output, &value) };
        Bls12_381Fr { limbs: output.l }
    }

    fn cpu_fr_pow(mut base: Bls12_381Fr, mut exponent: u64) -> Bls12_381Fr {
        let mut accumulator = cpu_fr_one();
        while exponent != 0 {
            if exponent & 1 != 0 {
                accumulator = cpu_fr_mul(accumulator, base);
            }
            base = cpu_fr_mul(base, base);
            exponent >>= 1;
        }
        accumulator
    }
}
