// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

//! Callback-scoped, resident KZG multi-opening polynomial pipeline.

use super::*;

const MULTIOPEN_BLOCK_SIZE: usize = 256;

const SOURCE_POLYNOMIALS: usize = 0;
const Q_POLYNOMIALS_A: usize = 1;
const Q_POLYNOMIALS_B: usize = 2;
const F_POLYNOMIAL: usize = 3;
const FINAL_POLYNOMIAL: usize = 4;
const PI_POLYNOMIAL: usize = 5;
const DIVISION_TRANSFORMS: usize = 6;
const DIVISION_INPUTS: usize = 7;
const EVAL_TRANSFORMS: usize = 8;
const EVALUATIONS: usize = 9;
const Q_OFFSETS: usize = 10;
const Q_POLY_INDICES: usize = 11;
const Q_COEFFICIENTS: usize = 12;
const POINT_OFFSETS: usize = 13;
const POINTS: usize = 14;
const F_WEIGHTS: usize = 15;
const PI_WEIGHTS: usize = 16;
const DIVISION_LOCAL_MULTIPLIERS: usize = 17;
const BUFFER_COUNT: usize = 18;

#[repr(C)]
#[derive(Clone, Copy)]
struct LincombParameters {
    poly_len: u32,
    poly_count: u32,
    q_count: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DivisionParameters {
    poly_len: u32,
    poly_count: u32,
    block_count: u32,
    block_size: u32,
    round: u32,
    use_direct_point: u32,
    reserved_0: u32,
    reserved_1: u32,
    direct_point: Bls12_381Fr,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EvalParameters {
    poly_len: u32,
    poly_count: u32,
    block_count: u32,
    block_size: u32,
    point: Bls12_381Fr,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CombineParameters {
    poly_len: u32,
    poly_count: u32,
    reserved_0: u32,
    reserved_1: u32,
}

const _: [(); 16] = [(); size_of::<LincombParameters>()];
const _: [(); 64] = [(); size_of::<DivisionParameters>()];
const _: [(); 32] = [(); offset_of!(DivisionParameters, direct_point)];
const _: [(); 48] = [(); size_of::<EvalParameters>()];
const _: [(); 16] = [(); offset_of!(EvalParameters, point)];
const _: [(); 16] = [(); size_of::<CombineParameters>()];

struct ValidatedMultiopen<'a> {
    request: MetalMultiopenRequest<'a>,
    poly_len: usize,
    poly_count: usize,
    q_count: usize,
    block_count: usize,
    division_rounds: usize,
    precombined_q: bool,
}

/// Scoped owner for q-polynomials, f, and π in reduced Montgomery form.
///
/// Values remain device-resident until the callback passed to
/// [`MetalSession::with_bls12_381_multiopen`] returns. Dropping the arena
/// securely clears every request allocation before it can re-enter the pool.
pub struct MetalMultiopenArena<'session> {
    session: &'session MetalSession,
    buffers: Vec<SecretBufferLease>,
    poly_len: usize,
    q_count: usize,
    block_count: usize,
    divided_q_buffer: usize,
    pi_ready: bool,
}

impl MetalSession {
    /// Runs q construction and f construction, then lends the resident arena to
    /// a transcript callback that may evaluate q and construct π.
    ///
    /// The callback is the only lifetime in which resident polynomial leases
    /// can be committed. Cleanup is mandatory on success, error, and panic; no
    /// CPU polynomial fallback is attempted.
    pub fn with_bls12_381_multiopen<T, F>(
        &self,
        request: MetalMultiopenRequest<'_>,
        callback: F,
    ) -> Result<T, MetalError>
    where
        F: FnOnce(&mut MetalMultiopenArena<'_>) -> Result<T, MetalError>,
    {
        let validated = validate_request(request)?;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let mut arena = {
                let _execution = self.execution_guard();
                self.ensure_session_healthy()?;
                let mut arena = MetalMultiopenArena::allocate(self, &validated)?;
                arena.prepare(&validated)?;
                arena
            };

            // f and π commitments run synchronously on the standalone MSM
            // queue. Do not leave the proof-runtime queue idle behind its host
            // mutex while those independent commands are in flight.
            let callback_outcome = catch_unwind(AssertUnwindSafe(|| callback(&mut arena)));
            let execution = self.execution_guard();
            match callback_outcome {
                Ok(operation) => arena.finish(operation),
                Err(payload) => {
                    let _ = arena.cleanup();
                    drop(execution);
                    resume_unwind(payload)
                }
            }
        }));
        match outcome {
            Ok(result) => result,
            Err(payload) => resume_unwind(payload),
        }
    }

    /// Commits one callback-scoped resident polynomial against prepared bases.
    ///
    /// The standalone MSM worker performs no scalar readback and returns no CPU
    /// curve-arithmetic fallback.
    pub fn bls12_381_msm_resident_polynomial(
        &self,
        fingerprint: [u8; 32],
        basis: Bls12_381MsmBasis,
        polynomial: &MetalResidentPolynomialMsmLease<'_>,
        window: Bls12_381MsmWindow,
    ) -> Result<Bls12_381G1Projective, MetalError> {
        if polynomial.session_id() != self.session_id
            || polynomial.scalars().device_registry_id() != self.device_registry_id
        {
            return Err(MetalError::Multiopen(
                "resident polynomial belongs to a different Metal session".into(),
            ));
        }
        let mut outputs = msm_worker::execute_resident_batch(
            self.device_registry_id,
            fingerprint,
            basis,
            polynomial.scalars().clone(),
            polynomial.len(),
            1,
            window,
        )?;
        outputs
            .pop()
            .ok_or_else(|| MetalError::Msm("resident polynomial MSM returned no output".into()))
    }
}

impl<'session> MetalMultiopenArena<'session> {
    fn allocate(
        session: &'session MetalSession,
        validated: &ValidatedMultiopen<'_>,
    ) -> Result<Self, MetalError> {
        let mut arena = Self {
            session,
            buffers: Vec::with_capacity(BUFFER_COUNT),
            poly_len: validated.poly_len,
            q_count: validated.q_count,
            block_count: validated.block_count,
            divided_q_buffer: Q_POLYNOMIALS_A,
            pi_ready: false,
        };
        arena.allocate_buffers(validated)?;
        arena.upload_request(validated)?;
        Ok(arena)
    }

    fn allocate_buffers(&mut self, validated: &ValidatedMultiopen<'_>) -> Result<(), MetalError> {
        let source_elements = checked_product(
            validated.poly_len,
            validated.poly_count,
            "source polynomial element count",
        )?;
        let q_elements = checked_product(
            validated.poly_len,
            validated.q_count,
            "q-polynomial element count",
        )?;
        let block_elements = checked_product(
            validated.block_count,
            validated.q_count,
            "multi-open block element count",
        )?;
        let request = validated.request;
        let specifications = [
            (
                fr_bytes(source_elements)?,
                "midnight.multiopen.source_polynomials",
            ),
            (fr_bytes(q_elements)?, "midnight.multiopen.q_polynomials_a"),
            (fr_bytes(q_elements)?, "midnight.multiopen.q_polynomials_b"),
            (
                fr_bytes(validated.poly_len)?,
                "midnight.multiopen.f_polynomial",
            ),
            (
                fr_bytes(validated.poly_len)?,
                "midnight.multiopen.final_polynomial",
            ),
            (
                fr_bytes(validated.poly_len)?,
                "midnight.multiopen.pi_polynomial",
            ),
            (
                fr_bytes(checked_product(
                    block_elements,
                    2,
                    "division transform count",
                )?)?,
                "midnight.multiopen.division_transforms",
            ),
            (
                fr_bytes(block_elements)?,
                "midnight.multiopen.division_inputs",
            ),
            (
                fr_bytes(checked_product(
                    block_elements,
                    2,
                    "evaluation transform count",
                )?)?,
                "midnight.multiopen.eval_transforms",
            ),
            (
                fr_bytes(validated.q_count.max(1))?,
                "midnight.multiopen.evaluations",
            ),
            (
                byte_len(request.q_offsets())?,
                "midnight.multiopen.q_offsets",
            ),
            (
                byte_len(request.q_poly_indices())?,
                "midnight.multiopen.q_poly_indices",
            ),
            (
                byte_len(request.q_coefficients())?,
                "midnight.multiopen.q_coefficients",
            ),
            (
                byte_len(request.point_offsets())?,
                "midnight.multiopen.point_offsets",
            ),
            (byte_len(request.points())?, "midnight.multiopen.points"),
            (
                byte_len(request.f_weights())?,
                "midnight.multiopen.f_weights",
            ),
            (
                fr_bytes(validated.q_count + 1)?,
                "midnight.multiopen.pi_weights",
            ),
            (
                fr_bytes(checked_product(
                    validated.q_count,
                    MULTIOPEN_BLOCK_SIZE,
                    "division local multiplier count",
                )?)?,
                "midnight.multiopen.division_local_multipliers",
            ),
        ];
        for (bytes, label) in specifications {
            self.buffers
                .push(self.session.secret_buffer(clearable_bytes(bytes)?, label)?);
        }
        Ok(())
    }

    fn upload_request(&mut self, validated: &ValidatedMultiopen<'_>) -> Result<(), MetalError> {
        let request = validated.request;
        let polynomial_buffer = if validated.precombined_q {
            Q_POLYNOMIALS_A
        } else {
            SOURCE_POLYNOMIALS
        };
        for (poly_index, polynomial) in request.polynomials().iter().copied().enumerate() {
            let offset = checked_product(poly_index, validated.poly_len, "polynomial offset")?;
            self.copy_values_at(polynomial_buffer, offset, polynomial)?;
        }
        self.copy_values(Q_OFFSETS, request.q_offsets())?;
        self.copy_values(Q_POLY_INDICES, request.q_poly_indices())?;
        self.copy_values(Q_COEFFICIENTS, request.q_coefficients())?;
        self.copy_values(POINT_OFFSETS, request.point_offsets())?;
        self.copy_values(POINTS, request.points())?;
        self.copy_values(F_WEIGHTS, request.f_weights())?;
        Ok(())
    }

    fn prepare(&mut self, validated: &ValidatedMultiopen<'_>) -> Result<(), MetalError> {
        let command = self.session.command_buffer("midnight.multiopen.prepare")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        let lincomb = LincombParameters {
            poly_len: u32_len(validated.poly_len)?,
            poly_count: u32_len(validated.poly_count)?,
            q_count: u32_len(validated.q_count)?,
            reserved: 0,
        };
        if !validated.precombined_q {
            encoder.setComputePipelineState(&self.session.multiopen.lincomb);
            self.bind(&encoder, SOURCE_POLYNOMIALS, 0);
            self.bind(&encoder, Q_OFFSETS, 1);
            self.bind(&encoder, Q_POLY_INDICES, 2);
            self.bind(&encoder, Q_COEFFICIENTS, 3);
            self.bind(&encoder, Q_POLYNOMIALS_A, 4);
            bind_parameters(&encoder, &lincomb, 5);
            self.session.dispatch(
                &encoder,
                &self.session.multiopen.lincomb,
                checked_product(validated.poly_len, validated.q_count, "lincomb dispatch")?,
            );
        }

        let mut source = Q_POLYNOMIALS_A;
        let mut destination = Q_POLYNOMIALS_B;
        for round in 0..validated.division_rounds {
            let parameters = DivisionParameters {
                poly_len: u32_len(validated.poly_len)?,
                poly_count: u32_len(validated.q_count)?,
                block_count: u32_len(validated.block_count)?,
                block_size: MULTIOPEN_BLOCK_SIZE as u32,
                round: u32_len(round)?,
                use_direct_point: 0,
                reserved_0: 0,
                reserved_1: 0,
                direct_point: Bls12_381Fr::default(),
            };
            self.encode_division_round(&encoder, source, destination, &parameters)?;
            source = destination;
            destination = if source == Q_POLYNOMIALS_B {
                // Source polynomials are dead after q construction and have at
                // least q_count*n capacity, so their arena becomes the second
                // division ping-pong slot while the original q buffer remains
                // immutable for transcript evaluation and π.
                SOURCE_POLYNOMIALS
            } else {
                Q_POLYNOMIALS_B
            };
        }
        self.divided_q_buffer = source;

        let combine = CombineParameters {
            poly_len: u32_len(validated.poly_len)?,
            poly_count: u32_len(validated.q_count)?,
            reserved_0: 0,
            reserved_1: 0,
        };
        encoder.setComputePipelineState(&self.session.multiopen.combine);
        self.bind(&encoder, self.divided_q_buffer, 0);
        self.bind(&encoder, F_WEIGHTS, 1);
        self.bind(&encoder, F_POLYNOMIAL, 2);
        bind_parameters(&encoder, &combine, 3);
        self.session.dispatch(
            &encoder,
            &self.session.multiopen.combine,
            validated.poly_len,
        );
        encoder.endEncoding();
        self.submit(command)
    }

    /// Returns f as an opaque reduced-Montgomery view for the standalone MSM.
    pub fn f_polynomial(&self) -> Result<MetalResidentPolynomialMsmLease<'_>, MetalError> {
        self.resident_polynomial(F_POLYNOMIAL)
    }

    /// Evaluates every sorted q-polynomial at one transcript challenge and
    /// reads back only that compact evaluation vector.
    pub fn evaluate_q(&mut self, point: Bls12_381Fr) -> Result<Vec<Bls12_381Fr>, MetalError> {
        validate_reduced("q evaluation point", &[point])?;
        let _execution = self.session.execution_guard();
        self.session.ensure_session_healthy()?;
        let command = self
            .session
            .command_buffer("midnight.multiopen.evaluate_q")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        let parameters = EvalParameters {
            poly_len: u32_len(self.poly_len)?,
            poly_count: u32_len(self.q_count)?,
            block_count: u32_len(self.block_count)?,
            block_size: MULTIOPEN_BLOCK_SIZE as u32,
            point,
        };
        self.encode_evaluation(&encoder, Q_POLYNOMIALS_A, &parameters)?;
        encoder.endEncoding();
        self.submit(command)?;
        self.session.read_values_from_raw(
            &self.buffers[EVALUATIONS].raw,
            self.buffers[EVALUATIONS].bytes,
            self.q_count,
        )
    }

    /// Constructs `(sum q_i*x4^i + f*x4^q_count - v) / (X - x3)` and returns
    /// π as an opaque resident polynomial for immediate commitment.
    pub fn compute_pi(
        &mut self,
        x3: Bls12_381Fr,
        combination_weights: &[Bls12_381Fr],
    ) -> Result<MetalResidentPolynomialMsmLease<'_>, MetalError> {
        if combination_weights.len() != self.q_count + 1 {
            return Err(multiopen_error(
                "π combination weight count is inconsistent",
            ));
        }
        validate_reduced("π division point", &[x3])?;
        validate_reduced("π combination weight", combination_weights)?;
        let _execution = self.session.execution_guard();
        self.session.ensure_session_healthy()?;
        self.copy_values(PI_WEIGHTS, combination_weights)?;

        let command = self
            .session
            .command_buffer("midnight.multiopen.compute_pi")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        let combine = CombineParameters {
            poly_len: u32_len(self.poly_len)?,
            poly_count: u32_len(self.q_count)?,
            reserved_0: 0,
            reserved_1: 0,
        };
        encoder.setComputePipelineState(&self.session.multiopen.combine_final);
        self.bind(&encoder, Q_POLYNOMIALS_A, 0);
        self.bind(&encoder, F_POLYNOMIAL, 1);
        self.bind(&encoder, PI_WEIGHTS, 2);
        self.bind(&encoder, FINAL_POLYNOMIAL, 3);
        bind_parameters(&encoder, &combine, 4);
        self.session.dispatch(
            &encoder,
            &self.session.multiopen.combine_final,
            self.poly_len,
        );

        let evaluation = EvalParameters {
            poly_len: u32_len(self.poly_len)?,
            poly_count: 1,
            block_count: u32_len(self.block_count)?,
            block_size: MULTIOPEN_BLOCK_SIZE as u32,
            point: x3,
        };
        self.encode_evaluation(&encoder, FINAL_POLYNOMIAL, &evaluation)?;
        encoder.setComputePipelineState(&self.session.multiopen.subtract_evaluation);
        self.bind(&encoder, FINAL_POLYNOMIAL, 0);
        self.bind(&encoder, EVALUATIONS, 1);
        self.session
            .dispatch(&encoder, &self.session.multiopen.subtract_evaluation, 1);

        let division = DivisionParameters {
            poly_len: u32_len(self.poly_len)?,
            poly_count: 1,
            block_count: u32_len(self.block_count)?,
            block_size: MULTIOPEN_BLOCK_SIZE as u32,
            round: 0,
            use_direct_point: 1,
            reserved_0: 0,
            reserved_1: 0,
            direct_point: x3,
        };
        self.encode_division_round(&encoder, FINAL_POLYNOMIAL, PI_POLYNOMIAL, &division)?;
        encoder.endEncoding();
        self.submit(command)?;
        self.pi_ready = true;
        self.resident_polynomial(PI_POLYNOMIAL)
    }

    /// Number of coefficients in each q, f, and π arena slot.
    pub const fn polynomial_len(&self) -> usize {
        self.poly_len
    }

    /// Number of sorted q-polynomials and compact q evaluations.
    pub const fn q_count(&self) -> usize {
        self.q_count
    }

    fn encode_division_round(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        source: usize,
        destination: usize,
        parameters: &DivisionParameters,
    ) -> Result<(), MetalError> {
        encoder.setComputePipelineState(&self.session.multiopen.division_summarize);
        self.bind(encoder, source, 0);
        self.bind(encoder, destination, 1);
        self.bind(encoder, POINT_OFFSETS, 2);
        self.bind(encoder, POINTS, 3);
        self.bind(encoder, DIVISION_TRANSFORMS, 4);
        self.bind(encoder, DIVISION_LOCAL_MULTIPLIERS, 5);
        bind_parameters(encoder, parameters, 6);
        let scratch_bytes = MULTIOPEN_BLOCK_SIZE
            .checked_mul(2)
            .and_then(|elements| elements.checked_mul(size_of::<Bls12_381Fr>()))
            .ok_or(MetalError::Dispatch(
                "multi-open division scratch size overflow",
            ))?;
        let total_threadgroup_bytes = scratch_bytes
            .checked_add(
                self.session
                    .multiopen
                    .division_summarize
                    .staticThreadgroupMemoryLength(),
            )
            .ok_or(MetalError::Dispatch(
                "multi-open division scratch size overflow",
            ))?;
        if total_threadgroup_bytes > self.session.device.maxThreadgroupMemoryLength() {
            return Err(MetalError::Dispatch(
                "multi-open division scratch exceeds the device limit",
            ));
        }
        // SAFETY: binding zero is the kernel's sole dynamic threadgroup
        // allocation and the complete allocation was checked above.
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(scratch_bytes, 0);
        }
        self.session.dispatch_threadgroups(
            encoder,
            &self.session.multiopen.division_summarize,
            MTLSize {
                width: parameters.block_count as usize,
                height: parameters.poly_count as usize,
                depth: 1,
            },
            MTLSize {
                width: MULTIOPEN_BLOCK_SIZE,
                height: 1,
                depth: 1,
            },
        )?;

        encoder.setComputePipelineState(&self.session.multiopen.division_scan_blocks);
        self.bind(encoder, DIVISION_TRANSFORMS, 0);
        self.bind(encoder, DIVISION_INPUTS, 1);
        bind_parameters(encoder, parameters, 2);
        self.session.dispatch(
            encoder,
            &self.session.multiopen.division_scan_blocks,
            parameters.poly_count as usize,
        );

        encoder.setComputePipelineState(&self.session.multiopen.division_fill);
        self.bind(encoder, destination, 0);
        self.bind(encoder, POINT_OFFSETS, 1);
        self.bind(encoder, POINTS, 2);
        self.bind(encoder, DIVISION_INPUTS, 3);
        self.bind(encoder, DIVISION_LOCAL_MULTIPLIERS, 4);
        bind_parameters(encoder, parameters, 5);
        self.session.dispatch(
            encoder,
            &self.session.multiopen.division_fill,
            checked_product(
                parameters.poly_count as usize,
                parameters.poly_len as usize,
                "division fill coefficient count",
            )?,
        );
        Ok(())
    }

    fn encode_evaluation(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        source: usize,
        parameters: &EvalParameters,
    ) -> Result<(), MetalError> {
        encoder.setComputePipelineState(&self.session.multiopen.eval_summarize);
        self.bind(encoder, source, 0);
        self.bind(encoder, EVAL_TRANSFORMS, 1);
        bind_parameters(encoder, parameters, 2);
        let scratch_bytes = MULTIOPEN_BLOCK_SIZE
            .checked_mul(2)
            .and_then(|elements| elements.checked_mul(size_of::<Bls12_381Fr>()))
            .ok_or(MetalError::Dispatch(
                "multi-open evaluation scratch size overflow",
            ))?;
        let total_threadgroup_bytes = scratch_bytes
            .checked_add(
                self.session
                    .multiopen
                    .eval_summarize
                    .staticThreadgroupMemoryLength(),
            )
            .ok_or(MetalError::Dispatch(
                "multi-open evaluation scratch size overflow",
            ))?;
        if total_threadgroup_bytes > self.session.device.maxThreadgroupMemoryLength() {
            return Err(MetalError::Dispatch(
                "multi-open evaluation scratch exceeds the device limit",
            ));
        }
        // SAFETY: binding zero is the kernel's sole dynamic threadgroup
        // allocation and the complete allocation was checked above.
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(scratch_bytes, 0);
        }
        self.session.dispatch_threadgroups(
            encoder,
            &self.session.multiopen.eval_summarize,
            MTLSize {
                width: parameters.block_count as usize,
                height: parameters.poly_count as usize,
                depth: 1,
            },
            MTLSize {
                width: MULTIOPEN_BLOCK_SIZE,
                height: 1,
                depth: 1,
            },
        )?;

        encoder.setComputePipelineState(&self.session.multiopen.eval_finish);
        self.bind(encoder, EVAL_TRANSFORMS, 0);
        self.bind(encoder, EVALUATIONS, 1);
        bind_parameters(encoder, parameters, 2);
        self.session.dispatch(
            encoder,
            &self.session.multiopen.eval_finish,
            parameters.poly_count as usize,
        );
        Ok(())
    }

    fn resident_polynomial(
        &self,
        buffer_index: usize,
    ) -> Result<MetalResidentPolynomialMsmLease<'_>, MetalError> {
        if buffer_index == PI_POLYNOMIAL && !self.pi_ready {
            return Err(multiopen_error("π was requested before construction"));
        }
        let scalars = ResidentMontgomeryScalars::from_synchronized_mtl_buffer(
            self.buffers[buffer_index].raw.clone(),
            0,
            self.poly_len,
        )
        .map_err(|error| MetalError::Msm(error.to_string()))?;
        Ok(MetalResidentPolynomialMsmLease::new(
            scalars,
            self.poly_len,
            self.session.session_id,
        ))
    }

    fn bind(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        buffer_index: usize,
        argument_index: usize,
    ) {
        // SAFETY: every buffer index is a fixed allocation owned by this arena;
        // the command retains all of them through completion.
        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(&self.buffers[buffer_index].raw),
                0,
                argument_index,
            );
        }
    }

    fn submit(&self, command: CommandBuffer) -> Result<(), MetalError> {
        self.session.submit_and_wait(PendingCommand {
            command,
            _resources: self
                .buffers
                .iter()
                .map(|buffer| buffer.raw.clone())
                .collect(),
        })
    }

    fn copy_values<T: Copy>(
        &mut self,
        buffer_index: usize,
        values: &[T],
    ) -> Result<(), MetalError> {
        self.copy_values_at(buffer_index, 0, values)
    }

    fn copy_values_at<T: Copy>(
        &mut self,
        buffer_index: usize,
        element_offset: usize,
        values: &[T],
    ) -> Result<(), MetalError> {
        let offset = element_offset
            .checked_mul(size_of::<T>())
            .ok_or(MetalError::LengthOverflow(element_offset))?;
        let bytes = byte_len(values)?;
        let end = offset
            .checked_add(bytes)
            .ok_or(MetalError::LengthOverflow(values.len()))?;
        let buffer = &self.buffers[buffer_index];
        if end > buffer.bytes {
            return Err(MetalError::BufferBounds {
                requested: end,
                available: buffer.bytes,
            });
        }
        // SAFETY: bounds above cover the exact POD-like source bytes, and no
        // command can access this arena while the serialized host upload runs.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                buffer.raw.contents().as_ptr().cast::<u8>().add(offset),
                bytes,
            );
        }
        self.session
            .activity
            .host_write_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    fn finish<T>(mut self, operation: Result<T, MetalError>) -> Result<T, MetalError> {
        let cleanup = self.cleanup();
        match (operation, cleanup) {
            (result, Ok(())) => result,
            (Ok(_), Err(cleanup)) => Err(cleanup),
            (Err(operation), Err(cleanup)) => Err(MetalError::SessionPoisoned {
                reason: format!("{cleanup}; original multi-open error: {operation}"),
            }),
        }
    }

    fn cleanup(&mut self) -> Result<(), MetalError> {
        if self.buffers.is_empty() {
            return Ok(());
        }
        self.session
            .clear_and_recycle_secret_buffers(std::mem::take(&mut self.buffers))
    }
}

impl Drop for MetalMultiopenArena<'_> {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

fn validate_request(
    request: MetalMultiopenRequest<'_>,
) -> Result<ValidatedMultiopen<'_>, MetalError> {
    let polynomials = request.polynomials();
    let poly_len = polynomials
        .first()
        .map(|polynomial| polynomial.len())
        .ok_or_else(|| multiopen_error("source polynomial list is empty"))?;
    if poly_len < 2
        || polynomials
            .iter()
            .any(|polynomial| polynomial.len() != poly_len)
    {
        return Err(multiopen_error(
            "source polynomials must have one equal length of at least two",
        ));
    }
    let q_offsets = request.q_offsets();
    if q_offsets.len() < 2 || q_offsets[0] != 0 {
        return Err(multiopen_error(
            "q offsets must start at zero and contain one set",
        ));
    }
    let q_count = q_offsets.len() - 1;
    if request.f_weights().len() != q_count
        || request.point_offsets().len() != q_count + 1
        || request.point_offsets()[0] != 0
    {
        return Err(multiopen_error("q, point-set, and f-weight counts differ"));
    }
    if request.q_poly_indices().len() != request.q_coefficients().len()
        || request.q_poly_indices().len() != polynomials.len()
        || q_offsets.last().copied().map(|value| value as usize)
            != Some(request.q_poly_indices().len())
        || request
            .point_offsets()
            .last()
            .copied()
            .map(|value| value as usize)
            != Some(request.points().len())
    {
        return Err(multiopen_error(
            "flattened metadata lengths disagree with their offsets",
        ));
    }
    if q_offsets.windows(2).any(|pair| pair[0] >= pair[1])
        || request
            .point_offsets()
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || request
            .q_poly_indices()
            .iter()
            .any(|&index| index as usize >= polynomials.len())
    {
        return Err(multiopen_error(
            "multi-open offsets or polynomial indices are invalid",
        ));
    }
    let mut seen_polynomials = vec![false; polynomials.len()];
    for &index in request.q_poly_indices() {
        let seen = &mut seen_polynomials[index as usize];
        if *seen {
            return Err(multiopen_error(
                "q-polynomial sources are not a permutation of input polynomials",
            ));
        }
        *seen = true;
    }
    let division_rounds = request
        .point_offsets()
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as usize)
        .max()
        .ok_or_else(|| multiopen_error("point-set list is empty"))?;
    if division_rounds >= poly_len {
        return Err(multiopen_error(
            "point-set cardinality exceeds the polynomial degree",
        ));
    }
    for (index, polynomial) in polynomials.iter().copied().enumerate() {
        validate_reduced(&format!("source polynomial {index}"), polynomial)?;
    }
    validate_reduced("q coefficient", request.q_coefficients())?;
    validate_reduced("division point", request.points())?;
    validate_reduced("f weight", request.f_weights())?;
    u32_len(poly_len)?;
    u32_len(polynomials.len())?;
    u32_len(q_count)?;
    let block_count = poly_len.div_ceil(MULTIOPEN_BLOCK_SIZE);
    u32_len(block_count)?;
    checked_product(
        poly_len,
        polynomials.len(),
        "source polynomial element count",
    )?;
    checked_product(poly_len, q_count, "q-polynomial element count")?;
    let precombined_q = polynomials.len() == q_count
        && request
            .q_offsets()
            .iter()
            .copied()
            .enumerate()
            .all(|(index, offset)| offset as usize == index)
        && request
            .q_poly_indices()
            .iter()
            .copied()
            .enumerate()
            .all(|(index, poly_index)| poly_index as usize == index)
        && request
            .q_coefficients()
            .iter()
            .all(|coefficient| coefficient.limbs == generated::FR_R);
    Ok(ValidatedMultiopen {
        request,
        poly_len,
        poly_count: polynomials.len(),
        q_count,
        block_count,
        division_rounds,
        precombined_q,
    })
}

fn validate_reduced(label: &str, values: &[Bls12_381Fr]) -> Result<(), MetalError> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| fr_greater_or_equal_host(value.limbs, generated::FR_MODULUS))
    {
        Err(multiopen_error(format!(
            "{label} element {index} is not reduced"
        )))
    } else {
        Ok(())
    }
}

fn bind_parameters<T>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    parameters: &T,
    argument_index: usize,
) {
    // SAFETY: each parameter type has a host/Metal size and offset assertion;
    // `setBytes` copies it before this stack reference expires.
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::from(parameters).cast::<c_void>(),
            size_of::<T>(),
            argument_index,
        );
    }
}

fn checked_product(lhs: usize, rhs: usize, label: &str) -> Result<usize, MetalError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| multiopen_error(format!("{label} overflows usize")))
}

fn byte_len<T>(values: &[T]) -> Result<usize, MetalError> {
    values
        .len()
        .checked_mul(size_of::<T>())
        .ok_or(MetalError::LengthOverflow(values.len()))
}

fn fr_bytes(elements: usize) -> Result<usize, MetalError> {
    elements
        .checked_mul(size_of::<Bls12_381Fr>())
        .ok_or(MetalError::LengthOverflow(elements))
}

fn clearable_bytes(bytes: usize) -> Result<usize, MetalError> {
    if bytes == 0 {
        return Err(multiopen_error("multi-open buffer length is zero"));
    }
    bytes
        .checked_add(size_of::<u64>() - 1)
        .map(|rounded| rounded & !(size_of::<u64>() - 1))
        .ok_or(MetalError::LengthOverflow(bytes))
}

fn u32_len(value: usize) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| MetalError::LengthOverflow(value))
}

fn multiopen_error(reason: impl Into<String>) -> MetalError {
    MetalError::Multiopen(reason.into())
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    #[test]
    fn resident_multiopen_matches_cpu_across_partial_blocks() {
        const POLY_LEN: usize = MULTIOPEN_BLOCK_SIZE * 2 + 5;

        let p0 = (0..POLY_LEN)
            .map(|index| fr_from_u64((3 * index + 5) as u64))
            .collect::<Vec<_>>();
        let p1 = (0..POLY_LEN)
            .map(|index| fr_from_u64((7 * index + 11) as u64))
            .collect::<Vec<_>>();
        let p2 = (0..POLY_LEN)
            .map(|index| fr_from_u64((13 * index + 17) as u64))
            .collect::<Vec<_>>();
        let polynomials = [p0.as_slice(), p1.as_slice(), p2.as_slice()];

        let one = fr_from_u64(1);
        let x1 = fr_from_u64(19);
        let x2 = fr_from_u64(23);
        let x3 = fr_from_u64(29);
        let x4 = fr_from_u64(31);
        let q_offsets = [0, 1, 3];
        let q_poly_indices = [2, 0, 1];
        let q_coefficients = [one, one, x1];
        let point_offsets = [0, 1, 3];
        let points = [fr_from_u64(37), fr_from_u64(41), fr_from_u64(43)];
        let f_weights = [one, x2];

        let q0 = p2.clone();
        let q1 = p0
            .iter()
            .copied()
            .zip(p1.iter().copied())
            .map(|(lhs, rhs)| fr_add(lhs, fr_mul(x1, rhs)))
            .collect::<Vec<_>>();
        let mut divided_q0 = kate_division(&q0, points[0]);
        let mut divided_q1 = kate_division(&kate_division(&q1, points[1]), points[2]);
        divided_q0.resize(POLY_LEN, Bls12_381Fr::default());
        divided_q1.resize(POLY_LEN, Bls12_381Fr::default());
        let expected_f = divided_q0
            .iter()
            .copied()
            .zip(divided_q1.iter().copied())
            .map(|(lhs, rhs)| fr_add(lhs, fr_mul(x2, rhs)))
            .collect::<Vec<_>>();
        let expected_evaluations = [evaluate(&q0, x3), evaluate(&q1, x3)];
        let x4_squared = fr_mul(x4, x4);
        let mut final_polynomial = q0
            .iter()
            .copied()
            .zip(q1.iter().copied())
            .zip(expected_f.iter().copied())
            .map(|((q0, q1), f)| fr_add(fr_add(q0, fr_mul(x4, q1)), fr_mul(x4_squared, f)))
            .collect::<Vec<_>>();
        let final_evaluation = evaluate(&final_polynomial, x3);
        final_polynomial[0] = fr_sub(final_polynomial[0], final_evaluation);
        let mut expected_pi = kate_division(&final_polynomial, x3);
        expected_pi.resize(POLY_LEN, Bls12_381Fr::default());

        let request = MetalMultiopenRequest::new(
            &polynomials,
            &q_offsets,
            &q_poly_indices,
            &q_coefficients,
            &point_offsets,
            &points,
            &f_weights,
        );
        let session = MetalSession::new().expect("initialize Metal runtime");
        session
            .with_bls12_381_multiopen(request, |arena| {
                let actual_f = session.read_values_from_raw::<Bls12_381Fr>(
                    &arena.buffers[F_POLYNOMIAL].raw,
                    arena.buffers[F_POLYNOMIAL].bytes,
                    POLY_LEN,
                )?;
                assert_eq!(actual_f, expected_f);

                assert_eq!(arena.evaluate_q(x3)?, expected_evaluations);

                {
                    let pi = arena.compute_pi(x3, &[one, x4, x4_squared])?;
                    assert_eq!(pi.len(), POLY_LEN);
                }
                let actual_pi = session.read_values_from_raw::<Bls12_381Fr>(
                    &arena.buffers[PI_POLYNOMIAL].raw,
                    arena.buffers[PI_POLYNOMIAL].bytes,
                    POLY_LEN,
                )?;
                assert_eq!(actual_pi, expected_pi);
                Ok(())
            })
            .expect("execute resident multi-open pipeline");
    }

    #[test]
    #[ignore = "release-mode resident multi-open stage diagnostic"]
    fn report_resident_multiopen_stage_latency() {
        const POLY_LEN: usize = 1 << 15;
        const Q_COUNT: usize = 5;

        let polynomials = (0..Q_COUNT)
            .map(|poly_index| {
                (0..POLY_LEN)
                    .map(|coefficient| fr_from_u64((17 * poly_index + 3 * coefficient + 1) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let polynomial_refs = polynomials.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let q_offsets = [0, 1, 2, 3, 4, 5];
        let q_poly_indices = [0, 1, 2, 3, 4];
        let q_coefficients = [fr_from_u64(1); Q_COUNT];
        let point_offsets = [0, 1, 3, 6, 9, 13];
        let points = (0..13)
            .map(|index| fr_from_u64((37 + 2 * index) as u64))
            .collect::<Vec<_>>();
        let f_weights = [1, 23, 529, 12_167, 279_841].map(fr_from_u64);
        let request = MetalMultiopenRequest::new(
            &polynomial_refs,
            &q_offsets,
            &q_poly_indices,
            &q_coefficients,
            &point_offsets,
            &points,
            &f_weights,
        );
        let session = MetalSession::new().expect("initialize Metal runtime");
        for round in 0..4 {
            let started = Instant::now();
            let mut prepare = None;
            let mut evaluation = None;
            let mut pi = None;
            session
                .with_bls12_381_multiopen(request, |arena| {
                    prepare = Some(started.elapsed());
                    let stage = Instant::now();
                    arena.evaluate_q(fr_from_u64(29))?;
                    evaluation = Some(stage.elapsed());
                    let stage = Instant::now();
                    let weights = [1, 31, 961, 29_791, 923_521, 28_629_151].map(fr_from_u64);
                    drop(arena.compute_pi(fr_from_u64(29), &weights)?);
                    pi = Some(stage.elapsed());
                    Ok(())
                })
                .expect("execute resident multi-open diagnostic");
            eprintln!(
                "round={round} prepare_ms={:.3} eval_ms={:.3} pi_ms={:.3} total_ms={:.3}",
                prepare.expect("callback ran").as_secs_f64() * 1_000.0,
                evaluation.expect("evaluation ran").as_secs_f64() * 1_000.0,
                pi.expect("pi ran").as_secs_f64() * 1_000.0,
                started.elapsed().as_secs_f64() * 1_000.0,
            );
        }
    }

    fn kate_division(polynomial: &[Bls12_381Fr], point: Bls12_381Fr) -> Vec<Bls12_381Fr> {
        assert!(polynomial.len() >= 2);
        let mut quotient = vec![Bls12_381Fr::default(); polynomial.len() - 1];
        let mut state = *polynomial.last().expect("non-empty polynomial");
        for index in (0..quotient.len()).rev() {
            if index + 1 != polynomial.len() - 1 {
                state = fr_add(polynomial[index + 1], fr_mul(point, state));
            }
            quotient[index] = state;
        }
        quotient
    }

    fn evaluate(polynomial: &[Bls12_381Fr], point: Bls12_381Fr) -> Bls12_381Fr {
        polynomial
            .iter()
            .rev()
            .copied()
            .fold(Bls12_381Fr::default(), |value, coefficient| {
                fr_add(fr_mul(value, point), coefficient)
            })
    }

    fn fr_from_u64(value: u64) -> Bls12_381Fr {
        let canonical = [value, 0, 0, 0];
        let mut output = blst::blst_fr::default();
        // SAFETY: BLST receives four initialized canonical little-endian limbs.
        unsafe { blst::blst_fr_from_uint64(&mut output, canonical.as_ptr()) };
        Bls12_381Fr { limbs: output.l }
    }

    fn fr_add(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
        let mut output = blst::blst_fr::default();
        let lhs = blst::blst_fr { l: lhs.limbs };
        let rhs = blst::blst_fr { l: rhs.limbs };
        // SAFETY: initialized BLST operands and disjoint output.
        unsafe { blst::blst_fr_add(&mut output, &lhs, &rhs) };
        Bls12_381Fr { limbs: output.l }
    }

    fn fr_sub(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
        let mut output = blst::blst_fr::default();
        let lhs = blst::blst_fr { l: lhs.limbs };
        let rhs = blst::blst_fr { l: rhs.limbs };
        // SAFETY: initialized BLST operands and disjoint output.
        unsafe { blst::blst_fr_sub(&mut output, &lhs, &rhs) };
        Bls12_381Fr { limbs: output.l }
    }

    fn fr_mul(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
        let mut output = blst::blst_fr::default();
        let lhs = blst::blst_fr { l: lhs.limbs };
        let rhs = blst::blst_fr { l: rhs.limbs };
        // SAFETY: initialized BLST operands and disjoint output.
        unsafe { blst::blst_fr_mul(&mut output, &lhs, &rhs) };
        Bls12_381Fr { limbs: output.l }
    }
}
