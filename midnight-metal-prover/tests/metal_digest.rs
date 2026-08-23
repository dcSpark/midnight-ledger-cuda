// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use midnight_metal_prover::{MetalSession, metal_status, proof_byte_digest_cpu};

#[test]
fn metal_digest_matches_cpu_across_boundary_lengths() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    session.reset_activity();
    let lengths = [0usize, 1, 31, 32, 1_024, 4_097];

    for length in lengths {
        let bytes = (0..length)
            .map(|index| index.wrapping_mul(17).wrapping_add(23) as u8)
            .collect::<Vec<_>>();
        let metal = session
            .proof_byte_digest(&bytes)
            .expect("execute Metal digest");

        assert_eq!(metal, proof_byte_digest_cpu(&bytes), "length {length}");
    }

    let activity = session.activity();
    assert_eq!(activity.command_buffers, lengths.len() as u64);
    assert_eq!(activity.dispatches, (2 * lengths.len()) as u64);
    assert_eq!(activity.host_waits, lengths.len() as u64);
    assert_eq!(activity.allocations, (3 * lengths.len()) as u64);
    let input_bytes = lengths
        .iter()
        .copied()
        .map(|length| length.max(1) as u64)
        .sum::<u64>();
    let host_write_bytes = input_bytes + (size_of::<u64>() * lengths.len()) as u64;
    assert_eq!(activity.host_write_bytes, host_write_bytes);
    assert_eq!(activity.host_read_bytes, (32 * lengths.len()) as u64);
    assert_eq!(
        activity.allocated_bytes,
        host_write_bytes + activity.host_read_bytes
    );
    assert_eq!(activity.blit_bytes, 0);
    assert_eq!(activity.execution_failures, 0);
}

#[test]
fn probe_separates_runtime_from_proof_readiness() {
    let status = metal_status();

    assert!(status.runtime_compiled());
    assert!(status.runtime_available());
    assert!(status.device().is_some());
    assert!(!status.proof_backend_available());
    assert!(status.runtime_unavailable_reason().is_none());
    assert!(
        status
            .proof_backend_unavailable_reason()
            .contains("proof execution kernels")
    );
}
