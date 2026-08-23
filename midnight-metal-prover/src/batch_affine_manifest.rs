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

use std::{collections::BTreeMap, fmt::Write as _, path::Path};

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;

const MANIFEST_SCHEMA: &str = "midnight-metal-msm-replay-manifest-v1";
const SOURCE_WORKLOAD_SHA256: &str =
    "47a6510def79f770e1efe57c8efd0630d916e1f60037865a8f9a0a295d4e27f1";
const SOURCE_HISTOGRAM_SHA256: &str =
    "ab3843f6d6ec98b223b2adaf2f3803b99439aa8312f2121df781726980af1ee0";
const SELECTED_LABELS: [&str; 4] = ["min", "low_mid", "high_mid", "max"];

#[derive(Debug, Error)]
pub(super) enum AuthenticManifestError {
    #[error("could not read the authentic replay manifest: {0}")]
    Read(#[from] std::io::Error),
    #[error("could not decode the authentic replay manifest: {0}")]
    Decode(#[from] serde_json::Error),
    #[error("authentic replay manifest invariant failed: {0}")]
    Invariant(&'static str),
    #[error("authentic replay manifest allocation failed: {0}")]
    Allocation(&'static str),
}

#[derive(Debug)]
pub(super) struct AuthenticReplayWorkload {
    pub(super) chunks: Box<[AuthenticReplayChunk]>,
    pub(super) total_records: u64,
    pub(super) total_runs: u64,
    pub(super) total_pairs: u64,
}

#[derive(Debug)]
pub(super) struct AuthenticReplayChunk {
    pub(super) label: Box<str>,
    pub(super) run_lengths: Box<[usize]>,
    pub(super) records: u64,
    pub(super) pairs: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    source_anchor: SourceAnchor,
    selection: Selection,
    chunks: Vec<ManifestChunk>,
    selected_totals: SelectedTotals,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SourceAnchor {
    source_workload_sha256: String,
    call_count: u32,
    logical_shapes: Vec<[u32; 2]>,
    logical_positions: u64,
    total_records: u64,
    total_nonempty_runs: u64,
    total_pair_reductions: u64,
    max_run: u32,
    run_histogram_sha256: String,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Selection {
    logical_lengths: Vec<u32>,
    lpt_bin_count: u32,
    max_calls_per_bin: u32,
    selected_load_ranks: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestChunk {
    label: String,
    load_rank: u8,
    call_count: u8,
    call_hashes: Vec<String>,
    total_records: u64,
    total_nonempty_runs: u64,
    total_pair_reductions: u64,
    max_run: u32,
    run_histogram: Vec<[u32; 2]>,
    chunk_sha256: String,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SelectedTotals {
    chunk_count: u32,
    call_count: u32,
    logical_shapes: Vec<[u32; 2]>,
    logical_positions: u64,
    total_records: u64,
    total_nonempty_runs: u64,
    total_pair_reductions: u64,
    max_run: u32,
    run_histogram_sha256: String,
}

pub(super) fn load_authentic_replay_manifest(
    path: &Path,
) -> Result<AuthenticReplayWorkload, AuthenticManifestError> {
    let bytes = std::fs::read(path)?;
    let mut value = serde_json::from_slice::<Value>(&bytes)?;
    let stored_checksum = value
        .as_object_mut()
        .and_then(|object| object.remove("manifest_sha256"))
        .and_then(|checksum| checksum.as_str().map(str::to_owned))
        .ok_or(AuthenticManifestError::Invariant(
            "manifest checksum is absent or malformed",
        ))?;
    let calculated_checksum = sha256(&serde_json::to_vec(&value)?);
    if !valid_sha256(&stored_checksum) || stored_checksum != calculated_checksum {
        return Err(AuthenticManifestError::Invariant(
            "manifest checksum is invalid",
        ));
    }

    // The checksum is an accidental-tamper check, not an authenticity proof.
    // Authenticity comes from regenerating this sensitive file locally from
    // the source capture and matching the published source-population anchor.
    let manifest = serde_json::from_value::<Manifest>(value)?;
    validate_manifest(&manifest)?;

    let mut chunks = Vec::new();
    chunks
        .try_reserve_exact(manifest.chunks.len())
        .map_err(|_| AuthenticManifestError::Allocation("chunk plans"))?;
    for chunk in &manifest.chunks {
        let expanded = expand_chunk(chunk)?;
        chunks.push(AuthenticReplayChunk {
            label: chunk.label.clone().into_boxed_str(),
            run_lengths: expanded.into_boxed_slice(),
            records: chunk.total_records,
            pairs: chunk.total_pair_reductions,
        });
    }

    Ok(AuthenticReplayWorkload {
        chunks: chunks.into_boxed_slice(),
        total_records: manifest.selected_totals.total_records,
        total_runs: manifest.selected_totals.total_nonempty_runs,
        total_pairs: manifest.selected_totals.total_pair_reductions,
    })
}

fn validate_manifest(manifest: &Manifest) -> Result<(), AuthenticManifestError> {
    validate_manifest_header(manifest)?;

    let mut merged_histogram = BTreeMap::<u32, u64>::new();
    let mut selected_calls = 0u64;
    let mut selected_records = 0u64;
    let mut selected_runs = 0u64;
    let mut selected_pairs = 0u64;
    let mut prior_pair_load = None;
    for ((chunk, expected_label), expected_rank) in manifest
        .chunks
        .iter()
        .zip(SELECTED_LABELS)
        .zip(&manifest.selection.selected_load_ranks)
    {
        validate_chunk(
            chunk,
            expected_label,
            *expected_rank,
            manifest.selection.max_calls_per_bin,
            &manifest.source_anchor,
            &mut merged_histogram,
        )?;
        if prior_pair_load.is_some_and(|prior| chunk.total_pair_reductions < prior) {
            return Err(AuthenticManifestError::Invariant(
                "selected chunk loads are not rank ordered",
            ));
        }
        prior_pair_load = Some(chunk.total_pair_reductions);
        selected_calls = checked_add(selected_calls, u64::from(chunk.call_count))?;
        selected_records = checked_add(selected_records, chunk.total_records)?;
        selected_runs = checked_add(selected_runs, chunk.total_nonempty_runs)?;
        selected_pairs = checked_add(selected_pairs, chunk.total_pair_reductions)?;
    }

    validate_selected_shapes(
        &manifest.selected_totals,
        &manifest.selection,
        selected_calls,
    )?;
    let merged = merged_histogram
        .into_iter()
        .map(|(run_length, count)| [u64::from(run_length), count])
        .collect::<Vec<_>>();
    let selected_histogram_sha256 = sha256(&serde_json::to_vec(&merged)?);
    let selected_max_run = u32::try_from(
        merged
            .last()
            .map(|[run_length, _]| *run_length)
            .unwrap_or(0),
    )
    .map_err(|_| AuthenticManifestError::Invariant("selected maximum run is invalid"))?;
    let selected = &manifest.selected_totals;
    if u64::from(selected.chunk_count) != manifest.chunks.len() as u64
        || u64::from(selected.call_count) != selected_calls
        || selected.total_records != selected_records
        || selected.total_nonempty_runs != selected_runs
        || selected.total_pair_reductions != selected_pairs
        || selected.max_run != selected_max_run
        || !valid_sha256(&selected.run_histogram_sha256)
        || selected.run_histogram_sha256 != selected_histogram_sha256
    {
        return Err(AuthenticManifestError::Invariant(
            "selected chunk totals do not reconcile",
        ));
    }
    if selected_calls > u64::from(manifest.source_anchor.call_count)
        || selected_records > manifest.source_anchor.total_records
        || selected_runs > manifest.source_anchor.total_nonempty_runs
        || selected_pairs > manifest.source_anchor.total_pair_reductions
        || selected_max_run > manifest.source_anchor.max_run
    {
        return Err(AuthenticManifestError::Invariant(
            "selected workload exceeds the anchored source population",
        ));
    }
    Ok(())
}

fn validate_manifest_header(manifest: &Manifest) -> Result<(), AuthenticManifestError> {
    if manifest.schema != MANIFEST_SCHEMA {
        return Err(AuthenticManifestError::Invariant(
            "manifest schema is not supported",
        ));
    }
    let expected_source = SourceAnchor {
        source_workload_sha256: SOURCE_WORKLOAD_SHA256.to_owned(),
        call_count: 62,
        logical_shapes: vec![[32_767, 2], [32_768, 60]],
        logical_positions: 2_031_614,
        total_records: 27_886_288,
        total_nonempty_runs: 634_284,
        total_pair_reductions: 27_252_004,
        max_run: 16_797,
        run_histogram_sha256: SOURCE_HISTOGRAM_SHA256.to_owned(),
    };
    if manifest.source_anchor != expected_source {
        return Err(AuthenticManifestError::Invariant(
            "source population does not match seed 24302",
        ));
    }
    let expected_selection = Selection {
        logical_lengths: vec![32_767, 32_768],
        lpt_bin_count: 16,
        max_calls_per_bin: 4,
        selected_load_ranks: vec![0, 5, 10, 15],
    };
    if manifest.selection != expected_selection {
        return Err(AuthenticManifestError::Invariant(
            "representative selection policy changed",
        ));
    }
    if manifest.chunks.len() != SELECTED_LABELS.len() {
        return Err(AuthenticManifestError::Invariant(
            "representative chunk count changed",
        ));
    }
    Ok(())
}

fn validate_chunk(
    chunk: &ManifestChunk,
    expected_label: &str,
    expected_rank: u32,
    max_calls_per_bin: u32,
    source: &SourceAnchor,
    merged_histogram: &mut BTreeMap<u32, u64>,
) -> Result<(), AuthenticManifestError> {
    if chunk.label != expected_label
        || u32::from(chunk.load_rank) != expected_rank
        || chunk.call_count == 0
        || u32::from(chunk.call_count) > max_calls_per_bin
        || chunk.call_hashes.len() != usize::from(chunk.call_count)
        || !valid_sha256(&chunk.chunk_sha256)
        || !chunk.call_hashes.iter().all(|hash| valid_sha256(hash))
        || !chunk.call_hashes.windows(2).all(|pair| pair[0] <= pair[1])
    {
        return Err(AuthenticManifestError::Invariant(
            "representative chunk policy is invalid",
        ));
    }

    let replay_payload = json!({
        "call_hashes": chunk.call_hashes,
        "max_run": chunk.max_run,
        "run_histogram": chunk.run_histogram,
        "total_nonempty_runs": chunk.total_nonempty_runs,
        "total_pair_reductions": chunk.total_pair_reductions,
        "total_records": chunk.total_records,
    });
    if sha256(&serde_json::to_vec(&replay_payload)?) != chunk.chunk_sha256 {
        return Err(AuthenticManifestError::Invariant(
            "representative chunk checksum is invalid",
        ));
    }

    if chunk.total_records > source.total_records
        || chunk.total_nonempty_runs > source.total_nonempty_runs
        || chunk.total_pair_reductions > source.total_pair_reductions
        || chunk.max_run > source.max_run
    {
        return Err(AuthenticManifestError::Invariant(
            "representative chunk exceeds the anchored source population",
        ));
    }
    let mut prior = 0u32;
    let mut records = 0u64;
    let mut runs = 0u64;
    let mut pairs = 0u64;
    for &[run_length, count] in &chunk.run_histogram {
        if run_length == 0 || run_length <= prior || run_length > source.max_run || count == 0 {
            return Err(AuthenticManifestError::Invariant(
                "chunk histogram is not positive and strictly ascending",
            ));
        }
        prior = run_length;
        let count64 = u64::from(count);
        records = checked_add(records, u64::from(run_length) * count64)?;
        runs = checked_add(runs, count64)?;
        pairs = checked_add(pairs, u64::from(run_length - 1) * count64)?;
    }
    if (records, runs, pairs)
        != (
            chunk.total_records,
            chunk.total_nonempty_runs,
            chunk.total_pair_reductions,
        )
        || prior != chunk.max_run
    {
        return Err(AuthenticManifestError::Invariant(
            "chunk histogram arithmetic does not reconcile",
        ));
    }
    for &[run_length, count] in &chunk.run_histogram {
        let merged_count = merged_histogram.entry(run_length).or_default();
        *merged_count = checked_add(*merged_count, u64::from(count))?;
    }
    Ok(())
}

fn validate_selected_shapes(
    selected: &SelectedTotals,
    selection: &Selection,
    expected_call_count: u64,
) -> Result<(), AuthenticManifestError> {
    let mut prior = 0u32;
    let mut call_count = 0u64;
    let mut logical_positions = 0u64;
    for &[logical_length, count] in &selected.logical_shapes {
        if logical_length == 0
            || count == 0
            || logical_length <= prior
            || !selection.logical_lengths.contains(&logical_length)
        {
            return Err(AuthenticManifestError::Invariant(
                "selected logical shapes are invalid",
            ));
        }
        prior = logical_length;
        call_count = checked_add(call_count, u64::from(count))?;
        logical_positions = checked_add(
            logical_positions,
            u64::from(logical_length) * u64::from(count),
        )?;
    }
    if call_count != expected_call_count
        || call_count != u64::from(selected.call_count)
        || logical_positions != selected.logical_positions
    {
        return Err(AuthenticManifestError::Invariant(
            "selected logical shapes do not reconcile",
        ));
    }
    Ok(())
}

fn expand_chunk(chunk: &ManifestChunk) -> Result<Vec<usize>, AuthenticManifestError> {
    let reserve = usize::try_from(chunk.total_nonempty_runs).map_err(|_| {
        AuthenticManifestError::Invariant("chunk run count does not fit host memory")
    })?;
    let mut run_lengths = Vec::new();
    run_lengths
        .try_reserve_exact(reserve)
        .map_err(|_| AuthenticManifestError::Allocation("expanded run lengths"))?;
    for &[run_length, count] in &chunk.run_histogram {
        let run_length = usize::try_from(run_length).map_err(|_| {
            AuthenticManifestError::Invariant("run length does not fit host memory")
        })?;
        let count = usize::try_from(count).map_err(|_| {
            AuthenticManifestError::Invariant("histogram count does not fit host memory")
        })?;
        run_lengths.extend(std::iter::repeat_n(run_length, count));
    }
    debug_assert_eq!(run_lengths.len(), reserve);
    Ok(run_lengths)
}

fn checked_add(left: u64, right: u64) -> Result<u64, AuthenticManifestError> {
    left.checked_add(right)
        .ok_or(AuthenticManifestError::Invariant(
            "manifest arithmetic overflowed",
        ))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing into a String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_TEMPORARY_FILE: AtomicU64 = AtomicU64::new(0);

    fn chunk_value(label: &str, load_rank: u8, run_length: u32) -> Value {
        let call_hashes = vec![format!("{run_length:064x}")];
        let run_histogram = vec![[run_length, 1u32]];
        let replay_payload = json!({
            "call_hashes": call_hashes,
            "max_run": run_length,
            "run_histogram": run_histogram,
            "total_nonempty_runs": 1,
            "total_pair_reductions": run_length - 1,
            "total_records": run_length,
        });
        let chunk_sha256 = sha256(&serde_json::to_vec(&replay_payload).expect("encode chunk"));
        json!({
            "call_count": 1,
            "call_hashes": call_hashes,
            "chunk_sha256": chunk_sha256,
            "label": label,
            "load_rank": load_rank,
            "max_run": run_length,
            "run_histogram": run_histogram,
            "total_nonempty_runs": 1,
            "total_pair_reductions": run_length - 1,
            "total_records": run_length,
        })
    }

    fn reseal_manifest(value: &mut Value) {
        value
            .as_object_mut()
            .expect("manifest object")
            .remove("manifest_sha256");
        let checksum = sha256(&serde_json::to_vec(value).expect("encode manifest body"));
        value
            .as_object_mut()
            .expect("manifest object")
            .insert("manifest_sha256".to_owned(), Value::String(checksum));
    }

    fn generated_source_tied_manifest() -> Value {
        let run_lengths = [1u32, 2, 3, 4];
        let chunks = SELECTED_LABELS
            .into_iter()
            .zip([0u8, 5, 10, 15])
            .zip(run_lengths)
            .map(|((label, rank), run_length)| chunk_value(label, rank, run_length))
            .collect::<Vec<_>>();
        let records = run_lengths.iter().copied().map(u64::from).sum::<u64>();
        let pairs = run_lengths
            .iter()
            .copied()
            .map(|run_length| u64::from(run_length - 1))
            .sum::<u64>();
        let merged_histogram = run_lengths
            .iter()
            .copied()
            .map(|run_length| [u64::from(run_length), 1u64])
            .collect::<Vec<_>>();
        let selected_histogram_sha256 =
            sha256(&serde_json::to_vec(&merged_histogram).expect("encode selected histogram"));
        let mut manifest = json!({
            "chunks": chunks,
            "schema": MANIFEST_SCHEMA,
            "selected_totals": {
                "call_count": run_lengths.len(),
                "chunk_count": chunks.len(),
                "logical_positions": 32_768 * run_lengths.len(),
                "logical_shapes": [[32_768, run_lengths.len()]],
                "max_run": run_lengths.last().copied().unwrap_or(0),
                "run_histogram_sha256": selected_histogram_sha256,
                "total_nonempty_runs": run_lengths.len(),
                "total_pair_reductions": pairs,
                "total_records": records,
            },
            "selection": {
                "logical_lengths": [32_767, 32_768],
                "lpt_bin_count": 16,
                "max_calls_per_bin": 4,
                "selected_load_ranks": [0, 5, 10, 15],
            },
            "source_anchor": {
                "call_count": 62,
                "logical_positions": 2_031_614,
                "logical_shapes": [[32_767, 2], [32_768, 60]],
                "max_run": 16_797,
                "run_histogram_sha256": SOURCE_HISTOGRAM_SHA256,
                "source_workload_sha256": SOURCE_WORKLOAD_SHA256,
                "total_nonempty_runs": 634_284,
                "total_pair_reductions": 27_252_004,
                "total_records": 27_886_288,
            },
        });
        reseal_manifest(&mut manifest);
        manifest
    }

    fn load_value(value: &Value) -> Result<AuthenticReplayWorkload, AuthenticManifestError> {
        let sequence = NEXT_TEMPORARY_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "midnight-metal-manifest-test-{}-{sequence}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            serde_json::to_vec(value).expect("encode test manifest"),
        )
        .expect("write test manifest");
        let result = load_authentic_replay_manifest(&path);
        std::fs::remove_file(path).expect("remove test manifest");
        result
    }

    #[test]
    fn checksum_and_sha_syntax_are_strict() {
        assert_eq!(
            sha256(b"midnight-metal"),
            "01f99491dcb94e781c500f09fe8a04c5b014a936e5511a4046396ac2011d7ff0"
        );
        assert!(valid_sha256(SOURCE_WORKLOAD_SHA256));
        assert!(!valid_sha256(&SOURCE_WORKLOAD_SHA256.to_uppercase()));
        assert!(!valid_sha256(&SOURCE_WORKLOAD_SHA256[1..]));
    }

    #[test]
    fn source_tied_manifest_does_not_require_selected_fingerprints() {
        let workload = load_value(&generated_source_tied_manifest()).expect("load manifest");
        assert_eq!(workload.chunks.len(), SELECTED_LABELS.len());
        assert_eq!(
            workload.total_records,
            workload
                .chunks
                .iter()
                .flat_map(|chunk| chunk.run_lengths.iter())
                .map(|run_length| *run_length as u64)
                .sum::<u64>()
        );
        assert_eq!(
            workload.total_runs,
            workload
                .chunks
                .iter()
                .map(|chunk| chunk.run_lengths.len() as u64)
                .sum::<u64>()
        );
        assert_eq!(
            workload.total_pairs,
            workload
                .chunks
                .iter()
                .flat_map(|chunk| chunk.run_lengths.iter())
                .map(|run_length| (*run_length - 1) as u64)
                .sum::<u64>()
        );
    }

    #[test]
    fn whole_manifest_tampering_is_rejected_before_expansion() {
        let mut manifest = generated_source_tied_manifest();
        manifest["chunks"][0]["total_nonempty_runs"] = json!(u64::MAX);
        let error = load_value(&manifest).expect_err("stale checksum must fail");
        assert!(matches!(
            error,
            AuthenticManifestError::Invariant("manifest checksum is invalid")
        ));
    }

    #[test]
    fn chunk_tampering_is_rejected_before_expansion() {
        let mut manifest = generated_source_tied_manifest();
        manifest["chunks"][0]["total_nonempty_runs"] = json!(u64::MAX);
        reseal_manifest(&mut manifest);
        let error = load_value(&manifest).expect_err("stale chunk checksum must fail");
        assert!(matches!(
            error,
            AuthenticManifestError::Invariant("representative chunk checksum is invalid")
        ));
    }

    #[test]
    fn authentic_manifest_loads_when_the_private_capture_is_present() {
        let path = Path::new("/tmp/metal-msm-replay-seed24302.json");
        if !path.exists() {
            return;
        }
        let workload = load_authentic_replay_manifest(path).expect("load authentic manifest");
        assert_eq!(workload.chunks.len(), SELECTED_LABELS.len());
        assert_eq!(
            workload.total_records,
            workload
                .chunks
                .iter()
                .map(|chunk| chunk.records)
                .sum::<u64>()
        );
        assert_eq!(
            workload.total_runs,
            workload
                .chunks
                .iter()
                .map(|chunk| chunk.run_lengths.len() as u64)
                .sum::<u64>()
        );
        assert_eq!(
            workload.total_pairs,
            workload.chunks.iter().map(|chunk| chunk.pairs).sum::<u64>()
        );
        for (chunk, expected_label) in workload.chunks.iter().zip(SELECTED_LABELS) {
            assert_eq!(chunk.label.as_ref(), expected_label);
            assert_eq!(
                chunk.records,
                chunk
                    .run_lengths
                    .iter()
                    .map(|run_length| *run_length as u64)
                    .sum::<u64>()
            );
            assert_eq!(
                chunk.pairs,
                chunk
                    .run_lengths
                    .iter()
                    .map(|run_length| (*run_length - 1) as u64)
                    .sum::<u64>()
            );
        }
    }
}
