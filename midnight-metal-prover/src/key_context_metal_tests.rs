// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Device-backed tests for immutable key-context ownership and caching.

use super::*;
use crate::key_context::{OwnedMetalKeyContextInput, OwnedV2Input};

#[test]
fn key_context_cache_cold_hit_content_domain_and_drop() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    let fixture = OwnedMetalKeyContextInput::representative();
    session.reset_activity();

    let first = session
        .get_or_create_key_context(fixture.input())
        .expect("build first immutable key context");
    let cold = session.activity();
    assert_eq!(cold.key_cache_hits, 0);
    assert_eq!(cold.key_cache_misses, 1);
    assert_eq!(cold.key_context_builds, 1);
    assert_eq!(cold.key_context_build_failures, 0);
    assert_eq!(cold.live_key_contexts, 1);
    assert_eq!(first.sections(), fixture.sections);
    assert_eq!(first.challenge_bound(), fixture.challenge_bound);
    assert_eq!(
        cold.immutable_key_live_bytes,
        first.immutable_bytes() as u64
    );
    assert_eq!(cold.allocations, 9);
    assert_eq!(cold.command_buffers, 0);
    assert_eq!(cold.dispatches, 0);
    assert_eq!(cold.host_waits, 0);

    let allocations = cold.allocations;
    let written = cold.host_write_bytes;
    let same = session
        .get_or_create_key_context(fixture.input())
        .expect("reuse immutable key context");
    let warm = session.activity();
    assert_eq!(warm.key_cache_hits, 1);
    assert_eq!(warm.key_cache_misses, 1);
    assert_eq!(warm.key_context_builds, 1);
    assert_eq!(warm.allocations, allocations);
    assert_eq!(warm.host_write_bytes, written);
    assert_eq!(warm.live_key_contexts, 1);
    assert_eq!(first.identity(), same.identity());

    drop(first);
    drop(same);
    assert_eq!(session.activity().live_key_contexts, 1);
    let retained = session
        .get_or_create_key_context(fixture.input())
        .expect("reuse a context retained only by the warm cache");
    assert_eq!(session.activity().key_cache_hits, 2);
    assert_eq!(session.activity().key_context_builds, 1);

    let mut changed = OwnedMetalKeyContextInput::representative();
    changed.fixed_cosets[0].limbs[0] ^= 1;
    let changed_context = session
        .get_or_create_key_context(changed.input())
        .expect("content change builds a distinct context");
    assert_ne!(
        retained.identity().content_fingerprint(),
        changed_context.identity().content_fingerprint()
    );

    let mut extended = OwnedMetalKeyContextInput::representative();
    extended.n_ext = 8;
    extended.fixed_cosets.resize(8, extended.constants[0]);
    extended.selectors.resize(24, extended.constants[0]);
    let extended_context = session
        .get_or_create_key_context(extended.input())
        .expect("domain change builds a distinct context");
    assert_ne!(
        retained.identity().n_ext(),
        extended_context.identity().n_ext()
    );
    assert_eq!(session.activity().key_cache_misses, 3);

    drop(retained);
    drop(changed_context);
    drop(extended_context);
    assert_eq!(session.activity().live_key_contexts, 2);
    assert_ne!(session.activity().immutable_key_live_bytes, 0);

    let rebuilt = session
        .get_or_create_key_context(fixture.input())
        .expect("expired weak entry rebuilds");
    let rebuilt_activity = session.activity();
    assert_eq!(rebuilt_activity.key_cache_misses, 4);
    assert_eq!(rebuilt_activity.key_context_builds, 4);
    assert_eq!(rebuilt_activity.live_key_contexts, 2);
    assert_eq!(
        session
            .key_contexts
            .lock()
            .expect("key cache lock")
            .weak
            .len(),
        2,
        "the weak index contains only the bounded warm set after eviction"
    );
    drop(rebuilt);
}

#[test]
fn key_context_larger_than_warm_budget_is_not_strongly_retained() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    let fixture = OwnedMetalKeyContextInput::representative();
    {
        let mut cache = session.key_contexts.lock().expect("key cache lock");
        cache.max_bytes = 0;
    }

    let context = session
        .get_or_create_key_context(fixture.input())
        .expect("build uncached immutable key context");
    assert_eq!(session.activity().live_key_contexts, 1);
    assert!(
        session
            .key_contexts
            .lock()
            .expect("key cache lock")
            .warm
            .is_empty()
    );

    drop(context);
    assert_eq!(session.activity().live_key_contexts, 0);
    assert_eq!(session.activity().immutable_key_live_bytes, 0);
}

#[test]
fn key_context_has_explicit_session_ownership() {
    let first_session = MetalSession::new().expect("initialize first Metal runtime");
    let second_session = MetalSession::new().expect("initialize second Metal runtime");
    let fixture = OwnedMetalKeyContextInput::representative();

    let first = first_session
        .get_or_create_key_context(fixture.input())
        .expect("build first-session context");
    first_session
        .validate_key_context(&first)
        .expect("owning session accepts context");
    assert!(matches!(
        second_session.validate_key_context(&first),
        Err(MetalError::ForeignKeyContext)
    ));

    second_session.reset_activity();
    let second = second_session
        .get_or_create_key_context(fixture.input())
        .expect("second session builds independently");
    assert_eq!(second_session.activity().key_cache_misses, 1);
    assert_eq!(second_session.activity().key_cache_hits, 0);
    second_session
        .validate_key_context(&second)
        .expect("second session accepts its context");

    let first_view = first_session
        .immutable_fr_view(&first, MetalKeyFrSource::FixedCosetCompact)
        .expect("resolve first-session immutable view");
    let mut resources = Vec::new();
    assert!(matches!(
        second_session.bind_immutable_fr_view(
            &first,
            &first_view,
            MetalKeyFrSource::FixedCosetCompact,
            &mut resources,
        ),
        Err(MetalError::ForeignKeyContext)
    ));
    assert!(resources.is_empty());

    let second_local = second_session
        .immutable_fr_view(&second, MetalKeyFrSource::FixedCosetCompact)
        .expect("resolve second-session immutable view");
    assert_eq!(first_view.context_id, second_local.context_id);
    assert_eq!(first_view.allocation_id, second_local.allocation_id);
    assert!(matches!(
        second_session.bind_immutable_fr_view(
            &second,
            &first_view,
            MetalKeyFrSource::FixedCosetCompact,
            &mut resources,
        ),
        Err(MetalError::ResidentState(_))
    ));
    assert!(resources.is_empty());
}

#[test]
fn immutable_fr_views_bind_exact_sources_domains_and_columns() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    let fixture = OwnedMetalKeyContextInput::representative();
    let context = session
        .get_or_create_key_context(fixture.input())
        .expect("build immutable key context");
    let expected_domain = FrDomainTag::for_ntt(
        bls12_381_ntt_domain(2).expect("construct canonical immutable domain"),
        FrBasis::ExtendedCoset,
    );

    let fixed = session
        .immutable_fr_view(&context, MetalKeyFrSource::FixedCosetCompact)
        .expect("resolve compact fixed-coset source");
    assert_eq!(fixed.session_id, session.session_id);
    assert_eq!(fixed.context_id, context.inner.context_id);
    assert_eq!(fixed.source_id, MetalKeyFrSource::FixedCosetCompact);
    assert_eq!(fixed.domain, expected_domain);
    assert_eq!(fixed.logical_len, 4);
    assert_eq!(fixed.stride, 4);
    assert_eq!(fixed.batch_count, 1);
    assert_ne!(fixed.allocation_id, 0);

    let selectors = session
        .immutable_fr_view(&context, MetalKeyFrSource::Selectors)
        .expect("resolve selector source");
    assert_eq!(selectors.domain, expected_domain);
    assert_eq!(selectors.batch_count, 3);
    let tail = selectors
        .columns(1, 2)
        .expect("select l_last and l_active columns");
    assert_eq!(tail.byte_offset, 4 * size_of::<Bls12_381Fr>());
    assert_eq!(tail.batch_count, 2);
    assert_eq!(tail.domain, expected_domain);
    assert!(matches!(
        selectors.columns(0, 0),
        Err(MetalError::ResidentState(
            "immutable key-view column range escapes its source"
        ))
    ));
    assert!(matches!(
        selectors.columns(2, 2),
        Err(MetalError::ResidentState(
            "immutable key-view column range escapes its source"
        ))
    ));

    let mut resources = Vec::new();
    session
        .bind_immutable_fr_view(
            &context,
            &fixed,
            MetalKeyFrSource::FixedCosetCompact,
            &mut resources,
        )
        .expect("bind fixed source into a command epoch");
    session
        .bind_immutable_fr_view(&context, &tail, MetalKeyFrSource::Selectors, &mut resources)
        .expect("bind selector subview into a command epoch");
    assert_eq!(resources.len(), 2);
}

#[test]
fn immutable_fr_views_reject_cross_context_source_allocation_and_basis_confusion() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    let fixture = OwnedMetalKeyContextInput::representative();
    let first = session
        .get_or_create_key_context(fixture.input())
        .expect("build first key context");
    let mut changed = OwnedMetalKeyContextInput::representative();
    changed.fixed_cosets[0].limbs[0] ^= 1;
    let second = session
        .get_or_create_key_context(changed.input())
        .expect("build same-shape second key context");
    let expected_domain = FrDomainTag::for_ntt(
        bls12_381_ntt_domain(2).expect("construct canonical immutable domain"),
        FrBasis::ExtendedCoset,
    );

    let first_view = session
        .immutable_fr_view(&first, MetalKeyFrSource::FixedCosetCompact)
        .expect("resolve first fixed source");
    let mut resources = Vec::new();
    assert!(matches!(
        session.bind_immutable_fr_view(
            &second,
            &first_view,
            MetalKeyFrSource::FixedCosetCompact,
            &mut resources,
        ),
        Err(MetalError::ResidentState(_))
    ));
    assert!(resources.is_empty());

    let mut wrong_allocation = session
        .immutable_fr_view(&first, MetalKeyFrSource::FixedCosetCompact)
        .expect("resolve allocation-identity test view");
    wrong_allocation.allocation_id ^= 1;
    assert!(matches!(
        session.bind_immutable_fr_view(
            &first,
            &wrong_allocation,
            MetalKeyFrSource::FixedCosetCompact,
            &mut resources,
        ),
        Err(MetalError::ResidentState(_))
    ));

    let mut wrong_basis = session
        .immutable_fr_view(&first, MetalKeyFrSource::FixedCosetCompact)
        .expect("resolve basis test view");
    wrong_basis.domain.basis = FrBasis::Lagrange;
    assert!(matches!(
        session.bind_immutable_fr_view(
            &first,
            &wrong_basis,
            MetalKeyFrSource::FixedCosetCompact,
            &mut resources,
        ),
        Err(MetalError::ResidentState(_))
    ));

    let mut shortened = session
        .immutable_fr_view(&first, MetalKeyFrSource::FixedCosetCompact)
        .expect("resolve canonical-shape test view");
    shortened.logical_len = 2;
    shortened.stride = 2;
    shortened.domain = FrDomainTag::for_ntt(
        bls12_381_ntt_domain(1).expect("construct forged short domain"),
        FrBasis::ExtendedCoset,
    );
    assert!(matches!(
        session.bind_immutable_fr_view(
            &first,
            &shortened,
            MetalKeyFrSource::FixedCosetCompact,
            &mut resources,
        ),
        Err(MetalError::ResidentState(_))
    ));

    let mut wrong_roots = session
        .immutable_fr_view(&first, MetalKeyFrSource::FixedCosetCompact)
        .expect("resolve exact-domain test view");
    let mut alternate_domain = expected_domain
        .ntt_domain
        .expect("evaluation basis carries an exact domain");
    std::mem::swap(
        &mut alternate_domain.root,
        &mut alternate_domain.inverse_root,
    );
    wrong_roots.domain = FrDomainTag::for_ntt(alternate_domain, FrBasis::ExtendedCoset);
    assert!(matches!(
        session.bind_immutable_fr_view(
            &first,
            &wrong_roots,
            MetalKeyFrSource::FixedCosetCompact,
            &mut resources,
        ),
        Err(MetalError::ResidentState(_))
    ));

    let selector_buffer = first
        .inner
        .buffers
        .lagrange_selectors
        .as_ref()
        .expect("selector allocation");
    let forged_source = ImmutableFrView {
        session_id: session.session_id,
        context_id: first.inner.context_id,
        source_id: MetalKeyFrSource::FixedCosetCompact,
        allocation_id: selector_buffer.allocation_id,
        buffer: selector_buffer,
        byte_offset: 0,
        logical_len: 4,
        stride: 4,
        batch_count: 1,
        domain: expected_domain,
    };
    assert!(matches!(
        session.bind_immutable_fr_view(
            &first,
            &forged_source,
            MetalKeyFrSource::FixedCosetCompact,
            &mut resources,
        ),
        Err(MetalError::ResidentState(_))
    ));
}

#[test]
fn physical_context_generations_change_after_same_content_rebuild() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    let fixture = OwnedMetalKeyContextInput::representative();
    let first = session
        .get_or_create_key_context(fixture.input())
        .expect("build first key context");
    let first_context_id = first.inner.context_id;
    let first_fixed_allocation_id = first
        .inner
        .buffers
        .fixed_cosets
        .as_ref()
        .expect("fixed-coset allocation")
        .allocation_id;
    drop(first);

    let mut changed_fixed = OwnedMetalKeyContextInput::representative();
    changed_fixed.fixed_cosets[0].limbs[0] ^= 1;
    let first_eviction_context = session
        .get_or_create_key_context(changed_fixed.input())
        .expect("build first eviction context");
    let mut changed_selector = OwnedMetalKeyContextInput::representative();
    changed_selector.selectors[0].limbs[0] ^= 1;
    let second_eviction_context = session
        .get_or_create_key_context(changed_selector.input())
        .expect("build second eviction context");
    drop(first_eviction_context);
    drop(second_eviction_context);

    let rebuilt = session
        .get_or_create_key_context(fixture.input())
        .expect("rebuild expired same-content context");
    assert_ne!(first_context_id, rebuilt.inner.context_id);
    assert_ne!(
        first_fixed_allocation_id,
        rebuilt
            .inner
            .buffers
            .fixed_cosets
            .as_ref()
            .expect("rebuilt fixed-coset allocation")
            .allocation_id
    );
}

#[test]
fn key_context_identity_exhaustion_is_counted_before_allocation() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    session
        .next_key_context_id
        .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    session.reset_activity();

    let fixture = OwnedMetalKeyContextInput::representative();
    assert!(matches!(
        session.get_or_create_key_context(fixture.input()),
        Err(MetalError::KeyContext(reason)) if reason.contains("identity space is exhausted")
    ));
    let activity = session.activity();
    assert_eq!(activity.key_cache_misses, 1);
    assert_eq!(activity.key_context_builds, 0);
    assert_eq!(activity.key_context_build_failures, 1);
    assert_eq!(activity.allocations, 0);
    assert_eq!(activity.host_write_bytes, 0);
    assert_eq!(activity.live_key_contexts, 0);
}

#[test]
fn immutable_allocation_identity_exhaustion_is_counted_before_allocation() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    session
        .next_immutable_allocation_id
        .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    session.reset_activity();

    let fixture = OwnedMetalKeyContextInput::representative();
    assert!(matches!(
        session.get_or_create_key_context(fixture.input()),
        Err(MetalError::KeyContext(reason)) if reason.contains("allocation identity space is exhausted")
    ));
    let activity = session.activity();
    assert_eq!(activity.key_cache_misses, 1);
    assert_eq!(activity.key_context_builds, 0);
    assert_eq!(activity.key_context_build_failures, 1);
    assert_eq!(activity.allocations, 0);
    assert_eq!(activity.host_write_bytes, 0);
    assert_eq!(activity.live_key_contexts, 0);
}

#[test]
fn zero_compact_fixed_columns_are_a_valid_absent_source() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    let mut fixture = OwnedMetalKeyContextInput::representative();
    fixture.fixed_map.clear();
    fixture.fixed_cosets.clear();
    let context = session
        .get_or_create_key_context(fixture.input())
        .expect("build context without compact fixed columns");
    assert_eq!(context.inner.fixed_column_count, 0);
    assert!(context.inner.buffers.fixed_cosets.is_none());
    assert!(matches!(
        session.immutable_fr_view(&context, MetalKeyFrSource::FixedCosetCompact),
        Err(MetalError::KeyContext(reason)) if reason.contains("no fixed-coset columns")
    ));
}

#[test]
fn malformed_key_context_is_rejected_before_metal_activity() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    let mut malformed = OwnedMetalKeyContextInput::representative();
    malformed.descriptors[0].scratch_capacity = 16;
    session.reset_activity();

    assert!(matches!(
        session.get_or_create_key_context(malformed.input()),
        Err(MetalError::KeyContext(_))
    ));
    let activity = session.activity();
    assert_eq!(activity.allocations, 0);
    assert_eq!(activity.allocated_bytes, 0);
    assert_eq!(activity.host_write_bytes, 0);
    assert_eq!(activity.command_buffers, 0);
    assert_eq!(activity.dispatches, 0);
    assert_eq!(activity.key_cache_hits, 0);
    assert_eq!(activity.key_cache_misses, 0);
    assert_eq!(activity.key_context_builds, 0);
    assert_eq!(activity.key_context_build_failures, 0);
    assert_eq!(activity.live_key_contexts, 0);
}

#[test]
fn key_context_v2_binds_all_content_and_reuses_only_validated_srs() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    let fixture = OwnedV2Input::representative();
    let input = fixture.input();
    session.reset_activity();

    let first = session
        .get_or_create_key_context_v2(input)
        .expect("build complete schema-v2 key context");
    let identity = first.identity();
    assert_eq!(identity.schema_version(), 2);
    assert_eq!(
        identity.domain_fingerprint(),
        Some(input.domain.neutral_fingerprint)
    );
    assert_eq!(
        identity.prepared_key_fingerprint(),
        Some(input.prepared_key_fingerprint)
    );
    assert_eq!(
        identity.prepared_srs_fingerprint(),
        Some(input.prepared_srs_fingerprint)
    );
    assert_eq!(identity.base_ntt_domain(), Some(input.domain.base_ntt));
    assert_eq!(
        identity.extended_ntt_domain(),
        Some(input.domain.extended_ntt)
    );
    assert_eq!(identity.ntt_abi_version(), Some(NTT_PARAMETER_ABI_VERSION));
    assert_eq!(
        identity.ntt_abi_fingerprint(),
        Some(NTT_PARAMETER_ABI_FINGERPRINT)
    );

    let buffers = &first.inner.buffers;
    assert!(buffers.descriptors.is_some());
    assert!(buffers.constants.is_some());
    assert!(buffers.rotations.is_some());
    assert!(buffers.operations.is_some());
    assert!(buffers.fixed_map.is_some());
    assert!(buffers.advice_map.is_some());
    assert!(buffers.instance_map.is_some());
    assert!(buffers.fixed_cosets.is_some());
    assert!(buffers.lagrange_selectors.is_some());
    assert!(buffers.permutation_columns.is_some());
    assert!(buffers.trash_selector_fixed_indices.is_some());
    assert!(buffers.fixed_lagrange.is_some());
    assert!(buffers.permutation_cosets.is_some());
    assert!(buffers.extended_coset_points.is_some());
    assert!(buffers.permutation_lagrange.is_some());
    assert!(buffers.coefficient_columns.is_some());
    assert!(buffers.vanishing_inverses.is_some());
    assert!(buffers.coefficient_bases.is_some());
    assert!(buffers.lagrange_bases.is_some());
    assert_eq!(buffers.checked_bytes().unwrap(), first.immutable_bytes());
    let cold = session.activity();
    assert_eq!(cold.allocations, 19);
    assert_eq!(cold.host_write_bytes, first.immutable_bytes() as u64);
    assert_eq!(cold.key_cache_misses, 1);
    assert_eq!(cold.key_context_builds, 1);
    assert_eq!(session.srs_subgroup_validations.load(Ordering::Relaxed), 1);

    let same = session
        .get_or_create_key_context_v2(fixture.input())
        .expect("reuse exact schema-v2 context");
    assert_eq!(same.identity(), first.identity());
    assert_eq!(session.srs_subgroup_validations.load(Ordering::Relaxed), 1);

    let mut changed_key = OwnedV2Input::representative();
    changed_key.permutation_columns.swap(0, 1);
    changed_key.refresh_neutral_fingerprints();
    let changed_key_context = session
        .get_or_create_key_context_v2(changed_key.input())
        .expect("build another PK over the already-validated SRS");
    assert_ne!(
        changed_key_context.identity().prepared_key_fingerprint(),
        first.identity().prepared_key_fingerprint()
    );
    assert_eq!(
        changed_key_context.identity().prepared_srs_fingerprint(),
        first.identity().prepared_srs_fingerprint()
    );
    assert_eq!(session.srs_subgroup_validations.load(Ordering::Relaxed), 1);

    let mut changed_srs = OwnedV2Input::representative();
    changed_srs.coefficient_bases.swap(0, 1);
    changed_srs.refresh_neutral_fingerprints();
    let changed_srs_context = session
        .get_or_create_key_context_v2(changed_srs.input())
        .expect("build context over a distinct valid SRS");
    assert_ne!(
        changed_srs_context.identity().prepared_srs_fingerprint(),
        first.identity().prepared_srs_fingerprint()
    );
    assert_eq!(session.srs_subgroup_validations.load(Ordering::Relaxed), 2);
    let validated_srs = session.validated_srs.lock().expect("validated-SRS cache");
    assert_eq!(validated_srs.max_entries, VALIDATED_SRS_MAX_ENTRIES);
    assert_eq!(validated_srs.fingerprints.len(), 2);
    assert!(validated_srs.fingerprints.len() <= validated_srs.max_entries);
    drop(validated_srs);

    let activity = session.activity();
    assert_eq!(activity.key_cache_hits, 1);
    assert_eq!(activity.key_cache_misses, 3);
    assert_eq!(activity.key_context_builds, 3);
    assert_eq!(activity.key_context_build_failures, 0);
}

#[test]
fn key_context_v2_views_preserve_exact_basis_period_bounds_and_ownership() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    let fixture = OwnedV2Input::representative();
    let input = fixture.input();
    let context = session
        .get_or_create_key_context_v2(input)
        .expect("build complete schema-v2 key context");
    let base = input.domain.base_ntt;
    let extended = input.domain.extended_ntt;

    let fixed_coset = session
        .immutable_fr_view(&context, MetalKeyFrSource::FixedCosetCompact)
        .expect("resolve compact fixed cosets");
    assert_eq!(
        fixed_coset.domain,
        FrDomainTag::for_ntt(extended, FrBasis::ExtendedCoset)
    );
    assert_eq!(fixed_coset.logical_len, 4);
    assert_eq!(fixed_coset.batch_count, 1);

    let selectors = session
        .immutable_fr_view(&context, MetalKeyFrSource::Selectors)
        .expect("resolve selectors");
    assert_eq!(selectors.domain.ntt_domain, Some(extended));
    assert_eq!(selectors.batch_count, 3);

    let fixed_lagrange = session
        .immutable_fr_view(&context, MetalKeyFrSource::FixedLagrangeCompact)
        .expect("resolve compact fixed Lagrange columns");
    assert_eq!(
        fixed_lagrange.domain,
        FrDomainTag::for_ntt(base, FrBasis::Lagrange)
    );
    assert_eq!(fixed_lagrange.logical_len, 2);

    let permutation_cosets = session
        .immutable_fr_view(&context, MetalKeyFrSource::PermutationCosets)
        .expect("resolve permutation cosets");
    assert_eq!(permutation_cosets.domain.ntt_domain, Some(extended));
    assert_eq!(permutation_cosets.batch_count, 3);

    let permutation_lagrange = session
        .immutable_fr_view(&context, MetalKeyFrSource::PermutationLagrange)
        .expect("resolve permutation Lagrange columns");
    assert_eq!(permutation_lagrange.domain.ntt_domain, Some(base));
    assert_eq!(permutation_lagrange.batch_count, 3);

    let coefficients = session
        .immutable_fr_view(&context, MetalKeyFrSource::CoefficientColumns)
        .expect("resolve coefficient columns");
    assert_eq!(coefficients.domain, FrDomainTag::coefficient(base.log_size));
    assert_eq!(coefficients.batch_count, 6);
    assert!(matches!(
        coefficients.columns(5, 2),
        Err(MetalError::ResidentState(
            "immutable key-view column range escapes its source"
        ))
    ));

    let mut resources = Vec::new();
    for (view, source) in [
        (&fixed_coset, MetalKeyFrSource::FixedCosetCompact),
        (&selectors, MetalKeyFrSource::Selectors),
        (&fixed_lagrange, MetalKeyFrSource::FixedLagrangeCompact),
        (&permutation_cosets, MetalKeyFrSource::PermutationCosets),
        (&permutation_lagrange, MetalKeyFrSource::PermutationLagrange),
        (&coefficients, MetalKeyFrSource::CoefficientColumns),
    ] {
        session
            .bind_immutable_fr_view(&context, view, source, &mut resources)
            .expect("bind exact immutable Fr source");
    }
    assert_eq!(resources.len(), 6);
    assert!(matches!(
        session.bind_immutable_fr_view(
            &context,
            &fixed_lagrange,
            MetalKeyFrSource::PermutationLagrange,
            &mut resources,
        ),
        Err(MetalError::ResidentState(_))
    ));

    let periodic = session
        .immutable_periodic_fr_view(&context, MetalKeyPeriodicFrSource::VanishingInverses)
        .expect("resolve periodic vanishing table");
    assert_eq!(periodic.period, 2);
    session
        .bind_immutable_periodic_fr_view(
            &context,
            &periodic,
            MetalKeyPeriodicFrSource::VanishingInverses,
            &mut resources,
        )
        .expect("bind periodic vanishing table");

    let coefficient_bases = session
        .immutable_g1_view(&context, MetalKeyG1Source::CoefficientBases)
        .expect("resolve coefficient-basis SRS");
    assert_eq!(coefficient_bases.basis, G1Basis::Coefficient);
    assert_eq!(coefficient_bases.logical_len, 2);
    let coefficient_tail = coefficient_bases
        .bases(1, 1)
        .expect("select a bounded coefficient-basis suffix");
    session
        .bind_immutable_g1_view(
            &context,
            &coefficient_tail,
            MetalKeyG1Source::CoefficientBases,
            &mut resources,
        )
        .expect("bind bounded coefficient-basis suffix");
    assert!(matches!(
        coefficient_bases.bases(2, 1),
        Err(MetalError::BufferBounds { .. })
    ));
    assert!(matches!(
        coefficient_bases.bases(0, 0),
        Err(MetalError::ResidentState(
            "immutable G1 view range must be nonzero"
        ))
    ));

    let mut wrong_basis = session
        .immutable_g1_view(&context, MetalKeyG1Source::CoefficientBases)
        .expect("resolve basis-confusion test view");
    wrong_basis.basis = G1Basis::Lagrange;
    assert!(matches!(
        session.bind_immutable_g1_view(
            &context,
            &wrong_basis,
            MetalKeyG1Source::CoefficientBases,
            &mut resources,
        ),
        Err(MetalError::ResidentState(_))
    ));
    assert!(matches!(
        session.bind_immutable_g1_view(
            &context,
            &coefficient_bases,
            MetalKeyG1Source::LagrangeBases,
            &mut resources,
        ),
        Err(MetalError::ResidentState(_))
    ));

    let foreign_session = MetalSession::new().expect("initialize foreign Metal runtime");
    let mut foreign_resources = Vec::new();
    assert!(matches!(
        foreign_session.bind_immutable_fr_view(
            &context,
            &fixed_coset,
            MetalKeyFrSource::FixedCosetCompact,
            &mut foreign_resources,
        ),
        Err(MetalError::ForeignKeyContext)
    ));
    assert!(matches!(
        foreign_session.bind_immutable_periodic_fr_view(
            &context,
            &periodic,
            MetalKeyPeriodicFrSource::VanishingInverses,
            &mut foreign_resources,
        ),
        Err(MetalError::ForeignKeyContext)
    ));
    assert!(matches!(
        foreign_session.bind_immutable_g1_view(
            &context,
            &coefficient_bases,
            MetalKeyG1Source::CoefficientBases,
            &mut foreign_resources,
        ),
        Err(MetalError::ForeignKeyContext)
    ));
    assert!(foreign_resources.is_empty());

    let v1 = session
        .get_or_create_key_context(OwnedMetalKeyContextInput::representative().input())
        .expect("build legacy schema-v1 context");
    assert!(matches!(
        session.immutable_fr_view(&v1, MetalKeyFrSource::FixedLagrangeCompact),
        Err(MetalError::KeyContext(reason)) if reason.contains("schema-v2")
    ));
    assert!(matches!(
        session.immutable_periodic_fr_view(
            &v1,
            MetalKeyPeriodicFrSource::VanishingInverses,
        ),
        Err(MetalError::KeyContext(reason)) if reason.contains("schema-v2")
    ));
    assert!(matches!(
        session.immutable_g1_view(&v1, MetalKeyG1Source::CoefficientBases),
        Err(MetalError::KeyContext(reason)) if reason.contains("schema-v2")
    ));
}

#[test]
fn invalid_v2_context_never_allocates_or_publishes_srs_validation() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    session.reset_activity();

    let mut bad_fingerprint = OwnedV2Input::representative();
    bad_fingerprint.prepared_key_fingerprint[0] ^= 1;
    assert!(matches!(
        session.get_or_create_key_context_v2(bad_fingerprint.input()),
        Err(MetalError::KeyContext(_))
    ));
    let cheap = session.activity();
    assert_eq!(cheap.allocations, 0);
    assert_eq!(cheap.key_cache_misses, 0);
    assert_eq!(cheap.key_context_build_failures, 0);
    assert_eq!(session.srs_subgroup_validations.load(Ordering::Relaxed), 0);

    let mut off_curve = OwnedV2Input::representative();
    off_curve.coefficient_bases[0].y = [0; 6];
    off_curve.refresh_neutral_fingerprints();
    assert!(matches!(
        session.get_or_create_key_context_v2(off_curve.input()),
        Err(MetalError::KeyContext(_))
    ));
    let first_scan = session.activity();
    assert_eq!(first_scan.allocations, 0);
    assert_eq!(first_scan.host_write_bytes, 0);
    assert_eq!(first_scan.key_cache_misses, 1);
    assert_eq!(first_scan.key_context_build_failures, 1);
    assert_eq!(session.srs_subgroup_validations.load(Ordering::Relaxed), 1);
    assert!(
        session
            .validated_srs
            .lock()
            .expect("validated-SRS cache")
            .fingerprints
            .is_empty()
    );

    assert!(matches!(
        session.get_or_create_key_context_v2(off_curve.input()),
        Err(MetalError::KeyContext(_))
    ));
    assert_eq!(session.activity().allocations, 0);
    assert_eq!(session.srs_subgroup_validations.load(Ordering::Relaxed), 2);
    assert!(
        session
            .validated_srs
            .lock()
            .expect("validated-SRS cache")
            .fingerprints
            .is_empty()
    );
}
