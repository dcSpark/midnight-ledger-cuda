//! Vendor-local preparation of immutable proving-key material for resident backends.

use std::sync::Arc;

use ff::PrimeField;
use midnight_curves::Fq;
use midnight_gpu_prover::{
    PreparedResidentProvingKey, PreparedResidentProvingKeyInput, RawFr, ResidentColumnKind,
    ResidentColumnRef, ResidentDomainManifest, ResidentDomainManifestInput,
    expression::GraphColumnRequirements,
};

use super::{
    ProvingKey,
    circuit::{Any, Expression},
    evaluation::{checked_raw_fr, compile_resident_protocol_graph_family_with_required_columns},
};
use crate::poly::{Polynomial, commitment::PolynomialCommitmentScheme};

impl<CS: PolynomialCommitmentScheme<Fq>> ProvingKey<Fq, CS> {
    /// Returns one immutable neutral preparation shared by every backend import.
    #[allow(dead_code)]
    pub(crate) fn prepared_resident_proving_key(
        &self,
    ) -> Result<Arc<PreparedResidentProvingKey>, String> {
        self.prepared_resident_key
            .get_or_init(|| build_prepared_resident_proving_key(self).map(Arc::new))
            .clone()
    }
}

fn build_prepared_resident_proving_key<CS: PolynomialCommitmentScheme<Fq>>(
    proving_key: &ProvingKey<Fq, CS>,
) -> Result<PreparedResidentProvingKey, String> {
    let mut required_fixed = Vec::new();
    let mut required_advice = Vec::new();
    let mut required_instance = Vec::new();
    let mut logical_permutation_columns =
        Vec::with_capacity(proving_key.vk.cs.permutation.columns.len());

    // Preserve the constraint system's insertion order. The compact family map
    // is used only to rewrite each descriptor's index after graph compilation.
    for column in &proving_key.vk.cs.permutation.columns {
        let logical_index = abi_index("permutation column", column.index())?;
        let kind = match column.column_type() {
            Any::Fixed => {
                required_fixed.push(logical_index);
                ResidentColumnKind::Fixed
            }
            Any::Advice(_) => {
                required_advice.push(logical_index);
                ResidentColumnKind::Advice
            }
            Any::Instance => {
                required_instance.push(logical_index);
                ResidentColumnKind::Instance
            }
        };
        logical_permutation_columns.push((kind, logical_index));
    }

    // Selectors are consumed outside graph operands, so they must participate
    // in compaction even when the corresponding graph never loads the selector.
    let mut logical_trash_selectors = Vec::with_capacity(proving_key.vk.cs.trashcans.len());
    for argument in &proving_key.vk.cs.trashcans {
        let query = match argument.selector() {
            Expression::Fixed(query) => query,
            _ => return Err("resident trash selector is not a fixed-column query".into()),
        };
        if query.rotation().0 != 0 {
            return Err("resident trash selector is not queried at the current rotation".into());
        }
        let logical_index = abi_index("trash selector fixed column", query.column_index())?;
        required_fixed.push(logical_index);
        logical_trash_selectors.push(logical_index);
    }

    let graphs = compile_resident_protocol_graph_family_with_required_columns(
        &proving_key.ev,
        GraphColumnRequirements::new(&required_fixed, &required_advice, &required_instance),
    )
    .map_err(|error| error.to_string())?;
    let packed = graphs.packed();

    let permutation_columns = logical_permutation_columns
        .into_iter()
        .map(|(kind, logical_index)| {
            compact_column_index(packed, kind, logical_index)
                .map(|index| ResidentColumnRef::new(kind, index))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let trash_selector_fixed_indices = logical_trash_selectors
        .into_iter()
        .map(|logical_index| {
            compact_index("fixed", packed.fixed_compact_to_logical(), logical_index)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let domain = build_domain_manifest(proving_key)?;
    let n = domain.n() as usize;
    let n_ext = domain.n_ext() as usize;

    // Fixed arrays follow the canonical compact-map order, not logical column
    // order. The two bases deliberately use the identical map.
    let fixed_cosets_compact = flatten_compact_polynomials(
        "compact fixed cosets",
        &proving_key.fixed_cosets,
        packed.fixed_compact_to_logical(),
        n_ext,
    )?;
    let fixed_lagrange_compact = flatten_compact_polynomials(
        "compact fixed Lagrange columns",
        &proving_key.fixed_values,
        packed.fixed_compact_to_logical(),
        n,
    )?;

    // Permutation arrays retain permutation insertion order exactly.
    let permutation_cosets =
        flatten_polynomials("permutation cosets", &proving_key.permutation.cosets, n_ext)?;
    let permutation_lagrange = flatten_polynomials(
        "permutation Lagrange columns",
        &proving_key.permutation.permutations,
        n,
    )?;

    // Coefficients are protocol-wide logical fixed columns first, followed by
    // the permutation coefficients in their original insertion order.
    let coefficient_capacity = checked_flat_capacity(
        "coefficient columns",
        proving_key
            .fixed_polys
            .len()
            .checked_add(proving_key.permutation.polys.len())
            .ok_or_else(|| "coefficient column count overflowed usize".to_string())?,
        n,
    )?;
    let mut coefficient_columns = Vec::new();
    coefficient_columns
        .try_reserve_exact(coefficient_capacity)
        .map_err(|error| format!("failed to reserve coefficient columns: {error}"))?;
    append_polynomials(
        "logical fixed coefficient columns",
        &mut coefficient_columns,
        &proving_key.fixed_polys,
        n,
    )?;
    append_polynomials(
        "permutation coefficient columns",
        &mut coefficient_columns,
        &proving_key.permutation.polys,
        n,
    )?;

    // Selectors are always packed in this protocol order.
    let mut extended_selectors = Vec::new();
    extended_selectors
        .try_reserve_exact(checked_flat_capacity("extended selectors", 3, n_ext)?)
        .map_err(|error| format!("failed to reserve extended selectors: {error}"))?;
    append_polynomial(
        "l0 extended selector",
        &mut extended_selectors,
        &proving_key.l0,
        n_ext,
    )?;
    append_polynomial(
        "l_last extended selector",
        &mut extended_selectors,
        &proving_key.l_last,
        n_ext,
    )?;
    append_polynomial(
        "l_active extended selector",
        &mut extended_selectors,
        &proving_key.l_active_row,
        n_ext,
    )?;

    PreparedResidentProvingKey::try_new(PreparedResidentProvingKeyInput {
        domain,
        graphs,
        permutation_columns: permutation_columns.into_boxed_slice(),
        trash_selector_fixed_indices: trash_selector_fixed_indices.into_boxed_slice(),
        fixed_cosets_compact: fixed_cosets_compact.into_boxed_slice(),
        fixed_lagrange_compact: fixed_lagrange_compact.into_boxed_slice(),
        permutation_cosets: permutation_cosets.into_boxed_slice(),
        permutation_lagrange: permutation_lagrange.into_boxed_slice(),
        coefficient_columns: coefficient_columns.into_boxed_slice(),
        extended_selectors: extended_selectors.into_boxed_slice(),
    })
    .map_err(|error| error.to_string())
}

fn build_domain_manifest<CS: PolynomialCommitmentScheme<Fq>>(
    proving_key: &ProvingKey<Fq, CS>,
) -> Result<ResidentDomainManifest, String> {
    let values = proving_key.vk.domain.resident_values();
    ResidentDomainManifest::try_new(ResidentDomainManifestInput {
        k: values.k,
        extended_k: values.extended_k,
        constraint_system_degree: abi_index(
            "constraint-system degree",
            proving_key.vk.cs.degree(),
        )?,
        blinding_factors: abi_index(
            "constraint-system blinding factors",
            proving_key.vk.cs.blinding_factors(),
        )?,
        omega: raw_fr(values.omega)?,
        omega_inv: raw_fr(values.omega_inv)?,
        extended_omega: raw_fr(values.extended_omega)?,
        extended_omega_inv: raw_fr(values.extended_omega_inv)?,
        n_inv: raw_fr(values.ifft_divisor)?,
        n_ext_inv: raw_fr(values.extended_ifft_divisor)?,
        coset_generator: raw_fr(values.g_coset)?,
        inverse_coset_generator: raw_fr(values.g_coset_inv)?,
        barycentric_weight: raw_fr(values.barycentric_weight)?,
        vanishing_inverses: values
            .t_evaluations
            .iter()
            .copied()
            .map(raw_fr)
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice(),
        delta: raw_fr(Fq::DELTA)?,
    })
    .map_err(|error| error.to_string())
}

fn compact_column_index(
    packed: &midnight_gpu_prover::expression::CompiledGraphFamily,
    kind: ResidentColumnKind,
    logical_index: u32,
) -> Result<u32, String> {
    match kind {
        ResidentColumnKind::Fixed => {
            compact_index("fixed", packed.fixed_compact_to_logical(), logical_index)
        }
        ResidentColumnKind::Advice => {
            compact_index("advice", packed.advice_compact_to_logical(), logical_index)
        }
        ResidentColumnKind::Instance => compact_index(
            "instance",
            packed.instance_compact_to_logical(),
            logical_index,
        ),
    }
}

fn compact_index(kind: &str, map: &[u32], logical_index: u32) -> Result<u32, String> {
    let index = map.binary_search(&logical_index).map_err(|_| {
        format!("logical {kind} column {logical_index} is missing from the resident compact map")
    })?;
    abi_index("resident compact column", index)
}

fn flatten_compact_polynomials<B>(
    name: &str,
    polynomials: &[Polynomial<Fq, B>],
    compact_to_logical: &[u32],
    expected_len: usize,
) -> Result<Vec<RawFr>, String> {
    let capacity = checked_flat_capacity(name, compact_to_logical.len(), expected_len)?;
    let mut flattened = Vec::new();
    flattened
        .try_reserve_exact(capacity)
        .map_err(|error| format!("failed to reserve {name}: {error}"))?;
    for &logical_index in compact_to_logical {
        let logical_index = usize::try_from(logical_index)
            .map_err(|_| format!("logical {name} index does not fit usize"))?;
        let polynomial = polynomials.get(logical_index).ok_or_else(|| {
            format!(
                "logical {name} column {logical_index} is outside the {} available columns",
                polynomials.len()
            )
        })?;
        append_polynomial(name, &mut flattened, polynomial, expected_len)?;
    }
    Ok(flattened)
}

fn flatten_polynomials<B>(
    name: &str,
    polynomials: &[Polynomial<Fq, B>],
    expected_len: usize,
) -> Result<Vec<RawFr>, String> {
    let capacity = checked_flat_capacity(name, polynomials.len(), expected_len)?;
    let mut flattened = Vec::new();
    flattened
        .try_reserve_exact(capacity)
        .map_err(|error| format!("failed to reserve {name}: {error}"))?;
    append_polynomials(name, &mut flattened, polynomials, expected_len)?;
    Ok(flattened)
}

fn append_polynomials<B>(
    name: &str,
    flattened: &mut Vec<RawFr>,
    polynomials: &[Polynomial<Fq, B>],
    expected_len: usize,
) -> Result<(), String> {
    for polynomial in polynomials {
        append_polynomial(name, flattened, polynomial, expected_len)?;
    }
    Ok(())
}

fn append_polynomial<B>(
    name: &str,
    flattened: &mut Vec<RawFr>,
    polynomial: &Polynomial<Fq, B>,
    expected_len: usize,
) -> Result<(), String> {
    if polynomial.len() != expected_len {
        return Err(format!(
            "{name} polynomial length mismatch: expected {expected_len}, got {}",
            polynomial.len()
        ));
    }
    for value in polynomial.iter().copied() {
        flattened.push(raw_fr(value)?);
    }
    Ok(())
}

fn checked_flat_capacity(name: &str, columns: usize, stride: usize) -> Result<usize, String> {
    columns
        .checked_mul(stride)
        .ok_or_else(|| format!("{name} flattened length overflowed usize"))
}

fn abi_index(name: &str, value: usize) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{name} {value} does not fit the resident ABI"))
}

fn raw_fr(value: Fq) -> Result<RawFr, String> {
    checked_raw_fr(value).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ff::Field;
    use midnight_curves::{Bls12, Fq};

    use super::{ResidentColumnKind, raw_fr};
    use crate::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            Advice, Circuit, Column, ConstraintSystem, Constraints, Error, Fixed, ProvingKey,
            Selector, keygen_pk, keygen_vk_with_k,
        },
        poly::{Rotation, commitment::PolynomialCommitmentScheme, kzg::KZGCommitmentScheme},
        utils::SerdeFormat,
    };

    type Scheme = KZGCommitmentScheme<Bls12>;

    #[derive(Clone, Debug)]
    struct TestConfig {
        trash_advice: Column<Advice>,
        permutation_advice: Column<Advice>,
        graph_advice: Column<Advice>,
        graph_fixed: Column<Fixed>,
        permutation_fixed: Column<Fixed>,
        trash_selector: Selector,
    }

    #[derive(Clone, Debug, Default)]
    struct TestCircuit;

    impl Circuit<Fq> for TestCircuit {
        type Config = TestConfig;
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(meta: &mut ConstraintSystem<Fq>) -> Self::Config {
            let trash_advice = meta.advice_column(); // logical 0
            let _unused_advice = meta.advice_column(); // logical 1
            let permutation_advice = meta.advice_column(); // logical 2
            let graph_advice = meta.advice_column(); // logical 3

            let _unused_fixed = meta.fixed_column(); // logical 0
            let graph_fixed = meta.fixed_column(); // logical 1
            let _second_unused_fixed = meta.fixed_column(); // logical 2
            let permutation_fixed = meta.fixed_column(); // logical 3

            let _unused_instance = meta.instance_column(); // logical 0
            let graph_instance = meta.instance_column(); // logical 1

            // This insertion order is protocol-significant and intentionally
            // differs from both kind order and logical-index order.
            meta.enable_equality(permutation_advice);
            meta.enable_equality(permutation_fixed);
            meta.enable_equality(graph_instance);
            meta.enable_equality(graph_advice);

            meta.create_gate("resident graph", |meta| {
                let advice = meta.query_advice(graph_advice, Rotation::cur());
                let fixed = meta.query_fixed(graph_fixed, Rotation::cur());
                let instance = meta.query_instance(graph_instance, Rotation::cur());
                Constraints::without_selector(vec![advice + fixed - instance])
            });

            let trash_selector = meta.complex_selector();
            meta.create_gate("resident trash", |meta| {
                let advice = meta.query_advice(trash_advice, Rotation::cur());
                Constraints::with_additive_selector(
                    trash_selector,
                    vec![advice.clone().square() - advice],
                )
            });

            TestConfig {
                trash_advice,
                permutation_advice,
                graph_advice,
                graph_fixed,
                permutation_fixed,
                trash_selector,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fq>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "resident key fixture",
                |mut region| {
                    config.trash_selector.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "trash advice",
                        config.trash_advice,
                        0,
                        || Value::known(Fq::ONE),
                    )?;
                    region.assign_advice(
                        || "permutation advice",
                        config.permutation_advice,
                        0,
                        || Value::known(Fq::from(2)),
                    )?;
                    region.assign_advice(
                        || "graph advice",
                        config.graph_advice,
                        0,
                        || Value::known(Fq::from(3)),
                    )?;
                    region.assign_fixed(
                        || "graph fixed",
                        config.graph_fixed,
                        0,
                        || Value::known(Fq::from(5)),
                    )?;
                    region.assign_fixed(
                        || "permutation fixed",
                        config.permutation_fixed,
                        0,
                        || Value::known(Fq::from(7)),
                    )?;
                    Ok(())
                },
            )
        }
    }

    fn proving_key() -> ProvingKey<Fq, Scheme> {
        let parameters = Scheme::gen_params(4);
        let circuit = TestCircuit;
        let verifying_key = keygen_vk_with_k::<Fq, Scheme, _>(&parameters, &circuit, 4).unwrap();
        keygen_pk(verifying_key, &circuit).unwrap()
    }

    #[test]
    fn preparation_is_cached_and_preserves_every_protocol_order() {
        let proving_key = proving_key();
        let prepared = proving_key.prepared_resident_proving_key().unwrap();
        let cached = proving_key.prepared_resident_proving_key().unwrap();
        assert!(Arc::ptr_eq(&prepared, &cached));

        let cloned_key = proving_key.clone();
        let cloned_cached = cloned_key.prepared_resident_proving_key().unwrap();
        assert!(Arc::ptr_eq(&prepared, &cloned_cached));

        let packed = prepared.graphs().packed();
        assert_eq!(packed.fixed_compact_to_logical(), &[1, 3, 4]);
        assert_eq!(packed.advice_compact_to_logical(), &[0, 2, 3]);
        assert_eq!(packed.instance_compact_to_logical(), &[1]);

        let permutation = prepared.permutation_columns();
        assert_eq!(permutation.len(), 4);
        assert_eq!(permutation[0].kind(), ResidentColumnKind::Advice);
        assert_eq!(permutation[0].index(), 1);
        assert_eq!(permutation[1].kind(), ResidentColumnKind::Fixed);
        assert_eq!(permutation[1].index(), 1);
        assert_eq!(permutation[2].kind(), ResidentColumnKind::Instance);
        assert_eq!(permutation[2].index(), 0);
        assert_eq!(permutation[3].kind(), ResidentColumnKind::Advice);
        assert_eq!(permutation[3].index(), 2);
        assert_eq!(prepared.trash_selector_fixed_indices(), &[2]);

        let n = prepared.domain().n() as usize;
        let n_ext = prepared.domain().n_ext() as usize;
        for (compact, &logical) in packed.fixed_compact_to_logical().iter().enumerate() {
            let coset = proving_key.fixed_cosets[logical as usize]
                .iter()
                .copied()
                .map(raw_fr)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(
                &prepared.fixed_cosets_compact()[compact * n_ext..(compact + 1) * n_ext],
                coset
            );
            let lagrange = proving_key.fixed_values[logical as usize]
                .iter()
                .copied()
                .map(raw_fr)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(
                &prepared.fixed_lagrange_compact()[compact * n..(compact + 1) * n],
                lagrange
            );
        }

        for (index, polynomial) in proving_key.permutation.cosets.iter().enumerate() {
            let expected = polynomial
                .iter()
                .copied()
                .map(raw_fr)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(
                &prepared.permutation_cosets()[index * n_ext..(index + 1) * n_ext],
                expected
            );
        }
        for (index, polynomial) in proving_key.permutation.permutations.iter().enumerate() {
            let expected = polynomial
                .iter()
                .copied()
                .map(raw_fr)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(
                &prepared.permutation_lagrange()[index * n..(index + 1) * n],
                expected
            );
        }

        let expected_coefficients = proving_key
            .fixed_polys
            .iter()
            .chain(&proving_key.permutation.polys)
            .flat_map(|polynomial| polynomial.iter().copied())
            .map(raw_fr)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(prepared.coefficient_columns(), expected_coefficients);
        assert_eq!(
            prepared.fixed_coefficient_count() as usize,
            proving_key.fixed_polys.len()
        );
        assert_eq!(
            prepared.permutation_coefficient_count() as usize,
            proving_key.permutation.polys.len()
        );

        let expected_selectors = [
            &proving_key.l0,
            &proving_key.l_last,
            &proving_key.l_active_row,
        ]
        .into_iter()
        .flat_map(|polynomial| polynomial.iter().copied())
        .map(raw_fr)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
        assert_eq!(prepared.extended_selectors(), expected_selectors);
        assert_eq!(prepared.extended_selectors().len(), 3 * n_ext);
    }

    #[test]
    fn deserialized_key_starts_with_an_independent_cache() {
        let proving_key = proving_key();
        let original = proving_key.prepared_resident_proving_key().unwrap();
        let bytes = proving_key.to_bytes(SerdeFormat::RawBytesUnchecked);
        let deserialized = ProvingKey::<Fq, Scheme>::from_bytes::<TestCircuit>(
            &bytes,
            SerdeFormat::RawBytesUnchecked,
        )
        .unwrap();

        let first = deserialized.prepared_resident_proving_key().unwrap();
        let second = deserialized.prepared_resident_proving_key().unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert!(!Arc::ptr_eq(&original, &first));
        assert_eq!(original.fingerprint(), first.fingerprint());
    }
}
