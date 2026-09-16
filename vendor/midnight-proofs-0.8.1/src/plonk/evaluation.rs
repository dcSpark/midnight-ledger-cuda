use ff::{PrimeField, WithSmallOrderMulGroup};
use group::ff::Field;
#[cfg(feature = "gpu-blitzar")]
use std::{
    any::TypeId,
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use super::{ConstraintSystem, Expression};
use crate::{
    plonk::{Any, lookup, permutation, trash},
    poly::{EvaluationDomain, Polynomial, PolynomialRepresentation, Rotation},
    utils::arithmetic::parallelize,
};

#[cfg(feature = "gpu-blitzar")]
use crate::poly::ExtendedLagrangeCoeff;

#[cfg(feature = "gpu-blitzar")]
use midnight_sppark_msm::{
    Bls12_381Fr, Bls12_381FrDeviceVec, Bls12_381NumeratorContext, Bls12_381NumeratorContextInputs,
    Bls12_381NumeratorEvalInputs, Bls12_381TraceLookupInputs, Bls12_381TracePermutationInputs,
    Bls12_381TraceTrashInputs, RawCalculation, RawColumnRef, RawGraph, RawValueSource,
};

/// Return the index in the polynomial of size `isize` after rotation `rot`.
#[inline(always)]
pub(crate) fn get_rotation_idx(idx: usize, rot: i32, rot_scale: i32, isize: i32) -> usize {
    (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
}

/// Value used in a calculation
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd)]
pub enum ValueSource {
    /// This is a constant value
    Constant(usize),
    /// This is an intermediate value
    Intermediate(usize),
    /// This is a fixed column
    Fixed(usize, usize),
    /// This is an advice (witness) column
    Advice(usize, usize),
    /// This is an instance (external) column
    Instance(usize, usize),
    /// This is a challenge
    Challenge(usize),
    /// beta
    Beta(),
    /// gamma
    Gamma(),
    /// theta
    Theta(),
    /// trash challenge
    TrashChallenge(),
    /// y
    Y(),
    /// Previous value
    PreviousValue(),
}

impl Default for ValueSource {
    fn default() -> Self {
        ValueSource::Constant(0)
    }
}

impl ValueSource {
    /// Get the value for this source
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn get<F: Field, B: PolynomialRepresentation>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        trash_challenge: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        match self {
            ValueSource::Constant(idx) => constants[*idx],
            ValueSource::Intermediate(idx) => intermediates[*idx],
            ValueSource::Fixed(column_index, rotation) => {
                fixed_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Advice(column_index, rotation) => {
                advice_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Instance(column_index, rotation) => {
                instance_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Challenge(index) => challenges[*index],
            ValueSource::Beta() => *beta,
            ValueSource::Gamma() => *gamma,
            ValueSource::Theta() => *theta,
            ValueSource::TrashChallenge() => *trash_challenge,
            ValueSource::Y() => *y,
            ValueSource::PreviousValue() => *previous_value,
        }
    }
}

/// Calculation
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Calculation {
    /// This is an addition
    Add(ValueSource, ValueSource),
    /// This is a subtraction
    Sub(ValueSource, ValueSource),
    /// This is a product
    Mul(ValueSource, ValueSource),
    /// This is a square
    Square(ValueSource),
    /// This is a double
    Double(ValueSource),
    /// This is a negation
    Negate(ValueSource),
    /// This is Horner's rule: `val = a; val = val * c + b[]`
    Horner(ValueSource, Vec<ValueSource>, ValueSource),
    /// This is a simple assignment
    Store(ValueSource),
}

impl Calculation {
    /// Get the resulting value of this calculation
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<F: Field, B: PolynomialRepresentation>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        trash_challenge: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        macro_rules! get_value {
            ($value:expr) => {
                $value.get(
                    rotations,
                    constants,
                    intermediates,
                    fixed_values,
                    advice_values,
                    instance_values,
                    challenges,
                    beta,
                    gamma,
                    theta,
                    trash_challenge,
                    y,
                    previous_value,
                )
            };
        }

        match self {
            Calculation::Add(a, b) => get_value!(a) + get_value!(b),
            Calculation::Sub(a, b) => get_value!(a) - get_value!(b),
            Calculation::Mul(a, b) => get_value!(a) * get_value!(b),
            Calculation::Square(v) => get_value!(v).square(),
            Calculation::Double(v) => get_value!(v).double(),
            Calculation::Negate(v) => -get_value!(v),
            Calculation::Horner(start_value, parts, factor) => {
                let factor = get_value!(factor);
                let mut value = get_value!(start_value);
                for part in parts.iter() {
                    value = value * factor + get_value!(part);
                }
                value
            }
            Calculation::Store(v) => get_value!(v),
        }
    }
}

/// Evaluator
#[derive(Clone, Default, Debug)]
pub struct Evaluator<F: PrimeField> {
    ///  Custom gates evalution
    pub custom_gates: GraphEvaluator<F>,
    ///  Lookups evalution
    pub lookups: Vec<GraphEvaluator<F>>,
    /// Lookup input compression graphs.
    pub lookup_inputs: Vec<GraphEvaluator<F>>,
    /// Lookup table compression graphs.
    pub lookup_tables: Vec<GraphEvaluator<F>>,
    ///  Trashcans evalution
    pub trashcans: Vec<GraphEvaluator<F>>,
}

/// GraphEvaluator
#[derive(Clone, Debug)]
pub struct GraphEvaluator<F: PrimeField> {
    /// Constants
    pub constants: Vec<F>,
    /// Rotations
    pub rotations: Vec<i32>,
    /// Calculations
    pub calculations: Vec<CalculationInfo>,
    /// Number of intermediates
    pub num_intermediates: usize,
}

/// EvaluationData
#[derive(Default, Debug)]
pub struct EvaluationData<F: PrimeField> {
    /// Intermediates
    pub intermediates: Vec<F>,
    /// Rotations
    pub rotations: Vec<usize>,
}

/// CaluclationInfo
#[derive(Clone, Debug)]
pub struct CalculationInfo {
    /// Calculation
    pub calculation: Calculation,
    /// Target
    pub target: usize,
}

#[cfg(feature = "gpu-blitzar")]
struct SpparkGraphBuffers {
    constants: Vec<Bls12_381Fr>,
    rotations: Vec<i32>,
    calculations: Vec<RawCalculation>,
    horner_parts: Vec<RawValueSource>,
    num_intermediates: u32,
}

#[cfg(feature = "gpu-blitzar")]
struct IntermediateSlotPlan {
    slots: Vec<u32>,
    capacity: u32,
}

#[cfg(feature = "gpu-blitzar")]
fn visit_calculation_sources(calculation: &Calculation, mut visit: impl FnMut(&ValueSource)) {
    match calculation {
        Calculation::Add(lhs, rhs)
        | Calculation::Sub(lhs, rhs)
        | Calculation::Mul(lhs, rhs) => {
            visit(lhs);
            visit(rhs);
        }
        Calculation::Square(source)
        | Calculation::Double(source)
        | Calculation::Negate(source)
        | Calculation::Store(source) => visit(source),
        Calculation::Horner(start, parts, factor) => {
            visit(start);
            visit(factor);
            for part in parts {
                visit(part);
            }
        }
    }
}

#[cfg(feature = "gpu-blitzar")]
fn intermediate_slot_plan<F: PrimeField>(graph: &GraphEvaluator<F>) -> Option<IntermediateSlotPlan> {
    let mut remaining_uses = vec![0usize; graph.num_intermediates];
    for calculation in &graph.calculations {
        let mut valid = true;
        visit_calculation_sources(&calculation.calculation, |source| {
            if let ValueSource::Intermediate(index) = source {
                if let Some(uses) = remaining_uses.get_mut(*index) {
                    *uses = uses.checked_add(1).unwrap_or_else(|| {
                        valid = false;
                        *uses
                    });
                } else {
                    valid = false;
                }
            }
        });
        if !valid {
            return None;
        }
    }

    let mut slots = vec![None; graph.num_intermediates];
    let mut free_slots = Vec::new();
    let mut next_slot = 0u32;

    for calculation in &graph.calculations {
        let target = calculation.target;
        if target >= slots.len() || slots[target].is_some() {
            return None;
        }

        let mut valid = true;
        visit_calculation_sources(&calculation.calculation, |source| {
            let ValueSource::Intermediate(index) = source else {
                return;
            };
            let Some(Some(slot)) = slots.get(*index) else {
                valid = false;
                return;
            };
            let Some(uses) = remaining_uses.get_mut(*index) else {
                valid = false;
                return;
            };
            let Some(next_uses) = uses.checked_sub(1) else {
                valid = false;
                return;
            };
            *uses = next_uses;
            if next_uses == 0 {
                free_slots.push(*slot);
            }
        });
        if !valid {
            return None;
        }

        let slot = if let Some(slot) = free_slots.pop() {
            slot
        } else {
            let slot = next_slot;
            next_slot = next_slot.checked_add(1)?;
            slot
        };
        slots[target] = Some(slot);
        if remaining_uses[target] == 0 {
            free_slots.push(slot);
        }
    }

    Some(IntermediateSlotPlan {
        slots: slots.into_iter().collect::<Option<Vec<_>>>()?,
        capacity: next_slot,
    })
}

#[cfg(feature = "gpu-blitzar")]
impl SpparkGraphBuffers {
    fn from_graph<F: PrimeField>(graph: &GraphEvaluator<F>) -> Option<Self> {
        let slot_plan = intermediate_slot_plan(graph)?;
        if slot_plan.capacity as usize > SPPARK_MAX_GRAPH_INTERMEDIATES {
            return None;
        }

        let constants = graph
            .constants
            .iter()
            .copied()
            .map(raw_from_field)
            .collect::<Vec<_>>();
        let rotations = graph.rotations.clone();
        let mut horner_parts = Vec::new();
        let mut calculations = Vec::with_capacity(graph.calculations.len());

        for calc in &graph.calculations {
            let (op, lhs, rhs, parts_offset, parts_len) = match &calc.calculation {
                Calculation::Add(a, b) => (
                    0,
                    raw_source(a, &slot_plan.slots)?,
                    raw_source(b, &slot_plan.slots)?,
                    0,
                    0,
                ),
                Calculation::Sub(a, b) => (
                    1,
                    raw_source(a, &slot_plan.slots)?,
                    raw_source(b, &slot_plan.slots)?,
                    0,
                    0,
                ),
                Calculation::Mul(a, b) => (
                    2,
                    raw_source(a, &slot_plan.slots)?,
                    raw_source(b, &slot_plan.slots)?,
                    0,
                    0,
                ),
                Calculation::Square(a) => (
                    3,
                    raw_source(a, &slot_plan.slots)?,
                    RawValueSource::default(),
                    0,
                    0,
                ),
                Calculation::Double(a) => (
                    4,
                    raw_source(a, &slot_plan.slots)?,
                    RawValueSource::default(),
                    0,
                    0,
                ),
                Calculation::Negate(a) => (
                    5,
                    raw_source(a, &slot_plan.slots)?,
                    RawValueSource::default(),
                    0,
                    0,
                ),
                Calculation::Horner(start, parts, factor) => {
                    let parts_offset = u32::try_from(horner_parts.len()).ok()?;
                    for part in parts {
                        horner_parts.push(raw_source(part, &slot_plan.slots)?);
                    }
                    (
                        6,
                        raw_source(start, &slot_plan.slots)?,
                        raw_source(factor, &slot_plan.slots)?,
                        parts_offset,
                        u32::try_from(parts.len()).ok()?,
                    )
                }
                Calculation::Store(a) => (
                    7,
                    raw_source(a, &slot_plan.slots)?,
                    RawValueSource::default(),
                    0,
                    0,
                ),
            };
            calculations.push(RawCalculation {
                op,
                target: *slot_plan.slots.get(calc.target)?,
                parts_offset,
                parts_len,
                lhs,
                rhs,
            });
        }

        Some(Self {
            constants,
            rotations,
            calculations,
            horner_parts,
            num_intermediates: slot_plan.capacity,
        })
    }

    fn as_raw(&self) -> RawGraph {
        RawGraph {
            constants: self.constants.as_ptr(),
            constants_len: self.constants.len(),
            rotations: self.rotations.as_ptr(),
            rotations_len: self.rotations.len(),
            calculations: self.calculations.as_ptr(),
            calculations_len: self.calculations.len(),
            horner_parts: self.horner_parts.as_ptr(),
            horner_parts_len: self.horner_parts.len(),
            num_intermediates: self.num_intermediates,
        }
    }
}

#[cfg(feature = "gpu-blitzar")]
const SPPARK_MAX_GRAPH_INTERMEDIATES: usize = 256;

#[cfg(feature = "gpu-blitzar")]
pub(super) fn sppark_resident_evaluator_supported<F: PrimeField>(evaluator: &Evaluator<F>) -> bool {
    std::iter::once(&evaluator.custom_gates)
        .chain(&evaluator.lookups)
        .chain(&evaluator.lookup_inputs)
        .chain(&evaluator.lookup_tables)
        .chain(&evaluator.trashcans)
        .all(|graph| {
            intermediate_slot_plan(graph)
                .map(|plan| plan.capacity as usize <= SPPARK_MAX_GRAPH_INTERMEDIATES)
                .unwrap_or(false)
        })
}

#[cfg(feature = "gpu-blitzar")]
fn raw_source(source: &ValueSource, intermediate_slots: &[u32]) -> Option<RawValueSource> {
    let raw = match source {
        ValueSource::Constant(idx) => RawValueSource {
            kind: 0,
            a: u32::try_from(*idx).ok()?,
            b: 0,
        },
        ValueSource::Intermediate(idx) => RawValueSource {
            kind: 1,
            a: *intermediate_slots.get(*idx)?,
            b: 0,
        },
        ValueSource::Fixed(column, rotation) => RawValueSource {
            kind: 2,
            a: u32::try_from(*column).ok()?,
            b: u32::try_from(*rotation).ok()?,
        },
        ValueSource::Advice(column, rotation) => RawValueSource {
            kind: 3,
            a: u32::try_from(*column).ok()?,
            b: u32::try_from(*rotation).ok()?,
        },
        ValueSource::Instance(column, rotation) => RawValueSource {
            kind: 4,
            a: u32::try_from(*column).ok()?,
            b: u32::try_from(*rotation).ok()?,
        },
        ValueSource::Challenge(index) => RawValueSource {
            kind: 5,
            a: u32::try_from(*index).ok()?,
            b: 0,
        },
        ValueSource::Beta() => RawValueSource {
            kind: 6,
            a: 0,
            b: 0,
        },
        ValueSource::Gamma() => RawValueSource {
            kind: 7,
            a: 0,
            b: 0,
        },
        ValueSource::Theta() => RawValueSource {
            kind: 8,
            a: 0,
            b: 0,
        },
        ValueSource::TrashChallenge() => RawValueSource {
            kind: 9,
            a: 0,
            b: 0,
        },
        ValueSource::Y() => RawValueSource {
            kind: 10,
            a: 0,
            b: 0,
        },
        ValueSource::PreviousValue() => RawValueSource {
            kind: 11,
            a: 0,
            b: 0,
        },
    };
    Some(raw)
}

#[cfg(feature = "gpu-blitzar")]
#[allow(unsafe_code)]
pub(super) fn raw_from_field<F: PrimeField>(value: F) -> Bls12_381Fr {
    unsafe { std::mem::transmute_copy(&value) }
}

#[cfg(feature = "gpu-blitzar")]
#[allow(unsafe_code)]
pub(super) fn field_from_raw<F: PrimeField>(value: Bls12_381Fr) -> F {
    unsafe { std::mem::transmute_copy(&value) }
}

#[cfg(feature = "gpu-blitzar")]
fn append_polys<F: PrimeField, B>(out: &mut Vec<Bls12_381Fr>, polys: &[Polynomial<F, B>]) {
    out.reserve(polys.iter().map(|poly| poly.values.len()).sum());
    for poly in polys {
        out.extend(poly.values.iter().copied().map(raw_from_field));
    }
}

#[cfg(feature = "gpu-blitzar")]
fn raw_column_ref(column_type: &Any, index: usize) -> Option<RawColumnRef> {
    let kind = match column_type {
        Any::Fixed => 0,
        Any::Advice(_) => 1,
        Any::Instance => 2,
    };
    Some(RawColumnRef {
        kind,
        index: u32::try_from(index).ok()?,
    })
}

#[cfg(feature = "gpu-blitzar")]
fn empty_raw_graph() -> RawGraph {
    RawGraph {
        constants: std::ptr::null(),
        constants_len: 0,
        rotations: std::ptr::null(),
        rotations_len: 0,
        calculations: std::ptr::null(),
        calculations_len: 0,
        horner_parts: std::ptr::null(),
        horner_parts_len: 0,
        num_intermediates: 0,
    }
}

#[cfg(feature = "gpu-blitzar")]
#[derive(Default)]
struct SpparkColumnUsage {
    fixed: BTreeMap<usize, usize>,
    advice: BTreeMap<usize, usize>,
    instance: BTreeMap<usize, usize>,
}

#[cfg(feature = "gpu-blitzar")]
impl SpparkColumnUsage {
    fn note_raw_graph(&mut self, graph: &SpparkGraphBuffers) {
        for calculation in &graph.calculations {
            self.note_raw_source(calculation.lhs);
            self.note_raw_source(calculation.rhs);
        }
        for source in &graph.horner_parts {
            self.note_raw_source(*source);
        }
    }

    fn note_raw_source(&mut self, source: RawValueSource) {
        match source.kind {
            2 => self.note_fixed(source.a as usize),
            3 => self.note_advice(source.a as usize),
            4 => self.note_instance(source.a as usize),
            _ => {}
        }
    }

    fn note_column_ref(&mut self, column: RawColumnRef) {
        match column.kind {
            0 => self.note_fixed(column.index as usize),
            1 => self.note_advice(column.index as usize),
            2 => self.note_instance(column.index as usize),
            _ => {}
        }
    }

    fn note_fixed(&mut self, index: usize) {
        self.fixed.entry(index).or_insert(usize::MAX);
    }

    fn note_advice(&mut self, index: usize) {
        self.advice.entry(index).or_insert(usize::MAX);
    }

    fn note_instance(&mut self, index: usize) {
        self.instance.entry(index).or_insert(usize::MAX);
    }

    fn finalize(&mut self) {
        assign_compact_indices(&mut self.fixed);
        assign_compact_indices(&mut self.advice);
        assign_compact_indices(&mut self.instance);
    }

    fn remap_raw_graph(&self, graph: &mut SpparkGraphBuffers) -> Option<()> {
        for calculation in &mut graph.calculations {
            self.remap_raw_source(&mut calculation.lhs)?;
            self.remap_raw_source(&mut calculation.rhs)?;
        }
        for source in &mut graph.horner_parts {
            self.remap_raw_source(source)?;
        }
        Some(())
    }

    fn remap_raw_source(&self, source: &mut RawValueSource) -> Option<()> {
        let map = match source.kind {
            2 => &self.fixed,
            3 => &self.advice,
            4 => &self.instance,
            _ => return Some(()),
        };
        source.a = u32::try_from(*map.get(&(source.a as usize))?).ok()?;
        Some(())
    }

    fn remap_column_ref(&self, column: &mut RawColumnRef) -> Option<()> {
        let map = match column.kind {
            0 => &self.fixed,
            1 => &self.advice,
            2 => &self.instance,
            _ => return None,
        };
        column.index = u32::try_from(*map.get(&(column.index as usize))?).ok()?;
        Some(())
    }
}

#[cfg(feature = "gpu-blitzar")]
fn assign_compact_indices(map: &mut BTreeMap<usize, usize>) {
    for (compact, (_, value)) in map.iter_mut().enumerate() {
        *value = compact;
    }
}

#[cfg(feature = "gpu-blitzar")]
fn compact_polys<F: PrimeField, B>(
    polys: &[Polynomial<F, B>],
    map: &BTreeMap<usize, usize>,
    size: usize,
) -> Option<Vec<Bls12_381Fr>> {
    let mut ordered = vec![None; map.len()];
    for (original, compact) in map {
        let poly = polys.get(*original)?;
        if poly.values.len() != size {
            return None;
        }
        ordered[*compact] = Some(poly);
    }

    let mut out = Vec::with_capacity(map.len() * size);
    for poly in ordered {
        out.extend(poly?.values.iter().copied().map(raw_from_field));
    }
    Some(out)
}

#[cfg(feature = "gpu-blitzar")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_evaluate_graphs_sppark<F, B>(
    graphs: &[&GraphEvaluator<F>],
    domain: &EvaluationDomain<F>,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
    beta: F,
    gamma: F,
    theta: F,
    trash_challenge: F,
    y: F,
) -> Result<Option<Vec<Polynomial<F, B>>>, String>
where
    F: WithSmallOrderMulGroup<3> + 'static,
    B: PolynomialRepresentation,
{
    if !midnight_curves::gpu_sppark::enabled()
        || TypeId::of::<F>() != TypeId::of::<midnight_curves::Fq>()
    {
        return Ok(None);
    }
    if graphs.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let size = B::len(domain);
    let mut graph_buffers = graphs
        .iter()
        .map(|graph| SpparkGraphBuffers::from_graph(graph))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| "an expression graph is not supported by the CUDA evaluator".to_string())?;
    let mut usage = SpparkColumnUsage::default();
    for graph in &graph_buffers {
        usage.note_raw_graph(graph);
    }
    usage.finalize();
    for graph in &mut graph_buffers {
        usage
            .remap_raw_graph(graph)
            .ok_or_else(|| "failed to remap batched CUDA expression columns".to_string())?;
    }

    let fixed = compact_polys(fixed, &usage.fixed, size)
        .ok_or_else(|| "fixed expression columns do not match the domain".to_string())?;
    let advice = compact_polys(advice, &usage.advice, size)
        .ok_or_else(|| "advice expression columns do not match the domain".to_string())?;
    let instance = compact_polys(instance, &usage.instance, size)
        .ok_or_else(|| "instance expression columns do not match the domain".to_string())?;
    let challenges = challenges
        .iter()
        .copied()
        .map(raw_from_field)
        .collect::<Vec<_>>();
    let raw_graphs = graph_buffers
        .iter()
        .map(SpparkGraphBuffers::as_raw)
        .collect::<Vec<_>>();
    let mut out = vec![Bls12_381Fr::default(); size * graphs.len()];
    midnight_sppark_msm::bls12_381_eval_graph_batch(
        &mut out,
        size,
        &fixed,
        usage.fixed.len(),
        &advice,
        usage.advice.len(),
        &instance,
        usage.instance.len(),
        &challenges,
        raw_from_field(beta),
        raw_from_field(gamma),
        raw_from_field(theta),
        raw_from_field(trash_challenge),
        raw_from_field(y),
        1i32 << (B::k(domain) - domain.k()),
        &raw_graphs,
    )
    .map_err(|error| error.to_string())?;

    Ok(Some(
        out.chunks_exact(size)
            .map(|values| Polynomial {
                values: values.iter().copied().map(field_from_raw).collect(),
                _marker: std::marker::PhantomData,
            })
            .collect(),
    ))
}

#[cfg(feature = "gpu-blitzar")]
#[derive(Debug)]
pub(super) struct SpparkResidentNumeratorContext {
    context: Bls12_381NumeratorContext,
    fixed_coefficients: Arc<Bls12_381FrDeviceVec>,
    fixed_poly_count: usize,
    permutation_poly_count: usize,
    advice_columns: BTreeMap<usize, usize>,
    instance_columns: BTreeMap<usize, usize>,
    input_len: usize,
    extended_len: usize,
    extended_log_n: u32,
    g_coset: Bls12_381Fr,
    g_coset_inv: Bls12_381Fr,
    permutation_column_count: usize,
    lookup_count: usize,
    trash_count: usize,
}

#[cfg(feature = "gpu-blitzar")]
#[derive(Debug)]
pub(super) struct SpparkResidentQuotient<F> {
    pub(super) quotient: Bls12_381FrDeviceVec,
    pub(super) dynamic_coefficients: Bls12_381FrDeviceVec,
    pub(super) dynamic_offsets: HashMap<usize, u32>,
    pub(super) _marker: std::marker::PhantomData<F>,
}

#[cfg(feature = "gpu-blitzar")]
impl SpparkResidentNumeratorContext {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build<F: WithSmallOrderMulGroup<3> + 'static>(
        evaluator: &Evaluator<F>,
        domain: &EvaluationDomain<F>,
        cs: &ConstraintSystem<F>,
        fixed_coefficients: &[Polynomial<F, crate::poly::Coeff>],
        permutation_coefficients: &[Polynomial<F, crate::poly::Coeff>],
        fixed_lagrange: &[Polynomial<F, crate::poly::LagrangeCoeff>],
        permutation_lagrange: &[Polynomial<F, crate::poly::LagrangeCoeff>],
        fixed: &[Polynomial<F, ExtendedLagrangeCoeff>],
        l0: &Polynomial<F, ExtendedLagrangeCoeff>,
        l_last: &Polynomial<F, ExtendedLagrangeCoeff>,
        l_active_row: &Polynomial<F, ExtendedLagrangeCoeff>,
        permutation_pk_cosets: &[Polynomial<F, ExtendedLagrangeCoeff>],
    ) -> Result<Self, String> {
        if TypeId::of::<F>() != TypeId::of::<midnight_curves::Fq>() {
            return Err("resident numerator only supports the BLS12-381 scalar field".into());
        }

        let size = domain.extended_len();
        let mut custom_graph =
            SpparkGraphBuffers::from_graph(&evaluator.custom_gates).ok_or_else(|| {
                format!(
                    "custom-gate graph is not supported by the CUDA evaluator: intermediates={} calculations={}",
                    evaluator.custom_gates.num_intermediates,
                    evaluator.custom_gates.calculations.len()
                )
            })?;
        let mut lookup_graphs = evaluator
            .lookups
            .iter()
            .map(SpparkGraphBuffers::from_graph)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "lookup graph is not supported by the CUDA evaluator".to_string())?;
        let mut lookup_input_graphs = evaluator
            .lookup_inputs
            .iter()
            .map(SpparkGraphBuffers::from_graph)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                "lookup input graph is not supported by the CUDA evaluator".to_string()
            })?;
        let mut lookup_table_graphs = evaluator
            .lookup_tables
            .iter()
            .map(SpparkGraphBuffers::from_graph)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                "lookup table graph is not supported by the CUDA evaluator".to_string()
            })?;
        let mut trash_graphs = evaluator
            .trashcans
            .iter()
            .map(SpparkGraphBuffers::from_graph)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "trash graph is not supported by the CUDA evaluator".to_string())?;

        if lookup_graphs.len() != cs.lookups.len()
            || lookup_input_graphs.len() != lookup_graphs.len()
            || lookup_table_graphs.len() != lookup_graphs.len()
            || trash_graphs.len() != cs.trashcans.len()
        {
            return Err("CUDA graph count does not match the constraint system".into());
        }

        let mut permutation_columns = cs
            .permutation
            .columns
            .iter()
            .map(|column| raw_column_ref(column.column_type(), column.index()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "permutation column index does not fit the CUDA ABI".to_string())?;
        let mut trash_selector_fixed_indices = cs
            .trashcans
            .iter()
            .map(|argument| match argument.selector() {
                Expression::Fixed(query) => u32::try_from(query.column_index()).ok(),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "trash selector is not a fixed-column query".to_string())?;

        let mut column_usage = SpparkColumnUsage::default();
        column_usage.note_raw_graph(&custom_graph);
        for graph in &lookup_graphs {
            column_usage.note_raw_graph(graph);
        }
        for graph in &lookup_input_graphs {
            column_usage.note_raw_graph(graph);
        }
        for graph in &lookup_table_graphs {
            column_usage.note_raw_graph(graph);
        }
        for graph in &trash_graphs {
            column_usage.note_raw_graph(graph);
        }
        for column in &permutation_columns {
            column_usage.note_column_ref(*column);
        }
        for selector in &trash_selector_fixed_indices {
            column_usage.note_fixed(*selector as usize);
        }
        column_usage.finalize();

        column_usage
            .remap_raw_graph(&mut custom_graph)
            .ok_or_else(|| "failed to remap custom-gate columns".to_string())?;
        for graph in &mut lookup_graphs {
            column_usage
                .remap_raw_graph(graph)
                .ok_or_else(|| "failed to remap lookup columns".to_string())?;
        }
        for graph in &mut lookup_input_graphs {
            column_usage
                .remap_raw_graph(graph)
                .ok_or_else(|| "failed to remap lookup input columns".to_string())?;
        }
        for graph in &mut lookup_table_graphs {
            column_usage
                .remap_raw_graph(graph)
                .ok_or_else(|| "failed to remap lookup table columns".to_string())?;
        }
        for graph in &mut trash_graphs {
            column_usage
                .remap_raw_graph(graph)
                .ok_or_else(|| "failed to remap trash columns".to_string())?;
        }
        for column in &mut permutation_columns {
            column_usage
                .remap_column_ref(column)
                .ok_or_else(|| "failed to remap a permutation column".to_string())?;
        }
        for selector in &mut trash_selector_fixed_indices {
            *selector = u32::try_from(
                *column_usage
                    .fixed
                    .get(&(*selector as usize))
                    .ok_or_else(|| {
                        "trash selector is missing from fixed-column usage".to_string()
                    })?,
            )
            .map_err(|_| "remapped trash selector does not fit the CUDA ABI".to_string())?;
        }

        let fixed_flat = compact_polys(fixed, &column_usage.fixed, size)
            .ok_or_else(|| "fixed coset layout does not match the extended domain".to_string())?;
        let input_len = 1usize << domain.k();
        let fixed_lagrange_flat = compact_polys(fixed_lagrange, &column_usage.fixed, input_len)
            .ok_or_else(|| "fixed Lagrange layout does not match the base domain".to_string())?;
        let mut permutation_pk_flat = Vec::with_capacity(permutation_pk_cosets.len() * size);
        append_polys(&mut permutation_pk_flat, permutation_pk_cosets);
        let mut permutation_pk_lagrange_flat =
            Vec::with_capacity(permutation_lagrange.len() * input_len);
        append_polys(&mut permutation_pk_lagrange_flat, permutation_lagrange);
        let lookup_raw = lookup_graphs
            .iter()
            .map(SpparkGraphBuffers::as_raw)
            .collect::<Vec<_>>();
        let lookup_input_raw = lookup_input_graphs
            .iter()
            .map(SpparkGraphBuffers::as_raw)
            .collect::<Vec<_>>();
        let lookup_table_raw = lookup_table_graphs
            .iter()
            .map(SpparkGraphBuffers::as_raw)
            .collect::<Vec<_>>();
        let trash_raw = trash_graphs
            .iter()
            .map(SpparkGraphBuffers::as_raw)
            .collect::<Vec<_>>();
        let extended_log_n = domain.extended_k();
        let rot_scale = 1i32
            .checked_shl(extended_log_n - domain.k())
            .ok_or_else(|| "extended-domain rotation scale does not fit i32".to_string())?;
        let last_rotation = -i32::try_from(cs.blinding_factors() + 1)
            .map_err(|_| "blinding rotation does not fit i32".to_string())?;
        let chunk_len = u32::try_from(
            cs.degree()
                .checked_sub(2)
                .ok_or_else(|| "constraint-system degree is below three".to_string())?,
        )
        .map_err(|_| "permutation chunk length does not fit the CUDA ABI".to_string())?;
        let g_coset = raw_from_field(ExtendedLagrangeCoeff::g_coset(domain));
        let g_coset_inv = raw_from_field(ExtendedLagrangeCoeff::g_coset(domain).square());
        let mut fixed_coefficient_values = Vec::with_capacity(
            (fixed_coefficients.len() + permutation_coefficients.len()) * (1usize << domain.k()),
        );
        append_polys(&mut fixed_coefficient_values, fixed_coefficients);
        append_polys(&mut fixed_coefficient_values, permutation_coefficients);
        let fixed_coefficient_arena = Arc::new(
            Bls12_381FrDeviceVec::upload(&fixed_coefficient_values)
                .map_err(|error| error.to_string())?,
        );
        let context = Bls12_381NumeratorContext::new(Bls12_381NumeratorContextInputs {
            len: size,
            input_len,
            fixed: &fixed_flat,
            fixed_lagrange: &fixed_lagrange_flat,
            fixed_cols: column_usage.fixed.len(),
            l0: &l0
                .values
                .iter()
                .copied()
                .map(raw_from_field)
                .collect::<Vec<_>>(),
            l_last: &l_last
                .values
                .iter()
                .copied()
                .map(raw_from_field)
                .collect::<Vec<_>>(),
            l_active: &l_active_row
                .values
                .iter()
                .copied()
                .map(raw_from_field)
                .collect::<Vec<_>>(),
            permutation_columns: &permutation_columns,
            permutation_pk_cosets: &permutation_pk_flat,
            permutation_pk_lagrange: &permutation_pk_lagrange_flat,
            permutation_pk_cosets_len: permutation_pk_cosets.len(),
            trash_selector_fixed_indices: &trash_selector_fixed_indices,
            custom_graph: custom_graph.as_raw(),
            lookup_graphs: &lookup_raw,
            lookup_input_graphs: &lookup_input_raw,
            lookup_table_graphs: &lookup_table_raw,
            trash_graphs: &trash_raw,
            omega: raw_from_field(ExtendedLagrangeCoeff::omega(domain)),
            trace_omega: raw_from_field(domain.get_omega()),
            g_coset,
            delta_const: raw_from_field(F::DELTA),
            rot_scale,
            last_rotation,
            chunk_len,
        })
        .map_err(|error| error.to_string())?;

        Ok(Self {
            context,
            fixed_coefficients: fixed_coefficient_arena,
            fixed_poly_count: fixed_coefficients.len(),
            permutation_poly_count: permutation_coefficients.len(),
            advice_columns: column_usage.advice,
            instance_columns: column_usage.instance,
            input_len,
            extended_len: size,
            extended_log_n,
            g_coset,
            g_coset_inv,
            permutation_column_count: permutation_columns.len(),
            lookup_count: lookup_graphs.len(),
            trash_count: trash_graphs.len(),
        })
    }

    pub(super) fn trace_trash_lagrange<F: WithSmallOrderMulGroup<3> + 'static>(
        &self,
        advice: &[Polynomial<F, crate::poly::LagrangeCoeff>],
        resident_advice: Option<&Bls12_381FrDeviceVec>,
        instance: &[Polynomial<F, crate::poly::LagrangeCoeff>],
        challenges: &[F],
        trash_challenge: F,
    ) -> Result<Option<Bls12_381FrDeviceVec>, String> {
        if self.trash_count == 0 {
            return Ok(None);
        }
        let advice_cols = self.advice_columns.len();
        let Some(resident_advice) = resident_advice.filter(|resident| {
            resident.len() == advice.len().saturating_mul(self.input_len)
                && advice_cols == advice.len()
                && self
                    .advice_columns
                    .iter()
                    .all(|(&column_index, &arena_index)| column_index == arena_index)
        }) else {
            return Ok(None);
        };
        let instance_values = compact_polys(instance, &self.instance_columns, self.input_len)
            .ok_or_else(|| {
                "trace instance Lagrange layout does not match the proving key".to_string()
            })?;
        let instance_cols = self.instance_columns.len();
        let resident_instance =
            Bls12_381FrDeviceVec::upload(&instance_values).map_err(|error| error.to_string())?;
        let challenges = challenges
            .iter()
            .copied()
            .map(raw_from_field)
            .collect::<Vec<_>>();
        self.context
            .evaluate_trace_trash(Bls12_381TraceTrashInputs {
                advice: resident_advice,
                advice_cols,
                instance: &resident_instance,
                instance_cols,
                challenges: &challenges,
                trash_challenge: raw_from_field(trash_challenge),
            })
            .map(Some)
            .map_err(|error| error.to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn trace_permutation_products<F: WithSmallOrderMulGroup<3> + 'static>(
        &self,
        advice: &[Polynomial<F, crate::poly::LagrangeCoeff>],
        resident_advice: Option<&Bls12_381FrDeviceVec>,
        instance: &[Polynomial<F, crate::poly::LagrangeCoeff>],
        beta: F,
        gamma: F,
        blind_values: &[F],
        blinding_factors: usize,
    ) -> Result<Option<Bls12_381FrDeviceVec>, String> {
        let advice_cols = self.advice_columns.len();
        let Some(resident_advice) = resident_advice.filter(|resident| {
            resident.len() == advice.len().saturating_mul(self.input_len)
                && advice_cols == advice.len()
                && self
                    .advice_columns
                    .iter()
                    .all(|(&column_index, &arena_index)| column_index == arena_index)
        }) else {
            return Ok(None);
        };
        let instance_values = compact_polys(instance, &self.instance_columns, self.input_len)
            .ok_or_else(|| {
                "trace instance Lagrange layout does not match the proving key".to_string()
            })?;
        let instance_cols = self.instance_columns.len();
        let resident_instance =
            Bls12_381FrDeviceVec::upload(&instance_values).map_err(|error| error.to_string())?;
        let mut delta = F::ONE;
        let delta_powers = (0..self.permutation_column_count)
            .map(|_| {
                let current = delta;
                delta *= F::DELTA;
                raw_from_field(current)
            })
            .collect::<Vec<_>>();
        let blind_values = blind_values
            .iter()
            .copied()
            .map(raw_from_field)
            .collect::<Vec<_>>();
        self.context
            .trace_permutation_products(Bls12_381TracePermutationInputs {
                advice: resident_advice,
                advice_cols,
                instance: &resident_instance,
                instance_cols,
                beta: raw_from_field(beta),
                gamma: raw_from_field(gamma),
                delta_powers: &delta_powers,
                blind_values: &blind_values,
                blinding_factors,
            })
            .map(Some)
            .map_err(|error| error.to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn trace_lookup_permutation<F: WithSmallOrderMulGroup<3> + 'static>(
        &self,
        advice: &[Polynomial<F, crate::poly::LagrangeCoeff>],
        resident_advice: Option<&Bls12_381FrDeviceVec>,
        instance: &[Polynomial<F, crate::poly::LagrangeCoeff>],
        challenges: &[F],
        theta: F,
        usable_rows: usize,
        input_blinds: &[F],
        table_blinds: &[F],
    ) -> Result<Option<Bls12_381FrDeviceVec>, String> {
        let advice_cols = self.advice_columns.len();
        let Some(resident_advice) = resident_advice.filter(|resident| {
            resident.len() == advice.len().saturating_mul(self.input_len)
                && advice_cols == advice.len()
                && self
                    .advice_columns
                    .iter()
                    .all(|(&column_index, &arena_index)| column_index == arena_index)
        }) else {
            return Ok(None);
        };
        let instance_values = compact_polys(instance, &self.instance_columns, self.input_len)
            .ok_or_else(|| {
                "trace instance Lagrange layout does not match the proving key".to_string()
            })?;
        let instance_cols = self.instance_columns.len();
        let resident_instance =
            Bls12_381FrDeviceVec::upload(&instance_values).map_err(|error| error.to_string())?;
        let challenges = challenges
            .iter()
            .copied()
            .map(raw_from_field)
            .collect::<Vec<_>>();
        let input_blinds = input_blinds
            .iter()
            .copied()
            .map(raw_from_field)
            .collect::<Vec<_>>();
        let table_blinds = table_blinds
            .iter()
            .copied()
            .map(raw_from_field)
            .collect::<Vec<_>>();
        self.context
            .trace_lookup_permutation(Bls12_381TraceLookupInputs {
                advice: resident_advice,
                advice_cols,
                instance: &resident_instance,
                instance_cols,
                challenges: &challenges,
                theta: raw_from_field(theta),
                usable_rows,
                input_blinds: &input_blinds,
                table_blinds: &table_blinds,
            })
            .map(Some)
            .map_err(|error| error.to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn quotient_coefficients<F: WithSmallOrderMulGroup<3> + 'static>(
        &self,
        domain: &EvaluationDomain<F>,
        advice: &[Polynomial<F, crate::poly::Coeff>],
        resident_advice: Option<&Bls12_381FrDeviceVec>,
        instance: &[Polynomial<F, crate::poly::Coeff>],
        resident_instance: Option<&Bls12_381FrDeviceVec>,
        challenges: &[F],
        y: F,
        beta: F,
        gamma: F,
        theta: F,
        trash_challenge: F,
        lookups: &[lookup::prover::Committed<F>],
        trashcans: &[trash::prover::Committed<F>],
        permutation: &permutation::prover::Committed<F>,
    ) -> Result<SpparkResidentQuotient<F>, String> {
        if domain.extended_len() != self.extended_len
            || lookups.len() != self.lookup_count
            || trashcans.len() != self.trash_count
        {
            return Err("proof trace does not match the resident numerator context".into());
        }

        let advice_cols = self.advice_columns.len();
        let instance_cols = self.instance_columns.len();
        let permutation_sets = permutation.sets.len();
        let argument_residency = std::env::var("MIDNIGHT_GPU_RESIDENT_ARGUMENTS")
            .unwrap_or_else(|_| "all".to_string())
            .to_ascii_lowercase();
        let use_resident_permutation = matches!(
            argument_residency.as_str(),
            "permutation"
                | "perm"
                | "debug-permutation"
                | "sync-permutation"
                | "all"
                | "debug-all"
                | "on"
                | "true"
                | "1"
        );
        let use_resident_lookup = matches!(
            argument_residency.as_str(),
            "lookup" | "debug-lookup" | "sync-lookup" | "all" | "debug-all" | "on" | "true" | "1"
        );
        let resident_advice = resident_advice.filter(|resident| {
            resident.len() == advice.len().saturating_mul(self.input_len)
                && advice_cols == advice.len()
                && self
                    .advice_columns
                    .iter()
                    .all(|(&column_index, &arena_index)| column_index == arena_index)
        });
        let resident_instance = resident_instance.filter(|resident| {
            resident.len() == instance.len().saturating_mul(self.input_len)
                && instance_cols == instance.len()
                && self
                    .instance_columns
                    .iter()
                    .all(|(&column_index, &arena_index)| column_index == arena_index)
        });
        let resident_permutation = permutation
            .resident_coefficients
            .as_deref()
            .filter(|_| use_resident_permutation)
            .filter(|resident| resident.len() == permutation_sets.saturating_mul(self.input_len));
        let resident_lookup_arc = lookups
            .first()
            .and_then(|lookup| lookup.resident_product_coefficients.as_ref())
            .filter(|_| use_resident_lookup)
            .filter(|resident| {
                resident.len() == lookups.len().saturating_mul(self.input_len)
                    && lookups.iter().all(|lookup| {
                        lookup
                            .resident_product_coefficients
                            .as_ref()
                            .is_some_and(|candidate| Arc::ptr_eq(candidate, resident))
                    })
            });
        let resident_lookup_products = resident_lookup_arc.map(AsRef::as_ref);
        let resident_lookup_permuted_arc = lookups
            .first()
            .and_then(|lookup| lookup.resident_permuted_coefficients.as_ref())
            .filter(|_| use_resident_lookup)
            .filter(|resident| {
                resident.len() == lookups.len().saturating_mul(self.input_len) * 2
                    && lookups.iter().enumerate().all(|(index, lookup)| {
                        lookup
                            .resident_permuted_coefficients
                            .as_ref()
                            .is_some_and(|candidate| Arc::ptr_eq(candidate, resident))
                            && lookup.resident_input_index == Some(index)
                            && lookup.resident_table_index == Some(lookups.len() + index)
                    })
            });
        let resident_lookup_permuted = resident_lookup_permuted_arc.map(AsRef::as_ref);
        let resident_trash_arc = trashcans
            .first()
            .and_then(|trash| trash.resident_coefficients.as_ref())
            .filter(|resident| {
                resident.len() == trashcans.len().saturating_mul(self.input_len)
                    && trashcans.iter().enumerate().all(|(index, trash)| {
                        trash
                            .resident_coefficients
                            .as_ref()
                            .is_some_and(|candidate| Arc::ptr_eq(candidate, resident))
                            && trash.resident_index == Some(index)
                    })
            });
        let resident_trash = resident_trash_arc.map(AsRef::as_ref);

        let mut fallback_numerator_for_debug = None;
        if argument_residency.starts_with("debug") {
            let check_resident = |
                label: &str,
                resident: &Bls12_381FrDeviceVec,
                expected: Vec<Bls12_381Fr>,
            | -> Result<(), String> {
                let actual = resident.download().map_err(|error| error.to_string())?;
                if actual == expected {
                    return Ok(());
                }
                let mismatch = actual
                    .iter()
                    .zip(&expected)
                    .position(|(actual, expected)| actual != expected)
                    .unwrap_or(actual.len().min(expected.len()));
                Err(format!("resident {label} coefficient mismatch at {mismatch}"))
            };
            if let Some(resident) = resident_advice {
                check_resident(
                    "advice",
                    resident,
                    advice
                        .iter()
                        .flat_map(|poly| poly.iter().copied())
                        .map(raw_from_field)
                        .collect(),
                )?;
            }
            if let Some(resident) = resident_instance {
                check_resident(
                    "instance",
                    resident,
                    instance
                        .iter()
                        .flat_map(|poly| poly.iter().copied())
                        .map(raw_from_field)
                        .collect(),
                )?;
            }
            if let Some(resident) = resident_permutation {
                check_resident(
                    "permutation",
                    resident,
                    permutation
                        .sets
                        .iter()
                        .flat_map(|set| set.permutation_product_poly.iter().copied())
                        .map(raw_from_field)
                        .collect(),
                )?;
            }
            if let Some(resident) = resident_lookup_products {
                check_resident(
                    "lookup product",
                    resident,
                    lookups
                        .iter()
                        .flat_map(|lookup| lookup.product_poly.iter().copied())
                        .map(raw_from_field)
                        .collect(),
                )?;
            }
            if let Some(resident) = resident_lookup_permuted {
                check_resident(
                    "lookup permuted",
                    resident,
                    lookups
                        .iter()
                        .flat_map(|lookup| lookup.permuted_input_poly.iter().copied())
                        .chain(
                            lookups
                                .iter()
                                .flat_map(|lookup| lookup.permuted_table_poly.iter().copied()),
                        )
                        .map(raw_from_field)
                        .collect(),
                )?;
            }
            if let Some(resident) = resident_trash {
                check_resident(
                    "trash",
                    resident,
                    trashcans
                        .iter()
                        .flat_map(|trash| trash.trash_poly.iter().copied())
                        .map(raw_from_field)
                        .collect(),
                )?;
            }
        }

        let mut device_sources = Vec::new();
        let mut device_columns = 0usize;
        let mut advice_offset = usize::MAX;
        if let Some(resident) = resident_advice {
            advice_offset = device_columns;
            device_sources.push(resident);
            device_columns += advice_cols;
        }
        let mut instance_offset = usize::MAX;
        if let Some(resident) = resident_instance {
            instance_offset = device_columns;
            device_sources.push(resident);
            device_columns += instance_cols;
        }
        let mut permutation_offset = usize::MAX;
        if let Some(resident) = resident_permutation {
            permutation_offset = device_columns;
            device_sources.push(resident);
            device_columns += permutation_sets;
        }
        let mut lookup_product_offset = usize::MAX;
        if let Some(resident) = resident_lookup_products {
            lookup_product_offset = device_columns;
            device_sources.push(resident);
            device_columns += lookups.len();
        }
        let mut lookup_permuted_offset = usize::MAX;
        if let Some(resident) = resident_lookup_permuted {
            lookup_permuted_offset = device_columns;
            device_sources.push(resident);
            device_columns += lookups.len() * 2;
        }
        let mut trash_offset = usize::MAX;
        if let Some(resident) = resident_trash {
            trash_offset = device_columns;
            device_sources.push(resident);
            device_columns += trashcans.len();
        }

        let mut dynamic_offsets = HashMap::new();
        let mut columns = Vec::new();
        let mut host_cursor = device_columns;
        if resident_advice.is_none() {
            advice_offset = host_cursor;
            columns.extend(
                compact_polys(advice, &self.advice_columns, self.input_len).ok_or_else(|| {
                    "advice coefficient layout does not match the proving key".to_string()
                })?,
            );
            host_cursor += advice_cols;
        }
        for (&column_index, &arena_index) in &self.advice_columns {
            dynamic_offsets.insert(
                &advice[column_index] as *const Polynomial<F, crate::poly::Coeff> as usize,
                u32::try_from(advice_offset + arena_index)
                    .map_err(|_| "advice arena index does not fit u32".to_string())?,
            );
        }

        if resident_instance.is_none() {
            instance_offset = host_cursor;
            columns.extend(
                compact_polys(instance, &self.instance_columns, self.input_len).ok_or_else(
                    || "instance coefficient layout does not match the proving key".to_string(),
                )?,
            );
            host_cursor += instance_cols;
        }
        for (&column_index, &arena_index) in &self.instance_columns {
            dynamic_offsets.insert(
                &instance[column_index] as *const Polynomial<F, crate::poly::Coeff> as usize,
                u32::try_from(instance_offset + arena_index)
                    .map_err(|_| "instance arena index does not fit u32".to_string())?,
            );
        }

        if resident_permutation.is_none() {
            permutation_offset = host_cursor;
            for set in &permutation.sets {
                append_polys(
                    &mut columns,
                    std::slice::from_ref(&set.permutation_product_poly),
                );
            }
            host_cursor += permutation_sets;
        }
        for (set_index, set) in permutation.sets.iter().enumerate() {
            dynamic_offsets.insert(
                &set.permutation_product_poly as *const Polynomial<F, crate::poly::Coeff> as usize,
                u32::try_from(permutation_offset + set_index)
                    .map_err(|_| "permutation arena index does not fit u32".to_string())?,
            );
        }

        if resident_lookup_products.is_none() {
            lookup_product_offset = host_cursor;
            for lookup in lookups {
                append_polys(
                    &mut columns,
                    std::slice::from_ref(lookup.product_poly.as_ref()),
                );
            }
            host_cursor += lookups.len();
        }
        for (lookup_index, lookup) in lookups.iter().enumerate() {
            dynamic_offsets.insert(
                lookup.product_poly.as_ref() as *const Polynomial<F, crate::poly::Coeff> as usize,
                u32::try_from(lookup_product_offset + lookup_index)
                    .map_err(|_| "lookup product arena index does not fit u32".to_string())?,
            );
        }
        let (lookup_input_offset, lookup_table_offset) = if resident_lookup_permuted.is_some() {
            (
                lookup_permuted_offset,
                lookup_permuted_offset + lookups.len(),
            )
        } else {
            let input_offset = host_cursor;
            for lookup in lookups {
                append_polys(
                    &mut columns,
                    std::slice::from_ref(lookup.permuted_input_poly.as_ref()),
                );
            }
            host_cursor += lookups.len();
            let table_offset = host_cursor;
            for lookup in lookups {
                append_polys(
                    &mut columns,
                    std::slice::from_ref(lookup.permuted_table_poly.as_ref()),
                );
            }
            host_cursor += lookups.len();
            (input_offset, table_offset)
        };
        for (lookup_index, lookup) in lookups.iter().enumerate() {
            dynamic_offsets.insert(
                lookup.permuted_input_poly.as_ref() as *const Polynomial<F, crate::poly::Coeff>
                    as usize,
                u32::try_from(lookup_input_offset + lookup_index)
                    .map_err(|_| "lookup input arena index does not fit u32".to_string())?,
            );
            dynamic_offsets.insert(
                lookup.permuted_table_poly.as_ref() as *const Polynomial<F, crate::poly::Coeff>
                    as usize,
                u32::try_from(lookup_table_offset + lookup_index)
                    .map_err(|_| "lookup table arena index does not fit u32".to_string())?,
            );
        }
        if resident_trash.is_none() {
            trash_offset = host_cursor;
        }
        for (trash_index, trash) in trashcans.iter().enumerate() {
            dynamic_offsets.insert(
                &trash.trash_poly as *const Polynomial<F, crate::poly::Coeff> as usize,
                u32::try_from(trash_offset + trash_index)
                    .map_err(|_| "trash arena index does not fit u32".to_string())?,
            );
            if resident_trash.is_none() {
                append_polys(&mut columns, std::slice::from_ref(&trash.trash_poly));
            }
        }
        if resident_trash.is_none() {
            host_cursor += trashcans.len();
        }
        let mut total_columns = host_cursor;

        let host_columns = total_columns.saturating_sub(device_columns);
        if columns.len() != host_columns.saturating_mul(self.input_len) {
            return Err("dynamic polynomial arena has an inconsistent column layout".into());
        }
        if total_columns == 0 {
            columns.resize(self.input_len, Bls12_381Fr::default());
            total_columns = 1;
        }

        let coefficient_columns = crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.resident.upload_coefficients",
            total_columns,
            || Bls12_381FrDeviceVec::concat_many_host(&device_sources, &columns),
        )
        .map_err(|error| error.to_string())?;
        if argument_residency.starts_with("debug") {
            let actual = coefficient_columns
                .download()
                .map_err(|error| error.to_string())?;
            let mut expected = Vec::with_capacity(actual.len());
            for source in &device_sources {
                expected.extend(source.download().map_err(|error| error.to_string())?);
            }
            expected.extend_from_slice(&columns);
            if actual != expected {
                let mismatch = actual
                    .iter()
                    .zip(&expected)
                    .position(|(actual, expected)| actual != expected)
                    .unwrap_or(actual.len().min(expected.len()));
                return Err(format!("resident dynamic arena mismatch at {mismatch}"));
            }
        }
        let resident_columns = crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.resident.coefficients_to_cosets",
            total_columns,
            || {
                coefficient_columns.coeffs_to_extended_from_device(
                    self.input_len,
                    total_columns,
                    self.extended_log_n,
                    self.g_coset,
                    self.g_coset_inv,
                )
            },
        )
        .map_err(|error| error.to_string())?;
        let challenges = challenges
            .iter()
            .copied()
            .map(raw_from_field)
            .collect::<Vec<_>>();
        let mut numerator = crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.resident.evaluate_numerator",
            self.extended_len,
            || {
                self.context.evaluate(Bls12_381NumeratorEvalInputs {
                    columns: Some(&resident_columns),
                    total_columns,
                    advice_offset,
                    advice_cols,
                    instance_offset,
                    instance_cols,
                    challenges: &challenges,
                    beta: raw_from_field(beta),
                    gamma: raw_from_field(gamma),
                    theta: raw_from_field(theta),
                    trash_challenge: raw_from_field(trash_challenge),
                    y: raw_from_field(y),
                    permutation_offset,
                    permutation_sets,
                    lookup_product_offset,
                    lookup_input_offset,
                    lookup_table_offset,
                    trash_offset,
                })
            },
        )
        .map_err(|error| error.to_string())?;

        if argument_residency.starts_with("sync") {
            let _ = numerator.download().map_err(|error| error.to_string())?;
        }

        if argument_residency.starts_with("debug") {
            let mut fallback = compact_polys(advice, &self.advice_columns, self.input_len)
                .ok_or_else(|| {
                    "debug advice coefficient layout does not match the proving key".to_string()
                })?;
            fallback.extend(
                compact_polys(instance, &self.instance_columns, self.input_len).ok_or_else(
                    || {
                        "debug instance coefficient layout does not match the proving key"
                            .to_string()
                    },
                )?,
            );
            for set in &permutation.sets {
                append_polys(
                    &mut fallback,
                    std::slice::from_ref(&set.permutation_product_poly),
                );
            }
            for lookup in lookups {
                append_polys(
                    &mut fallback,
                    std::slice::from_ref(lookup.product_poly.as_ref()),
                );
            }
            for lookup in lookups {
                append_polys(
                    &mut fallback,
                    std::slice::from_ref(lookup.permuted_input_poly.as_ref()),
                );
            }
            for lookup in lookups {
                append_polys(
                    &mut fallback,
                    std::slice::from_ref(lookup.permuted_table_poly.as_ref()),
                );
            }
            for trash in trashcans {
                append_polys(&mut fallback, std::slice::from_ref(&trash.trash_poly));
            }
            let fallback_coefficients =
                Bls12_381FrDeviceVec::upload(&fallback).map_err(|error| error.to_string())?;
            let fallback_columns = fallback_coefficients
                .coeffs_to_extended_from_device(
                    self.input_len,
                    total_columns,
                    self.extended_log_n,
                    self.g_coset,
                    self.g_coset_inv,
                )
                .map_err(|error| error.to_string())?;
            let fallback_instance_offset = advice_cols;
            let fallback_permutation_offset = fallback_instance_offset + instance_cols;
            let fallback_lookup_product_offset = fallback_permutation_offset + permutation_sets;
            let fallback_lookup_input_offset = fallback_lookup_product_offset + lookups.len();
            let fallback_lookup_table_offset = fallback_lookup_input_offset + lookups.len();
            let fallback_trash_offset = fallback_lookup_table_offset + lookups.len();
            let fallback_numerator = self
                .context
                .evaluate(Bls12_381NumeratorEvalInputs {
                    columns: Some(&fallback_columns),
                    total_columns,
                    advice_offset: 0,
                    advice_cols,
                    instance_offset: fallback_instance_offset,
                    instance_cols,
                    challenges: &challenges,
                    beta: raw_from_field(beta),
                    gamma: raw_from_field(gamma),
                    theta: raw_from_field(theta),
                    trash_challenge: raw_from_field(trash_challenge),
                    y: raw_from_field(y),
                    permutation_offset: fallback_permutation_offset,
                    permutation_sets,
                    lookup_product_offset: fallback_lookup_product_offset,
                    lookup_input_offset: fallback_lookup_input_offset,
                    lookup_table_offset: fallback_lookup_table_offset,
                    trash_offset: fallback_trash_offset,
                })
                .map_err(|error| error.to_string())?;
            let actual = numerator.download().map_err(|error| error.to_string())?;
            let expected = fallback_numerator
                .download()
                .map_err(|error| error.to_string())?;
            if actual != expected {
                let mismatch = actual
                    .iter()
                    .zip(&expected)
                    .position(|(actual, expected)| actual != expected)
                    .unwrap_or(actual.len().min(expected.len()));
                return Err(format!("resident numerator mismatch at {mismatch}"));
            }
            fallback_numerator_for_debug = Some(fallback_numerator);
        }

        let (factors, scale, first, second) = domain.sppark_quotient_data();
        let factors = factors
            .iter()
            .copied()
            .map(raw_from_field)
            .collect::<Vec<_>>();
        crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.resident.quotient_to_coeff",
            self.extended_len,
            || {
                numerator.divide_ntt_zeta(
                    &factors,
                    raw_from_field(scale),
                    raw_from_field(first),
                    raw_from_field(second),
                )
            },
        )
        .map_err(|error| error.to_string())?;
        if let Some(mut fallback_numerator) = fallback_numerator_for_debug {
            fallback_numerator
                .divide_ntt_zeta(
                    &factors,
                    raw_from_field(scale),
                    raw_from_field(first),
                    raw_from_field(second),
                )
                .map_err(|error| error.to_string())?;
            let actual = numerator.download().map_err(|error| error.to_string())?;
            let expected = fallback_numerator
                .download()
                .map_err(|error| error.to_string())?;
            if actual != expected {
                let mismatch = actual
                    .iter()
                    .zip(&expected)
                    .position(|(actual, expected)| actual != expected)
                    .unwrap_or(actual.len().min(expected.len()));
                return Err(format!("resident quotient mismatch at {mismatch}"));
            }
        }
        Ok(SpparkResidentQuotient {
            quotient: numerator,
            dynamic_coefficients: coefficient_columns,
            dynamic_offsets,
            _marker: std::marker::PhantomData,
        })
    }

    pub(super) fn fixed_coefficients(&self) -> Arc<Bls12_381FrDeviceVec> {
        self.fixed_coefficients.clone()
    }

    pub(super) fn fixed_poly_count(&self) -> usize {
        self.fixed_poly_count
    }

    pub(super) fn permutation_poly_count(&self) -> usize {
        self.permutation_poly_count
    }
}

#[cfg(feature = "gpu-blitzar")]
fn numerator_full_backend_enabled() -> bool {
    matches!(
        std::env::var("MIDNIGHT_GPU_NUMERATOR_BACKEND")
            .unwrap_or_else(|_| "off".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on" | "gpu" | "cuda" | "sppark" | "full"
    )
}

#[cfg(feature = "gpu-blitzar")]
fn numerator_tail_backend_enabled() -> bool {
    matches!(
        std::env::var("MIDNIGHT_GPU_NUMERATOR_BACKEND")
            .unwrap_or_else(|_| "off".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "tail" | "partial" | "hybrid" | "after-custom"
    )
}

#[cfg(feature = "gpu-blitzar")]
fn numerator_lookup_tail_backend_enabled() -> bool {
    matches!(
        std::env::var("MIDNIGHT_GPU_NUMERATOR_BACKEND")
            .unwrap_or_else(|_| "off".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "lookup-tail" | "after-permutation" | "lookup-trash"
    )
}

impl<F: WithSmallOrderMulGroup<3>> Evaluator<F> {
    /// Creates a new evaluation structure
    pub fn new(cs: &ConstraintSystem<F>) -> Self {
        let mut ev = Evaluator::default();

        // Custom gates
        let mut parts = Vec::new();
        for gate in cs.gates.iter() {
            parts.extend(
                gate.polynomials()
                    .iter()
                    .map(|poly| ev.custom_gates.add_expression(poly)),
            );
        }
        ev.custom_gates.add_calculation(Calculation::Horner(
            ValueSource::PreviousValue(),
            parts,
            ValueSource::Y(),
        ));

        // Lookups
        for lookup in cs.lookups.iter() {
            let mut input_graph = GraphEvaluator::default();
            let input_parts = lookup
                .input_expressions
                .iter()
                .map(|expr| input_graph.add_expression(expr))
                .collect();
            input_graph.add_calculation(Calculation::Horner(
                ValueSource::Constant(0),
                input_parts,
                ValueSource::Theta(),
            ));
            ev.lookup_inputs.push(input_graph);

            let mut table_graph = GraphEvaluator::default();
            let table_parts = lookup
                .table_expressions
                .iter()
                .map(|expr| table_graph.add_expression(expr))
                .collect();
            table_graph.add_calculation(Calculation::Horner(
                ValueSource::Constant(0),
                table_parts,
                ValueSource::Theta(),
            ));
            ev.lookup_tables.push(table_graph);

            let mut graph = GraphEvaluator::default();

            let mut evaluate_lc = |expressions: &Vec<Expression<_>>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            // Input coset
            let compressed_input_coset = evaluate_lc(&lookup.input_expressions);
            // table coset
            let compressed_table_coset = evaluate_lc(&lookup.table_expressions);
            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            let right_gamma = graph.add_calculation(Calculation::Add(
                compressed_table_coset,
                ValueSource::Gamma(),
            ));
            let lc = graph.add_calculation(Calculation::Add(
                compressed_input_coset,
                ValueSource::Beta(),
            ));
            graph.add_calculation(Calculation::Mul(lc, right_gamma));

            ev.lookups.push(graph);
        }

        // Trashcans
        for trash in cs.trashcans.iter() {
            let mut graph = GraphEvaluator::default();

            let parts = trash
                .constraint_expressions()
                .iter()
                .map(|expr| graph.add_expression(expr))
                .collect();

            graph.add_calculation(Calculation::Horner(
                ValueSource::Constant(0),
                parts,
                ValueSource::TrashChallenge(),
            ));

            ev.trashcans.push(graph);
        }

        ev
    }

    #[cfg(feature = "gpu-blitzar")]
    #[allow(clippy::too_many_arguments)]
    fn try_evaluate_numerator_sppark<B: PolynomialRepresentation>(
        &self,
        initial_values: Option<&[F]>,
        include_permutation: bool,
        include_lookups: bool,
        include_trash: bool,
        domain: &EvaluationDomain<F>,
        cs: &ConstraintSystem<F>,
        advice: &[Polynomial<F, B>],
        instance: &[Polynomial<F, B>],
        fixed: &[Polynomial<F, B>],
        challenges: &[F],
        y: F,
        beta: F,
        gamma: F,
        theta: F,
        trash_challenge: F,
        lookups: &[lookup::prover::Committed<F>],
        trashcans: &[trash::prover::Committed<F>],
        permutation: &permutation::prover::Committed<F>,
        l0: &Polynomial<F, B>,
        l_last: &Polynomial<F, B>,
        l_active_row: &Polynomial<F, B>,
        permutation_pk_cosets: &[Polynomial<F, B>],
    ) -> Option<Polynomial<F, B>>
    where
        F: 'static,
    {
        if !midnight_curves::gpu_sppark::enabled()
            || TypeId::of::<F>() != TypeId::of::<midnight_curves::Fq>()
        {
            return None;
        }
        if initial_values.is_none() && !numerator_full_backend_enabled() {
            return None;
        }
        if !include_permutation && !include_lookups && !include_trash {
            return None;
        }

        let size = B::len(domain);
        if initial_values.is_some_and(|values| values.len() != size) {
            return None;
        }
        let rot_scale = 1 << (B::k(domain) - domain.k());
        let last_rotation = -((cs.blinding_factors() + 1) as i32);
        let chunk_len = cs.degree().checked_sub(2)?;
        let p = &cs.permutation;

        let mut custom_graph = if initial_values.is_none() {
            Some(SpparkGraphBuffers::from_graph(&self.custom_gates)?)
        } else {
            None
        };
        let mut lookup_graphs = if include_lookups {
            self.lookups
                .iter()
                .map(SpparkGraphBuffers::from_graph)
                .collect::<Option<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let mut trash_graphs = if include_trash {
            self.trashcans
                .iter()
                .map(SpparkGraphBuffers::from_graph)
                .collect::<Option<Vec<_>>>()?
        } else {
            Vec::new()
        };

        if (include_lookups && lookup_graphs.len() != lookups.len())
            || (include_trash && trash_graphs.len() != trashcans.len())
        {
            return None;
        }

        let permutation_product_cosets = if include_permutation {
            permutation
                .sets
                .iter()
                .map(|set| B::coeff_to_self(domain, set.permutation_product_poly.clone()))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let mut permutation_products = Vec::with_capacity(permutation_product_cosets.len() * size);
        append_polys(&mut permutation_products, &permutation_product_cosets);

        let mut permutation_columns = if include_permutation {
            p.columns
                .iter()
                .map(|column| raw_column_ref(column.column_type(), column.index()))
                .collect::<Option<Vec<_>>>()?
        } else {
            Vec::new()
        };

        let mut permutation_pk_flat = Vec::with_capacity(if include_permutation {
            permutation_pk_cosets.len() * size
        } else {
            0
        });
        if include_permutation {
            append_polys(&mut permutation_pk_flat, permutation_pk_cosets);
        }

        let mut lookup_product_flat = Vec::with_capacity(if include_lookups {
            lookups.len() * size
        } else {
            0
        });
        let mut lookup_input_flat = Vec::with_capacity(lookup_product_flat.capacity());
        let mut lookup_table_flat = Vec::with_capacity(lookup_product_flat.capacity());
        if include_lookups {
            for lookup in lookups {
                let product = B::coeff_to_self(domain, (*lookup.product_poly).clone());
                let input = B::coeff_to_self(domain, (*lookup.permuted_input_poly).clone());
                let table = B::coeff_to_self(domain, (*lookup.permuted_table_poly).clone());
                append_polys(&mut lookup_product_flat, &[product]);
                append_polys(&mut lookup_input_flat, &[input]);
                append_polys(&mut lookup_table_flat, &[table]);
            }
        }

        let mut trash_flat = Vec::with_capacity(if include_trash {
            trashcans.len() * size
        } else {
            0
        });
        let mut trash_selector_fixed_indices = Vec::with_capacity(trashcans.len());
        if include_trash {
            for (trash, argument) in trashcans.iter().zip(cs.trashcans.iter()) {
                let trash_poly = B::coeff_to_self(domain, trash.trash_poly.clone());
                append_polys(&mut trash_flat, &[trash_poly]);
                let selector = match argument.selector() {
                    Expression::Fixed(query) => query.column_index(),
                    _ => return None,
                };
                trash_selector_fixed_indices.push(u32::try_from(selector).ok()?);
            }
        }

        let mut column_usage = SpparkColumnUsage::default();
        if let Some(custom_graph) = custom_graph.as_ref() {
            column_usage.note_raw_graph(custom_graph);
        }
        for graph in &lookup_graphs {
            column_usage.note_raw_graph(graph);
        }
        for graph in &trash_graphs {
            column_usage.note_raw_graph(graph);
        }
        for column in &permutation_columns {
            column_usage.note_column_ref(*column);
        }
        for selector in &trash_selector_fixed_indices {
            column_usage.note_fixed(*selector as usize);
        }
        column_usage.finalize();

        if let Some(custom_graph) = custom_graph.as_mut() {
            column_usage.remap_raw_graph(custom_graph)?;
        }
        for graph in &mut lookup_graphs {
            column_usage.remap_raw_graph(graph)?;
        }
        for graph in &mut trash_graphs {
            column_usage.remap_raw_graph(graph)?;
        }
        for column in &mut permutation_columns {
            column_usage.remap_column_ref(column)?;
        }
        for selector in &mut trash_selector_fixed_indices {
            *selector = u32::try_from(*column_usage.fixed.get(&(*selector as usize))?).ok()?;
        }

        let fixed_flat = compact_polys(fixed, &column_usage.fixed, size)?;
        let advice_flat = compact_polys(advice, &column_usage.advice, size)?;
        let instance_flat = compact_polys(instance, &column_usage.instance, size)?;
        let challenges_raw = challenges
            .iter()
            .copied()
            .map(raw_from_field)
            .collect::<Vec<_>>();

        let custom_raw = custom_graph
            .as_ref()
            .map(SpparkGraphBuffers::as_raw)
            .unwrap_or_else(empty_raw_graph);
        let lookup_raw = lookup_graphs
            .iter()
            .map(SpparkGraphBuffers::as_raw)
            .collect::<Vec<_>>();
        let trash_raw = trash_graphs
            .iter()
            .map(SpparkGraphBuffers::as_raw)
            .collect::<Vec<_>>();

        let mut out = vec![Bls12_381Fr::default(); size];
        let initial_raw = initial_values.map(|values| {
            values
                .iter()
                .copied()
                .map(raw_from_field)
                .collect::<Vec<_>>()
        });
        let _timer = crate::perf_timer::ScopedTimer::with_count(
            if initial_values.is_some() && include_permutation {
                "proof_server.prover.plonk.evaluate_numerator.tail.gpu_sppark"
            } else if initial_values.is_some() {
                "proof_server.prover.plonk.evaluate_numerator.lookup_tail.gpu_sppark"
            } else {
                "proof_server.prover.plonk.evaluate_numerator.full.gpu_sppark"
            },
            Some(size),
        );
        midnight_sppark_msm::bls12_381_eval_numerator(
            &mut out,
            initial_raw.as_deref(),
            &fixed_flat,
            column_usage.fixed.len(),
            &advice_flat,
            column_usage.advice.len(),
            &instance_flat,
            column_usage.instance.len(),
            &challenges_raw,
            raw_from_field(beta),
            raw_from_field(gamma),
            raw_from_field(theta),
            raw_from_field(trash_challenge),
            raw_from_field(y),
            raw_from_field(B::omega(domain)),
            raw_from_field(beta * B::g_coset(domain)),
            raw_from_field(F::DELTA),
            rot_scale,
            last_rotation,
            u32::try_from(chunk_len).ok()?,
            &l0.values
                .iter()
                .copied()
                .map(raw_from_field)
                .collect::<Vec<_>>(),
            &l_last
                .values
                .iter()
                .copied()
                .map(raw_from_field)
                .collect::<Vec<_>>(),
            &l_active_row
                .values
                .iter()
                .copied()
                .map(raw_from_field)
                .collect::<Vec<_>>(),
            &permutation_products,
            permutation_product_cosets.len(),
            &permutation_columns,
            &permutation_pk_flat,
            if include_permutation {
                permutation_pk_cosets.len()
            } else {
                0
            },
            &lookup_product_flat,
            &lookup_input_flat,
            &lookup_table_flat,
            if include_lookups { lookups.len() } else { 0 },
            &trash_flat,
            &trash_selector_fixed_indices,
            custom_raw,
            &lookup_raw,
            &trash_raw,
        )
        .ok()?;

        Some(Polynomial {
            values: out.into_iter().map(field_from_raw).collect(),
            _marker: std::marker::PhantomData,
        })
    }

    /// Evaluate numerator polynomial `nu(X)` of the quotient polynomial
    /// `h(X) = nu(X) / (X^n-1)`
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_numerator<B: PolynomialRepresentation>(
        &self,
        domain: &EvaluationDomain<F>,
        cs: &ConstraintSystem<F>,
        advice: &[&[Polynomial<F, B>]],
        instance: &[&[Polynomial<F, B>]],
        fixed: &[Polynomial<F, B>],
        challenges: &[F],
        y: F,
        beta: F,
        gamma: F,
        theta: F,
        trash_challenge: F,
        lookups: &[Vec<lookup::prover::Committed<F>>],
        trashcans: &[Vec<trash::prover::Committed<F>>],
        permutations: &[permutation::prover::Committed<F>],
        l0: &Polynomial<F, B>,
        l_last: &Polynomial<F, B>,
        l_active_row: &Polynomial<F, B>,
        permutation_pk_cosets: &[Polynomial<F, B>],
    ) -> Polynomial<F, B> {
        let size = B::len(domain);
        let rot_scale = 1 << (B::k(domain) - domain.k());
        let omega = B::omega(domain);
        let isize = size as i32;
        let one = F::ONE;

        let p = &cs.permutation;

        #[cfg(feature = "gpu-blitzar")]
        let can_use_numerator_tail = advice.len() == 1
            && instance.len() == 1
            && lookups.len() == 1
            && trashcans.len() == 1
            && permutations.len() == 1;

        #[cfg(feature = "gpu-blitzar")]
        if can_use_numerator_tail {
            if let Some(values) = self.try_evaluate_numerator_sppark(
                None,
                true,
                true,
                true,
                domain,
                cs,
                advice[0],
                instance[0],
                fixed,
                challenges,
                y,
                beta,
                gamma,
                theta,
                trash_challenge,
                &lookups[0],
                &trashcans[0],
                &permutations[0],
                l0,
                l_last,
                l_active_row,
                permutation_pk_cosets,
            ) {
                return values;
            }
        }

        let mut values = B::empty(domain);

        // Core expression evaluations
        let num_threads = rayon::current_num_threads();
        for ((((advice, instance), lookups), trashcans), permutation) in advice
            .iter()
            .zip(instance.iter())
            .zip(lookups.iter())
            .zip(trashcans.iter())
            .zip(permutations.iter())
        {
            // Custom gates
            crate::perf_timer::timed_count(
                "proof_server.prover.plonk.evaluate_numerator.custom_gates",
                size,
                || {
                    rayon::scope(|scope| {
                        let chunk_size = size.div_ceil(num_threads);
                        for (thread_idx, values) in values.chunks_mut(chunk_size).enumerate() {
                            let start = thread_idx * chunk_size;
                            scope.spawn(move |_| {
                                let mut eval_data = self.custom_gates.instance();
                                for (i, value) in values.iter_mut().enumerate() {
                                    let idx = start + i;
                                    *value = self.custom_gates.evaluate::<B>(
                                        &mut eval_data,
                                        fixed,
                                        advice,
                                        instance,
                                        challenges,
                                        &beta,
                                        &gamma,
                                        &theta,
                                        &trash_challenge,
                                        &y,
                                        value,
                                        idx,
                                        rot_scale,
                                        isize,
                                    );
                                }
                            });
                        }
                    });
                },
            );

            #[cfg(feature = "gpu-blitzar")]
            if can_use_numerator_tail && numerator_tail_backend_enabled() {
                if let Some(values) = self.try_evaluate_numerator_sppark(
                    Some(&values),
                    true,
                    true,
                    true,
                    domain,
                    cs,
                    advice,
                    instance,
                    fixed,
                    challenges,
                    y,
                    beta,
                    gamma,
                    theta,
                    trash_challenge,
                    lookups,
                    trashcans,
                    permutation,
                    l0,
                    l_last,
                    l_active_row,
                    permutation_pk_cosets,
                ) {
                    return values;
                }
            }

            // Permutations
            let sets = &permutation.sets;
            if !sets.is_empty() {
                crate::perf_timer::timed_count(
                    "proof_server.prover.plonk.evaluate_numerator.permutation",
                    size,
                    || {
                        let blinding_factors = cs.blinding_factors();
                        let last_rotation = Rotation(-((blinding_factors + 1) as i32));
                        let chunk_len = cs.degree() - 2;
                        let delta_start = beta * &B::g_coset(domain);

                        let permutation_product_cosets: Vec<Polynomial<F, B>> = sets
                            .iter()
                            .map(|set| {
                                B::coeff_to_self(domain, set.permutation_product_poly.clone())
                            })
                            .collect();

                        let first_set_permutation_product_coset =
                            permutation_product_cosets.first().unwrap();
                        let last_set_permutation_product_coset =
                            permutation_product_cosets.last().unwrap();

                        // Permutation constraints
                        parallelize(&mut values, |values, start| {
                            let mut beta_term = omega.pow_vartime([start as u64, 0, 0, 0]);
                            for (i, value) in values.iter_mut().enumerate() {
                                let idx = start + i;
                                let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                                let r_last =
                                    get_rotation_idx(idx, last_rotation.0, rot_scale, isize);

                                // Enforce only for the first set.
                                // l_0(X) * (1 - z_0(X)) = 0
                                *value = *value * y
                                    + ((one - first_set_permutation_product_coset[idx]) * l0[idx]);
                                // Enforce only for the last set.
                                // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
                                *value = *value * y
                                    + ((last_set_permutation_product_coset[idx]
                                        * last_set_permutation_product_coset[idx]
                                        - last_set_permutation_product_coset[idx])
                                        * l_last[idx]);
                                // Except for the first set, enforce.
                                // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
                                for set_idx in 0..sets.len() {
                                    if set_idx != 0 {
                                        *value = *value * y
                                            + ((permutation_product_cosets[set_idx][idx]
                                                - permutation_product_cosets[set_idx - 1][r_last])
                                                * l0[idx]);
                                    }
                                }
                                // And for all the sets we enforce:
                                // (1 - (l_last(X) + l_blind(X))) * (
                                //   z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
                                // - z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma)
                                // )
                                let mut current_delta = delta_start * beta_term;
                                for ((permutation_product_coset, columns), cosets) in
                                    permutation_product_cosets
                                        .iter()
                                        .zip(p.columns.chunks(chunk_len))
                                        .zip(permutation_pk_cosets.chunks(chunk_len))
                                {
                                    let mut left = permutation_product_coset[r_next];
                                    for (values, permutation) in columns
                                        .iter()
                                        .map(|&column| match column.column_type() {
                                            Any::Advice(_) => &advice[column.index()],
                                            Any::Fixed => &fixed[column.index()],
                                            Any::Instance => &instance[column.index()],
                                        })
                                        .zip(cosets.iter())
                                    {
                                        left *= values[idx] + beta * permutation[idx] + gamma;
                                    }

                                    let mut right = permutation_product_coset[idx];
                                    for values in
                                        columns.iter().map(|&column| match column.column_type() {
                                            Any::Advice(_) => &advice[column.index()],
                                            Any::Fixed => &fixed[column.index()],
                                            Any::Instance => &instance[column.index()],
                                        })
                                    {
                                        right *= values[idx] + current_delta + gamma;
                                        current_delta *= &F::DELTA;
                                    }

                                    *value = *value * y + ((left - right) * l_active_row[idx]);
                                }
                                beta_term *= &omega;
                            }
                        });
                    },
                );
            }

            #[cfg(feature = "gpu-blitzar")]
            if can_use_numerator_tail && numerator_lookup_tail_backend_enabled() {
                if let Some(values) = self.try_evaluate_numerator_sppark(
                    Some(&values),
                    false,
                    true,
                    true,
                    domain,
                    cs,
                    advice,
                    instance,
                    fixed,
                    challenges,
                    y,
                    beta,
                    gamma,
                    theta,
                    trash_challenge,
                    lookups,
                    trashcans,
                    permutation,
                    l0,
                    l_last,
                    l_active_row,
                    permutation_pk_cosets,
                ) {
                    return values;
                }
            }

            // Lookups
            if !lookups.is_empty() {
                crate::perf_timer::timed_count(
                    "proof_server.prover.plonk.evaluate_numerator.lookup",
                    size * lookups.len(),
                    || {
                        // Polynomials required for each lookup. Calculated here so they only
                        // have to be kept in memory for the short time they are actually needed.
                        let lookup_cosets = lookups
                            .iter()
                            .map(|lookup| {
                                (
                                    B::coeff_to_self(domain, (*lookup.product_poly).clone()),
                                    B::coeff_to_self(domain, (*lookup.permuted_input_poly).clone()),
                                    B::coeff_to_self(domain, (*lookup.permuted_table_poly).clone()),
                                )
                            })
                            .collect::<Vec<_>>();

                        // Lookup constraints. Keep the same per-row Horner order as the original
                        // per-lookup passes, while streaming the output vector only once.
                        parallelize(&mut values, |values, start| {
                            let mut eval_data = self
                                .lookups
                                .iter()
                                .map(GraphEvaluator::instance)
                                .collect::<Vec<_>>();
                            for (i, value) in values.iter_mut().enumerate() {
                                let idx = start + i;
                                let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                                let r_prev = get_rotation_idx(idx, -1, rot_scale, isize);

                                for (
                                    n,
                                    (product_coset, permuted_input_coset, permuted_table_coset),
                                ) in lookup_cosets.iter().enumerate()
                                {
                                    let lookup_evaluator = &self.lookups[n];
                                    let table_value = lookup_evaluator.evaluate(
                                        &mut eval_data[n],
                                        fixed,
                                        advice,
                                        instance,
                                        challenges,
                                        &beta,
                                        &gamma,
                                        &theta,
                                        &trash_challenge,
                                        &y,
                                        &F::ZERO,
                                        idx,
                                        rot_scale,
                                        isize,
                                    );

                                    let a_minus_s =
                                        permuted_input_coset[idx] - permuted_table_coset[idx];
                                    // l_0(X) * (1 - z(X)) = 0
                                    *value = *value * y + ((one - product_coset[idx]) * l0[idx]);
                                    // l_last(X) * (z(X)^2 - z(X)) = 0
                                    *value = *value * y
                                        + ((product_coset[idx] * product_coset[idx]
                                            - product_coset[idx])
                                            * l_last[idx]);
                                    // (1 - (l_last(X) + l_blind(X))) * (
                                    //   z(omega X) (a'(X) + beta) (s'(X) + gamma)
                                    //   - z(X) (...) (...))
                                    // = 0
                                    *value = *value * y
                                        + ((product_coset[r_next]
                                            * (permuted_input_coset[idx] + beta)
                                            * (permuted_table_coset[idx] + gamma)
                                            - product_coset[idx] * table_value)
                                            * l_active_row[idx]);
                                    // l_0(X) * (a'(X) - s'(X)) = 0
                                    *value = *value * y + (a_minus_s * l0[idx]);
                                    // (1 - (l_last + l_blind)) *
                                    // (a'(X) - s'(X)) * (a'(X) - a'(omega^-1 X)) = 0
                                    *value = *value * y
                                        + (a_minus_s
                                            * (permuted_input_coset[idx]
                                                - permuted_input_coset[r_prev])
                                            * l_active_row[idx]);
                                }
                            }
                        });
                    },
                );
            }

            // Trashcans
            for (n, trash) in trashcans.iter().enumerate() {
                crate::perf_timer::timed_count(
                    "proof_server.prover.plonk.evaluate_numerator.trash",
                    size,
                    || {
                        // Polynomials required for this trash argument.
                        // Calculated here so these only have to be kept in memory for the short time
                        // they are actually needed.
                        let trash_poly = B::coeff_to_self(domain, trash.trash_poly.clone());

                        // Trash argument constraints.
                        parallelize(&mut values, |values, start| {
                            let trash_evaluator = &self.trashcans[n];
                            let argument = &cs.trashcans[n];
                            let mut eval_data = trash_evaluator.instance();
                            for (i, value) in values.iter_mut().enumerate() {
                                let idx = start + i;

                                let compressed_expression = trash_evaluator.evaluate(
                                    &mut eval_data,
                                    fixed,
                                    advice,
                                    instance,
                                    challenges,
                                    &beta,
                                    &gamma,
                                    &theta,
                                    &trash_challenge,
                                    &y,
                                    &F::ZERO,
                                    idx,
                                    rot_scale,
                                    isize,
                                );

                                let q = match argument.selector() {
                                    Expression::Fixed(query) => fixed[query.column_index()][idx],
                                    _ => unreachable!(),
                                };

                                // compressed_expressions - (1 - q) * trash
                                *value = *value * y
                                    + (compressed_expression - (one - q) * trash_poly[idx]);
                            }
                        });
                    },
                );
            }
        }
        values
    }
}

impl<F: PrimeField> Default for GraphEvaluator<F> {
    fn default() -> Self {
        Self {
            // Fixed positions to allow easy access
            constants: vec![F::ZERO, F::ONE, F::from(2u64)],
            rotations: Vec::new(),
            calculations: Vec::new(),
            num_intermediates: 0,
        }
    }
}

impl<F: PrimeField> GraphEvaluator<F> {
    #[cfg(feature = "gpu-blitzar")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_evaluate_sppark<B: PolynomialRepresentation>(
        &self,
        domain: &EvaluationDomain<F>,
        fixed: &[Polynomial<F, B>],
        advice: &[Polynomial<F, B>],
        instance: &[Polynomial<F, B>],
        challenges: &[F],
        beta: F,
        gamma: F,
        theta: F,
        trash_challenge: F,
        y: F,
    ) -> Result<Option<Polynomial<F, B>>, String>
    where
        F: WithSmallOrderMulGroup<3> + 'static,
    {
        if !midnight_curves::gpu_sppark::enabled()
            || TypeId::of::<F>() != TypeId::of::<midnight_curves::Fq>()
        {
            return Ok(None);
        }

        let size = B::len(domain);
        let mut graph = SpparkGraphBuffers::from_graph(self)
            .ok_or_else(|| "expression graph is not supported by the CUDA evaluator".to_string())?;
        let mut usage = SpparkColumnUsage::default();
        usage.note_raw_graph(&graph);
        usage.finalize();
        usage
            .remap_raw_graph(&mut graph)
            .ok_or_else(|| "failed to remap CUDA expression columns".to_string())?;

        let fixed = compact_polys(fixed, &usage.fixed, size)
            .ok_or_else(|| "fixed expression columns do not match the domain".to_string())?;
        let advice = compact_polys(advice, &usage.advice, size)
            .ok_or_else(|| "advice expression columns do not match the domain".to_string())?;
        let instance = compact_polys(instance, &usage.instance, size)
            .ok_or_else(|| "instance expression columns do not match the domain".to_string())?;
        let challenges = challenges
            .iter()
            .copied()
            .map(raw_from_field)
            .collect::<Vec<_>>();
        let mut out = vec![Bls12_381Fr::default(); size];
        midnight_sppark_msm::bls12_381_eval_numerator(
            &mut out,
            None,
            &fixed,
            usage.fixed.len(),
            &advice,
            usage.advice.len(),
            &instance,
            usage.instance.len(),
            &challenges,
            raw_from_field(beta),
            raw_from_field(gamma),
            raw_from_field(theta),
            raw_from_field(trash_challenge),
            raw_from_field(y),
            raw_from_field(B::omega(domain)),
            raw_from_field(F::ONE),
            raw_from_field(F::DELTA),
            1i32 << (B::k(domain) - domain.k()),
            0,
            1,
            &[],
            &[],
            &[],
            &[],
            0,
            &[],
            &[],
            0,
            &[],
            &[],
            &[],
            0,
            &[],
            &[],
            graph.as_raw(),
            &[],
            &[],
        )
        .map_err(|error| error.to_string())?;

        Ok(Some(Polynomial {
            values: out.into_iter().map(field_from_raw).collect(),
            _marker: std::marker::PhantomData,
        }))
    }

    /// Adds a rotation
    fn add_rotation(&mut self, rotation: &Rotation) -> usize {
        let position = self.rotations.iter().position(|&c| c == rotation.0);
        match position {
            Some(pos) => pos,
            None => {
                self.rotations.push(rotation.0);
                self.rotations.len() - 1
            }
        }
    }

    /// Adds a constant
    fn add_constant(&mut self, constant: &F) -> ValueSource {
        let position = self.constants.iter().position(|&c| c == *constant);
        ValueSource::Constant(match position {
            Some(pos) => pos,
            None => {
                self.constants.push(*constant);
                self.constants.len() - 1
            }
        })
    }

    /// Adds a calculation.
    /// Currently does the simplest thing possible: just stores the
    /// resulting value so the result can be reused  when that calculation
    /// is done multiple times.
    fn add_calculation(&mut self, calculation: Calculation) -> ValueSource {
        let existing_calculation = self
            .calculations
            .iter()
            .find(|c| c.calculation == calculation);
        match existing_calculation {
            Some(existing_calculation) => ValueSource::Intermediate(existing_calculation.target),
            None => {
                let target = self.num_intermediates;
                self.calculations.push(CalculationInfo {
                    calculation,
                    target,
                });
                self.num_intermediates += 1;
                ValueSource::Intermediate(target)
            }
        }
    }

    /// Generates an optimized evaluation for the expression
    fn add_expression(&mut self, expr: &Expression<F>) -> ValueSource {
        match expr {
            Expression::Constant(scalar) => self.add_constant(scalar),
            Expression::Selector(_selector) => unreachable!(),
            Expression::Fixed(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Fixed(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Advice(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Advice(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Instance(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Instance(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Challenge(challenge) => self.add_calculation(Calculation::Store(
                ValueSource::Challenge(challenge.index()),
            )),
            Expression::Negated(a) => match **a {
                Expression::Constant(scalar) => self.add_constant(&-scalar),
                _ => {
                    let result_a = self.add_expression(a);
                    match result_a {
                        ValueSource::Constant(0) => result_a,
                        _ => self.add_calculation(Calculation::Negate(result_a)),
                    }
                }
            },
            Expression::Sum(a, b) => {
                // Undo subtraction stored as a + (-b) in expressions
                match &**b {
                    Expression::Negated(b_int) => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b_int);
                        if result_a == ValueSource::Constant(0) {
                            self.add_calculation(Calculation::Negate(result_b))
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else {
                            self.add_calculation(Calculation::Sub(result_a, result_b))
                        }
                    }
                    _ => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b);
                        if result_a == ValueSource::Constant(0) {
                            result_b
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else if result_a <= result_b {
                            self.add_calculation(Calculation::Add(result_a, result_b))
                        } else {
                            self.add_calculation(Calculation::Add(result_b, result_a))
                        }
                    }
                }
            }
            Expression::Product(a, b) => {
                let result_a = self.add_expression(a);
                let result_b = self.add_expression(b);
                if result_a == ValueSource::Constant(0) || result_b == ValueSource::Constant(0) {
                    ValueSource::Constant(0)
                } else if result_a == ValueSource::Constant(1) {
                    result_b
                } else if result_b == ValueSource::Constant(1) {
                    result_a
                } else if result_a == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_b))
                } else if result_b == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_a))
                } else if result_a == result_b {
                    self.add_calculation(Calculation::Square(result_a))
                } else if result_a <= result_b {
                    self.add_calculation(Calculation::Mul(result_a, result_b))
                } else {
                    self.add_calculation(Calculation::Mul(result_b, result_a))
                }
            }
            Expression::Scaled(a, f) => {
                if *f == F::ZERO {
                    ValueSource::Constant(0)
                } else if *f == F::ONE {
                    self.add_expression(a)
                } else {
                    let cst = self.add_constant(f);
                    let result_a = self.add_expression(a);
                    self.add_calculation(Calculation::Mul(result_a, cst))
                }
            }
        }
    }

    /// Creates a new evaluation structure
    pub fn instance(&self) -> EvaluationData<F> {
        EvaluationData {
            intermediates: vec![F::ZERO; self.num_intermediates],
            rotations: vec![0usize; self.rotations.len()],
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn evaluate<B: PolynomialRepresentation>(
        &self,
        data: &mut EvaluationData<F>,
        fixed: &[Polynomial<F, B>],
        advice: &[Polynomial<F, B>],
        instance: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        trash_challenge: &F,
        y: &F,
        previous_value: &F,
        idx: usize,
        rot_scale: i32,
        isize: i32,
    ) -> F {
        // All rotation index values
        for (rot_idx, rot) in self.rotations.iter().enumerate() {
            data.rotations[rot_idx] = get_rotation_idx(idx, *rot, rot_scale, isize);
        }

        // All calculations, with cached intermediate results
        for calc in self.calculations.iter() {
            data.intermediates[calc.target] = calc.calculation.evaluate(
                &data.rotations,
                &self.constants,
                &data.intermediates,
                fixed,
                advice,
                instance,
                challenges,
                beta,
                gamma,
                theta,
                trash_challenge,
                y,
                previous_value,
            );
        }

        // Return the result of the last calculation (if any)
        if let Some(calc) = self.calculations.last() {
            data.intermediates[calc.target]
        } else {
            F::ZERO
        }
    }
}

/// Simple evaluation of an expression
pub fn evaluate<F: Field, B: PolynomialRepresentation>(
    expression: &Expression<F>,
    size: usize,
    rot_scale: i32,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
) -> Vec<F> {
    let mut values = vec![F::ZERO; size];
    let isize = size as i32;
    parallelize(&mut values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;
            *value = expression.evaluate(
                &|scalar| scalar,
                &|_| panic!("virtual selectors are removed during optimization"),
                &|query| {
                    fixed[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    advice[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    instance[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|challenge| challenges[challenge.index()],
                &|a| -a,
                &|a, b| a + &b,
                &|a, b| a * b,
                &|a, scalar| a * scalar,
            );
        }
    });
    values
}

#[cfg(all(test, feature = "gpu-blitzar"))]
mod sppark_slot_tests {
    use super::*;

    fn graph(calculations: Vec<CalculationInfo>) -> GraphEvaluator<midnight_curves::Fq> {
        GraphEvaluator {
            constants: vec![midnight_curves::Fq::ZERO],
            rotations: Vec::new(),
            num_intermediates: calculations.len(),
            calculations,
        }
    }

    #[test]
    fn intermediate_slots_are_reused_after_last_use() {
        let mut calculations = vec![CalculationInfo {
            calculation: Calculation::Store(ValueSource::Constant(0)),
            target: 0,
        }];
        for target in 1..401 {
            calculations.push(CalculationInfo {
                calculation: Calculation::Add(
                    ValueSource::Intermediate(target - 1),
                    ValueSource::Constant(0),
                ),
                target,
            });
        }

        let graph = graph(calculations);
        let plan = intermediate_slot_plan(&graph).expect("chain graph should be valid");
        assert_eq!(plan.capacity, 1);
        assert!(SpparkGraphBuffers::from_graph(&graph).is_some());
    }

    #[test]
    fn live_fanout_values_keep_distinct_slots() {
        let graph = graph(vec![
            CalculationInfo {
                calculation: Calculation::Store(ValueSource::Constant(0)),
                target: 0,
            },
            CalculationInfo {
                calculation: Calculation::Add(
                    ValueSource::Intermediate(0),
                    ValueSource::Constant(0),
                ),
                target: 1,
            },
            CalculationInfo {
                calculation: Calculation::Add(
                    ValueSource::Intermediate(0),
                    ValueSource::Intermediate(1),
                ),
                target: 2,
            },
        ]);

        let plan = intermediate_slot_plan(&graph).expect("fanout graph should be valid");
        assert_eq!(plan.capacity, 2);
        assert_ne!(plan.slots[0], plan.slots[1]);
    }
}
