//! Contains utilities for performing polynomial arithmetic over an evaluation
//! domain that is of a suitable size for the application.

use std::marker::PhantomData;

use ff::WithSmallOrderMulGroup;
use group::ff::{BatchInvert, Field};
use midnight_curves::fft::best_fft;

use super::{Coeff, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation};
use crate::utils::{arithmetic::parallelize, rational::Rational};

/// This structure contains precomputed constants and other details needed for
/// performing operations on an evaluation domain of size $2^k$ and an extended
/// domain of size $2^{k} * j$ with $j \neq 0$.
#[derive(Clone, Debug)]
pub struct EvaluationDomain<F: Field> {
    pub(crate) n: u64,
    k: u32,
    extended_k: u32,
    omega: F,
    omega_inv: F,
    extended_omega: F,
    extended_omega_inv: F,
    pub(crate) g_coset: F,
    g_coset_inv: F,
    quotient_poly_degree: u64,
    ifft_divisor: F,
    extended_ifft_divisor: F,
    t_evaluations: Vec<F>,
    barycentric_weight: F,
}

/// Exact private domain values needed to prepare a backend-neutral resident key.
#[cfg(feature = "gpu-resident")]
pub(crate) struct ResidentDomainValues<'a, F: Field> {
    pub(crate) k: u32,
    pub(crate) extended_k: u32,
    pub(crate) omega: F,
    pub(crate) omega_inv: F,
    pub(crate) extended_omega: F,
    pub(crate) extended_omega_inv: F,
    pub(crate) ifft_divisor: F,
    pub(crate) extended_ifft_divisor: F,
    pub(crate) g_coset: F,
    pub(crate) g_coset_inv: F,
    pub(crate) barycentric_weight: F,
    pub(crate) t_evaluations: &'a [F],
}

impl<F: WithSmallOrderMulGroup<3>> EvaluationDomain<F> {
    /// This constructs a new evaluation domain object based on the provided
    /// values $j, k$.
    pub fn new(j: u32, k: u32) -> Self {
        // quotient_poly_degree * params.n - 1 is the degree of the quotient polynomial
        let quotient_poly_degree = (j - 1) as u64;

        // n = 2^k
        let n = 1u64 << k;

        // We need to work within an extended domain, not params.k but params.k + i
        // for some integer i such that 2^(params.k + i) is sufficiently large to
        // describe the quotient polynomial.
        let mut extended_k = k;
        while (1 << extended_k) < (n * quotient_poly_degree) {
            extended_k += 1;
        }

        // ensure extended_k <= S
        assert!(extended_k <= F::S);

        let mut extended_omega = F::ROOT_OF_UNITY;

        // Get extended_omega, the 2^{extended_k}'th root of unity
        // The loop computes extended_omega = omega^{2 ^ (S - extended_k)}
        // Notice that extended_omega ^ {2 ^ extended_k} = omega ^ {2^S} = 1.
        for _ in extended_k..F::S {
            extended_omega = extended_omega.square();
        }
        let extended_omega = extended_omega;
        let mut extended_omega_inv = extended_omega; // Inversion computed later

        // Get omega, the 2^{k}'th root of unity (i.e. n'th root of unity)
        // The loop computes omega = extended_omega ^ {2 ^ (extended_k - k)}
        //           = (omega^{2 ^ (S - extended_k)})  ^ {2 ^ (extended_k - k)}
        //           = omega ^ {2 ^ (S - k)}.
        // Notice that omega ^ {2^k} = omega ^ {2^S} = 1.
        let mut omega = extended_omega;
        for _ in k..extended_k {
            omega = omega.square();
        }
        let omega = omega;
        let mut omega_inv = omega; // Inversion computed later

        // The coset evaluation domain is:
        // zeta {1, extended_omega, extended_omega^2, ...,
        // extended_omega^{(2^extended_k) - 1}}
        let g_coset = F::ZETA;
        let g_coset_inv = g_coset.square();

        let mut t_evaluations = Vec::with_capacity(1 << (extended_k - k));
        {
            // Compute the evaluations of t(X) = X^n - 1 in the coset evaluation domain.
            // We don't have to compute all of them, because it will repeat.
            let orig = g_coset.pow_vartime([n, 0, 0, 0]);
            let step = extended_omega.pow_vartime([n, 0, 0, 0]);
            let mut cur = orig;
            loop {
                t_evaluations.push(cur);
                cur *= &step;
                if cur == orig {
                    break;
                }
            }
            assert_eq!(t_evaluations.len(), 1 << (extended_k - k));

            // Subtract 1 from each to give us t_evaluations[i] = t(zeta * extended_omega^i)
            for coeff in &mut t_evaluations {
                *coeff -= &F::ONE;
            }

            // Invert, because we're dividing by this polynomial.
            // We invert in a batch, below.
        }

        let mut ifft_divisor = F::from(1 << k); // Inversion computed later
        let mut extended_ifft_divisor = F::from(1 << extended_k); // Inversion computed later

        // The barycentric weight of 1 over the evaluation domain
        // 1 / \prod_{i != 0} (1 - omega^i)
        let mut barycentric_weight = F::from(n); // Inversion computed later

        // Compute batch inversion
        t_evaluations
            .iter_mut()
            .chain(Some(&mut ifft_divisor))
            .chain(Some(&mut extended_ifft_divisor))
            .chain(Some(&mut barycentric_weight))
            .chain(Some(&mut extended_omega_inv))
            .chain(Some(&mut omega_inv))
            .batch_invert();

        EvaluationDomain {
            n,
            k,
            extended_k,
            omega,
            omega_inv,
            extended_omega,
            extended_omega_inv,
            g_coset,
            g_coset_inv,
            quotient_poly_degree,
            ifft_divisor,
            extended_ifft_divisor,
            t_evaluations,
            barycentric_weight,
        }
    }

    /// Borrows the exact domain constants used by resident key preparation.
    #[cfg(feature = "gpu-resident")]
    pub(crate) fn resident_values(&self) -> ResidentDomainValues<'_, F> {
        ResidentDomainValues {
            k: self.k,
            extended_k: self.extended_k,
            omega: self.omega,
            omega_inv: self.omega_inv,
            extended_omega: self.extended_omega,
            extended_omega_inv: self.extended_omega_inv,
            ifft_divisor: self.ifft_divisor,
            extended_ifft_divisor: self.extended_ifft_divisor,
            g_coset: self.g_coset,
            g_coset_inv: self.g_coset_inv,
            barycentric_weight: self.barycentric_weight,
            t_evaluations: &self.t_evaluations,
        }
    }

    /// Obtains a polynomial in Lagrange form when given a vector of Lagrange
    /// coefficients of size `n`; panics if the provided vector is the wrong
    /// length.
    pub fn lagrange_from_vec(&self, values: Vec<F>) -> Polynomial<F, LagrangeCoeff> {
        assert_eq!(values.len(), self.n as usize);

        Polynomial {
            values,
            _marker: PhantomData,
        }
    }

    /// Obtains a polynomial in coefficient form when given a vector of
    /// coefficients of size `n`; panics if the provided vector is the wrong
    /// length.
    pub fn coeff_from_vec(&self, values: Vec<F>) -> Polynomial<F, Coeff> {
        assert_eq!(values.len(), self.n as usize);

        Polynomial {
            values,
            _marker: PhantomData,
        }
    }

    /// Returns an empty (zero) polynomial in the coefficient representation
    pub fn empty_coeff(&self) -> Polynomial<F, Coeff> {
        Polynomial {
            values: vec![F::ZERO; self.n as usize],
            _marker: PhantomData,
        }
    }

    /// Returns an empty (zero) polynomial in the Lagrange coefficient
    /// representation
    pub fn empty_lagrange(&self) -> Polynomial<F, LagrangeCoeff> {
        Polynomial {
            values: vec![F::ZERO; self.n as usize],
            _marker: PhantomData,
        }
    }

    /// Returns an empty (zero) polynomial in the Lagrange coefficient
    /// representation, with deferred inversions.
    pub fn empty_lagrange_rational(&self) -> Polynomial<Rational<F>, LagrangeCoeff> {
        Polynomial {
            values: vec![F::ZERO.into(); self.n as usize],
            _marker: PhantomData,
        }
    }

    /// Returns a constant polynomial in the Lagrange coefficient representation
    pub fn constant_lagrange(&self, scalar: F) -> Polynomial<F, LagrangeCoeff> {
        Polynomial {
            values: vec![scalar; self.n as usize],
            _marker: PhantomData,
        }
    }

    /// Returns an empty (zero) polynomial in the extended Lagrange coefficient
    /// representation
    pub fn empty_extended(&self) -> Polynomial<F, ExtendedLagrangeCoeff> {
        Polynomial {
            values: vec![F::ZERO; self.extended_len()],
            _marker: PhantomData,
        }
    }

    /// Returns a constant polynomial in the extended Lagrange coefficient
    /// representation
    pub fn constant_extended(&self, scalar: F) -> Polynomial<F, ExtendedLagrangeCoeff> {
        Polynomial {
            values: vec![scalar; self.extended_len()],
            _marker: PhantomData,
        }
    }

    /// This takes us from an n-length vector into the coefficient form.
    ///
    /// This function will panic if the provided vector does not have the
    /// correct length.
    pub fn lagrange_to_coeff(&self, mut a: Polynomial<F, LagrangeCoeff>) -> Polynomial<F, Coeff> {
        assert_eq!(a.values.len(), 1 << self.k);

        // Perform inverse FFT to obtain the polynomial in coefficient form
        Self::ifft(&mut a.values, self.omega_inv, self.k, self.ifft_divisor);

        Polynomial {
            values: a.values,
            _marker: PhantomData,
        }
    }

    /// Converts one same-domain Lagrange batch to coefficient form.
    ///
    /// Metal-enabled builds submit eligible BLS12-381 batches as one inverse
    /// transform; all other fields and shapes retain the established CPU path.
    pub fn lagranges_to_coeff(
        &self,
        polynomials: Vec<Polynomial<F, LagrangeCoeff>>,
    ) -> Vec<Polynomial<F, Coeff>>
    where
        F: 'static,
    {
        #[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
        {
            let inputs = polynomials
                .iter()
                .map(|polynomial| polynomial.values.as_slice())
                .collect::<Vec<_>>();
            if let Some(values) =
                midnight_curves::gpu_metal::try_bls12_381_fr_lagrange_to_coeff_batch(
                    &inputs, self.k,
                )
            {
                return values
                    .into_iter()
                    .map(|values| Polynomial {
                        values,
                        _marker: PhantomData,
                    })
                    .collect();
            }
        }

        polynomials
            .into_iter()
            .map(|polynomial| self.lagrange_to_coeff(polynomial))
            .collect()
    }

    /// This takes us from an n-length coefficient vector into the
    /// lagrange evaluation domain.
    pub fn coeff_to_lagrange(&self, mut a: Polynomial<F, Coeff>) -> Polynomial<F, LagrangeCoeff> {
        assert_eq!(a.values.len(), 1 << self.k);

        best_fft(&mut a.values, self.omega, self.k);

        Polynomial {
            values: a.values,
            _marker: PhantomData,
        }
    }

    /// This takes us from an n-length coefficient vector into a coset of the
    /// extended evaluation domain, rotating by `rotation` if desired.
    pub fn coeff_to_extended(
        &self,
        mut a: Polynomial<F, Coeff>,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        assert_eq!(a.values.len(), 1 << self.k);

        self.distribute_powers_zeta(&mut a.values, true);
        a.values.resize(self.extended_len(), F::ZERO);
        best_fft(&mut a.values, self.extended_omega, self.extended_k);

        Polynomial {
            values: a.values,
            _marker: PhantomData,
        }
    }

    /// Converts coefficient-basis polynomials into the extended coset
    /// evaluation domain, batching the GPU path when available.
    pub fn coeffs_to_extended(
        &self,
        polynomials: &[Polynomial<F, Coeff>],
    ) -> Vec<Polynomial<F, ExtendedLagrangeCoeff>>
    where
        F: 'static,
    {
        #[cfg(feature = "gpu-blitzar")]
        {
            let inputs = polynomials
                .iter()
                .map(|polynomial| polynomial.values.as_slice())
                .collect::<Vec<_>>();
            if let Some(values) =
                midnight_curves::gpu_sppark::try_bls12_381_fr_coeff_to_extended_batch(
                    &inputs,
                    self.extended_len(),
                    self.extended_k,
                    self.g_coset,
                    self.g_coset_inv,
                )
            {
                return values
                    .into_iter()
                    .map(|values| Polynomial {
                        values,
                        _marker: PhantomData,
                    })
                    .collect();
            }
        }

        #[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
        {
            let inputs = polynomials
                .iter()
                .map(|polynomial| polynomial.values.as_slice())
                .collect::<Vec<_>>();
            if let Some(values) =
                midnight_curves::gpu_metal::try_bls12_381_fr_coeff_to_extended_batch(
                    &inputs,
                    self.extended_len(),
                    self.extended_k,
                )
            {
                return values
                    .into_iter()
                    .map(|values| Polynomial {
                        values,
                        _marker: PhantomData,
                    })
                    .collect();
            }
        }

        polynomials
            .iter()
            .cloned()
            .map(|poly| self.coeff_to_extended(poly))
            .collect()
    }

    /// This takes us from the extended evaluation domain and gets us the
    /// quotient polynomial coefficients.
    ///
    /// This function will panic if the provided vector does not have the
    /// correct length.
    pub fn extended_to_coeff(&self, mut a: Polynomial<F, ExtendedLagrangeCoeff>) -> Vec<F>
    where
        F: 'static,
    {
        assert_eq!(a.values.len(), self.extended_len());

        #[cfg(feature = "gpu-blitzar")]
        {
            let sppark_scale = F::from(1u64 << self.extended_k) * self.extended_ifft_divisor;
            if midnight_curves::gpu_sppark::try_bls12_381_fr_ntt_zeta(
                &mut a.values,
                true,
                sppark_scale,
                self.g_coset_inv,
                self.g_coset,
            ) {
                return a.values;
            }
        }

        // Inverse FFT
        Self::ifft(
            &mut a.values,
            self.extended_omega_inv,
            self.extended_k,
            self.extended_ifft_divisor,
        );

        // Distribute powers to move from coset; opposite from the
        // transformation we performed earlier.
        self.distribute_powers_zeta(&mut a.values, false);

        a.values
    }

    /// This takes us from the extended evaluation domain and gets us to the
    /// small evaluation domain (of size `n`).
    ///
    /// This function will panic if the provided vector does not have the
    /// correct length.
    pub fn extended_to_lagrange(
        &self,
        mut a: Polynomial<F, ExtendedLagrangeCoeff>,
    ) -> Polynomial<F, LagrangeCoeff> {
        assert_eq!(a.values.len(), self.extended_len());

        Self::ifft(
            &mut a.values,
            self.extended_omega_inv,
            self.extended_k,
            self.extended_ifft_divisor,
        );

        a.values.truncate(self.n as usize);
        self.distribute_powers_zeta(&mut a.values, false);

        best_fft(&mut a.values, self.omega, self.k);

        Polynomial {
            values: a.values,
            _marker: PhantomData,
        }
    }

    /// This divides the polynomial (in the extended domain) by the vanishing
    /// polynomial of the $2^k$ size domain.
    pub fn divide_by_vanishing_poly(
        &self,
        mut a: Polynomial<F, ExtendedLagrangeCoeff>,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        assert_eq!(a.values.len(), self.extended_len());

        // Divide to obtain the quotient polynomial in the coset evaluation
        // domain.
        #[cfg(feature = "gpu-blitzar")]
        if midnight_curves::gpu_sppark::try_bls12_381_fr_mul_assign_repeating(
            &mut a.values,
            &self.t_evaluations,
        ) {
            return Polynomial {
                values: a.values,
                _marker: PhantomData,
            };
        }

        parallelize(&mut a.values, |h, mut index| {
            for h in h {
                *h *= &self.t_evaluations[index % self.t_evaluations.len()];
                index += 1;
            }
        });

        Polynomial {
            values: a.values,
            _marker: PhantomData,
        }
    }

    /// Divides an extended-domain numerator by the vanishing polynomial and
    /// converts the quotient to coefficient form using one fused GPU bridge
    /// when available.
    pub fn try_divide_by_vanishing_to_coeff(
        &self,
        a: &mut Polynomial<F, ExtendedLagrangeCoeff>,
    ) -> bool
    where
        F: 'static,
    {
        assert_eq!(a.values.len(), self.extended_len());

        #[cfg(feature = "gpu-blitzar")]
        {
            let sppark_scale = F::from(1u64 << self.extended_k) * self.extended_ifft_divisor;
            midnight_curves::gpu_sppark::try_bls12_381_fr_divide_ntt_zeta(
                &mut a.values,
                &self.t_evaluations,
                sppark_scale,
                self.g_coset_inv,
                self.g_coset,
            )
        }

        #[cfg(not(feature = "gpu-blitzar"))]
        {
            false
        }
    }

    #[cfg(feature = "gpu-blitzar")]
    pub(crate) fn sppark_quotient_data(&self) -> (&[F], F, F, F) {
        let scale = F::from(1u64 << self.extended_k) * self.extended_ifft_divisor;
        (&self.t_evaluations, scale, self.g_coset_inv, self.g_coset)
    }

    /// Given a slice of group elements `[a_0, a_1, a_2, ...]`, this returns
    /// `[a_0, [zeta]a_1, [zeta^2]a_2, a_3, [zeta]a_4, [zeta^2]a_5, a_6, ...]`,
    /// where zeta is a cube root of unity in the multiplicative subgroup with
    /// order (p - 1), i.e. zeta^3 = 1.
    ///
    /// `into_coset` should be set to `true` when moving into the coset,
    /// and `false` when moving out. This toggles the choice of `zeta`.
    fn distribute_powers_zeta(&self, a: &mut [F], into_coset: bool) {
        let coset_powers = if into_coset {
            [self.g_coset, self.g_coset_inv]
        } else {
            [self.g_coset_inv, self.g_coset]
        };

        #[cfg(feature = "gpu-blitzar")]
        if midnight_curves::gpu_sppark::try_bls12_381_fr_mul_assign_zeta(
            a,
            coset_powers[0],
            coset_powers[1],
        ) {
            return;
        }

        parallelize(a, |a, mut index| {
            for a in a {
                // Distribute powers to move into/from coset
                let i = index % (coset_powers.len() + 1);
                if i != 0 {
                    *a *= &coset_powers[i - 1];
                }
                index += 1;
            }
        });
    }

    fn ifft(a: &mut [F], omega_inv: F, log_n: u32, divisor: F) {
        best_fft(a, omega_inv, log_n);
        #[cfg(feature = "gpu-blitzar")]
        if midnight_curves::gpu_sppark::try_bls12_381_fr_mul_assign(a, divisor) {
            return;
        }

        parallelize(a, |a, _| {
            for a in a {
                // Finish iFFT
                *a *= &divisor;
            }
        });
    }

    /// Get the size of the domain
    pub fn k(&self) -> u32 {
        self.k
    }

    /// Get the size of the extended domain
    pub fn extended_k(&self) -> u32 {
        self.extended_k
    }

    /// Get the size of the extended domain
    pub fn extended_len(&self) -> usize {
        1 << self.extended_k
    }

    /// Get $\omega$, the generator of the $2^k$ order multiplicative subgroup.
    pub fn get_omega(&self) -> F {
        self.omega
    }

    /// Get $\omega^{-1}$, the inverse of the generator of the $2^k$ order
    /// multiplicative subgroup.
    pub fn get_omega_inv(&self) -> F {
        self.omega_inv
    }

    /// Get the generator of the extended domain's multiplicative subgroup.
    pub fn get_extended_omega(&self) -> F {
        self.extended_omega
    }

    /// Multiplies a value by some power of $\omega$, essentially rotating over
    /// the domain.
    pub fn rotate_omega(&self, value: F, rotation: Rotation) -> F {
        let mut point = value;
        if rotation.0 >= 0 {
            point *= &self.get_omega().pow_vartime([rotation.0 as u64]);
        } else {
            point *= &self
                .get_omega_inv()
                .pow_vartime([(rotation.0 as i64).unsigned_abs()]);
        }
        point
    }

    /// Computes evaluations (at the point `x`, where `xn = x^n`) of Lagrange
    /// polynomials `l_i(X)` defined such that `l_i(omega^i) = 1` and
    /// `l_i(omega^j) = 0` for all `j != i` at each provided rotation `i`.
    ///
    /// # Implementation
    ///
    /// The polynomial
    ///     $$\prod_{j=0,j \neq i}^{n - 1} (X - \omega^j)$$
    /// has a root at all points in the domain except $\omega^i$, where it
    /// evaluates to     $$\prod_{j=0,j \neq i}^{n - 1} (\omega^i -
    /// \omega^j)$$ and so we divide that polynomial by this value to obtain
    /// $l_i(X)$. Since     $$\prod_{j=0,j \neq i}^{n - 1} (X - \omega^j)
    ///       = \frac{X^n - 1}{X - \omega^i}$$
    /// then $l_i(x)$ for some $x$ is evaluated as
    ///     $$\left(\frac{x^n - 1}{x - \omega^i}\right)
    ///       \cdot \left(\frac{1}{\prod_{j=0,j \neq i}^{n - 1} (\omega^i -
    /// \omega^j)}\right).$$ We refer to
    ///     $$1 \over \prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)$$
    /// as the barycentric weight of $\omega^i$.
    ///
    /// We know that for $i = 0$
    ///     $$\frac{1}{\prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)} =
    /// \frac{1}{n}.$$
    ///
    /// If we multiply $(1 / n)$ by $\omega^i$ then we obtain
    ///     $$\frac{1}{\prod_{j=0,j \neq 0}^{n - 1} (\omega^i - \omega^j)}
    ///       = \frac{1}{\prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)}$$
    /// which is the barycentric weight of $\omega^i$.
    pub fn l_i_range<I: IntoIterator<Item = i32> + Clone>(
        &self,
        x: F,
        xn: F,
        rotations: I,
    ) -> Vec<F> {
        let mut results;
        {
            let rotations = rotations.clone().into_iter();
            results = Vec::with_capacity(rotations.size_hint().1.unwrap_or(0));
            for rotation in rotations {
                let rotation = Rotation(rotation);
                let result = x - self.rotate_omega(F::ONE, rotation);
                results.push(result);
            }
            results.iter_mut().batch_invert();
        }

        let common = (xn - F::ONE) * self.barycentric_weight;
        for (rotation, result) in rotations.into_iter().zip(results.iter_mut()) {
            let rotation = Rotation(rotation);
            *result = self.rotate_omega(*result * common, rotation);
        }

        results
    }

    /// Gets the quotient polynomial's degree (as a multiple of n)
    pub fn get_quotient_poly_degree(&self) -> usize {
        self.quotient_poly_degree as usize
    }

    /// Obtain a pinned version of this evaluation domain; a structure with the
    /// minimal parameters needed to determine the rest of the evaluation
    /// domain.
    pub fn pinned(&self) -> PinnedEvaluationDomain<'_, F> {
        PinnedEvaluationDomain {
            k: &self.k,
            extended_k: &self.extended_k,
            omega: &self.omega,
        }
    }
}

/// Represents the minimal parameters that determine an `EvaluationDomain`.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PinnedEvaluationDomain<'a, F: Field> {
    k: &'a u32,
    extended_k: &'a u32,
    omega: &'a F,
}

#[test]
fn test_rotate() {
    use midnight_curves::Fq as Scalar;
    use rand_core::OsRng;

    use crate::utils::arithmetic::eval_polynomial;

    let domain = EvaluationDomain::<Scalar>::new(1, 3);
    let rng = OsRng;

    let mut poly = domain.empty_lagrange();
    assert_eq!(poly.len(), 8);
    for value in poly.iter_mut() {
        *value = Scalar::random(rng);
    }

    let poly_rotated_cur = poly.rotate(Rotation::cur());
    let poly_rotated_next = poly.rotate(Rotation::next());
    let poly_rotated_prev = poly.rotate(Rotation::prev());

    let poly = domain.lagrange_to_coeff(poly);
    let poly_rotated_cur = domain.lagrange_to_coeff(poly_rotated_cur);
    let poly_rotated_next = domain.lagrange_to_coeff(poly_rotated_next);
    let poly_rotated_prev = domain.lagrange_to_coeff(poly_rotated_prev);

    let x = Scalar::random(rng);

    assert_eq!(
        eval_polynomial(&poly[..], x),
        eval_polynomial(&poly_rotated_cur[..], x)
    );
    assert_eq!(
        eval_polynomial(&poly[..], x * domain.omega),
        eval_polynomial(&poly_rotated_next[..], x)
    );
    assert_eq!(
        eval_polynomial(&poly[..], x * domain.omega_inv),
        eval_polynomial(&poly_rotated_prev[..], x)
    );
}

#[test]
fn test_l_i() {
    use midnight_curves::Fq;
    use rand_core::OsRng;

    use crate::utils::arithmetic::{eval_polynomial, lagrange_interpolate};
    let domain = EvaluationDomain::<Fq>::new(1, 3);

    let mut l = vec![];
    let mut points = vec![];
    for i in 0..8 {
        points.push(domain.omega.pow([i]));
    }
    for i in 0..8 {
        let mut l_i = vec![Fq::ZERO; 8];
        l_i[i] = Fq::ONE;
        let l_i = lagrange_interpolate(&points[..], &l_i[..]);
        l.push(l_i);
    }

    let x = Fq::random(OsRng);
    let xn = x.pow([8]);

    let evaluations = domain.l_i_range(x, xn, -7..=7);
    for i in 0..8 {
        assert_eq!(eval_polynomial(&l[i][..], x), evaluations[7 + i]);
        assert_eq!(eval_polynomial(&l[(8 - i) % 8][..], x), evaluations[7 - i]);
    }
}
