use ff::Field;
#[cfg(feature = "gpu-sppark")]
use ff::PrimeField;
use group::{GroupOpsOwned, ScalarMulOwned};
#[cfg(feature = "gpu-sppark")]
use std::any::TypeId;

pub use crate::{CurveAffine, CurveExt};

/// This represents an element of a group with basic operations that can be
/// performed. This allows an FFT implementation (for example) to operate
/// generically over either a field or elliptic curve group.
pub trait FftGroup<Scalar: Field>:
    Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>
{
}

impl<T, Scalar> FftGroup<Scalar> for T
where
    Scalar: Field,
    T: Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>,
{
}

/// Performs a radix-$2$ Fast-Fourier Transformation (FFT) on a vector of size
/// $n = 2^k$, when provided `log_n` = $k$ and an element of multiplicative
/// order $n$ called `omega` ($\omega$). The result is that the vector `a`, when
/// interpreted as the coefficients of a polynomial of degree $n - 1$, is
/// transformed into the evaluations of this polynomial at each of the $n$
/// distinct powers of $\omega$. This transformation is invertible by providing
/// $\omega^{-1}$ in place of $\omega$ and dividing each resulting field element
/// by $n$.
///
/// This will use multithreading if beneficial.
pub fn best_fft<Scalar: Field + 'static, G: FftGroup<Scalar>>(
    a: &mut [G],
    omega: Scalar,
    log_n: u32,
) {
    let _timer = crate::perf_timer::ScopedTimer::with_count(
        "proof_server.prover.curves.best_fft",
        Some(a.len()),
    );

    #[cfg(feature = "gpu-sppark")]
    if try_sppark_fft(a, omega, log_n) {
        return;
    }

    fn bitreverse(mut n: usize, l: usize) -> usize {
        let mut r = 0;
        for _ in 0..l {
            r = (r << 1) | (n & 1);
            n >>= 1;
        }
        r
    }

    let threads = rayon::current_num_threads();
    let log_threads = threads.ilog2();
    let n = a.len();
    assert_eq!(n, 1 << log_n);

    for k in 0..n {
        let rk = bitreverse(k, log_n as usize);
        if k < rk {
            a.swap(rk, k);
        }
    }

    // precompute twiddle factors
    let twiddles: Vec<_> = (0..(n / 2))
        .scan(Scalar::ONE, |w, _| {
            let tw = *w;
            *w *= &omega;
            Some(tw)
        })
        .collect();

    if log_n <= log_threads {
        let mut chunk = 2_usize;
        let mut twiddle_chunk = n / 2;
        for _ in 0..log_n {
            a.chunks_mut(chunk).for_each(|coeffs| {
                let (left, right) = coeffs.split_at_mut(chunk / 2);

                // case when twiddle factor is one
                let (a, left) = left.split_at_mut(1);
                let (b, right) = right.split_at_mut(1);
                let t = b[0];
                b[0] = a[0];
                a[0] += &t;
                b[0] -= &t;

                left.iter_mut()
                    .zip(right.iter_mut())
                    .enumerate()
                    .for_each(|(i, (a, b))| {
                        let mut t = *b;
                        t *= &twiddles[(i + 1) * twiddle_chunk];
                        *b = *a;
                        *a += &t;
                        *b -= &t;
                    });
            });
            chunk *= 2;
            twiddle_chunk /= 2;
        }
    } else {
        recursive_butterfly_arithmetic(a, n, 1, &twiddles)
    }
}

#[cfg(feature = "gpu-sppark")]
#[allow(unsafe_code)]
fn try_sppark_fft<Scalar: Field + 'static, G: FftGroup<Scalar>>(
    a: &mut [G],
    omega: Scalar,
    log_n: u32,
) -> bool {
    if !sppark_fft_enabled(a.len())
        || TypeId::of::<Scalar>() != TypeId::of::<crate::Fq>()
        || TypeId::of::<G>() != TypeId::of::<crate::Fq>()
    {
        return false;
    }

    let omega = unsafe { *(&omega as *const Scalar as *const crate::Fq) };
    let mut expected_forward = crate::Fq::ROOT_OF_UNITY;
    for _ in log_n..crate::Fq::S {
        expected_forward = expected_forward.square();
    }
    let inverse = if omega == expected_forward {
        false
    } else if omega == expected_forward.invert().unwrap() {
        true
    } else {
        return false;
    };

    let values = unsafe { &mut *(a as *mut [G] as *mut [midnight_sppark_msm::Bls12_381Fr]) };
    midnight_sppark_msm::bls12_381_fr_ntt(values, inverse)
        .expect("sppark BLS12-381 scalar-field NTT failed");
    if inverse {
        let domain_size = crate::Fq::from(1u64 << log_n);
        let values = unsafe { &mut *(a as *mut [G] as *mut [crate::Fq]) };
        for value in values {
            *value *= &domain_size;
        }
    }
    true
}

#[cfg(feature = "gpu-sppark")]
fn sppark_fft_enabled(len: usize) -> bool {
    if !crate::gpu_sppark::enabled() {
        return false;
    }

    let backend_enabled = !matches!(
        std::env::var("MIDNIGHT_GPU_FFT_BACKEND")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "cpu" | "off" | "none"
    );
    if !backend_enabled {
        return false;
    }

    let min_len = std::env::var("MIDNIGHT_GPU_SPPARK_FFT_MIN_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1 << 15);
    len >= min_len
}

/// This perform recursive butterfly arithmetic
pub fn recursive_butterfly_arithmetic<Scalar: Field, G: FftGroup<Scalar>>(
    a: &mut [G],
    n: usize,
    twiddle_chunk: usize,
    twiddles: &[Scalar],
) {
    if n == 2 {
        let t = a[1];
        a[1] = a[0];
        a[0] += &t;
        a[1] -= &t;
    } else {
        let (left, right) = a.split_at_mut(n / 2);
        rayon::join(
            || recursive_butterfly_arithmetic(left, n / 2, twiddle_chunk * 2, twiddles),
            || recursive_butterfly_arithmetic(right, n / 2, twiddle_chunk * 2, twiddles),
        );

        // case when twiddle factor is one
        let (a, left) = left.split_at_mut(1);
        let (b, right) = right.split_at_mut(1);
        let t = b[0];
        b[0] = a[0];
        a[0] += &t;
        b[0] -= &t;

        left.iter_mut()
            .zip(right.iter_mut())
            .enumerate()
            .for_each(|(i, (a, b))| {
                let mut t = *b;
                t *= &twiddles[(i + 1) * twiddle_chunk];
                *b = *a;
                *a += &t;
                *b -= &t;
            });
    }
}
