#!/usr/bin/env python3
"""Reproduce the frozen eight-coefficient BLS12-381 AMNS candidate.

This script intentionally depends only on Python plus ``fpylll==0.6.4``.  It
rebuilds the kernel lattice, pins the complete reduced basis, deterministically
selects the reduction polynomial, derives its negative inverse modulo 2^64,
and emits the arithmetic and representation certificates used by the Rust
model.
"""

from __future__ import annotations

import json
from importlib.metadata import version

from fpylll import IntegerMatrix, LLL

P = int(
    "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787"
)
N = 8
LAMBDA = -2
PHI = 1 << 64
DELTA = 10
FROZEN_GAMMA = int(
    "89413f1d151a1a6e4666ea4ed3dc47ab1cb92213acc505243052888f88ccf35f60b63def51da8deac4a8cbd2f9ae056",
    16,
)

# fpylll 0.6.4, LLL(delta=0.99, eta=0.501), using the lattice constructed in
# main().  Freezing every row makes a library parameter update auditable.
FROZEN_BASIS = [
    [
        165287713379363,
        -5030420591247,
        137672150042232,
        119775351316119,
        -20239437355587,
        63135815810068,
        -43664550814409,
        -11848715286990,
    ],
    [
        74829388475034,
        4902425732877,
        -73494590381457,
        -149295178729809,
        -30065716019627,
        -162375608766907,
        121383494388985,
        34256165897936,
    ],
    [
        -90148039660281,
        78086606608915,
        -92905410826002,
        1904264076559,
        175110925282580,
        -58671794440980,
        -134358479080627,
        -45832564069768,
    ],
    [
        -76912876176448,
        -33191872534315,
        150407466557905,
        182487051264124,
        -120341750538278,
        -20111442497217,
        -1041743850707,
        -14144723400719,
    ],
    [
        80042099451959,
        -438404363789,
        -104528439513579,
        91613899373051,
        -7504120839914,
        125847515758073,
        -143766863997100,
        -95095973594275,
    ],
    [
        -566399222159,
        -40350879852804,
        62094071959361,
        -57809274215128,
        26607722801234,
        -66047920422524,
        -72688522983329,
        -160079600653178,
    ],
    [
        52008642367261,
        -51880647508891,
        -12296912151884,
        41816739565574,
        8488413809640,
        90751379147199,
        -168470322721775,
        146062872110829,
    ],
    [
        -46850226917644,
        -149969062194116,
        -77958611750545,
        28727851165227,
        27615563337131,
        -124805771907366,
        157911587397819,
        56639535506051,
    ],
]

# The selected odd-constant row, exactly as emitted by the pinned LLL run.
FROZEN_REDUCTION = [
    80042099451959,
    -438404363789,
    -104528439513579,
    91613899373051,
    -7504120839914,
    125847515758073,
    -143766863997100,
    -95095973594275,
]

FROZEN_INVERSE = [
    0x0010C6FB6F1F4907,
    0x7A708F072A6FC213,
    0xE6AF80B15B4D0BC4,
    0xA5EEDA87343CDD86,
    0x516FCFE42DE2B795,
    0x33826E370315ED0A,
    0xC7AF1C653AFFF478,
    0x1DB4899B375F66C0,
]


def multiply_mod_e(
    left: list[int], right: list[int], modulus: int | None = None
) -> list[int]:
    """Multiply modulo E=X^8+2, optionally reducing coefficients modulo m."""

    result = [0] * N
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            degree = left_index + right_index
            product = left_value * right_value
            if degree >= N:
                degree -= N
                product *= LAMBDA
            result[degree] += product
    if modulus is not None:
        result = [coefficient % modulus for coefficient in result]
    return result


def multiplication_matrix(polynomial: list[int]) -> list[list[int]]:
    """Return rows for multiplication by X^shift modulo E."""

    matrix = []
    for shift in range(N):
        row = [0] * N
        for index, original_coefficient in enumerate(polynomial):
            degree = shift + index
            coefficient = original_coefficient
            if degree >= N:
                degree -= N
                coefficient *= LAMBDA
            row[degree] += coefficient
        matrix.append(row)
    return matrix


def determinant_bareiss(matrix: list[list[int]]) -> int:
    """Compute an exact integer determinant without a symbolic dependency."""

    matrix = [row.copy() for row in matrix]
    sign = 1
    previous_pivot = 1
    for pivot_index in range(len(matrix) - 1):
        pivot_row = next(
            row
            for row in range(pivot_index, len(matrix))
            if matrix[row][pivot_index] != 0
        )
        if pivot_row != pivot_index:
            matrix[pivot_row], matrix[pivot_index] = (
                matrix[pivot_index],
                matrix[pivot_row],
            )
            sign = -sign
        pivot = matrix[pivot_index][pivot_index]
        for row in range(pivot_index + 1, len(matrix)):
            for column in range(pivot_index + 1, len(matrix)):
                numerator = (
                    matrix[row][column] * pivot
                    - matrix[row][pivot_index] * matrix[pivot_index][column]
                )
                assert numerator % previous_pivot == 0
                matrix[row][column] = numerator // previous_pivot
        previous_pivot = pivot
    return sign * matrix[-1][-1]


def negative_inverse(polynomial: list[int]) -> list[int]:
    """Derive -M^-1 in (Z/2^64 Z)[X]/(X^8+2) by Newton lifting."""

    # Because the external constant is even, E is X^8 modulo 2.  An odd
    # constant coefficient makes M invertible in the resulting truncated
    # polynomial ring.  This recurrence obtains the inverse modulo 2.
    parity = [coefficient & 1 for coefficient in polynomial]
    inverse = [0] * N
    inverse[0] = 1
    for degree in range(1, N):
        inverse[degree] = sum(
            parity[index] * inverse[degree - index]
            for index in range(1, degree + 1)
        ) & 1
    assert multiply_mod_e(polynomial, inverse, 2) == [1] + [0] * (N - 1)

    modulus = 2
    while modulus < PHI:
        next_modulus = min(modulus * modulus, PHI)
        product = multiply_mod_e(polynomial, inverse, next_modulus)
        correction = [(-coefficient) % next_modulus for coefficient in product]
        correction[0] = (2 - product[0]) % next_modulus
        inverse = multiply_mod_e(inverse, correction, next_modulus)
        modulus = next_modulus
        assert multiply_mod_e(polynomial, inverse, modulus) == [1] + [0] * (
            N - 1
        )
    return [(-coefficient) % PHI for coefficient in inverse]


def main() -> None:
    installed_fpylll = version("fpylll")
    assert installed_fpylll == "0.6.4", (
        f"expected fpylll 0.6.4, found {installed_fpylll}"
    )
    assert pow(FROZEN_GAMMA, N, P) == P - 2

    rows = [[P] + [0] * (N - 1)]
    for degree in range(1, N):
        row = [(-pow(FROZEN_GAMMA, degree, P)) % P] + [0] * (N - 1)
        row[degree] = 1
        rows.append(row)

    basis = IntegerMatrix.from_matrix(rows)
    LLL.reduction(basis, delta=0.99, eta=0.501)
    reduced_rows = [
        [int(basis[row, column]) for column in range(N)] for row in range(N)
    ]
    assert reduced_rows == FROZEN_BASIS
    assert all(
        sum(
            coefficient * pow(FROZEN_GAMMA, degree, P)
            for degree, coefficient in enumerate(row)
        )
        % P
        == 0
        for row in reduced_rows
    )

    candidates = [
        (max(map(abs, row)), sum(map(abs, row)), tuple(row))
        for row in reduced_rows
        if row[0] & 1
    ]
    reduction = list(min(candidates)[2])
    assert reduction == FROZEN_REDUCTION

    inverse = negative_inverse(reduction)
    assert inverse == FROZEN_INVERSE
    assert multiply_mod_e(reduction, inverse, PHI) == [PHI - 1] + [0] * (
        N - 1
    )

    basis_determinant = determinant_bareiss(reduced_rows)
    assert basis_determinant == P
    max_basis_row_l1 = max(sum(map(abs, row)) for row in reduced_rows)
    max_basis_column_l1 = max(
        sum(abs(reduced_rows[row][column]) for row in range(N))
        for column in range(N)
    )
    assert max_basis_row_l1 == 677_018_084_045_712
    assert max_basis_column_l1 == 843_285_565_234_751

    sigma = max(map(abs, reduction))
    rho = 2 * N * abs(LAMBDA) * sigma
    assert sigma == 143_766_863_997_100
    assert rho == 4_600_539_647_907_200
    assert 2 * rho > max_basis_row_l1
    assert 2 * rho > max_basis_column_l1

    capacity_floor = PHI // (2 * N * abs(LAMBDA) * rho)
    required_capacity = (DELTA + 1) ** 2
    assert capacity_floor == 125
    assert required_capacity == 121
    assert capacity_floor >= required_capacity

    external_weight = 1 + (N - 1) * abs(LAMBDA)
    lazy_limit = (DELTA + 1) * (rho - 1)
    product_bound = external_weight * lazy_limit**2
    correction_bound = external_weight * (PHI // 2) * sigma
    assert product_bound + correction_bound < rho * PHI

    # For the recursive K8 graph, a conservative bound on every held
    # convolution coefficient is 8^3=512 times the two input bounds.  Folding
    # X^8=-2 costs another factor of at most three.
    karatsuba_factor = 8**3
    karatsuba_numerator_bound = 3 * karatsuba_factor * (
        lazy_limit**2 + (PHI // 2) * sigma
    )
    assert karatsuba_numerator_bound < 1 << 127

    reduction_determinant = determinant_bareiss(multiplication_matrix(reduction))
    assert reduction_determinant == 33 * P

    print(
        json.dumps(
            {
                "fpylll_version": installed_fpylll,
                "lll_delta": 0.99,
                "lll_eta": 0.501,
                "n": N,
                "external_polynomial": "X^8 + 2",
                "lambda": LAMBDA,
                "phi": PHI,
                "delta": DELTA,
                "gamma_hex": hex(FROZEN_GAMMA),
                "gamma": str(FROZEN_GAMMA),
                "reduced_basis": reduced_rows,
                "basis_determinant": str(basis_determinant),
                "max_basis_row_l1": max_basis_row_l1,
                "max_basis_column_l1": max_basis_column_l1,
                "reduction_polynomial": reduction,
                "negative_inverse_mod_phi": inverse,
                "multiplication_matrix_determinant_factor": 33,
                "sigma": sigma,
                "rho": rho,
                "capacity_floor": capacity_floor,
                "required_capacity": required_capacity,
                "lazy_input_limit": lazy_limit,
                "closure_numerator_bound": product_bound + correction_bound,
                "karatsuba_i128_numerator_bound": karatsuba_numerator_bound,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
