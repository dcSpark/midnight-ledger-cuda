#!/usr/bin/env python3
"""Reproduce the frozen 25-coefficient BLS12-381 PMNS reduction basis."""

from __future__ import annotations

import json
from importlib.metadata import version

from fpylll import IntegerMatrix, LLL

P = int(
    "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787"
)
N = 25
LAMBDA = 2
PHI = 1 << 32
DELTA = 2
FROZEN_GAMMA = 511066730143548652923330221406862432842513716081412796986393940547425281101465886366786490203512647746495335967402
FROZEN_REDUCTION = [
    5265, -6042, -550, -3841, -46309, -3691, -13917, 54, 11281, -18175,
    -73, 6377, 11698, -9308, 1330, 2745, -40, 8179, -3550, -11129,
    -22495, 9238, 6654, 3275, 2537,
]
FROZEN_INVERSE = [
    4232023841, 3036869944, 3601667488, 3120418835, 3586139365,
    3598383083, 956870906, 3696527062, 2215698700, 1470334370,
    3997815005, 157132901, 563938266, 522095032, 3965809933,
    3693029335, 229979373, 472493717, 3637835460, 2878516501,
    1101624567, 3515217104, 3802752435, 2306994988, 4030515588,
]


def multiply_mod_e(left: list[int], right: list[int], modulus: int | None = None) -> list[int]:
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


def multiplication_norm(polynomial: list[int]) -> int:
    columns = []
    for column in range(N):
        column_sum = 0
        for shift in range(N):
            source = column - shift
            if source < 0:
                source += N
                column_sum += LAMBDA * abs(polynomial[source])
            else:
                column_sum += abs(polynomial[source])
        columns.append(column_sum)
    return max(columns)


def multiplication_matrix(polynomial: list[int]) -> list[list[int]]:
    matrix = []
    for shift in range(N):
        row = [0] * N
        for index, coefficient in enumerate(polynomial):
            degree = shift + index
            if degree >= N:
                degree -= N
                coefficient *= LAMBDA
            row[degree] += coefficient
        matrix.append(row)
    return matrix


def determinant_bareiss(matrix: list[list[int]]) -> int:
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
            matrix[pivot_row], matrix[pivot_index] = matrix[pivot_index], matrix[pivot_row]
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
    inverse = [0] * N
    parity = [coefficient & 1 for coefficient in polynomial]
    inverse[0] = 1
    for degree in range(1, N):
        inverse[degree] = sum(parity[index] * inverse[degree - index] for index in range(1, degree + 1)) & 1

    modulus = 2
    while modulus < PHI:
        next_modulus = min(modulus * modulus, PHI)
        product = multiply_mod_e(polynomial, inverse, next_modulus)
        correction = [(-coefficient) % next_modulus for coefficient in product]
        correction[0] = (2 - product[0]) % next_modulus
        inverse = multiply_mod_e(inverse, correction, next_modulus)
        modulus = next_modulus
        assert multiply_mod_e(polynomial, inverse, modulus) == [1] + [0] * (N - 1)
    return [(-coefficient) % PHI for coefficient in inverse]


def main() -> None:
    installed_fpylll = version("fpylll")
    assert installed_fpylll == "0.6.4", f"expected fpylll 0.6.4, found {installed_fpylll}"
    gamma = pow(LAMBDA, pow(N, -1, P - 1), P)
    assert (pow(gamma, N, P) - LAMBDA) % P == 0
    assert gamma == FROZEN_GAMMA

    rows = [[P] + [0] * (N - 1)]
    for degree in range(1, N):
        row = [(-pow(gamma, degree, P)) % P] + [0] * (N - 1)
        row[degree] = 1
        rows.append(row)

    basis = IntegerMatrix.from_matrix(rows)
    LLL.reduction(basis, delta=0.99, eta=0.501)
    reduced_rows = [[int(basis[row, column]) for column in range(N)] for row in range(N)]
    candidates = [row for row in reduced_rows if row[0] & 1]
    reduction = min(candidates, key=multiplication_norm)
    rho = multiplication_norm(reduction) - 1
    inverse = negative_inverse(reduction)

    assert reduction == FROZEN_REDUCTION
    assert inverse == FROZEN_INVERSE
    assert sum(coefficient * pow(gamma, degree, P) for degree, coefficient in enumerate(reduction)) % P == 0
    assert multiply_mod_e(reduction, inverse, PHI) == [PHI - 1] + [0] * (N - 1)
    required_phi = 2 * (1 + (N - 1) * LAMBDA) * (rho - 1) * (DELTA + 1) ** 2
    assert rho == 410_240
    assert required_phi == 361_830_798
    assert required_phi < PHI
    determinant = determinant_bareiss(multiplication_matrix(reduction))
    assert determinant == -19_431 * P

    print(
        json.dumps(
            {
                "fpylll_version": installed_fpylll,
                "lll_delta": 0.99,
                "lll_eta": 0.501,
                "n": N,
                "lambda": LAMBDA,
                "phi": PHI,
                "delta": DELTA,
                "gamma": str(gamma),
                "reduction_polynomial": reduction,
                "negative_inverse_mod_phi": inverse,
                "rho": rho,
                "required_phi": required_phi,
                "multiplication_matrix_determinant_factor": -19_431,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
