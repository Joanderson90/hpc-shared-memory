"""
Mesmo algoritmo das versoes em C (Eliminacao de Gauss-Jordan com
pivotamento parcial), implementado em Python puro -- apenas listas e
range, sem bibliotecas externas. Le o MESMO arquivo binario de matriz
usado pelas versoes em C (N doubles x N doubles, row-major, sem
cabecalho), garantindo volume de dados identico para comparacao justa.

Uso:
    python3 programa_serial.py <N> <matriz.bin> [inversa_saida.bin] [--verify]

Medicao de tempo: time.perf_counter()
============================================================================
"""

import sys
import struct
import time


def read_matrix(path, N):
    with open(path, "rb") as f:
        raw = f.read(N * N * 8)
    flat = struct.unpack(f"<{N * N}d", raw)
    return [list(flat[i * N:(i + 1) * N]) for i in range(N)]


def write_matrix(path, mat, N):
    flat = [v for row in mat for v in row]
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{N * N}d", *flat))


def build_augmented(A, N):
    W = 2 * N
    M = [[0.0] * W for _ in range(N)]
    for i in range(N):
        row = M[i]
        arow = A[i]
        for j in range(N):
            row[j] = arow[j]
        row[N + i] = 1.0
    return M


def partial_pivot(M, N, W, k):
    max_row = k
    max_val = abs(M[k][k])
    for i in range(k + 1, N):
        v = abs(M[i][k])
        if v > max_val:
            max_val = v
            max_row = i
    if max_row != k:
        M[k], M[max_row] = M[max_row], M[k]


def gauss_jordan_serial(M, N, W):
    for k in range(N):
        partial_pivot(M, N, W, k)

        row_k = M[k]
        pivot = row_k[k]
        for j in range(W):
            row_k[j] /= pivot

        for i in range(N):
            if i == k:
                continue
            row_i = M[i]
            factor = row_i[k]
            if factor == 0.0:
                continue
            for j in range(W):
                row_i[j] -= factor * row_k[j]


def extract_inverse(M, N, W):
    return [row[N:W] for row in M]


def verify_residual(A, Inv, N):
    residual_sq = 0.0
    for i in range(N):
        for j in range(N):
            s = 0.0
            for p in range(N):
                s += A[i][p] * Inv[p][j]
            diff = s - (1.0 if i == j else 0.0)
            residual_sq += diff * diff
    return residual_sq ** 0.5


def main():
    if len(sys.argv) < 3:
        print(f"Uso: {sys.argv[0]} <N> <matriz.bin> [inversa_saida.bin] [--verify]", file=sys.stderr)
        sys.exit(1)

    N = int(sys.argv[1])
    matrix_path = sys.argv[2]
    out_path = None
    do_verify = False
    for a in sys.argv[3:]:
        if a == "--verify":
            do_verify = True
        else:
            out_path = a
    W = 2 * N

    t0 = time.perf_counter()
    A = read_matrix(matrix_path, N)
    t1 = time.perf_counter()

    M = build_augmented(A, N)
    t2 = time.perf_counter()

    gauss_jordan_serial(M, N, W)
    t3 = time.perf_counter()

    Inv = extract_inverse(M, N, W)
    if out_path:
        write_matrix(out_path, Inv, N)
    t4 = time.perf_counter()

    print(f"=== PYTHON SERIAL (N={N}) ===")
    print(f"Leitura           : {t1 - t0:.6f} s")
    print(f"Setup (montagem)  : {t2 - t1:.6f} s")
    print(f"Eliminacao        : {t3 - t2:.6f} s")
    print(f"TOTAL             : {t4 - t0:.6f} s")

    if do_verify:
        residual = verify_residual(A, Inv, N)
        print(f"Residuo ||A*A^-1 - I||_F = {residual:.6e}")


if __name__ == "__main__":
    main()
