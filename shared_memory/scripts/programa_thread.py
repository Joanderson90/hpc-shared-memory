"""
Mesmo algoritmo, paralelizando o loop de atualizacao de linhas (for i)
com o modulo threading -- analogo direto ao "#pragma omp parallel for"
da versao C+OpenMP. Threads em CPython COMPARTILHAM o mesmo interpretador
e o mesmo GIL (Global Interpreter Lock): apenas UMA thread executa
bytecode Python por vez, mesmo em CPU multicore.

Uso:
    python3 programa_thread.py <N> <matriz.bin> <num_threads> [inversa_saida.bin] [tempo_serial_s]
"""

import sys
import struct
import time
import threading


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


def update_chunk(M, row_k, factor_col, start, end, k, W):
    for i in range(start, end):
        if i == k:
            continue
        row_i = M[i]
        factor = row_i[factor_col]
        if factor == 0.0:
            continue
        for j in range(W):
            row_i[j] -= factor * row_k[j]


def gauss_jordan_threading(M, N, W, num_threads):
    chunk_size = (N + num_threads - 1) // num_threads

    for k in range(N):
        partial_pivot(M, N, W, k)

        row_k = M[k]
        pivot = row_k[k]
        for j in range(W):
            row_k[j] /= pivot

        threads = []
        for t in range(num_threads):
            start = t * chunk_size
            end = min(start + chunk_size, N)
            if start >= end:
                continue
            th = threading.Thread(target=update_chunk, args=(M, row_k, k, start, end, k, W))
            threads.append(th)
            th.start()
        for th in threads:
            th.join()


def extract_inverse(M, N, W):
    return [row[N:W] for row in M]


def main():
    if len(sys.argv) < 4:
        print(f"Uso: {sys.argv[0]} <N> <matriz.bin> <num_threads> [inversa_saida.bin] [tempo_serial_s]", file=sys.stderr)
        sys.exit(1)

    N = int(sys.argv[1])
    matrix_path = sys.argv[2]
    num_threads = int(sys.argv[3])
    out_path = sys.argv[4] if len(sys.argv) >= 5 else None
    tempo_serial = float(sys.argv[5]) if len(sys.argv) >= 6 else None
    W = 2 * N

    t0 = time.perf_counter()
    A = read_matrix(matrix_path, N)
    t1 = time.perf_counter()

    M = build_augmented(A, N)
    t2 = time.perf_counter()

    gauss_jordan_threading(M, N, W, num_threads)
    t3 = time.perf_counter()

    Inv = extract_inverse(M, N, W)
    if out_path:
        write_matrix(out_path, Inv, N)
    t4 = time.perf_counter()

    print(f"=== PYTHON THREADING (N={N}, threads={num_threads}) ===")
    print(f"Leitura           : {t1 - t0:.6f} s")
    print(f"Setup (montagem)  : {t2 - t1:.6f} s")
    print(f"Eliminacao        : {t3 - t2:.6f} s")
    print(f"TOTAL             : {t4 - t0:.6f} s")

    if tempo_serial:
        speedup = tempo_serial / (t3 - t2)
        eficiencia = speedup / num_threads
        print(f"Speedup vs. serial ({tempo_serial:.6f}s): {speedup:.3f}x")
        print(f"Eficiencia                : {eficiencia * 100:.1f}%")


if __name__ == "__main__":
    main()
