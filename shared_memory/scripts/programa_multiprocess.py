"""
Mesmo algoritmo, paralelizado com o modulo multiprocessing. Diferente do
threading, cada processo tem sua PROPRIA instancia do interpretador
CPython (e seu proprio GIL) -- paralelismo real em codigo CPU-bound.

Modelo de memoria: processos NAO compartilham espaco de enderecamento
por padrao. Para evitar o pior overhead (serializar/copiar a matriz
inteira via pickle a cada uma das N iteracoes de k), usamos
multiprocessing.shared_memory: a matriz aumentada fica em um bloco de
memoria compartilhada, e cada processo trabalhador anexa a ela UMA
UNICA VEZ (no initializer do Pool). A cada iteracao k, so um pequeno
payload (start, end, k) e enviado a cada worker via IPC -- nao a matriz
inteira.

O Pool de processos e criado UMA VEZ (fora do loop de iteracoes) para
evitar pagar o custo de fork() repetidamente a cada k.

Uso:
    python3 programa_multiprocess.py <N> <matriz.bin> <num_processos> [inversa_saida.bin] [tempo_serial_s]
"""

import sys
import struct
import time
import multiprocessing as mp
from multiprocessing import shared_memory
import cProfile
import pstats

_shm = None
_buf = None
_N = None
_W = None


def _worker_init(shm_name, N, W):
    global _shm, _buf, _N, _W
    _shm = shared_memory.SharedMemory(name=shm_name)
    _buf = memoryview(_shm.buf).cast("d")
    _N, _W = N, W


def _update_chunk_worker(start, end, k):
    W = _W
    buf = _buf
    row_k_off = k * W
    for i in range(start, end):
        if i == k:
            continue
        row_i_off = i * W
        factor = buf[row_i_off + k]
        if factor == 0.0:
            continue
        for j in range(W):
            buf[row_i_off + j] -= factor * buf[row_k_off + j]


def read_matrix_flat(path, N):
    with open(path, "rb") as f:
        raw = f.read(N * N * 8)
    return list(struct.unpack(f"<{N * N}d", raw))


def write_matrix(path, flat_inv, N):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{N * N}d", *flat_inv))


def partial_pivot(buf, N, W, k):
    max_row = k
    max_val = abs(buf[k * W + k])
    for i in range(k + 1, N):
        v = abs(buf[i * W + k])
        if v > max_val:
            max_val = v
            max_row = i
    if max_row != k:
        off_k, off_m = k * W, max_row * W
        for j in range(W):
            buf[off_k + j], buf[off_m + j] = buf[off_m + j], buf[off_k + j]


def main():
    if len(sys.argv) < 4:
        print(f"Uso: {sys.argv[0]} <N> <matriz.bin> <num_processos> [inversa_saida.bin] [tempo_serial_s]", file=sys.stderr)
        sys.exit(1)

    N = int(sys.argv[1])
    matrix_path = sys.argv[2]
    num_proc = int(sys.argv[3])
    out_path = sys.argv[4] if len(sys.argv) >= 5 else None
    tempo_serial = float(sys.argv[5]) if len(sys.argv) >= 6 else None
    W = 2 * N

    t0 = time.perf_counter()
    A_flat = read_matrix_flat(matrix_path, N)
    t1 = time.perf_counter()

    # Monta a matriz aumentada diretamente em memoria compartilhada
    shm = shared_memory.SharedMemory(create=True, size=N * W * 8)
    buf = memoryview(shm.buf).cast("d")
    for i in range(N):
        off_a, off_m = i * N, i * W
        for j in range(N):
            buf[off_m + j] = A_flat[off_a + j]
        buf[off_m + N + i] = 1.0
    t2 = time.perf_counter()

    # Pool persistente: criado UMA VEZ (custo de fork medido separadamente)
    t_pool_start = time.perf_counter()
    pool = mp.Pool(processes=num_proc, initializer=_worker_init, initargs=(shm.name, N, W))
    t_pool_end = time.perf_counter()

    chunk_size = (N + num_proc - 1) // num_proc

    for k in range(N):
        partial_pivot(buf, N, W, k)

        off_k = k * W
        pivot = buf[off_k + k]
        for j in range(W):
            buf[off_k + j] /= pivot

        tasks = []
        for p in range(num_proc):
            start = p * chunk_size
            end = min(start + chunk_size, N)
            if start < end:
                tasks.append((start, end, k))
        pool.starmap(_update_chunk_worker, tasks)

    t3 = time.perf_counter()

    pool.close()
    pool.join()

    inv_flat = [buf[i * W + N + j] for i in range(N) for j in range(N)]
    buf.release()
    shm.close()
    shm.unlink()

    if out_path:
        write_matrix(out_path, inv_flat, N)
    t4 = time.perf_counter()

    print(f"=== PYTHON MULTIPROCESSING (N={N}, processos={num_proc}) ===")
    print(f"Leitura                : {t1 - t0:.6f} s")
    print(f"Setup (memoria compart.): {t2 - t1:.6f} s")
    print(f"Criacao do pool (fork) : {t_pool_end - t_pool_start:.6f} s")
    print(f"Eliminacao             : {t3 - t2:.6f} s  <-- hotspot (paralelismo real)")
    print(f"TOTAL                  : {t4 - t0:.6f} s")

    if tempo_serial:
        speedup = tempo_serial / (t3 - t2)
        eficiencia = speedup / num_proc
        print(f"Speedup vs. serial ({tempo_serial:.6f}s): {speedup:.3f}x")
        print(f"Eficiencia                : {eficiencia * 100:.1f}%")


if __name__ == "__main__":
    # --cprofile: perfila internamente, em vez de via "python -m cProfile".
    # Necessario porque "-m cProfile" executa o script fora do modulo real
    # __main__, quebrando o pickle que o multiprocessing precisa fazer de
    # _update_chunk_worker (o pickle procura a funcao em sys.modules['__main__'],
    # que sob "-m cProfile" aponta para o proprio cProfile.py, nao para este
    # script -- causando "PicklingError: attribute lookup ... on __main__ failed").
    if "--cprofile" in sys.argv:
        sys.argv.remove("--cprofile")
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats()
    else:
        main()