#!/usr/bin/env bash
# ==============================================================================
# run_all.sh — Paralelismo com Memoria Compartilhada
# ==============================================================================
# Automatiza: compilacao, geracao de dados, execucao das 5 versoes e coleta
# completa de profiling (/usr/bin/time -v, gprof, perf, Valgrind, strace,
# cProfile), reproduzindo os experimentos documentados no relatorio.
#
# Uso:
#   ./run_all.sh              # roda tudo (pode levar VARIAS HORAS - ver aviso)
#   ./run_all.sh --rapido     # pula as execucoes Python longas (>30min cada)
#   ./run_all.sh --so-c       # roda so C serial + C OpenMP (rapido, minutos)
#
# Requisitos: gcc, python3, perf, valgrind, strace, /usr/bin/time
# ==============================================================================

set -uo pipefail

N=2500
N_VALGRIND=250          # N reduzido para Callgrind/Cachegrind (custo de simulacao ~10x)
MATRIZ=matriz.bin
MATRIZ_PEQUENA=matriz_pequena.bin
RESULTS=resultados
MODO="${1:-completo}"

mkdir -p "$RESULTS"/{c_serial,c_openmp,python_serial,python_thread,python_multi,perf_data}

echo "=============================================================="
echo " Paralelismo com Memoria Compartilhada - run_all.sh"
echo " N=$N | modo=$MODO | resultados em ./$RESULTS/"
echo "=============================================================="

if [[ "$MODO" == "--rapido" || "$MODO" == "--so-c" ]]; then
  echo "AVISO: pulando execucoes Python longas (>30min cada)."
fi

# ------------------------------------------------------------------------
# 0. Compilacao e geracao de dados
# ------------------------------------------------------------------------
echo ""
echo "--- [0/7] Compilando e gerando matriz de entrada ---"
gcc -O2 -o gerador_matriz gerador_matriz.c -lm
gcc -O2 -g -o programa_serial programa_serial.c -lm
gcc -O2 -g -fopenmp -o programa_openmp programa_openmp.c -lm
gcc -pg -O2 -g -o programa_serial_gprof programa_serial.c -lm

[ -f "$MATRIZ" ] || ./gerador_matriz "$N" "$MATRIZ"
[ -f "$MATRIZ_PEQUENA" ] || ./gerador_matriz "$N_VALGRIND" "$MATRIZ_PEQUENA"

# ==========================================================================
# 1. C SERIAL
# ==========================================================================
echo ""
echo "--- [1/7] C Serial: /usr/bin/time, gprof, perf, Valgrind, strace ---"

/usr/bin/time -v ./programa_serial "$N" "$MATRIZ" linha "$RESULTS/c_serial/inversa.bin" \
  2> "$RESULTS/c_serial/time_v.txt"

./programa_serial_gprof "$N" "$MATRIZ" linha "$RESULTS/c_serial/inversa_gprof.bin"
gprof programa_serial_gprof gmon.out > "$RESULTS/c_serial/gprof.txt"
mv gmon.out "$RESULTS/c_serial/" 2>/dev/null

sudo perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses,branches,L1-dcache-load-misses,LLC-load-misses \
  ./programa_serial "$N" "$MATRIZ" linha "$RESULTS/c_serial/inversa_perf.bin" \
  > "$RESULTS/c_serial/perf_stat.txt" 2>&1

sudo perf record -g -o "$RESULTS/perf_data/c_serial.data" \
  ./programa_serial "$N" "$MATRIZ" linha "$RESULTS/c_serial/inversa_perf2.bin"
sudo perf report --stdio -i "$RESULTS/perf_data/c_serial.data" > "$RESULTS/c_serial/perf_report.txt"

valgrind --tool=callgrind --callgrind-out-file="$RESULTS/c_serial/callgrind.out" \
  ./programa_serial "$N_VALGRIND" "$MATRIZ_PEQUENA" linha "$RESULTS/c_serial/inversa_cg.bin"
callgrind_annotate "$RESULTS/c_serial/callgrind.out" > "$RESULTS/c_serial/callgrind.txt"

valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file="$RESULTS/c_serial/cachegrind.out" \
  ./programa_serial "$N_VALGRIND" "$MATRIZ_PEQUENA" linha "$RESULTS/c_serial/inversa_cache.bin"
cg_annotate "$RESULTS/c_serial/cachegrind.out" > "$RESULTS/c_serial/cachegrind.txt"

strace -c ./programa_serial "$N" "$MATRIZ" linha "$RESULTS/c_serial/inversa_strace.bin" \
  2> "$RESULTS/c_serial/strace.txt"

# ==========================================================================
# 2. C + OPENMP (1/2/4/8 threads)
# ==========================================================================
echo ""
echo "--- [2/7] C + OpenMP: /usr/bin/time e perf stat para 1/2/4/8 threads ---"

for t in 1 2 4 8; do
  /usr/bin/time -v env OMP_NUM_THREADS=$t ./programa_openmp "$N" "$MATRIZ" linha \
    "$RESULTS/c_openmp/inversa_${t}t.bin" 2> "$RESULTS/c_openmp/time_v_${t}t.txt"

  sudo env OMP_NUM_THREADS=$t perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses,branches,L1-dcache-load-misses,LLC-load-misses \
    ./programa_openmp "$N" "$MATRIZ" linha "$RESULTS/c_openmp/inversa_perf_${t}t.bin" \
    > "$RESULTS/c_openmp/perf_stat_${t}t.txt" 2>&1
done

# perf record na configuracao de referencia (4 threads), conforme enunciado (3.4.1)
sudo env OMP_NUM_THREADS=4 perf record -g -o "$RESULTS/perf_data/c_openmp_4t.data" \
  ./programa_openmp "$N" "$MATRIZ" linha "$RESULTS/c_openmp/inversa_perf_record.bin"
sudo perf report --stdio -i "$RESULTS/perf_data/c_openmp_4t.data" > "$RESULTS/c_openmp/perf_report_4t.txt"

if [[ "$MODO" == "--so-c" ]]; then
  echo ""
  echo "Modo --so-c: parando aqui (C serial + C OpenMP concluidos)."
  exit 0
fi

# ==========================================================================
# 3. PYTHON SERIAL  (~30min)
# ==========================================================================
echo ""
echo "--- [3/7] Python Serial (estimativa: ~30-35 minutos) ---"

if [[ "$MODO" != "--rapido" ]]; then
  /usr/bin/time -v python3 programa_serial.py "$N" "$MATRIZ" \
    "$RESULTS/python_serial/inversa.bin" 2> "$RESULTS/python_serial/time_v.txt"

  python3 -m cProfile -s cumulative programa_serial.py "$N" "$MATRIZ" \
    "$RESULTS/python_serial/inversa_cprofile.bin" > "$RESULTS/python_serial/cprofile.txt"

  sudo perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses,branches,L1-dcache-load-misses,LLC-load-misses \
    python3 programa_serial.py "$N" "$MATRIZ" "$RESULTS/python_serial/inversa_perf.bin" \
    > "$RESULTS/python_serial/perf_stat.txt" 2>&1

  strace -c python3 programa_serial.py "$N" "$MATRIZ" "$RESULTS/python_serial/inversa_strace.bin" \
    2> "$RESULTS/python_serial/strace.txt"
else
  echo "Pulado (--rapido)."
fi

# ==========================================================================
# 4. PYTHON THREADING (1/2/4/8) (~1h cada, exceto 1t)
# ==========================================================================
echo ""
echo "--- [4/7] Python Threading (estimativa: ~30min a ~1h20 por execucao) ---"

if [[ "$MODO" != "--rapido" ]]; then
  for t in 1 2 4 8; do
    /usr/bin/time -v python3 programa_thread.py "$N" "$MATRIZ" "$t" \
      "$RESULTS/python_thread/inversa_${t}t.bin" 2> "$RESULTS/python_thread/time_v_${t}t.txt"
  done

  python3 -m cProfile -s cumulative programa_thread.py "$N" "$MATRIZ" 4 \
    "$RESULTS/python_thread/inversa_cprofile.bin" > "$RESULTS/python_thread/cprofile_4t.txt"

  sudo perf stat python3 programa_thread.py "$N" "$MATRIZ" 4 \
    "$RESULTS/python_thread/inversa_perf.bin" > "$RESULTS/python_thread/perf_stat_4t.txt" 2>&1

  sudo perf record -g -o "$RESULTS/perf_data/python_thread_4t.data" \
    python3 programa_thread.py "$N" "$MATRIZ" 4 "$RESULTS/python_thread/inversa_perf_record.bin"
  sudo perf report --stdio -i "$RESULTS/perf_data/python_thread_4t.data" \
    > "$RESULTS/python_thread/perf_report_4t.txt"
else
  echo "Pulado (--rapido)."
fi

# ==========================================================================
# 5. PYTHON MULTIPROCESSING (1/2/4/8) (~30min a ~1h40 por execucao)
# ==========================================================================
echo ""
echo "--- [5/7] Python Multiprocessing (estimativa: ~30min a ~1h40 por execucao) ---"

if [[ "$MODO" != "--rapido" ]]; then
  for p in 1 2 4 8; do
    /usr/bin/time -v python3 programa_multiprocess.py "$N" "$MATRIZ" "$p" \
      "$RESULTS/python_multi/inversa_${p}p.bin" 2> "$RESULTS/python_multi/time_v_${p}p.txt"
  done

  # --cprofile: flag propria do script (necessaria porque "-m cProfile" quebra o
  # pickle de multiprocessing, ver Secao 7 do relatorio)
  python3 programa_multiprocess.py "$N" "$MATRIZ" 4 \
    "$RESULTS/python_multi/inversa_cprofile.bin" --cprofile > "$RESULTS/python_multi/cprofile_4p.txt"

  sudo perf stat python3 programa_multiprocess.py "$N" "$MATRIZ" 4 \
    "$RESULTS/python_multi/inversa_perf.bin" > "$RESULTS/python_multi/perf_stat_4p.txt" 2>&1

  echo "AVISO: perf record de multiprocessing gera arquivo muito grande (~9GB"
  echo "       observado neste trabalho) e pode ser impraticavel de processar."
  echo "       Descomente as linhas abaixo por sua conta e risco."
  # sudo perf record -g -o "$RESULTS/perf_data/python_multi_4p.data" \
  #   python3 programa_multiprocess.py "$N" "$MATRIZ" 4 "$RESULTS/python_multi/inversa_pr.bin"
else
  echo "Pulado (--rapido)."
fi

# ==========================================================================
# 6. Empacotar perf.data
# ==========================================================================
echo ""
echo "--- [6/7] Compactando arquivos perf.data ---"
tar -czf "$RESULTS/perf_data.tar.gz" -C "$RESULTS" perf_data/
echo "Gerado: $RESULTS/perf_data.tar.gz"

# ==========================================================================
# 7. Resumo
# ==========================================================================
echo ""
echo "--- [7/7] Concluido ---"
echo "Todos os resultados estao em ./$RESULTS/"
du -sh "$RESULTS" 2>/dev/null
