# Paralelismo com Memória Compartilhada

Comparação de desempenho entre cinco implementações de um mesmo algoritmo;
inversão de matriz por eliminação de Gauss-Jordan com pivotamento parcial
(N=2500); variando linguagem (C compilado vs. Python interpretado) e modelo
de paralelização (OpenMP, threading, multiprocessing). O trabalho aplica um
conjunto completo de ferramentas de profiling (gprof, perf, Valgrind,
strace, cProfile) para identificar hotspots, quantificar o overhead do
interpretador Python e diagnosticar a causa raiz da baixa escalabilidade
observada em todos os modelos testados.

## Resultados principais

- Overhead do interpretador Python: **80,3× mais instruções** e **63,9× mais
  lento** que o C compilado com `-O2`, apesar de IPC e taxa de cache-miss
  isoladamente favorecerem o Python.
- C + OpenMP não escala conforme a Lei de Amdahl prevê (7,83× teórico contra
  **1,16-1,22× medido**): o algoritmo é _memory-bound_, e a contenção de
  Hyper-Threading agrava o problema de 4 para 8 threads.
- Python Threading nunca demonstra paralelismo real em nenhuma configuração
  (%CPU nunca ultrapassa 102%) devido ao GIL.
- Python Multiprocessing demonstra paralelismo real (%CPU até 757%), mas o
  overhead estrutural de acesso à memória compartilhada consome quase todo o
  ganho frente ao Python Serial comum.

Detalhamento completo, com todas as tabelas, gráficos e discussão crítica,
está no [relatório técnico](shared_memory/report_technical/Relatorio_Paralelismo_Memoria_Compartilhada.pdf).

## Estrutura do repositório

```
hpc-shared-memory/
└── shared_memory/
    ├── perf/              Arquivos perf.data compactados
    ├── report_technical/  Relatório técnico final (PDF)
    ├── reports/           Saídas brutas de todas as ferramentas de profiling
    └── scripts/           Código-fonte das 5 versões + script de automação
```

### [`shared_memory/scripts/`](shared_memory/scripts/)

Código-fonte das cinco versões implementadas, mais os utilitários de apoio.

| Arquivo                                                                      | Descrição                                                                          |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| [`programa_serial.c`](shared_memory/scripts/programa_serial.c)               | Versão 1/5 — C serial                                                              |
| [`programa_openmp.c`](shared_memory/scripts/programa_openmp.c)               | Versão 2/5 — C + OpenMP (1/2/4/8 threads)                                          |
| [`programa_serial.py`](shared_memory/scripts/programa_serial.py)             | Versão 3/5 — Python serial                                                         |
| [`programa_thread.py`](shared_memory/scripts/programa_thread.py)             | Versão 4/5 — Python + threading                                                    |
| [`programa_multiprocess.py`](shared_memory/scripts/programa_multiprocess.py) | Versão 5/5 — Python + multiprocessing (memória compartilhada via `shared_memory`)  |
| [`gerador_matriz.c`](shared_memory/scripts/gerador_matriz.c)                 | Gera a matriz de entrada (N×N, diagonalmente dominante) usada por todas as versões |
| [`run_all.sh`](shared_memory/scripts/run_all.sh)                             | Automatiza compilação, execução e coleta de profiling das 5 versões                |

### [`shared_memory/reports/`](shared_memory/reports/)

Saídas brutas (texto) de cada ferramenta de profiling, uma execução por arquivo.

| Ferramenta                                          | Arquivos                                                             |
| --------------------------------------------------- | -------------------------------------------------------------------- |
| `gprof` (C serial)                                  | `gprof_resultado.txt`                                                |
| Valgrind — Callgrind / Cachegrind (C serial)        | `callgrind_resultado.txt`, `cachegrind_resultado.txt`                |
| `perf stat` (todas as versões, por thread/processo) | `perf_stat_*.txt`                                                    |
| `perf record` + `perf report`                       | `perf_report_*.txt`                                                  |
| `cProfile` (versões Python)                         | `cprofile_serial.txt`, `cprofile_thread4.txt`, `cprofile_multi4.txt` |
| `/usr/bin/time -v` (todas as versões)               | `time_*.txt`                                                         |
| Medição interna (`clock_gettime`)                   | `stats*.txt`                                                         |

### [`shared_memory/perf/`](shared_memory/perf/)

`perf_data.zip` — arquivos `perf.data` brutos (call graph completo), gerados
por `perf record -g`.

> **Nota:** os arquivos `perf.data` das versões em **Python** (Threading e
> Multiprocessing) não estão incluídos neste pacote. Nas execuções deste
> trabalho, esses arquivos atingiram a casa de gigabytes (o do
> Multiprocessing chegou a ~9 GB), inviável para versionamento no GitHub. A
> informação equivalente já processada está disponível em
> `shared_memory/reports/perf_report_*.txt`. Detalhes da causa dessa
> diferença de tamanho estão na Seção 7.4 do relatório técnico.

### [`shared_memory/report_technical/`](shared_memory/report_technical/)

O relatório técnico final em PDF, com toda a metodologia, resultados,
gráficos de escalabilidade, comparação entre os três modelos de
paralelização, discussão crítica e referências.

## Ambiente de teste

Intel Core i7-3770 @ 3,90 GHz (4 núcleos físicos / 8 lógicos via
Hyper-Threading), 16 GiB RAM, Linux.
