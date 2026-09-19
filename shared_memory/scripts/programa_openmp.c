/*
 * ============================================================================
 * Algoritmo: Inversao de Matriz (Gauss-Jordan) -- Paralelo com OpenMP
 * ============================================================================
 *
 * Identico a programa_serial.c em toda a estrutura (mesmas funcoes, mesmo
 * formato de arquivo, mesma interface de linha de comando), acrescentando
 * apenas as diretivas de paralelizacao OpenMP nos dois pontos de eliminacao:
 *
 *   gauss_jordan_row_oriented: loop externo "for i" (linhas) paralelizado.
 *   gauss_jordan_col_oriented: loop externo "for j" (colunas) paralelizado.
 *
 * Em ambos os casos, o loop paralelizado e' exatamente aquele ja identificado
 * como "externo" na versao serial -- cada iteracao (linha i, ou coluna j,
 * dependendo da orientacao) e' independente das demais dentro da mesma
 * iteracao k: nao ha dependencia de dados entre elas, apenas leituras da
 * linha/coluna pivo (ja calculada) e escritas em posicoes disjuntas da
 * matriz. Isso torna a paralelizacao direta, sem necessidade de secoes
 * criticas ou locks.
 *
 * O pivotamento parcial permanece serial nas duas orientacoes: seu custo e'
 * O(N) por iteracao k, ordem de grandeza menor que o O(N) por linha/coluna
 * x N linhas/colunas = O(N^2) da eliminacao -- paralelizar essa busca
 * acrescentaria overhead de sincronizacao sem ganho liquido.
 *
 * Compilacao:
 *   gcc -O2 -g -fopenmp -o programa_openmp programa_openmp.c -lm
 *
 * Uso (numero de threads via variavel de ambiente OMP_NUM_THREADS):
 *   OMP_NUM_THREADS=4 ./programa_openmp <N> <matriz.bin> <linha|coluna> [inversa_saida.bin] [stats.txt] [--verify]
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/* ---------------------------------------------------------------------- */
/* Utilitario de tempo                                                    */
/* ---------------------------------------------------------------------- */
static double timespec_diff_sec(struct timespec start, struct timespec end)
{
    double s = (double)(end.tv_sec - start.tv_sec);
    double ns = (double)(end.tv_nsec - start.tv_nsec) / 1e9;
    return s + ns;
}

/* ---------------------------------------------------------------------- */
/* Leitura da matriz do arquivo binario (N*N doubles, row-major, sem       */
/* cabecalho)                                                              */
/* ---------------------------------------------------------------------- */
static double *read_matrix(const char *path, int N)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        fprintf(stderr, "Erro: nao foi possivel abrir '%s'\n", path);
        return NULL;
    }

    size_t expected = (size_t)N * N;
    double *A = (double *)malloc(expected * sizeof(double));
    if (!A)
    {
        fclose(f);
        return NULL;
    }

    if (fread(A, sizeof(double), expected, f) != expected)
    {
        fprintf(stderr, "Erro: leitura incompleta de '%s'\n", path);
        free(A);
        A = NULL;
    }
    fclose(f);
    return A;
}

/* ---------------------------------------------------------------------- */
/* Escrita da matriz (N*N doubles, row-major, sem cabecalho)               */
/* ---------------------------------------------------------------------- */
static int write_matrix(const char *path, int N, const double *A)
{
    FILE *f = fopen(path, "wb");
    if (!f)
    {
        fprintf(stderr, "Erro: nao foi possivel criar '%s'\n", path);
        return 0;
    }
    size_t expected = (size_t)N * N;
    size_t written = fwrite(A, sizeof(double), expected, f);
    fclose(f);
    return written == expected;
}

/* ---------------------------------------------------------------------- */
/* Monta a matriz aumentada [A | I] de tamanho N x 2N                      */
/* Layout: M[i * (2N) + j], row-major                                      */
/* ---------------------------------------------------------------------- */
static double *build_augmented(int N, const double *A)
{
    int W = 2 * N;
    double *M = (double *)calloc((size_t)N * W, sizeof(double));
    if (!M)
        return NULL;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            M[(size_t)i * W + j] = A[(size_t)i * N + j];
        }
        M[(size_t)i * W + (N + i)] = 1.0; /* identidade na metade direita */
    }
    return M;
}

/* ---------------------------------------------------------------------- */
/* Pivotamento parcial (serial nas duas orientacoes -- custo desprezivel). */
/* ---------------------------------------------------------------------- */
static void partial_pivot(double *M, int N, int W, int k)
{
    int max_row = k;
    double max_val = fabs(M[(size_t)k * W + k]);

    for (int i = k + 1; i < N; i++)
    {
        double v = fabs(M[(size_t)i * W + k]);
        if (v > max_val)
        {
            max_val = v;
            max_row = i;
        }
    }

    if (max_row != k)
    {
        for (int j = 0; j < W; j++)
        {
            double tmp = M[(size_t)k * W + j];
            M[(size_t)k * W + j] = M[(size_t)max_row * W + j];
            M[(size_t)max_row * W + j] = tmp;
        }
    }
}

/* ---------------------------------------------------------------------- */
/* HOTSPOT (PARALELO): eliminacao de Gauss-Jordan, ORIENTADA A LINHAS      */
/* Loop externo em i (linhas) paralelizado com OpenMP -- cada linha i e    */
/* atualizada de forma independente dentro da mesma iteracao k.           */
/* ---------------------------------------------------------------------- */
static void gauss_jordan_row_oriented(double *M, int N)
{
    int W = 2 * N;

    for (int k = 0; k < N; k++)
    {
        partial_pivot(M, N, W, k);

        double pivot = M[(size_t)k * W + k];
        for (int j = 0; j < W; j++)
        {
            M[(size_t)k * W + j] /= pivot;
        }

#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
        { /* <-- externo: LINHAS (paralelizado) */
            if (i == k)
                continue;
            double factor = M[(size_t)i * W + k];
            if (factor == 0.0)
                continue;
            for (int j = 0; j < W; j++)
            { /* <-- interno: COLUNAS (sequencial, dentro de cada thread) */
                M[(size_t)i * W + j] -= factor * M[(size_t)k * W + j];
            }
        }
    }
}

/* ---------------------------------------------------------------------- */
/* HOTSPOT (PARALELO): eliminacao de Gauss-Jordan, ORIENTADA A COLUNAS     */
/* Loop externo em j (colunas) paralelizado com OpenMP -- cada coluna j e  */
/* atualizada de forma independente (le apenas factors[] e M[k][j], ja     */
/* fixos; escreve em posicoes M[i][j] disjuntas entre threads).            */
/* ---------------------------------------------------------------------- */
static void gauss_jordan_col_oriented(double *M, int N)
{
    int W = 2 * N;

    for (int k = 0; k < N; k++)
    {
        partial_pivot(M, N, W, k);

        double pivot = M[(size_t)k * W + k];
        for (int j = 0; j < W; j++)
        {
            M[(size_t)k * W + j] /= pivot;
        }

        /* pre-calcula os fatores de eliminacao de cada linha (serial, O(N)) */
        double *factors = (double *)malloc((size_t)N * sizeof(double));
        for (int i = 0; i < N; i++)
        {
            factors[i] = (i == k) ? 0.0 : M[(size_t)i * W + k];
        }

#pragma omp parallel for schedule(static)
        for (int j = 0; j < W; j++)
        { /* <-- externo: COLUNAS (paralelizado) */
            double pivot_val = M[(size_t)k * W + j];
            for (int i = 0; i < N; i++)
            { /* <-- interno: LINHAS (stride grande, dentro de cada thread) */
                if (i == k)
                    continue;
                if (factors[i] == 0.0)
                    continue;
                M[(size_t)i * W + j] -= factors[i] * pivot_val;
            }
        }

        free(factors);
    }
}

/* ---------------------------------------------------------------------- */
/* Extrai a metade direita da matriz aumentada (a inversa calculada)       */
/* ---------------------------------------------------------------------- */
static double *extract_inverse(const double *M, int N)
{
    int W = 2 * N;
    double *Inv = (double *)malloc((size_t)N * N * sizeof(double));
    if (!Inv)
        return NULL;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            Inv[(size_t)i * N + j] = M[(size_t)i * W + (N + j)];
        }
    }
    return Inv;
}

/* ---------------------------------------------------------------------- */
/* Verificacao opcional de corretude: computa ||A*Ainv - I||_F             */
/* (norma de Frobenius do residuo). Serial -- usada apenas com --verify.   */
/* ---------------------------------------------------------------------- */
static double verify_residual(const double *A, const double *Inv, int N)
{
    double residual_sq = 0.0;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double sum = 0.0;
            for (int p = 0; p < N; p++)
            {
                sum += A[(size_t)i * N + p] * Inv[(size_t)p * N + j];
            }
            double expected = (i == j) ? 1.0 : 0.0;
            double diff = sum - expected;
            residual_sq += diff * diff;
        }
    }
    return sqrt(residual_sq);
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        fprintf(stderr,
                "Uso: %s <N> <matriz.bin> <linha|coluna> [inversa_saida.bin] [stats.txt] [--verify]\n",
                argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    const char *matrix_path = argv[2];
    const char *orientacao = argv[3];

    const char *out_path = NULL;
    const char *stats_path = NULL;
    int do_verify = 0;

    for (int i = 4; i < argc; i++)
    {
        if (strcmp(argv[i], "--verify") == 0)
        {
            do_verify = 1;
        }
        else if (!out_path)
        {
            out_path = argv[i];
        }
        else if (!stats_path)
        {
            stats_path = argv[i];
        }
    }

    if (N <= 0)
    {
        fprintf(stderr, "Erro: N deve ser positivo\n");
        return 1;
    }
    if (strcmp(orientacao, "linha") != 0 && strcmp(orientacao, "coluna") != 0)
    {
        fprintf(stderr, "Erro: orientacao deve ser 'linha' ou 'coluna'\n");
        return 1;
    }

    int nthreads = omp_get_max_threads();

    struct timespec t_prog_start, t_prog_end;
    struct timespec t_read_start, t_read_end;
    struct timespec t_setup_start, t_setup_end;
    struct timespec t_elim_start, t_elim_end;
    struct timespec t_write_start, t_write_end;

    clock_gettime(CLOCK_MONOTONIC, &t_prog_start);

    /* ---- 1. Leitura da matriz ---- */
    clock_gettime(CLOCK_MONOTONIC, &t_read_start);
    double *A = read_matrix(matrix_path, N);
    if (!A)
        return 1;
    clock_gettime(CLOCK_MONOTONIC, &t_read_end);

    printf("Matriz %dx%d carregada de '%s' (orientacao: %s, threads: %d)\n",
           N, N, matrix_path, orientacao, nthreads);

    /* ---- 2. Montagem da matriz aumentada [A | I] ---- */
    clock_gettime(CLOCK_MONOTONIC, &t_setup_start);
    double *M = build_augmented(N, A);
    clock_gettime(CLOCK_MONOTONIC, &t_setup_end);

    if (!M)
    {
        fprintf(stderr, "Erro: falha ao alocar matriz aumentada (%dx%d)\n", N, 2 * N);
        free(A);
        return 1;
    }

    /* ---- 3. Eliminacao de Gauss-Jordan (paralela) ---- */
    clock_gettime(CLOCK_MONOTONIC, &t_elim_start);
    if (strcmp(orientacao, "linha") == 0)
    {
        gauss_jordan_row_oriented(M, N);
    }
    else
    {
        gauss_jordan_col_oriented(M, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &t_elim_end);

    /* ---- 4. Extracao e escrita da inversa ---- */
    double *Inv = extract_inverse(M, N);
    free(M);

    if (!Inv)
    {
        fprintf(stderr, "Erro: falha ao extrair a inversa\n");
        free(A);
        return 1;
    }

    double t_write_s = 0.0;
    if (out_path)
    {
        clock_gettime(CLOCK_MONOTONIC, &t_write_start);
        if (!write_matrix(out_path, N, Inv))
        {
            fprintf(stderr, "Aviso: falha ao gravar inversa em '%s'\n", out_path);
        }
        clock_gettime(CLOCK_MONOTONIC, &t_write_end);
        t_write_s = timespec_diff_sec(t_write_start, t_write_end);
    }

    /* ---- 5. Verificacao opcional de corretude ---- */
    double residual = -1.0;
    if (do_verify)
    {
        residual = verify_residual(A, Inv, N);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_prog_end);

    /* ---- 6. Relatorio ---- */
    double t_read_s = timespec_diff_sec(t_read_start, t_read_end);
    double t_setup_s = timespec_diff_sec(t_setup_start, t_setup_end);
    double t_elim_s = timespec_diff_sec(t_elim_start, t_elim_end);
    double t_total_s = timespec_diff_sec(t_prog_start, t_prog_end);

    printf("\n=== Medicao interna de tempo (clock_gettime, CLOCK_MONOTONIC) ===\n");
    printf("Leitura da matriz         : %.6f s\n", t_read_s);
    printf("Montagem matriz aumentada : %.6f s\n", t_setup_s);
    printf("Eliminacao Gauss-Jordan   : %.6f s  (%d threads)\n", t_elim_s, nthreads);

    if (out_path)
        printf("Escrita da inversa        : %.6f s\n", t_write_s);
    printf("TOTAL (programa)          : %.6f s\n", t_total_s);

    if (do_verify)
    {
        printf("\n=== Verificacao de corretude ===\n");
        printf("||A * A^-1 - I||_F = %.6e  (deve ser proximo de 0)\n", residual);
    }

    if (stats_path)
    {
        FILE *out = fopen(stats_path, "w");
        if (out)
        {
            fprintf(out, "versao=openmp\n");
            fprintf(out, "N=%d\n", N);
            fprintf(out, "orientacao=%s\n", orientacao);
            fprintf(out, "threads=%d\n", nthreads);
            fprintf(out, "tempo_leitura_s=%.6f\n", t_read_s);
            fprintf(out, "tempo_setup_s=%.6f\n", t_setup_s);
            fprintf(out, "tempo_eliminacao_s=%.6f\n", t_elim_s);
            fprintf(out, "tempo_escrita_s=%.6f\n", t_write_s);
            fprintf(out, "tempo_total_s=%.6f\n", t_total_s);
            if (do_verify)
                fprintf(out, "residual_frobenius=%.6e\n", residual);
            fclose(out);
            printf("\nEstatisticas gravadas em '%s'\n", stats_path);
        }
    }

    free(A);
    free(Inv);
    return 0;
}