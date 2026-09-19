/*
 * ============================================================================
 * Gerador de matrizes NxN em arquivo binario (linha por linha)
 *
 * Gera uma matriz quadrada de N x N elementos double, armazenada em ordem
 * row-major (linha por linha), sem cabecalho -- apenas N*N doubles crus,
 * conforme exigido no requisito 1.d do trabalho.
 *
 * A matriz e' gerada com dominancia diagonal estrita:
 *     A[i][i] > soma(|A[i][j]|, j != i)
 * o que garante que a matriz e' sempre invertivel (nao-singular) e bem
 * condicionada, evitando problemas de estabilidade numerica na eliminacao
 * de Gauss-Jordan sem a necessidade de pivotamento especial.
 *
 * Compilacao:
 *   gcc -O2 -o gerador_matriz gerador_matriz.c -lm
 *
 * Uso:
 *   ./gerador_matriz <N> <arquivo_saida.bin> [seed]
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <N> <arquivo_saida.bin> [seed]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    const char *out_path = argv[2];
    unsigned int seed = (argc >= 4) ? (unsigned int)atoi(argv[3]) : 42u;

    if (N <= 0) {
        fprintf(stderr, "Erro: N deve ser positivo\n");
        return 1;
    }

    double *A = (double *)malloc((size_t)N * N * sizeof(double));
    if (!A) {
        fprintf(stderr, "Erro: falha ao alocar matriz %dx%d\n", N, N);
        return 1;
    }

    srand(seed);

    /* Preenche off-diagonal com valores aleatorios em [-1, 1] */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                A[(size_t)i * N + j] = 0.0; /* preenchido depois */
            } else {
                double v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                A[(size_t)i * N + j] = v;
            }
        }
    }

    /* Dominancia diagonal estrita: garante matriz nao-singular */
    for (int i = 0; i < N; i++) {
        double soma_abs = 0.0;
        for (int j = 0; j < N; j++) {
            if (j != i) soma_abs += fabs(A[(size_t)i * N + j]);
        }
        double folga = 1.0 + ((double)rand() / RAND_MAX) * 4.0; /* [1,5) */
        A[(size_t)i * N + i] = soma_abs + folga;
    }

    FILE *f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "Erro ao criar '%s'\n", out_path);
        free(A);
        return 1;
    }

    size_t written = fwrite(A, sizeof(double), (size_t)N * N, f);
    fclose(f);
    free(A);

    if (written != (size_t)N * N) {
        fprintf(stderr, "Erro: escrita incompleta no arquivo\n");
        return 1;
    }

    printf("Matriz %dx%d gerada em '%s' (%.2f MB, seed=%u, diagonalmente dominante)\n",
           N, N, out_path, (N * (double)N * sizeof(double)) / (1024.0 * 1024.0), seed);

    return 0;
}
