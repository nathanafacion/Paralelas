#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N = 8

/* Nome: Nathana Facion RA:191079 */
/* Exercicio 7 - Matriz Add */
/* Data: 20/04/2017 */

__global__
void addMatriz(float *A,float *B, float *C, int linhas, int colunas ){
    int i = threadIdx.x + blockDim.x*blockIdx.x; // linha
    int j = threadIdx.y + blockDim.y*blockIdx.y; // coluna
    if (i < linhas && j < colunas){
	   C[i*colunas+j] = A[i*colunas+j] + B[i*colunas+j];
    }
}

int main()
{
    int *A, *B, *C;
    int *A, *B, *C;
    int i, j;

    // Declaracao do cuda	
    int *A_Cuda;
    int *B_Cuda;
    int *C_Cuda;
    //Input
    int linhas, colunas;

    const int size = linhas*colunas* sizeof(int);


    scanf("%d", &linhas);
    scanf("%d", &colunas);

    //Alocando memória na GPU
    A_Cuda = (int *)malloc(size);
    B_Cuda = (int *)malloc(size);
    C_Cuda = (int *)malloc(size);

    // Malloc para GPU
    cudaMalloc( (void**) & A_Cuda, size);
    cudaMalloc( (void**) & B_Cuda, size);
    cudaMalloc( (void**) & C_Cuda, size);

    //Inicializar
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            A[i*colunas+j] =  B[i*colunas+j] = i+j;
        }
    }


    // Copia para GPU
    cudaMemcpy(A, A_Cuca, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_Cuca, size, cudaMemcpyHostToDevice);
    cudaMemcpy(C, C_Cuca, size, cudaMemcpyHostToDevice);


    dim3 threadPorBloco(N, N);
    
    // O numero de blocos deve variar baseado na entrada 
    dim3 numeroBlocos( (int)ceil((float)linhas/threadPorBloco.x), (int)ceil((float)colunas/threadPorBloco.y) );
    addMatriz<<<numeroBlocos,threadPorBloco>>>(A_Cuda,B_Cuda,C_Cuda);   


    long long int somador=0;
    //Manter esta computação na CPU
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            somador+=C[i*colunas+j];   
        }
    }
    
    printf("%lli\n", somador);

    // Libera memoria da GPU
    cudaFree(A_Cuda);
    cudaFree(B_Cuda);
    cudaFree(C_Cuda);

    free(A);
    free(B);
    free(C);
}

