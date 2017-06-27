#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N = 8

/* Nome: Nathana Facion RA:191079 */
/* Exercicio 7 - Matriz Add */
/* Data: 20/04/2017 */

__global__ void addMatriz(int *A,int *B, int *C, int linhas, int colunas ){
    int i = threadIdx.x + blockDim.x*blockIdx.x; // linha
    int j = threadIdx.y + blockDim.y*blockIdx.y; // coluna
    if ((i < linhas) && (j < colunas)){
	   C[i*colunas+j] = A[i*colunas+j] + B[i*colunas+j];
    }
}

int main()
{
    int *A, *B, *C;
    int i, j;

    // Declaracao do cuda	
    int *A_Cuda;
    int *B_Cuda;
    int *C_Cuda;
    //Input
    int linhas, colunas;
    scanf("%d", &linhas);
    scanf("%d", &colunas);

    size_t size = linhas*colunas* sizeof(int);


    //Alocando memória na CPU
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    // Malloc para GPU
    cudaMalloc(&A_Cuda, size);
    cudaMalloc(&B_Cuda, size);
    cudaMalloc(&C_Cuda, size);

    //Inicializar
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            A[i*colunas+j] =  B[i*colunas+j] = i+j;
        }
    }


    // Copia para GPU
    cudaMemcpy(A_Cuda, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_Cuda, B, size, cudaMemcpyHostToDevice);

    dim3 threadPorBloco(8, 8);
    
    // O numero de blocos deve variar baseado na entrada 
    dim3 numeroBlocos( (int)ceil((float)linhas/threadPorBloco.x), (int)ceil((float)colunas/threadPorBloco.y) );

    addMatriz<<<numeroBlocos,threadPorBloco>>>(A_Cuda,B_Cuda,C_Cuda,linhas,colunas);   
    
    cudaMemcpy(C, C_Cuda, size, cudaMemcpyDeviceToHost);


    long long int somador=0;
    //Manter esta computação na CPU
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            somador+=C[i*colunas+j];   
        }
    }
    
    printf("%lli\n", somador);

    free(A);
    free(B);
    free(C);

    // Libera memoria da GPU
    cudaFree(A_Cuda);
    cudaFree(B_Cuda);
    cudaFree(C_Cuda);
}

