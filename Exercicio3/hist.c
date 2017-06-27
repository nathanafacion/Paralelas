#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>
#include <inttypes.h>

/* Nome: Nathana Facion RA:191079 */
/* Exercicio 3 - Histograma */
/* Data: 23/03/2017 */
/* Para rodar o sequencial:  gcc -g -Wall -o hist_s hist_s.c -lpthread*/
/* Para rodar o paralelo: gcc -g -Wall -o hist_p hist_p.c -lpthread*/
/* Resposta: ../hist_p < arqX.in> */

// Torna variaveis global
double h, *val, max, min;
int n, nval, *vet, size, **vectorthread;

/* funcao que calcula o minimo valor em um vetor */
double min_val(double * vet,int nval) {
	int i;
	double min;

	min = FLT_MAX;

	for(i=0;i<nval;i++) {
		if(vet[i] < min)
			min =  vet[i];
	}
	
	return min;
}

/* funcao que calcula o maximo valor em um vetor */
double max_val(double * vet, int nval) {
	int i;
	double max;

	max = FLT_MIN;

	for(i=0;i<nval;i++) {
		if(vet[i] > max)
			max =  vet[i];
	}
	
	return max;
}


/* conta quantos valores no vetor estao entre o minimo e o maximo passados como parametros */
void * count(void *rank) {
	int count;
	double min_t, max_t;
	int my_rank = (intptr_t)rank;
	int  thread_init = floor(my_rank*nval/(float)size);
	int  thread_end  = floor((my_rank+1)*nval/(float)size);
	for(int j=0;j<n;j++) {
		count = 0;
		min_t = min + j*h;
		max_t = min + (j+1)*h;
		for(int i=thread_init;i<thread_end;i++) {
			if(val[i] <= max_t && val[i] > min_t) {
				count++;
			}
		}
		vectorthread[my_rank][j] = count;
	}

	return NULL;

}

int main(int argc, char * argv[]) {
	int i;
	long unsigned int duracao;
	struct timeval start, end;
	pthread_t *thread = (pthread_t*)malloc(size*sizeof(pthread_t));
 	int errorCreate, errorJoin; // mensagens de erro

	/* numeros de threads */
	scanf("%d",&size);

	/* entrada do numero de dados */
	scanf("%d",&nval);
	/* numero de barras do histograma a serem calculadas */
	scanf("%d",&n); // nbins

	/* vetor com os dados */
	val = (double *)malloc(nval*sizeof(double));
	vet = (int *)malloc(n*sizeof(int));

	/* entrada dos dados */
	for(i=0;i<nval;i++) {
		scanf("%lf",&val[i]);
	}

	/* calcula o minimo e o maximo valores inteiros */

	min = floor(min_val(val,nval));
	max = ceil(max_val(val,nval));

	/* calcula o tamanho de cada barra */
	h = (max - min)/n;

	/* alocando memoria para a matriz */
	vectorthread = (int**)malloc(size*sizeof(int*));

	for (i=0; i < size ; i++){
		/* aloca linha para cada vetor */
		vectorthread[i] =  (int*)malloc(n*sizeof(int));
  		for (int j = 0; j < n; j++){  //Percorre o Vetor de Inteiros atual.
            		vectorthread[i][j] = 0; //Inicializa com 0.
  
		}

	}
	gettimeofday(&start, NULL);

 	// Cria as threads
	for (i=0; i< (size); i++){
		if((errorCreate = pthread_create(&thread[i], NULL, count, (void *)(intptr_t)i))) {
			 printf("Thread creation failed: %d\n", errorCreate);
		}
	}

	// Espera por outras threads
	for (i=0; i< (size); i++){
 		if((errorJoin = pthread_join(thread[i], NULL))) {
 			printf("Thread creation failed: %d\n", errorJoin);
		}
	}

	// Junta histrograma
	for(int k=0; k<size; k++) {
        	for(int j=0; j<n; j++) {
            		vet[j] += vectorthread[k][j];
        	}

    	}


	gettimeofday(&end, NULL);

	duracao = ((end.tv_sec * 1000000 + end.tv_usec) - \
	(start.tv_sec * 1000000 + start.tv_usec));

	printf("%.2lf",min);	
	for(i=1;i<=n;i++) {
		printf(" %.2lf",min + h*i);
	}
	printf("\n");

	/* imprime o histograma calculado */	
	printf("%d",vet[0]);
	for(i=1;i<n;i++) {
		printf(" %d",vet[i]);
	}
	printf("\n");

	/* imprime o tempo de duracao do calculo */
	printf("%lu\n",duracao);

	free(vectorthread);
	free(vet);
	free(val);

	return 0;
}


/*  Tarefa Complementar

PS: Para arq1 , arq2, arq 3 com 1 thread o valor eh 1 para todos os casos abaixo.

 		----------------------------------------------------------
      |         | Threads    |    2   |    4   |    8   |   16   |
      |---------|------------|--------|--------|--------|--------|
      | arq1.in | Speedup    |  1.1   |  1.2   |   1.3  |  1.47  |
      |         |------------|--------|--------|--------|--------|
      |         | Eficiencia |  0.55  |  0.3   |  0.16  |  0.09  |
      |---------|------------|--------|--------|--------|--------|
      | arq2.in | Speedup    |  1.8   |  2.3   |  2.1   |  1.9   |
      |         |------------|--------|--------|--------|--------|
      |         | Eficiencia |  0.9   |  0.57  |  0.26  |  0.11  |
      |---------|------------|--------|--------|--------|--------|
      | arq3.in | Speedup    |  2.22  |  2.7   |  2.3   |  2.6   |
      |         |------------|--------|--------|--------|--------|
      |         | Eficiencia |  1.11  |  0.675 |  0.287 |  0.165 |
      ------------------------------------------------------------

*/
