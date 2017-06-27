#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>
#include <inttypes.h>

/* Nome: Nathana Facion RA:191079 */
/* Exercicio 4 - PI de Monte Carlo */
/* Data: 29/03/2017 */
/* Para rodar o sequencial:  gcc -g -Wall -o pi_s pi_s.c -lpthread*/
/* Para rodar o paralelo: gcc -g -Wall -o pi_p pi.c -lpthread*/
/* Resposta: ../pi < arqX.in> */

/* Variaveis globais */

extern int rand_r(unsigned int *seedp); 
unsigned int n;
int size,*vectorthread;

void * monte_carlo_pi(void* rank) {
	int my_rank = (intptr_t)rank;
	double x, y, d;
	/* valor inicial e final da thread */
	int  thread_init = floor(my_rank*n/(float)size);
	int  thread_end  = floor((my_rank+1)*n/(float)size);
	
	/* seed para random */
	unsigned int seed = time(NULL);
	for(int i=thread_init;i<thread_end;i++) {

		x = ((rand_r(&seed) % 1000000)/500000.0)-1;
		y = ((rand_r(&seed) % 1000000)/500000.0)-1;
		d = ((x*x) + (y*y));
		if (d <= 1){
		 vectorthread[my_rank] +=1; // conta o valor para cada thread
		}
	}

	return NULL;
}

int main(void) {
	long unsigned int duracao;
	long long unsigned int in = 0;
	double pi;
	struct timeval start, end;
	pthread_t *thread = (pthread_t*)malloc(size*sizeof(pthread_t));
	scanf("%d %u",&size, &n);
 	int errorCreate, errorJoin; // mensagens de erro
	int i;

	vectorthread = (int*)malloc(size*sizeof(int));

	/* inicializa o vetor */
	for (i = 0; i < size ; i++){
      	vectorthread[i] = 0; //Inicializa com 0.
 	}

	gettimeofday(&start, NULL);

	// Cria as threads
	for (i = 0; i < size; i++){
		if((errorCreate = pthread_create(&thread[i], NULL, monte_carlo_pi, (void *)(intptr_t)i))) {
			 printf("Thread creation failed: %d\n", errorCreate);
		}
	}

	// Espera por outras threads
	for (i = 0; i < size; i++){
 		if((errorJoin = pthread_join(thread[i], NULL))) {
 			printf("Thread creation failed: %d\n", errorJoin);
		}
	}

	// Junta valores calculados
	for (i = 0; i < size; i++){
		in += vectorthread[i];
	}

	gettimeofday(&end, NULL);

	duracao = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));

	pi = 4*in/((double)n);
	printf("%lf\n%lu\n",pi,duracao);

	/* libera vetor */
	free(vectorthread);

	return 0;
}
