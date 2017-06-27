#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
/* Nome: Nathana Facion RA:191079 */
/* Exercicio 1 - Counting Sort */
/* Data: 08/03/2017 */
/* Para rodar o sequencial:  gcc -g -Wall -fopenmp -o T1-seq T1-seq.c */
/* Para rodar o paralelo: gcc -g -Wall -fopenmp -o T1-par T1-par.c */
/* Resposta: ./idade <arq1.in >minhaSaida1.res */

/* count sort parallel */
double count_sort_parallel(double a[], int n, int nt) {
	int i, j, count;
	double *temp;
	double start, end, duracao;
	int thread_count;

	temp = (double *)malloc(n*sizeof(double));

	/* quantidade de thread a ser usada */
	thread_count = nt;

	start = omp_get_wtime();

	/* parallel-for com suas variaveis privadas */
	#  pragma omp parallel for num_threads(thread_count) private(count,i,j)
	for (i = 0; i < n; i++) {
		count = 0;
		for (j = 0; j < n; j++)
			if (a[j] < a[i])
				count++;
			else if (a[j] == a[i] && j < i)
				count++;
		temp[count] = a[i];
	}
	end = omp_get_wtime();

	duracao = end - start;

	memcpy(a, temp, n*sizeof(double));
	free(temp);

	return duracao;
}

int main(int argc, char * argv[]) {
	int i, n, nt;
	double  * a, t_s;

	/* quantidade de threads */
	scanf("%d",&nt);
	
	/* numero de valores */
	scanf("%d",&n);

	/* aloca os vetores de valores para o teste em serial(b) e para o teste em paralelo(a) */
	a = (double *)malloc(n*sizeof(double));

	/* entrada dos valores */
	for(i=0;i<n;i++)
		scanf("%lf",&a[i]);
	
	/* chama as funcoes de count sort em paralelo e em serial */
	t_s = count_sort_parallel(a,n,nt);
	
	/* Imprime o vetor ordenado */
	for(i=0;i<n;i++)
		printf("%.2lf ",a[i]);

	printf("\n");

	/* imprime os tempos obtidos e o speedup */
	printf("%lf\n",t_s);

	return 0;
}
