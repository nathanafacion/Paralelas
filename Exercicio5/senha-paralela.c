#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>
#include <inttypes.h>
#define bool int
#define true 1
#define false 0

/* Nome: Nathana Facion RA:191079 */
/* Exercicio 5 - Senha */
/* Data: 07/04/2017 */
/* Para rodar o sequencial:  gcc -g -Wall -o senha-serial senha-serial.c -lpthread*/
/* Para rodar o paralelo: gcc -g -Wall -o senha-paralelo senha-paralelo.c -lpthread*/
/* Resposta: ../pi < arqX.i */


FILE *popen(const char *command, const char *type);


char filename[100];
int nt;
bool notpassword = true;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void * quebra_senha(void* rank) {
    FILE * fp;
    char finalcmd[300] = "unzip -P%d -t %s 2>&1";
    char ret[200];
    char cmd[400];
    int i;
    int chunck =10000; // chunck pedido no enunciado
	int my_rank = (intptr_t)rank; // rank de cada thread
	/* valor inicial e final da thread */
    int  thread_init;
    int  thread_end; 
    int j = 0; // variavel auxiliar para pegar proximo trecho

    while(notpassword){     
	    thread_init =(j + my_rank)*chunck; // Pega posicao inicial por meio do identificado da thread
	    thread_end  = thread_init + chunck;  // Pega a posicao final por meio do identificador da thread 
        i=thread_init; 
        notpassword = true; // Verifica se o password foi encontrado
        while(i < thread_end && notpassword == true){   // Se encontrar a senha para     
	        sprintf((char*)&cmd, finalcmd, i, filename);
            fp = popen(cmd, "r");	
	        while (!feof(fp)) {
		        fgets((char*)&ret, 200, fp);
                if (strcasestr(ret, "ok") != NULL) {
			        printf("Senha:%d\n", i);
			        notpassword = false;
                    i = thread_end; // condicao para sair do while
                    break;
		        }
	        }
            i++;
	        pclose(fp); 
        }   
    j+=nt; // Pega o proximo trecho da verificacao para essa thread
    }      
    return NULL;
}



int main ()
{

   int size;
   double t_start, t_end;
   int errorCreate, errorJoin; // mensagens de erro   
   int i;
   scanf("%d", &nt);
   scanf("%s", filename);

   size = nt;
   pthread_t *thread = (pthread_t*)malloc(size*sizeof(pthread_t));
   t_start = rtclock();
    // Cria as threads
	for (i = 0; i < size; i++){
        // Cria as threads e verifica possivel erro na criacao
		if((errorCreate = pthread_create(&thread[i], NULL, quebra_senha, (void *)(intptr_t)i))) {
			 printf("Thread creation failed: %d\n", errorCreate);
		}
	}

	// Espera por outras threads
	for (i = 0; i < size; i++){
        // Espera as threads e verifica possivel erros
 		if((errorJoin = pthread_join(thread[i], NULL))) {
 			printf("Thread Join failed: %d\n", errorJoin);
		}
	}

    t_end = rtclock();
 
  fprintf(stdout, "%0.6lf\n", t_end - t_start);  
}


/*

-----------arq1---------
Serial:
Senha:10000
12.214114

Paralelo:
Senha:10000
0.004802
-----------arq2---------
Serial:
Senha:100000
138.643915

Paralelo:
Senha:100000
95.612194
-----------arq3---------
Serial
Senha:450000
613.580795

Paralelo
Senha:450000
476.034665
-----------arq4----------
Serial
Senha:310000
390.858381

Paralelo:
Senha:310000
177.627720

-----------arq5----------
Serial
Senha:65000
74.972662

Paralelo:
Senha:65000
32.214463
----------arq6---------
Serial
Senha:245999
307.760002

Paralelo:
Senha:245999
202.279669






*/
