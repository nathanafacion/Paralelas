#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>
#ifdef MEDIUM
  #define N 2048
#elif LARGE
  #define N 4096
#elif EXTRALARGE
  #define N 8192
#endif
#define GPU 1

/*

Nathana Facion RA 191079
Exercicio 10 - Aclang

Caso sua resolucao de tela seja ruim para ver a tabela abaixo, acesse https://drive.google.com/file/d/0B5_XMklmZVg_X3M3TnEwVFZibVk/view?usp=sharing

                         +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
                         | offloading_read_only | offloading_read_only | offloading_read_only | offloading_read_write| offloading_read_write| execute_kernel|execute_kernel|read_buffer|read_buffer | 
-------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
vectorize e MEDIUM       |      2815776         |      2592            |       2304           |        2304          |      2336            |      2014528  |   979200     |  3904     |    3488    |
vectorize e LARGE        |     10119008         |      3968            |       3680           |        3840          |      3680            |      6330976  |  2002752     |  4672     |    4288    |  
vectorize e EXTRALARGE   |     36970752         |      6720            |       6432           |        6432          |      6432            |     20622176  |  4556128     |  6176     |    5792    |
none e MEDIUM            |      2710272         |      2624            |       2336           |        2336          |      2272            |      2035776  |   981632     |  3936     |    3456    |
none e LARGE             |     10082080         |      4096            |       3584           |        3616          |      3584            |      6427776  |  2011744     |  4544     |    4096    |
none e EXTRALARGE        |     39376704         |     16480            |       6432           |        6400          |      6400            |     20523296  |  4556000     |  6656     |    6176    |
tile e MEDIUM            |      2833888         |      2592            |       2304           |        2304          |      2304            |      2019584  |   976768     |  3904     |    3424    |
tile e LARGE             |     10253472         |      3904            |       3616           |        3648          |      3776            |      6333216  |  2005472     |  4544     |    4128    |
tile e EXTRALARGE        |     36511616         |      6752            |       6400           |        6592          |      6400            |     20521152  |  4563008     |  6240     |    5824    |
-------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Paralelo ( Rodando uma unica vez )
+---------------------------------------------+
         |-DMEDIUM |  -DLARGE  | -DEXTRALARGE |
+---------------------------------------------+
none     | 0.3096  |  0.3982   |  0.4129      |  
tile     |  0.3308 |  0.3312   |  0.3761      |
vectorize| 0.4306  |  0.4702   |  0.3729      |
-----------------------------------------------



Serial
+---------------------------------------------+
         |-DMEDIUM |  -DLARGE  | -DEXTRALARGE |
+---------------------------------------------+
none     | 0.0315  |   0.2085  |   0.8523     |     
tile     | 0.0297  |   0.1994  |   0.8711     |    
vectorize| 0.0308  |   0.2008  |   0.8611     |
-----------------------------------------------

Paralelo ( Rodando media)
+---------------------------------------------+
         |-DMEDIUM |  -DLARGE  | -DEXTRALARGE |
+---------------------------------------------+
none     | 0.0207  | 0.0318    | 0.0737       |  
tile     | 0.0206  | 0.0319    | 0.0752       |
vectorize| 0.0223  | 0.0327    | 0.0720       |
-----------------------------------------------

Speedup (Serial/Paralelo)
+---------------------------------------------+
         |-DMEDIUM |  -DLARGE  | -DEXTRALARGE |
+---------------------------------------------+
none     |  1,52   |   6,55    |      11,5    |
tile     |  1,44   |   6,25    |      11,5    |
vectorize|  1,38   |   6,14    |      11,9    |
 -----------------------------------------------



Resultados

     MEDIUM:
	 1433052672.0000  1434100096.0000
	 
	 LARGE:
	 11458845696.0000  11463036928.0000
	 
	 EXTRA LARGE:
	 91648991232.0000  91664662528.0000


Podemos concluir que quanto mais dados temos melhor eh o speedup. Podemos percber isso facilmente olhando MEDIUM, LARGE, DEXTRALARGE.
O mesmo ocorre com o offloading que aumenta conforme a quantidade de dados. No kernel e no read_buffer tambem temos as mesmas observacoes.

*/


double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void init_array(float *A,float *x1,float *x2,float *y1,float *y2){
  int i,j;
  for(i = 0 ; i < N ; i++){
    x1[i] = ((float)i)/N;
    x2[i] = ((float)i + 1)/N;
    y1[i] = ((float)i + 3)/N;
    y2[i] = ((float)i + 4)/N;
    for(j = 0 ; j < N ; j++)
      A[i*N + j] = ((float)i*j)/N;
  }
  return;
}

void runMvt(float *a,float *x1,float *x2,float *y1,float *y2){
  int i , j;

 #pragma omp target map(to: a[:N*N], y1[:N], y2[:N])  map(tofrom: x1[:N], x2[:N]) device(GPU)
 {
  #pragma omp parallel for collapse(1) 
  for(i = 0; i < N ; i++)
    for(j = 0 ; j < N ; j++)
      x1[i] += a[i*N + j] * y1[j];

  #pragma omp parallel for collapse(1)
  for(i = 0; i < N ; i++)
    for(j = 0 ; j < N ; j++)
      x2[i] += a[j*N + i] * y2[j];
  }

  return;
}

int main(){
  double t_start, t_end;

  float *A,*x1,*x2,*y1,*y2;
  A = (float*)malloc( N * N * sizeof(float) );
  x1 = (float*)malloc( N * sizeof(float) );
  x2 = (float*)malloc( N * sizeof(float) );
  y1 = (float*)malloc( N * sizeof(float) );
  y2 = (float*)malloc( N * sizeof(float) );

  init_array(A,x1,x2,y1,y2);

  t_start = rtclock();
  runMvt( A , x1 , x2 , y1 , y2 );
  t_end = rtclock();

  float m = 0 , n = 0;
  for(int i = 0 ; i < N ; i++)
    m += x1[i] , n += x2[i];

  fprintf(stdout, "%0.4lf  %0.4lf\n", m, n);
  fprintf(stdout, "%0.4lf\n", t_end - t_start);
}

