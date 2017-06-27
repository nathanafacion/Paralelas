#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
/* Nome: Nathana Facion RA:191079 */
/* Exercicio 2 - Produtor / Consumidor */
/* Data: 16/03/2017 */
/* Para rodar o sequencial:  gcc -g -std=c99 -pedantic -Wall -fopenmp -lm -o constSerial T2_ser.c */
/* Para rodar o paralelo: gcc -g -std=c99 -pedantic -Wall -fopenmp -lm -o constParalelo  prod_cons_paralelo.c */
/* Resposta: ../prod cons < arqX.in> */


// Paralelo 
void producer_consumer(int *buffer, int size, int *vec, int n, int thread_count) {
	int i, j, k;
	long long unsigned int sum = 0;
 
	# pragma omp parallel num_threads(thread_count) 	private(i,j,k)
	for(i=0;i<n;i++) {
		if(i % 2 == 0) {	// PRODUTOR
			#	pragma omp for			
			for(j=0;j<size;j++) {
				buffer[j] = vec[i] + j*vec[i+1];
			}
		}
		else {	// CONSUMIDOR
			#	pragma omp for	reduction(+:sum)	
			for(k=0;k<size;k++) {
		
				sum += buffer[k];
			}

	
		}

	}
	printf("%llu\n",sum);
}


int main(int argc, char * argv[]) {
	
	// Declara variavel
	double start, end;
	int i, n, size, nt;
	int *buff;
	int *vec;

	// Le entrada
	scanf("%d %d %d",&nt,&n,&size);
	
	// Aloca memoria
	buff = (int *)malloc(size*sizeof(int));
	vec = (int *)malloc(n*sizeof(int));

	// Le vetor
	for(i=0;i<n;i++)
		scanf("%d",&vec[i]);
	
	// Le tempo inicial
	start = omp_get_wtime();
	producer_consumer(buff, size, vec, n,nt);
	// Le tempo final	
	end = omp_get_wtime();

	printf("%lf\n",end-start);

	free(buff);
	free(vec);

	return 0;
}


// Complementar

//Exercicio 1 
/*processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 60
model name	: Intel(R) Core(TM) i5-4210H CPU @ 2.90GHz
stepping	: 3
microcode	: 0x1c
cpu MHz		: 3324.804
cache size	: 3072 KB
physical id	: 0
siblings	: 4
core id		: 0
cpu cores	: 2
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm epb tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt dtherm ida arat pln pts
bugs		:
bogomips	: 5786.64
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 1
vendor_id	: GenuineIntel
cpu family	: 6
model		: 60
model name	: Intel(R) Core(TM) i5-4210H CPU @ 2.90GHz
stepping	: 3
microcode	: 0x1c
cpu MHz		: 3376.007
cache size	: 3072 KB
physical id	: 0
siblings	: 4
core id		: 1
cpu cores	: 2
apicid		: 2
initial apicid	: 2
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm epb tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt dtherm ida arat pln pts
bugs		:
bogomips	: 5786.64
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 2
vendor_id	: GenuineIntel
cpu family	: 6
model		: 60
model name	: Intel(R) Core(TM) i5-4210H CPU @ 2.90GHz
stepping	: 3
microcode	: 0x1c
cpu MHz		: 3196.003
cache size	: 3072 KB
physical id	: 0
siblings	: 4
core id		: 0
cpu cores	: 2
apicid		: 1
initial apicid	: 1
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm epb tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt dtherm ida arat pln pts
bugs		:
bogomips	: 5786.64
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 3
vendor_id	: GenuineIntel
cpu family	: 6
model		: 60
model name	: Intel(R) Core(TM) i5-4210H CPU @ 2.90GHz
stepping	: 3
microcode	: 0x1c
cpu MHz		: 3326.730
cache size	: 3072 KB
physical id	: 0
siblings	: 4
core id		: 1
cpu cores	: 2
apicid		: 3
initial apicid	: 3
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm epb tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt dtherm ida arat pln pts
bugs		:
bogomips	: 5786.64
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

*/


// Exercicio 2

/*

-------------------------------------------------------------------------------------
arq1 - serial

  %   cumulative   self              self     total           
 time   seconds   seconds    calls  Ts/call  Ts/call  name    
  0.00      0.00     0.00        1     0.00     0.00  producer_consumer

--------------------------------------------------------------------------------------
arq2 - serial

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ms/call  ms/call  name    
100.76      0.32     0.32        1   322.44   322.44  producer_consumer
---------------------------------------------------------------------------------------
arq3 - serial

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls   s/call   s/call  name    
100.76      2.53     2.53        1     2.53     2.53  producer_consumer

---------------------------------------------------------------------------------------
Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  Ts/call  Ts/call  name    
100.56      0.03     0.03                             main
  0.00      0.03     0.00        1     0.00     0.00  producer_consumer
*/

// Exercicio 3

/*
arq1 - sem flag
0.008809
arq1 - com O0 
0.004540
arq1 - com O1
0.001993
arq1 - com O2
0.000551
arq1 - com O3
0.000300
----------------------------------------------------------------------------------------
arq2 - sem flag
0.321574
arq2 - com O0 
0.359480
arq2 - com O1
0.055236
arq2 - com O2
0.057894
arq3 - com O3
0.024267

----------------------------------------------------------------------------------------
arq3 - sem flag
2.534562
arq1 - com O0 
2.513621
arq1 - com O1
0.471755
arq2 - com O2
0.485840
arq3 - com O3
0.240609



*/












