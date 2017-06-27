/*
 *   MD5 Benchmark
 *   -------------
 *   File: md5_bmark.c
 *
 *   This is the main file for the md5 benchmark kernel. This benchmark was
 *   written as part of the StarBENCH benchmark suite at TU Berlin. It performs
 *   MD5 computation on a number of self-generated input buffers in parallel,
 *   automatically measuring execution time.
 *
 *   Copyright (C) 2011 Michael Andersch
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include "md5.h"
#include "md5_bmark.h"

typedef struct timeval timer;

#define TIME(x) gettimeofday(&x, NULL)

/* Function declarations */
int initialize(md5bench_t* args);
int finalize(md5bench_t* args);
void run(md5bench_t* args);
void process(uint8_t* in, uint8_t* out, int bufsize);
void listInputs();
long timediff(timer* starttime, timer* finishtime);
int nt;


// Input configurations
static data_t datasets[] = {
    {64, 512, 0},
    {64, 1024, 0},
    {64, 2048, 0},
    {64, 4096, 0},
    {128, 1024*512, 1},
    {128, 1024*1024, 1},
    {128, 1024*2048, 1},
    {128, 1024*4096, 1},
};

/*
 *   Function: initialize
 *   --------------------
 *   To initialize the benchmark parameters. Generates the input buffers from random data.
 */
int initialize(md5bench_t* args) {
    int index = args->input_set;
    if(index < 0 || index >= sizeof(datasets)/sizeof(datasets[0])) {
        fprintf(stderr, "Invalid input set specified! Clamping to set 0\n");
        index = 0;
    }

    args->numinputs = datasets[index].numbufs;
    args->size = datasets[index].bufsize;
    args->inputs = (uint8_t**)calloc(args->numinputs, sizeof(uint8_t*));
    args->out = (uint8_t*)calloc(args->numinputs, DIGEST_SIZE);
    if(args->inputs == NULL || args->out == NULL) {
        fprintf(stderr, "Memory Allocation Error\n");
        return -1;
    }

    //fprintf(stderr, "Reading input set: %d buffers, %d bytes per buffer\n", datasets[index].numbufs, datasets[index].bufsize);

    // Now the input buffers need to be generated, for replicability, use same seed
    srand(datasets[index].rseed);

    for(int i = 0; i < args->numinputs; i++) {
        args->inputs[i] = (uint8_t*)malloc(sizeof(uint8_t)*datasets[index].bufsize);
        uint8_t* p = args->inputs[i];
        if(p == NULL) {
            fprintf(stderr, "Memory Allocation Error\n");
            return -1;
        }
        for(int j = 0; j < datasets[index].bufsize; j++)
            *p++ = rand() % 255;
    }

    return 0;
}

/*
 *   Function: process
 *   -----------------
 *   Processes one input buffer, delivering the digest into out.
 */
void process(uint8_t* in, uint8_t* out, int bufsize) {
    MD5_CTX context;
    uint8_t digest[16];

    MD5_Init(&context);
    MD5_Update(&context, in, bufsize);
    MD5_Final(digest, &context);

    memcpy(out, digest, DIGEST_SIZE);
}

/*
 *   Function: run
 *   --------------------
 *   Main benchmarking function. If called, processes buffers with MD5
 *   until no more buffers available. The resulting message digests
 *   are written into consecutive locations in the preallocated output
 *   buffer.
 */
void run(md5bench_t* args) 
   
    {
    #pragma omp parallel num_threads(nt)
    #pragma omp single
    for(int i = 0; i < args->iterations; i++) {
        int buffers_to_process = args->numinputs;
        int next = 0;
        uint8_t** in = args->inputs;
        uint8_t* out = args->out;

        while(buffers_to_process > 0) {
            #pragma omp task 
            process(in[next], out+next*DIGEST_SIZE, args->size); // Neste trecho ocorre dependencia: dependencia(in: in[next]) dependencia(out: out)
            next++;
            buffers_to_process--;
        }
    }
}

/*
 *   Function: finalize
 *   ------------------
 *   Cleans up memory used by the benchmark for input and output buffers.
 */
int finalize(md5bench_t* args) {

    char buffer[64];
    int offset = 0;

    for(int i = 0; i < args->numinputs; i++) {
#ifdef DEBUG
        sprintf(buffer, "Buffer %d has checksum ", i);
        fwrite(buffer, sizeof(char), strlen(buffer)+1, stdout);
#endif

        for(int j = 0; j < DIGEST_SIZE*2; j+=2) {
            sprintf(buffer+j,   "%x", args->out[DIGEST_SIZE*i+j/2] & 0xf);
            sprintf(buffer+j+1, "%x", args->out[DIGEST_SIZE*i+j/2] & 0xf0);
        }
        buffer[32] = '\0';

#ifdef DEBUG            
        fwrite(buffer, sizeof(char), 32, stdout);
        fputc('\n', stdout);
#else
        printf("%s ", buffer);
#endif

    }
#ifndef DEBUG
    printf("\n");
#endif

    if(args->inputs) {
        for(int i = 0; i < args->numinputs; i++) {
            if(args->inputs[i])
                free(args->inputs[i]);
        }

        free(args->inputs);
    }

    if(args->out)
        free(args->out);

    return 0;
}


/*
 *   Function: timediff
 *   ------------------
 *   Compute the difference between timers starttime and finishtime in msecs.
 */
long timediff(timer* starttime, timer* finishtime)
{
    long msec;
    msec=(finishtime->tv_sec-starttime->tv_sec)*1000;
    msec+=(finishtime->tv_usec-starttime->tv_usec)/1000;
    return msec;
}

/** MAIN **/
int main(int argc, char** argv) {

    timer b_start, b_end;
    md5bench_t args;

    //Receber parâmetros
    scanf("%d", &nt);
    scanf("%d", &args.input_set);
    scanf("%d", &args.iterations);
    args.outflag = 1;


    // Parameter initialization
    if(initialize(&args)) {
        fprintf(stderr, "Initialization Error\n");
        exit(EXIT_FAILURE);
    }

    TIME(b_start);

    run(&args);

    TIME(b_end);

    // Free memory
    if(finalize(&args)) {
        fprintf(stderr, "Finalization Error\n");
        exit(EXIT_FAILURE);
    }


    double b_time = (double)timediff(&b_start, &b_end)/1000;

    printf("%.3f\n", b_time);

    return 0;
}


/*
--------arq1.in--------
Serial: 0.002
Paralelo: 0.001
SpeedUp: 2,00

--------arq2.in--------
Serial: 8.088
Paralelo: 2.571
SpeedUp: 3,2

--------arq3.in--------
Serial: 16.683
Paralelo: 5.535
SpeedUp: 3,1

Por meio do vtune podemos ver que a funcao run eh a que poderia ser alterada de forma a trazer ganho a aplicacao.
Com isso alteramos a run e obtivemos os resultados acima, por meio dos resultados podemos ver que o ganho realmente
foi significativo para o md5.

-------------------------************** SERIAL ************************* ---------------------------------------
// testando no arquivo 3 serial:
     

//./amplxe-cl -report summary  -result-dir serial -format text -report-output serial/serial_summary.txt


amplxe: Using result path `/opt/intel/vtune_amplifier_xe_2017.2.0.499904/bin64/p_res'
amplxe: Executing actions 20 % Resolving information for `libc-2.23.so'        
amplxe: Warning: Cannot locate debugging symbols for file `/opt/intel/vtune_amplifier_xe_2017.2.0.499904/bin64/amplxe-runss'.
amplxe: Executing actions 75 % Generating a report                             
Elapsed Time: 20.760s
    Clockticks: 71,673,500,000
    Instructions Retired: 128,020,500,000
    CPI Rate: 0.560
    MUX Reliability: 0.973
    Front-End Bound: 2.8% of Pipeline Slots
        Front-End Latency: 0.1% of Pipeline Slots
            ICache Misses: 0.1% of Clockticks
            ITLB Overhead: 0.0% of Clockticks
            Branch Resteers: 0.3% of Clockticks
            DSB Switches: 0.0% of Clockticks
            Length Changing Prefixes: 0.0% of Clockticks
            MS Switches: 1.6% of Clockticks
        Front-End Bandwidth: 2.8% of Pipeline Slots
            Front-End Bandwidth MITE: 0.4% of Clockticks
            Front-End Bandwidth DSB: 8.9% of Clockticks
            Front-End Bandwidth LSD: 0.0% of Clockticks
    Bad Speculation: 0.4% of Pipeline Slots
        Branch Mispredict: 0.4% of Pipeline Slots
        Machine Clears: 0.0% of Pipeline Slots
    Back-End Bound: 51.2% of Pipeline Slots
     | Identify slots where no uOps are delivered due to a lack of required
     | resources for accepting more uOps in the back-end of the pipeline. Back-
     | end metrics describe a portion of the pipeline where the out-of-order
     | scheduler dispatches ready uOps into their respective execution units,
     | and, once completed, these uOps get retired according to program order.
     | Stalls due to data-cache misses or stalls due to the overloaded divider
     | unit are examples of back-end bound issues.
     |
        Memory Bound: 2.6% of Pipeline Slots
            L1 Bound: 2.3% of Clockticks
                DTLB Overhead: 0.0% of Clockticks
                Loads Blocked by Store Forwarding: 0.0% of Clockticks
                Lock Latency: 0.0% of Clockticks
                Split Loads: 0.0% of Clockticks
                4K Aliasing: 6.5% of Clockticks
                FB Full: 0.0% of Clockticks
            L3 Bound: 0.0% of Clockticks
                Contested Accesses: 0.0% of Clockticks
                Data Sharing: 0.0% of Clockticks
                L3 Latency: 0.0% of Clockticks
                SQ Full: 0.0% of Clockticks
            DRAM Bound: 0.0% of Clockticks
                Memory Latency: 0.0% of Clockticks
                    LLC Miss: 0.0% of Clockticks
            Store Bound: 0.1% of Clockticks
                Store Latency: 80.3% of Clockticks
                False Sharing: 0.0% of Clockticks
                Split Stores: 0.0% of Clockticks
                DTLB Store Overhead: 0.0% of Clockticks
        Core Bound: 48.6% of Pipeline Slots
         | This metric represents how much Core non-memory issues were of a
         | bottleneck. Shortage in hardware compute resources, or dependencies
         | software's instructions are both categorized under Core Bound. Hence
         | it may indicate the machine ran out of an OOO resources, certain
         | execution units are overloaded or dependencies in program's data- or
         | instruction- flow are limiting the performance (e.g. FP-chained long-
         | latency arithmetic operations).
         |
            Divider: 0.0% of Clockticks
            Port Utilization: 48.7% of Clockticks
             | This metric represents a fraction of cycles during which an
             | application was stalled due to Core non-divider-related issues.
             | For example, heavy data-dependency between nearby instructions,
             | or a sequence of instructions that overloads specific ports.
             | Hint: Loop Vectorization - most compilers feature auto-
             | Vectorization options today - reduces pressure on the execution
             | ports as multiple elements are calculated with same uop.
             |
                Cycles of 0 Ports Utilized: 3.1% of Clockticks
                Cycles of 1 Port Utilized: 48.1% of Clockticks
                 | This metric represents cycles fraction where the CPU executed
                 | total of 1 uop per cycle on all execution ports. This can be
                 | due to heavy data-dependency among software instructions, or
                 | oversubscribing a particular hardware resource. In some other
                 | cases with high 1_Port_Utilized and L1 Bound, this metric can
                 | point to L1 data-cache latency bottleneck that may not
                 | necessarily manifest with complete execution starvation (due
                 | to the short L1 latency e.g. walking a linked list) - looking
                 | at the assembly can be helpful. Note that this metric value
                 | may be highlighted due to L1 Bound issue.
                 |
                Cycles of 2 Ports Utilized: 24.4% of Clockticks
                 | This metric represents cycles fraction CPU executed total of
                 | 2 uops per cycle on all execution ports. Tip: Loop
                 | Vectorization - most compilers feature auto-Vectorization
                 | options today- reduces pressure on the execution ports as
                 | multiple elements are calculated with same uop.
                 |
                Cycles of 3+ Ports Utilized: 21.5% of Clockticks
                    Port 0: 33.5% of Clockticks
                    Port 1: 36.9% of Clockticks
                    Port 2: 12.1% of Clockticks
                    Port 3: 13.0% of Clockticks
                    Port 4: 7.1% of Clockticks
                    Port 5: 31.7% of Clockticks
                    Port 6: 37.3% of Clockticks
                    Port 7: 4.6% of Clockticks
    Retiring: 45.5% of Pipeline Slots
        General Retirement: 45.3% of Pipeline Slots
        Microcode Sequencer: 0.3% of Pipeline Slots
            Assists: 0.0% of Clockticks
    Total Thread Count: 1
    Paused Time: 0s
Collection and Platform Info
    Application Command Line: ./md5 
    User Name: root
    Operating System: 4.4.0-64-generic NAME="elementary OS" VERSION="0.4 Loki" ID="elementary" ID_LIKE=ubuntu PRETTY_NAME="elementary OS 0.4 Loki" VERSION_ID="0.4" HOME_URL="http://elementary.io/" SUPPORT_URL="http://elementary.io/support/" BUG_REPORT_URL="https://bugs.launchpad.net/elementary/+filebug" VERSION_CODENAME=loki UBUNTU_CODENAME=loki
    Computer Name: nathana-GE60-2PL
    Result Size: 35 MB 
    Collection start time: 14:28:49 13/04/2017 UTC
    Collection stop time: 14:29:10 13/04/2017 UTC
    CPU
        Name: Intel(R) Core(TM) Processor code named Haswell
        Frequency: 2.893 GHz
        Logical CPU Count: 4
      


 ./amplxe-cl -report gprof-cc -result-dir serial -format text -report-output serial/serial_gprof.txt
Index  % CPU Time:Total  CPU Time:Self  CPU Time:Children  Name                 Index
-----  ----------------  -------------  -----------------  -------------------  -----
                         0.0            21.090               _start             [2]  
[1]    100.0             0.0            21.090             __libc_start_main    [1]  
                         0.0            21.090               main               [3]  
                                                                                     
                                                             <spontaneous>           
[2]    100.0             0.0            21.090             _start               [2]  
                         0.0            21.090               __libc_start_main  [1]  
                                                                                     
                         0.0            21.090               __libc_start_main  [1]  
[3]    100.0             0.0            21.090             main                 [3]  
                         1.800          2.712                initialize         [8]  
                         0.0            0.008                finalize           [12] 
                         0.0            16.570               run                [7]  
                                                                                     
                         0.0            16.570               process            [6]  
[4]    78.57             0.0            16.570             MD5_Update           [4]  
                         16.570         0.0                  body               [5]  
                                                                                     
                         16.570         0.0                  MD5_Update         [4]  
[5]    78.57             16.570         0.0                body                 [5]  
                                                                                     
                         0.0            16.570               run                [7]  
[6]    78.57             0.0            16.570             process              [6]  
                         0.0            16.570               MD5_Update         [4]  
                                                                                     
                         0.0            16.570               main               [3]  
[7]    78.57             0.0            16.570             run                  [7]  
                         0.0            16.570               process            [6]  
                                                                                     
                         1.800          2.712                main               [3]  
[8]    21.39             1.800          2.712              initialize           [8]  
                         2.596          0.0                  rand               [9]  
                         0.116          0.0                  func@0x4008c0      [10] 
                                                                                     
                         2.596          0.0                  initialize         [8]  
[9]    12.31             2.596          0.0                rand                 [9]  
                                                                                     
                         0.116          0.0                  initialize         [8]  
[10]   0.55              0.116          0.0                func@0x4008c0        [10] 
                                                                                     
                         0.008          0.0                  sprintf            [13] 
[11]   0.04              0.008          0.0                ___sprintf_chk       [11] 
                                                                                     
                         0.0            0.008                main               [3]  
[12]   0.04              0.0            0.008              finalize             [12] 
                         0.0            0.008                sprintf            [13] 
                                                                                     
                         0.0            0.008                finalize           [12] 
[13]   0.04              0.0            0.008              sprintf              [13] 
                         0.008          0.0                  ___sprintf_chk     [11] 
                                                                                     


Index by function name

Index  Function         
-----  -----------------
[4]    MD5_Update       
[11]   ___sprintf_chk   
[1]    __libc_start_main
[2]    _start           
[5]    body             
[12]   finalize         
[10]   func@0x4008c0    
[8]    initialize       
[3]    main             
[6]    process          
[9]    rand             
[7]    run              
[13]   sprintf          
-------------------------************** PARALELO ************************* ---------------------------------------
./amplxe-cl -report summary  -result-dir paralelo_res -format text -report-output paralelo_res/paralelo_summary.txt
// Testado no arquivo 3 pois eh o que da mais diferenca de desempenho
Elapsed Time: 10.050s
    Clockticks: 84,152,200,000
    Instructions Retired: 128,098,800,000
    CPI Rate: 0.657
    MUX Reliability: 0.980
    Front-End Bound: 7.1% of Pipeline Slots
        Front-End Latency: 3.7% of Pipeline Slots
            ICache Misses: 0.2% of Clockticks
            ITLB Overhead: 0.0% of Clockticks
            Branch Resteers: 0.3% of Clockticks
            DSB Switches: 0.3% of Clockticks
            Length Changing Prefixes: 0.0% of Clockticks
            MS Switches: 1.4% of Clockticks
        Front-End Bandwidth: 3.4% of Pipeline Slots
            Front-End Bandwidth MITE: 5.2% of Clockticks
            Front-End Bandwidth DSB: 7.5% of Clockticks
            Front-End Bandwidth LSD: 0.0% of Clockticks
    Bad Speculation: 0.4% of Pipeline Slots
        Branch Mispredict: 0.4% of Pipeline Slots
        Machine Clears: 0.0% of Pipeline Slots
    Back-End Bound: 54.8% of Pipeline Slots
     | Identify slots where no uOps are delivered due to a lack of required
     | resources for accepting more uOps in the back-end of the pipeline. Back-
     | end metrics describe a portion of the pipeline where the out-of-order
     | scheduler dispatches ready uOps into their respective execution units,
     | and, once completed, these uOps get retired according to program order.
     | Stalls due to data-cache misses or stalls due to the overloaded divider
     | unit are examples of back-end bound issues.
     |
        Memory Bound: 22.4% of Pipeline Slots
         | The metric value is high. This can indicate that the significant
         | fraction of execution pipeline slots could be stalled due to demand
         | memory load and stores. Use Memory Access analysis to have the metric
         | breakdown by memory hierarchy, memory bandwidth information,
         | correlation by memory objects.
         |
            L1 Bound: 11.8% of Clockticks
             | This metric shows how often machine was stalled without missing
             | the L1 data cache. The L1 cache typically has the shortest
             | latency. However, in certain cases like loads blocked on older
             | stores, a load might suffer a high latency even though it is
             | being satisfied by the L1. Note that this metric value may be
             | highlighted due to DTLB Overhead or Cycles of 1 Port Utilized
             | issues.
             |
                DTLB Overhead: 0.0% of Clockticks
                Loads Blocked by Store Forwarding: 0.0% of Clockticks
                Lock Latency: 0.0% of Clockticks
                 | A significant fraction of CPU cycles spent handling cache
                 | misses due to lock operations. Due to the microarchitecture
                 | handling of locks, they are classified as L1 Bound regardless
                 | of what memory source satisfied them. Note that this metric
                 | value may be highlighted due to Store Latency issue.
                 |
                Split Loads: 0.0% of Clockticks
                4K Aliasing: 0.0% of Clockticks
                FB Full: 0.0% of Clockticks
                 | This metric does a rough estimation of how often L1D Fill
                 | Buffer unavailability limited additional L1D miss memory
                 | access requests to proceed. The higher the metric value, the
                 | deeper the memory hierarchy level the misses are satisfied
                 | from. Often it hints on approaching bandwidth limits (to L2
                 | cache, L3 cache or external memory). Avoid adding software
                 | prefetches if indeed memory BW limited.
                 |
            L3 Bound: 0.0% of Clockticks
                Contested Accesses
                Data Sharing
                L3 Latency
                SQ Full: 0.0% of Clockticks
            DRAM Bound: 0.0% of Clockticks
                Memory Latency: 0.0% of Clockticks
                    LLC Miss: 0.0% of Clockticks
            Store Bound: 0.1% of Clockticks
                Store Latency: 56.1% of Clockticks
                False Sharing: 0.0% of Clockticks
                Split Stores: 0.0% of Clockticks
                DTLB Store Overhead: 0.0% of Clockticks
        Core Bound: 32.4% of Pipeline Slots
         | This metric represents how much Core non-memory issues were of a
         | bottleneck. Shortage in hardware compute resources, or dependencies
         | software's instructions are both categorized under Core Bound. Hence
         | it may indicate the machine ran out of an OOO resources, certain
         | execution units are overloaded or dependencies in program's data- or
         | instruction- flow are limiting the performance (e.g. FP-chained long-
         | latency arithmetic operations).
         |
            Divider: 0.0% of Clockticks
            Port Utilization: 19.9% of Clockticks
                Cycles of 0 Ports Utilized: 19.7% of Clockticks
                Cycles of 1 Port Utilized: 13.9% of Clockticks
                Cycles of 2 Ports Utilized: 30.2% of Clockticks
                Cycles of 3+ Ports Utilized: 43.8% of Clockticks
                    Port 0: 28.0% of Clockticks
                    Port 1: 34.0% of Clockticks
                    Port 2: 11.5% of Clockticks
                    Port 3: 11.6% of Clockticks
                    Port 4: 7.1% of Clockticks
                    Port 5: 26.5% of Clockticks
                    Port 6: 31.0% of Clockticks
                    Port 7: 4.6% of Clockticks
    Retiring: 37.7% of Pipeline Slots
        General Retirement: 37.5% of Pipeline Slots
        Microcode Sequencer: 0.2% of Pipeline Slots
            Assists: 0.0% of Clockticks
    Total Thread Count: 4
    Paused Time: 0s
Collection and Platform Info
    Application Command Line: ./md5_par 
    User Name: root
    Operating System: 4.4.0-64-generic NAME="elementary OS" VERSION="0.4 Loki" ID="elementary" ID_LIKE=ubuntu PRETTY_NAME="elementary OS 0.4 Loki" VERSION_ID="0.4" HOME_URL="http://elementary.io/" SUPPORT_URL="http://elementary.io/support/" BUG_REPORT_URL="https://bugs.launchpad.net/elementary/+filebug" VERSION_CODENAME=loki UBUNTU_CODENAME=loki
    Computer Name: nathana-GE60-2PL
    Result Size: 40 MB 
    Collection start time: 13:37:36 13/04/2017 UTC
    Collection stop time: 13:37:46 13/04/2017 UTC
    CPU
        Name: Intel(R) Core(TM) Processor code named Haswell
        Frequency: 2.893 GHz
        Logical CPU Count: 4


// ./amplxe-cl -report gprof-cc -result-dir paralelo_resArq31 -format text -report-output paralelo_resArq3/p_res_gprof.txt
ndex  % CPU Time:Total  CPU Time:Self  CPU Time:Children  Name                     Index
-----  ----------------  -------------  -----------------  -----------------------  -----
                                                             <spontaneous>               
[1]    81.05             19.959         0.0                body                     [1]  
                                                                                         
                                                             <spontaneous>               
[2]    7.02              1.730          0.0                initialize               [2]  
                                                                                         
                                                             <spontaneous>               
[3]    4.75              1.171          0.0                __random_r               [3]  
                                                                                         
                                                             <spontaneous>               
[4]    4.49              1.105          0.0                __random                 [4]  
                                                                                         
                                                             <spontaneous>               
[5]    1.22              0.300          0.0                rand                     [5]  
                                                                                         
                                                             <spontaneous>               
[6]    0.71              0.174          0.0                func@0x400a60            [6]  
                                                                                         
                                                             <spontaneous>               
[7]    0.22              0.054          0.0                __do_softirq             [7]  
                                                                                         
                                                             <spontaneous>               
[8]    0.11              0.026          0.0                clear_page_c_e           [8]  
                                                                                         
                                                             <spontaneous>               
[9]    0.07              0.016          0.0                __do_page_fault          [9]  
                                                                                         
                                                             <spontaneous>               
[10]   0.04              0.009          0.0                finish_task_switch       [10] 
                                                                                         
                                                             <spontaneous>               
[11]   0.04              0.009          0.0                func@0x11ad0             [11] 
                                                                                         
                                                             <spontaneous>               
[12]   0.03              0.007          0.0                get_page_from_freelist   [12] 
                                                                                         
                                                             <spontaneous>               
[13]   0.02              0.006          0.0                __lock_text_start        [13] 
                                                                                         
                                                             <spontaneous>               
[14]   0.02              0.006          0.0                exit_to_usermode_loop    [14] 
                                                                                         
                                                             <spontaneous>               
[15]   0.02              0.006          0.0                func@0x11c80             [15] 
                                                                                         
                                                             <spontaneous>               
[16]   0.02              0.005          0.0                free_hot_cold_page       [16] 
                                                                                         
                                                             <spontaneous>               
[17]   0.02              0.005          0.0                handle_mm_fault          [17] 
                                                                                         
                                                             <spontaneous>               
[18]   0.02              0.004          0.0                page_add_new_anon_rmap   [18] 
                                                                                         
                                                             <spontaneous>               
[19]   0.02              0.004          0.0                page_remove_rmap         [19] 
                                                                                         
                                                             <spontaneous>               
[20]   0.01              0.003          0.0                process                  [20] 
                                                                                         
                                                             <spontaneous>               
[21]   0.01              0.002          0.0                __GI_                    [21] 
                                                                                         
                                                             <spontaneous>               
[22]   0.01              0.002          0.0                anon_vma_prepare         [22] 
                                                                                         
                                                             <spontaneous>               
[23]   0.01              0.002          0.0                intel_pstate_timer_func  [23] 
                                                                                         
                                                             <spontaneous>               
[24]   0.01              0.002          0.0                run_timer_softirq        [24] 
                                                                                         
                                                             <spontaneous>               
[25]   0.01              0.002          0.0                unmap_page_range         [25] 
                                                                                         
                                                             <spontaneous>               
[26]   0.01              0.002          0.0                vmacache_find            [26] 
                                                                                         
                                                             <spontaneous>               
[27]   0.0               0.001          0.0                _IO_vfprintf_internal    [27] 
                                                                                         
                                                             <spontaneous>               
[28]   0.0               0.001          0.0                __alloc_pages_nodemask   [28] 
                                                                                         
                                                             <spontaneous>               
[29]   0.0               0.001          0.0                __lru_cache_add          [29] 
                                                                                         
                                                             <spontaneous>               
[30]   0.0               0.001          0.0                __mod_zone_page_state    [30] 
                                                                                         
                                                             <spontaneous>               
[31]   0.0               0.001          0.0                __schedule               [31] 
                                                                                         
                                                             <spontaneous>               
[32]   0.0               0.001          0.0                __zone_watermark_ok      [32] 
                                                                                         
                                                             <spontaneous>               
[33]   0.0               0.001          0.0                _raw_spin_lock           [33] 
                                                                                         
                                                             <spontaneous>               
[34]   0.0               0.001          0.0                down_read_trylock        [34] 
                                                                                         
                                                             <spontaneous>               
[35]   0.0               0.001          0.0                find_vma                 [35] 
                                                                                         
                                                             <spontaneous>               
[36]   0.0               0.001          0.0                inc_zone_page_state      [36] 
                                                                                         
                                                             <spontaneous>               
[37]   0.0               0.001          0.0                intel_pstate_set_pstate  [37] 
                                                                                         
                                                             <spontaneous>               
[38]   0.0               0.001          0.0                mem_cgroup_try_charge    [38] 
                                                                                         
                                                             <spontaneous>               
[39]   0.0               0.001          0.0                pfn_pte                  [39] 
                                                                                         
                                                             <spontaneous>               
[40]   0.0               0.001          0.0                policy_zonelist          [40] 
                                                                                         
                                                             <spontaneous>               
[41]   0.0               0.001          0.0                release_pages            [41] 
                                                                                         
                                                             <spontaneous>               
[42]   0.0               0.001          0.0                vma_adjust               [42] 
                                                                                         


Index by function name

Index  Function               
-----  -----------------------
[27]   _IO_vfprintf_internal  
[21]   __GI_                  
[28]   __alloc_pages_nodemask 
[9]    __do_page_fault        
[7]    __do_softirq           
[13]   __lock_text_start      
[29]   __lru_cache_add        
[30]   __mod_zone_page_state  
[4]    __random               
[3]    __random_r             
[31]   __schedule             
[32]   __zone_watermark_ok    
[33]   _raw_spin_lock         
[22]   anon_vma_prepare       
[1]    body                   
[8]    clear_page_c_e         
[34]   down_read_trylock      
[14]   exit_to_usermode_loop  
[35]   find_vma               
[10]   finish_task_switch     
[16]   free_hot_cold_page     
[11]   func@0x11ad0           
[15]   func@0x11c80           
[6]    func@0x400a60          
[12]   get_page_from_freelist 
[17]   handle_mm_fault        
[36]   inc_zone_page_state    
[2]    initialize             
[37]   intel_pstate_set_pstate
[23]   intel_pstate_timer_func
[38]   mem_cgroup_try_charge  
[18]   page_add_new_anon_rmap 
[19]   page_remove_rmap       
[39]   pfn_pte                
[40]   policy_zonelist        
[20]   process                
[5]    rand                   
[41]   release_pages          
[24]   run_timer_softirq      
[25]   unmap_page_range       
[42]   vma_adjust             
[26]   vmacache_find          


*/



