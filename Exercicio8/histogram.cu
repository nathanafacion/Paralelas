#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
/*
Nome: Nathana Facion RA: 191079
Exercicio 8 - Histograma na GPU

Contem relatorio na parte de baixo.

*/

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255
#define THREADBLOCK 8
#define BLOCK 64
typedef struct {
	unsigned char red, green, blue;
} PPMPixel;

typedef struct {
	int x, y;
	PPMPixel *data;
} PPMImage;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


static PPMImage *readPPM(const char *filename) {
	char buff[16];
	PPMImage *img;
	FILE *fp;
	int c, rgb_comp_color;
	fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	if (!fgets(buff, sizeof(buff), fp)) {
		perror(filename);
		exit(1);
	}

	if (buff[0] != 'P' || buff[1] != '6') {
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	img = (PPMImage *) malloc(sizeof(PPMImage));
	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	c = getc(fp);
	while (c == '#') {
		while (getc(fp) != '\n')
			;
		c = getc(fp);
	}

	ungetc(c, fp);
	if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
				filename);
		exit(1);
	}

	if (rgb_comp_color != RGB_COMPONENT_COLOR) {
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}

	while (fgetc(fp) != '\n')
		;
	img->data = (PPMPixel*) malloc(img->x * img->y * sizeof(PPMPixel));

	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	fclose(fp);
	return img;
}

__global__ void HistogramAux(PPMPixel *img, int n, int colunas, int* h){ // Number coluna 
    int x,y, k, i, j, count = 0;
    int l = threadIdx.x + blockDim.x*blockIdx.x; // linha
    int c = threadIdx.y + blockDim.y*blockIdx.y; // coluna
    i = l * colunas + c;
    if ((i < n)){
	for (j = 0; j <= 3; j++) {
		for (k = 0; k <= 3; k++) {
			for (l = 0; l <= 3; l++) {
				if (img[i].red == j && img[i].green == k && img[i].blue == l) {
					 x = j * 16 + k * 4 + l;
 					 atomicAdd(&(h[x]), 1);

				}
			}				
		}
	}
    }
}


void Histogram(PPMImage *image_host, float *hist) {
	int linhas, colunas;
	float n = image_host->y * image_host->x;
	colunas = image_host->x;
	linhas = image_host->y;
	PPMPixel *image_device;
	int *h_device, *h_host;
	int sizeImage = n*sizeof(PPMPixel);
	int i;
	float time_ms=0;

	for (i = 0; i < n; i++) {
		image_host->data[i].red = floor((image_host->data[i].red * 4) / 256);
		image_host->data[i].blue = floor((image_host->data[i].blue * 4) / 256);
		image_host->data[i].green = floor((image_host->data[i].green * 4) / 256);
	}

	h_host = (int *)malloc(BLOCK*sizeof(int));
	cudaEvent_t  init, stop;
    	cudaEventCreate(&init);
    	cudaEventCreate(&stop);

	// ------------------------ buffer --------------------------------	
	cudaEventRecord(init);

	cudaMalloc(&image_device,sizeImage);
	cudaMalloc(&h_device, sizeof(int)*BLOCK);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_ms, init, stop);
	// -------------------------host -> device ------------------------------

	cudaEventRecord(init);
	cudaMemcpy(image_device, image_host->data, sizeImage,cudaMemcpyHostToDevice);
	cudaMemset(h_device, 0, sizeof(int)*BLOCK);
    	cudaEventRecord(stop);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&time_ms, init, stop);

        dim3 threadPorBloco(THREADBLOCK, THREADBLOCK);
        dim3 numeroBlocos( (int)ceil((float)linhas/threadPorBloco.x), (int)ceil((float)colunas/threadPorBloco.y) );
	// --------------------- kernel --------------------------------------------
	cudaEventRecord(init);
        HistogramAux<<<numeroBlocos,threadPorBloco>>>(image_device, n, image_host->x, h_device);   
    	cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_ms, init, stop);
        // ----------------------------device -> host--------------------------------------
	cudaEventRecord(init);
        cudaMemcpy(h_host,h_device, sizeof(int)*BLOCK, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_ms, init, stop);

	for (i=0; i<BLOCK; i++){
		hist[i] = (float)h_host[i]/n;
	}

	free(h_host);
	cudaFree(h_device);
	cudaFree(image_device);

}

int main(int argc, char *argv[]) {

	if( argc != 2 ) {
		printf("Too many or no one arguments supplied.\n");
	}

	double t_start, t_end;
	int i;
	char *filename = argv[1]; //Recebendo o arquivo!;
	
	//scanf("%s", filename);
	PPMImage *image = readPPM(filename);

	float *h = (float*)malloc(sizeof(float) * 64);

	//Inicializar h
	for(i=0; i < BLOCK; i++) h[i] = 0.0;

	t_start = rtclock();
	Histogram(image, h);
	t_end = rtclock();

	for (i = 0; i < BLOCK; i++){
		printf("%0.3f ", h[i]);
	}
	printf("\n");
	fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);  
	free(h);
}


/************************  



    +-----------+----------+-----------+------------------+--------------------+------------------+------------------------+-----------+
    |  Arquivo  |   Serial | Buffer    |    host-> device |     Kernel         |  device->host    | Total GPU |  Paralelo  |Speedup    |
    |           |          |   ms      |       ms         |      ms            |       ms         |    ms     |            |P/S e S/GPU|
    +-----------+----------+-----------+------------------+--------------------+------------------+------------------------+-----------+
    | arq1.ppm  |   0.34  |   1.23    |        0.8722    |       1.890        |        0.031      |   4.02    |    0.34    | 1.0 / 85  |
    +-----------+----------+-----------+------------------+--------------------+------------------+------------------------+-----------+
    | arq2.ppm  |   0.59   |   1.27    |       1.35       |       3.320        |        0.037     |   5.97    |    0.38    | 1.5 / 98  |
    +-----------+----------+-----------+------------------+--------------------+------------------+------------------------+-----------+
    | arq3.ppm  |   1.80   |   1,2     |       4.48       |       12.46        |        0.03      |   18,17   |    0.55    |  3.2 /99  |
    +-----------+----------+-----------+------------------+--------------------+------------------+------------------------+-----------+


***************/
