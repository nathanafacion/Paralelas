#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define MASK_WIDTH 5
#define WIDTH 32
#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

/*
Nome: Nathana Facion RA: 191079
Exercicio 9 - Smooth

Contem relatorio na parte de baixo.

*/

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

void writePPM(PPMImage *img) {

    fprintf(stdout, "P6\n");
    fprintf(stdout, "# %s\n", COMMENT);
    fprintf(stdout, "%d %d\n", img->x, img->y);
    fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

    fwrite(img->data, 3 * img->x, img->y, stdout);
    fclose(stdout);
}

/*
#define masksize (MASK_WIDTH*MASK_WIDTH)
#define blockwidth (TILE_WIDTH + (MASK_WIDTH-1))
#define maskradius ((MASK_WIDTH-1)/2)

*/


__global__ void SmoothGPU (PPMPixel *image, PPMPixel *image_copy, int linhas, int colunas) {
    int total_red =0;
    int total_blue =0;
    int total_green = 0;
   
    __shared__ PPMPixel image_share[(WIDTH + MASK_WIDTH -1)*(WIDTH + MASK_WIDTH -1)];
    
    int x,y, k,index,l, count = 0;
    int id_x = 	threadIdx.x;
    int id_y = threadIdx.y; 
    int block_y = blockDim.y;
    int block_x = blockDim.x;
    int MASK_WIDTH_2 =(MASK_WIDTH*MASK_WIDTH); 
    int i = id_x + WIDTH*blockIdx.x; // linha
    int j = id_y + WIDTH*blockIdx.y; // coluna
    i = i - ((MASK_WIDTH-1)/2);
    j = j - ((MASK_WIDTH-1)/2);

 	  if ((0 <= i) && (0 <= j) && (i < linhas) && (j < colunas)){
                image_share[id_y * block_y + id_x].red   = image_copy[j * linhas + i].red;
                image_share[id_y * block_y + id_x].green = image_copy[j * linhas + i].green;
                image_share[id_y * block_y + id_x].blue  = image_copy[j * linhas + i].blue;
          } else {
                image_share[id_y * block_y + id_x].red = image_share[id_y * block_y + id_x].green = image_share[id_y * block_y + id_x].blue  = 0;
	  }

     __syncthreads();
    

    if (id_x >=(MASK_WIDTH-1)/2 && (id_x < WIDTH + (MASK_WIDTH-1) - (MASK_WIDTH-1)/2) &&
	id_y >=(MASK_WIDTH-1)/2 && (id_y < WIDTH + (MASK_WIDTH-1) - (MASK_WIDTH-1)/2)){
        for (int k = -(MASK_WIDTH-1)/2; k <= (MASK_WIDTH-1)/2; k++){
            for (int l = -(MASK_WIDTH-1)/2; l <= (MASK_WIDTH-1)/2; l++)
            {
 	  	if ((0 <= i) && (0 <= j) && (i < linhas) && (j < colunas)){
                	total_red += image_share[(id_y * block_y + id_x) + (k* block_x) + l].red;
                	total_green += image_share[(id_y * block_y + id_x) + (k* block_x) + l].green;
                	total_blue += image_share[(id_y * block_y + id_x) + (k* block_x) + l].blue;
		}
            }
	}

        image[j * linhas + i].red   = total_red / MASK_WIDTH_2;
        image[j * linhas + i].green = total_green / MASK_WIDTH_2;
        image[j * linhas + i].blue  = total_blue / MASK_WIDTH_2;
    }
}

void SmoothAux(PPMImage *in, PPMImage *out) {
    
    PPMPixel *in_image, *out_image;
    int linhas, colunas;
    colunas = in->x;
    linhas = in->y;
    int n = linhas * colunas;
    int sizeImage = n*sizeof(PPMPixel);
    double t_start, t_end;
    
    cudaMalloc((void**)&in_image,sizeImage);
    cudaMalloc((void**)&out_image, sizeImage);

    cudaMemcpy(in_image, in->data , sizeImage,cudaMemcpyHostToDevice);

    dim3 Grid((colunas -1)/(WIDTH+1), (linhas-1)/(WIDTH+1),1);
    dim3 numeroBlocos(WIDTH,WIDTH,1);

    t_start = rtclock();
    SmoothGPU<<<Grid,numeroBlocos>>>(in_image, out_image, linhas, colunas);
    t_end = rtclock();

    fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);     
    // Nao sei se precisa multiplicar pelo bloco
    cudaMemcpy(out->data, out_image, sizeImage, cudaMemcpyDeviceToHost);
    cudaFree(in_image);
    cudaFree(out_image);

}

int main(int argc, char *argv[]) {

    if( argc != 2 ) {
        printf("Too many or no one arguments supplied.\n");
    }


    int i;
    char *filename = argv[1]; //Recebendo o arquivo!;

    PPMImage *image = readPPM(filename);
    PPMImage *image_output = readPPM(filename);
   
    SmoothAux(image, image_output);

    free(image);
    free(image_output);
}

