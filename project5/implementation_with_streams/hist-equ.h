#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    



PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

__global__ void addH(int * input0, int * input1, int * output);
__global__ void equalizeIMG(unsigned char * input, unsigned char * output, int *lut, int size);
__global__ void createCDF(float *cdf, int *bins, int size);
__global__ void calcLut(float* cdf, int* lut, int min, int N_bins, int diff);
__global__ void calcH(unsigned char *input, int *bins, int size);

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin, float *cdf_d, int *lut);

//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in, int *bins, float *cdf, int *lut, unsigned char *out);

#endif
