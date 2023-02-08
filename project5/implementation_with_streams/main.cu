#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

void run_cpu_gray_test(PGM_IMG img_in, char *out_filename, int *bins, float *cdf_d, int *lut, unsigned char *out);



int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g;
    PGM_IMG img_obuf;

	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
	
    unsigned char *input, *output;
    unsigned char *d_input, *d_output;
    
    int *d_bins;
    int *bins;
	
    float *cdf;
	float *d_cdf;

    int *lut;
    int *d_lut;



    unsigned char *input0, *input1;
    


    int N_bins = 256;
    float kernel_time = 0;

    img_ibuf_g = read_pgm(argv[1]);
    img_obuf.img = (unsigned char *)malloc(img_ibuf_g.w * img_ibuf_g.h * sizeof(unsigned char));
    img_obuf.h = img_ibuf_g.h;
    img_obuf.w = img_ibuf_g.w;  
    
    
    int N = img_ibuf_g.h * img_ibuf_g.w;
    int div2 = N / 2;

    //printf("%d %d  %d\n",N, div2, N-div2);
    
    input0 = (unsigned char *)malloc(div2 * sizeof(unsigned char));
    input1 = (unsigned char *)malloc((N-div2) * sizeof(unsigned char)); 


    for (int i = 0; i < N; i++)
    {   
        if (i < div2)
            input0[i] = img_ibuf_g.img[i];
        else 
           input1[i] = img_ibuf_g.img[i]; 
    }

    // for (int i = div2; i < N; i++)
    // {
    //     input1[i] = img_ibuf_g.img[i];
    // }
    
    


    cudaEvent_t start_kernel, stop_kernel;
	cudaEventCreate(&start_kernel);
	cudaEventCreate(&stop_kernel);

    size_t bytes = img_ibuf_g.h * img_ibuf_g.w * sizeof(unsigned char);
    size_t bytes_bins = N_bins * sizeof(int);
	size_t bytes_bins_f = N_bins * sizeof(float);

    // Allocate host memory
    input = (unsigned char *) malloc(bytes);
    output = (unsigned char *) malloc(bytes);

    bins = (int *)malloc(bytes_bins);
    lut = (int *)malloc(bytes_bins);
    cdf = (float *)malloc(bytes_bins_f);



    // cudaMallocManaged(&input,bytes);
    // cudaMallocManaged(&bins, bytes_bins);

	// cudaMallocManaged(&cdf, bytes_bins_f);
	// cudaMallocManaged(&lut, bytes_bins);

    // cudaMallocManaged(&output, bytes);

    cudaMalloc((void **)&d_input, bytes);

    cudaMalloc((void **)&d_bins, bytes_bins);

    cudaMalloc((void **)&d_cdf, bytes_bins_f);

    cudaMalloc((void **)&d_lut, bytes_bins);

    cudaMalloc((void **)&d_output, bytes);

    printf("Running contrast enhancement for gray-scale images.\n\n");
    

    // Init input values
    for (int i = 0; i < N; i++) {
        input[i] = img_ibuf_g.img[i];
        output[i] = 0;
    }

    
    for (int i = 0; i < N_bins; i++) {
        bins[i] = 0;
        cdf[i] = 0.0;
        lut[i] = 0;
    }
    
 
    // Set grid dims
    int THREADS = 512;
    int gridDim = (N + 1) / THREADS;
    
    //briskoume min
	int min = 0;
    int i = 0;

    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_bins, bins, bytes_bins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cdf, cdf, bytes_bins_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lut, lut, bytes_bins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, bytes, cudaMemcpyHostToDevice);

    int *bins0;
	int *bins1;

    
	cudaMalloc((void **)&bins0, 256* sizeof(int));
	cudaMalloc((void **)&bins1, 256 * sizeof(int));

	unsigned char
		*d_Input0,
		*d_Input1;

    


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);




	//N * N * sizeof(unsigned char)/2 is half the image

    cudaMalloc((void **)&d_Input0, div2 * sizeof(unsigned char));
    cudaMalloc((void **)&d_Input1, (N -div2)* sizeof(unsigned char));

    cudaEventRecord(start_kernel);
    
	cudaMemcpyAsync(d_Input0, input, div2* sizeof(unsigned char), cudaMemcpyHostToDevice, stream0);
	cudaMemcpyAsync(d_Input1, input+div2* sizeof(unsigned char), (N -div2)* sizeof(unsigned char), cudaMemcpyHostToDevice, stream1);

    // cudaDeviceSynchronize();

    int gridDim0 = (div2+ 1) / THREADS;
    int gridDim1 = (N - div2+ 1) / THREADS;

	calcH << <gridDim0, THREADS, 0, stream0 >>> (d_Input0, bins0, div2);
	calcH << <gridDim1, THREADS, 0, stream1 >>> (d_Input1, bins1, N-div2);
    cudaDeviceSynchronize();

	//a8roisma twn histogram
	addH<<<1, 256 >>>(bins0, bins1, d_bins);//kai gia grid megalutero apo 1x1 
	cudaDeviceSynchronize();


    createCDF<<<1, 256>>>(d_cdf, d_bins, N_bins);
    cudaDeviceSynchronize();

    // Get calculated bins from device and cdf
    cudaMemcpy(bins, d_bins, bytes_bins, cudaMemcpyDeviceToHost);
    
    while (min == 0) {
        min = bins[i++];
    }


	calcLut <<<1, 256 >> > (d_cdf, d_lut, min, N_bins, N-min);
    cudaDeviceSynchronize();
    
    equalizeIMG<<<gridDim, THREADS>>>(d_input, d_output, d_lut, N);
    cudaDeviceSynchronize();

    cudaMemcpy(img_obuf.img, d_output, bytes, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
    printf("kernel time in ms: %f  \n", kernel_time);    
    
    
    // run_cpu_gray_test(img_ibuf_g, argv[2], bins, cdf, lut, output);
    write_pgm(img_obuf, argv[2]);
    free_pgm(img_obuf);    
    

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bins);
    cudaFree(d_cdf);
    cudaFree(d_lut);
    free_pgm(img_ibuf_g);

    cudaDeviceReset();
	return 0;
}



void run_cpu_gray_test(PGM_IMG img_in, char *out_filename, int *bins, float *cdf, int *lut, unsigned char *out)
{
    unsigned int timer = 0;
    PGM_IMG img_obuf;
    
    
    printf("\nStarting CPU processing...\n");
    img_obuf = contrast_enhancement_g(img_in, bins, cdf, lut, out);
    write_pgm(img_obuf, out_filename);
    free_pgm(img_obuf);
}


PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}

