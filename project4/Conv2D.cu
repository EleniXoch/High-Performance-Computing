/*
* This sample implements a separable convolution
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


unsigned int filter_radius = 16;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.5 

//__constant__ double filterConst[65];

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter,
	int imageW, int imageH, int filterR) {

	int x, y, k;

	for (y = filterR; y < imageH + filterR; y++) {
		for (x = filterR; x < imageH + filterR; x++) {
			double sum = 0;

			for (k = -filterR; k <= filterR; k++) {
				int d = x + k;

				sum += h_Src[y * (imageW + 2 * filterR) + d] * h_Filter[filterR - k];

			}
			h_Dst[y * (imageW + 2 * filterR) + x] = sum;
		}

	}

}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
	int imageW, int imageH, int filterR) {

	int x, y, k;

	for (y = filterR; y < imageH + filterR; y++) {
		for (x = filterR; x < imageH + filterR; x++) {
			double sum = 0;

			for (k = -filterR; k <= filterR; k++) {
				int d = y + k;
				sum += h_Src[d * (imageW + 2 * filterR) + x] * h_Filter[filterR - k];
			}
			h_Dst[y * (imageW + 2 * filterR) + x] = sum;
		}
	}

}

__global__ void RowGPU(double *d_Dst, const double * d_Src, double *d_Filter, int imageW, int imageH, int filterR) {


	int k;
	int indexX = threadIdx.x + blockDim.x * blockIdx.x;
	int indexY = threadIdx.y + blockDim.y * blockIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	int idx = indexY * (grid_width + 2 * filterR) + indexX;
	int padding_offset = (imageW + 2 * filterR)*filterR + filterR;
	int d;

	double sum = 0;
	for (k = -filterR; k <= filterR; k++) {
		d = indexX + k;
		sum += d_Src[indexY*(imageW + 2 * filterR) + d + padding_offset] * d_Filter[filterR - k];
	}
	d_Dst[idx + padding_offset] = sum;

}

__global__ void ColGPU(double *d_Dst, const double  *d_Src, double *d_Filter, int imageW, int imageH, int filterR) {


	int k;
	int indexX = threadIdx.x + blockDim.x * blockIdx.x;
	int indexY = threadIdx.y + blockDim.y * blockIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	int idx = indexY * (grid_width + 2 * filterR) + indexX;
	int padding_offset = (imageW + 2 * filterR)*filterR + filterR;
	int d;

	double sum = 0;
	for (k = -filterR; k <= filterR; k++) {
		d = indexY + k;
		sum += d_Src[d*(imageW + 2 * filterR) + indexX + padding_offset] * d_Filter[filterR - k];
	}
	d_Dst[idx + padding_offset] = sum;

}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

	double
		*h_Filter,
		*h_Input,
		*h_Buffer,
		*h_OutputCPU,
		*h_OutputGPU;

	double
        *d_Filter,
		*d_Input,
		*d_Buffer,
		*d_OutputGPU;


	int imageW;
	int imageH;
	int padding;
	int dim_padding;
	unsigned int i;

	struct timespec tv1, tv2;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	padding = 2 * filter_radius;

	printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
	scanf("%d", &imageW);
	//imageW = 8192;
	imageH = imageW;
	dim_padding = imageW + padding;

	printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
	printf("Allocating and initializing host arrays...\n");
	// Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
	h_Filter = (double *)malloc(FILTER_LENGTH * sizeof(double));
	h_Input = (double *)malloc(dim_padding * dim_padding * sizeof(double));
	h_Buffer = (double *)malloc(dim_padding * dim_padding * sizeof(double));
	h_OutputCPU = (double *)malloc(dim_padding * dim_padding * sizeof(double));
	h_OutputGPU = (double *)malloc(dim_padding * dim_padding * sizeof(double));

	// Allocate memory for the device
	cudaError_t mallocErr1 = cudaMalloc((void **)&d_Filter, FILTER_LENGTH * sizeof(double));
	cudaError_t mallocErr2 = cudaMalloc((void **)&d_Input, dim_padding * dim_padding * sizeof(double));
	cudaError_t mallocErr3 = cudaMalloc((void **)&d_Buffer, dim_padding * dim_padding * sizeof(double));
	cudaError_t mallocErr4 = cudaMalloc((void **)&d_OutputGPU, dim_padding * dim_padding * sizeof(double));


	if (!h_Filter || !h_Input || !h_Buffer || !h_OutputCPU || !h_OutputGPU) {
		fprintf(stderr, "malloc error\n");
		exit(1);
	}

	if (mallocErr2 != cudaSuccess ||
		mallocErr3 != cudaSuccess || mallocErr4 != cudaSuccess) {
		fprintf(stderr, "cudaMalloc error\n");
		exit(1);
	}


	// to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
	// arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
	// to convolution kai arxikopoieitai kai auth tuxaia.

	srand(200);

	for (i = 0; i < FILTER_LENGTH; i++) {
		h_Filter[i] = (double)(rand() % 16);
		//filterConst[i] = h_Filter[i];
	}

	//Initialize padding with zeros
	for (int i = 0; i < dim_padding; i++) {
		for (int j = 0; j < dim_padding; j++) {
			if (i < filter_radius || i > imageW + filter_radius - 1 || j < filter_radius || j > filter_radius + imageW - 1) {
				h_Input[i + j * dim_padding] = 0;
				h_Buffer[i + j * dim_padding] = 0;
				h_OutputCPU[i + j * dim_padding] = 0;
			}
			else {
				h_Input[i + j * dim_padding] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
			}
		}
	}


	// To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
	printf("CPU computation...\n");

	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

	convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
	convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	
	printf("%g\n",
	(double)(tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
	(double)(tv2.tv_sec - tv1.tv_sec));




	// Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
	// pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  

	dim3 grid_dim;
	dim3 block_dim;

	if (imageW > 32) {
		block_dim.x = 32;
		block_dim.y = 32;

		grid_dim.x = imageW / block_dim.x;
		grid_dim.y = imageH / block_dim.y;
	}
	else {
		grid_dim.x = 1;
		grid_dim.y = 1;

		block_dim.x = imageW;
		block_dim.y = imageH;
	}

	printf("GPU computation...\n");

	//Start measuring execution time of the two kernels
	cudaEventRecord(start);
	cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Input, h_Input, dim_padding * dim_padding * sizeof(double), cudaMemcpyHostToDevice);

	RowGPU << <grid_dim, block_dim >> > (d_Buffer, d_Input, d_Filter ,imageW, imageH, filter_radius);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		printf("CUDA Error1: %s\n", cudaGetErrorString(err));
		exit(-1);
	}

	cudaDeviceSynchronize();

	ColGPU << <grid_dim, block_dim >> > (d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);

	if (err != cudaSuccess)
	{
		printf("CUDA Error2: %s\n", cudaGetErrorString(err));
		exit(-1);
	}


	cudaDeviceSynchronize();

	cudaMemcpy(h_OutputGPU, d_OutputGPU, dim_padding * dim_padding * sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop);


	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("kernel time in ms: %f\n", milliseconds);


	for (i = 0; i < imageW * imageH; i++) {
		if (ABS(h_OutputGPU[i] - h_OutputCPU[i]) >= accuracy) {
			printf("error\n");
			break;
		}
	}

	// free all the allocated memory
	free(h_OutputCPU);
	free(h_Buffer);
	free(h_Input);
	free(h_Filter);
	free(h_OutputGPU);

	cudaFree(d_Filter);
	cudaFree(d_Input);
	cudaFree(d_Buffer);
	cudaFree(d_OutputGPU);
	// Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
	cudaDeviceReset();


	return 0;
}
