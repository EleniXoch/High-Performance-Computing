/*
* This sample implements a separable convolution
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


unsigned int filter_radius = 32;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.5 
#define PICS_SIZE 1024
#define PICS_SIZE_PADDED (PICS_SIZE + 2 * filter_radius)
// #define CPU
__constant__ double filterConst[513];

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

__global__ void RowGPU(double *d_Dst, double * d_Src, int imageW, int imageH, int filterR) {


	register int k;
	int indexX = threadIdx.x + blockDim.x * blockIdx.x;
	int indexY = threadIdx.y + blockDim.y * blockIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	register int idx = indexY * (grid_width + 2 * filterR) + indexX;
	register int padding_offset = (imageW + 2 * filterR)*filterR + filterR;
	register int d;

	register double sum = 0;
	for (k = -filterR; k <= filterR; k++) {
		d = indexX + k;
		sum += d_Src[indexY*(imageW + 2 * filterR) + d + padding_offset] * filterConst[filterR - k];
	}
	d_Dst[idx + padding_offset] = sum;

}

__global__ void ColGPU(double *d_Dst, const double  *d_Src, int imageW, int imageH, int filterR) {


	register int k;
	int indexX = threadIdx.x + blockDim.x * blockIdx.x;
	int indexY = threadIdx.y + blockDim.y * blockIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	register int idx = indexY * (grid_width + 2 * filterR) + indexX;
	register int padding_offset = (imageW + 2 * filterR)*filterR + filterR;
	register int d;

	register double sum = 0;
	for (k = -filterR; k <= filterR; k++) {
		d = indexY + k;
		sum += d_Src[d*(imageW + 2 * filterR) + indexX + padding_offset] * filterConst[filterR - k];
	}
	d_Dst[idx + padding_offset] = sum;

}

void printArray(double *input, int size) {

	printf("The array is\n\n");
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("%lf ", input[i*size + j]);
		}
		printf("\n");
	}

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
		*h_OutputGPU,
		*h_OutputGPU_pic,
		*h_small_pic;

	double
		*d_Input,
		*d_Buffer,
		*d_OutputGPU_pic;


	int imageW;
	int imageH;
	int padding;
	int dim_padding;
	unsigned int i;

#ifdef CPU
	struct timespec tv1, tv2;
#endif
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	padding = 2 * filter_radius;

	printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
	scanf("%d", &imageW);
	imageH = imageW;
	dim_padding = imageW + padding;

	int runTimes = (imageH / PICS_SIZE) * (imageH / PICS_SIZE);
	int width = sqrt(runTimes);


	printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
	printf("Allocating and initializing host arrays...\n");
	// Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
	h_Filter = (double *)malloc(FILTER_LENGTH * sizeof(double));
	h_small_pic = (double *)calloc(PICS_SIZE_PADDED * PICS_SIZE_PADDED, sizeof(double));
	h_Input= (double *)calloc(dim_padding *dim_padding, sizeof(double));
	h_Buffer = (double *)calloc(dim_padding * dim_padding, sizeof(double));
	h_OutputCPU = (double *)calloc(dim_padding * dim_padding, sizeof(double));
	h_OutputGPU = (double *)calloc(dim_padding * dim_padding, sizeof(double));
	h_OutputGPU_pic = (double *)calloc(PICS_SIZE_PADDED * PICS_SIZE_PADDED, sizeof(double));

	// Allocate memory for the device
	cudaError_t mallocErr2 = cudaMalloc((void **)&d_Input, PICS_SIZE_PADDED * PICS_SIZE_PADDED * sizeof(double));
	cudaError_t mallocErr3 = cudaMalloc((void **)&d_Buffer, PICS_SIZE_PADDED * PICS_SIZE_PADDED * sizeof(double));
	cudaError_t mallocErr4 = cudaMalloc((void **)&d_OutputGPU_pic, PICS_SIZE_PADDED * PICS_SIZE_PADDED * sizeof(double));


	if (!h_Filter || !h_Input || !h_Buffer || !h_OutputCPU || !h_OutputGPU || !h_OutputGPU_pic) {
		fprintf(stderr, "malloc error\n");
		exit(1);
	}

	if (mallocErr2 != cudaSuccess ||
		mallocErr3 != cudaSuccess || mallocErr4 != cudaSuccess) {
		fprintf(stderr, "cudaMalloc error\n");
		exit(1);
	}



	srand(200);

	for (i = 0; i < FILTER_LENGTH; i++) {
		h_Filter[i] = (double)(rand() % 16);
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
#ifdef CPU
	printf("CPU computation...\n");
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
	convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
	convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	printf("%g\n",
		(double)(tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
		(double)(tv2.tv_sec - tv1.tv_sec));
#endif


	dim3 grid_dim;
	dim3 block_dim;

	if (PICS_SIZE > 32) {
		block_dim.x = 32;
		block_dim.y = 32;

		grid_dim.x = PICS_SIZE / block_dim.x;
		grid_dim.y = PICS_SIZE / block_dim.y;
	}
	else {
		grid_dim.x = 1;
		grid_dim.y = 1;

		block_dim.x = PICS_SIZE;
		block_dim.y = PICS_SIZE;
	}

	printf("GPU computation...\n");

	//Start measuring execution time of the two kernels
	cudaEventRecord(start);
	cudaMemcpyToSymbol(filterConst, h_Filter, FILTER_LENGTH * sizeof(double));



	for (int id = 0; id < runTimes; id++) {
		//printf("(id%%width) %d (id / width) %d\n", (id%width), (id / width));
		for (int j = 0; j < PICS_SIZE_PADDED; j++) {
			for (int i = 0; i < PICS_SIZE_PADDED; i++) {
				h_small_pic[j*PICS_SIZE_PADDED + i] = h_Input[(i + (id%width)*PICS_SIZE) + (j + (id / width)*(PICS_SIZE))*(dim_padding)];
			}
		}


		//Copy small pic to d_Input
		cudaMemcpy(d_Input, h_small_pic, PICS_SIZE_PADDED * PICS_SIZE_PADDED * sizeof(double), cudaMemcpyHostToDevice);
		//Compute the rows and copy output to d_Buffer
		RowGPU << <grid_dim, block_dim >> > (d_Buffer, d_Input, PICS_SIZE, PICS_SIZE, filter_radius);

		cudaMemcpy(h_OutputGPU_pic, d_Buffer, PICS_SIZE_PADDED * PICS_SIZE_PADDED * sizeof(double), cudaMemcpyDeviceToHost);
		
		cudaDeviceSynchronize();


		cudaDeviceSynchronize();
		//transfer small to big array
		for (int j = filter_radius; j < PICS_SIZE + filter_radius; j++) {
			for (int i = filter_radius; i < PICS_SIZE + filter_radius; i++) {
				h_Buffer[(i + (id%width)*PICS_SIZE) + (j + (id / width)*(PICS_SIZE))*(dim_padding)] = h_OutputGPU_pic[j*PICS_SIZE_PADDED + i];
			}
		}


	}


	for (int id = 0; id < runTimes; id++) {

		for (int j = 0; j < PICS_SIZE_PADDED; j++) {
			for (int i = 0; i < PICS_SIZE_PADDED; i++) {
				h_small_pic[j*PICS_SIZE_PADDED + i] = h_Buffer[(i + (id%width)*PICS_SIZE) + (j + (id / width)*(PICS_SIZE))*(dim_padding)];
			}
		}


		//Copy small pic to d_Input
		cudaMemcpy(d_Input, h_small_pic, PICS_SIZE_PADDED * PICS_SIZE_PADDED * sizeof(double), cudaMemcpyHostToDevice);
		//Compute the rows and copy output to d_Buffer
		ColGPU << <grid_dim, block_dim >> > (d_Buffer, d_Input, PICS_SIZE, PICS_SIZE, filter_radius);
		cudaMemcpy(h_OutputGPU_pic, d_Buffer, PICS_SIZE_PADDED * PICS_SIZE_PADDED * sizeof(double), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
		//transfer small to big array
		for (int j = filter_radius; j < PICS_SIZE + filter_radius; j++) {
			for (int i = filter_radius; i < PICS_SIZE + filter_radius; i++) {
				h_OutputGPU[(i + (id%width)*PICS_SIZE) + (j + (id / width)*(PICS_SIZE))*(dim_padding)] = h_OutputGPU_pic[j*PICS_SIZE_PADDED + i];
			}
		}


	}
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("kernel time in ms: %f\n", milliseconds);

#ifdef CPU
	for (i = 0; i < dim_padding * dim_padding; i++) {
		if (ABS(h_OutputGPU[i] - h_OutputCPU[i]) >= accuracy) {
			printf("error\n");
			break;
		}
	}
#endif

	// free all the allocated memory
	free(h_OutputCPU);
	free(h_Buffer);
	free(h_Input);
	free(h_Filter);
	free(h_OutputGPU);
	free(h_small_pic);
	free(h_OutputGPU_pic);

	//cudaFree(d_Filter);
	cudaFree(d_Input);
	cudaFree(d_Buffer);
	cudaFree(d_OutputGPU_pic);
	// Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
	cudaDeviceReset();


	return 0;
}

