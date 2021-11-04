#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv)
{
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2)
    {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2)
    {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum)
    {
    case 1:
        labwork.labwork1_CPU();
        labwork.saveOutputImage("labwork2-cpu-out.jpg");
        printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        timer.start();
        labwork.labwork1_OpenMP();
        labwork.saveOutputImage("labwork2-openmp-out.jpg");
        break;
    case 2:
        labwork.labwork2_GPU();
        break;
    case 3:
        labwork.labwork3_GPU();
        labwork.saveOutputImage("labwork3-gpu-out.jpg");
        break;
    case 4:
        labwork.labwork4_GPU();
        labwork.saveOutputImage("labwork4-gpu-out.jpg");
        break;
    case 5:
        labwork.labwork5_CPU();
        labwork.saveOutputImage("labwork5-cpu-out.jpg");
        labwork.labwork5_GPU();
        labwork.saveOutputImage("labwork5-gpu-out.jpg");
        break;
    case 6:
        labwork.labwork6_GPU();
        labwork.saveOutputImage("labwork6-gpu-out.jpg");
        break;
    case 7:
        labwork.labwork7_GPU();
        printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        labwork.saveOutputImage("labwork7-gpu-out.jpg");
        break;
    case 8:
        labwork.labwork8_GPU();
        printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        labwork.saveOutputImage("labwork8-gpu-out.jpg");
        break;
    case 9:
        labwork.labwork9_GPU();
        printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        labwork.saveOutputImage("labwork9-gpu-out.jpg");
        break;
    case 10:
        labwork.labwork10_GPU();
        printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        labwork.saveOutputImage("labwork10-gpu-out.jpg");
        break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName)
{
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName)
{
    bool saveImageSuccess = jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU()
{
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++)
    { // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++)
        {
            outputImage[i * 3] = (char)(((int)inputImage->buffer[i * 3] + (int)inputImage->buffer[i * 3 + 1] +
                                         (int)inputImage->buffer[i * 3 + 2]) /
                                        3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP()
{
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    // do nothing here
    omp_set_num_threads(2);
#pragma omp parallel for
    for (int j = 0; j < 100; j++)
    { // let's do it 100 times, otherwise it's too fast!

        for (int i = 0; i < pixelCount; i++)
        {

            outputImage[i * 3] = (char)(((int)inputImage->buffer[i * 3] + (int)inputImage->buffer[i * 3 + 1] +
                                         (int)inputImage->buffer[i * 3 + 2]) /
                                        3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major)
    {
    case 2: // Fermi
        if (devProp.minor == 1)
            cores = mp * 48;
        else
            cores = mp * 32;
        break;
    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;
    case 6: // Pascal
        if (devProp.minor == 1)
            cores = mp * 128;
        else if (devProp.minor == 0)
            cores = mp * 64;
        else
            printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}

void Labwork::labwork2_GPU()
{
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        // get informations from individual device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device id: %d\n Name: %s\n Core clock speed: %d\n Core count: %d\n Warp size: %d\n",
               i,
               prop.name,
               prop.clockRate,
               prop.multiProcessorCount,
               prop.warpSize

        );

        // something more here
    }
}
__global__ void rgb2grayCUDA(uchar3 *input, uchar3 *output)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = (input[tid].x + input[tid].y +
                     input[tid].z) /
                    3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork3_GPU()
{
    // Calculate number of pixels
    // inputImage struct: width, height, buffer
    int pixelCount = inputImage->width * inputImage->height * 3;

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount);
    cudaMalloc(&devOutput, pixelCount);

    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount, cudaMemcpyHostToDevice);

    // Processing
    int blockSize = 64;
    int numBlock = pixelCount / (blockSize * 3);
    rgb2grayCUDA<<<numBlock, blockSize>>>(devInput, devOutput);

    // Copy CUDA Memory from GPU to CPU
    outputImage = (char *)malloc(pixelCount);
    cudaMemcpy(outputImage, devOutput, pixelCount, cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void rgb2grayCUDABlock(uchar3 *input, uchar3 *output)
{
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = tid_x + blockDim.x * gridDim.x * tid_y;
    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork4_GPU()
{

    int blockSizes[5] = {32, 8, 4};

    for (int i = 0; i < 3; i++)
    {
        Timer timer;
        timer.start();
        //
        dim3 blockSize = dim3(blockSizes[i], blockSizes[i]);
        dim3 gridSize = dim3(round(inputImage->width / blockSizes[i]), round(inputImage->height / blockSizes[i]));
        // dim3 gridSize = dim3(120,68);
        printf("gridSize: %d - blockSize: %d\n", gridSize, blockSize);

        // Calculate number of pixels
        // inputImage struct: width, height, buffer
        int pixelCount = inputImage->width * inputImage->height * 3;

        // Allocate CUDA memory
        uchar3 *devInput;
        uchar3 *devOutput;
        cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));

        // Copy CUDA Memory from CPU to GPU
        cudaMemcpy(devInput, inputImage->buffer, pixelCount, cudaMemcpyHostToDevice);

        // Processing
        // int blockSize = 64;
        // int numBlock = pixelCount / (blockSize * 3);
        // rgb2grayCUDA<<<numBlock, blockSize>>>(devInput, devOutput);

        rgb2grayCUDABlock<<<gridSize, blockSize>>>(devInput, devOutput);

        // Copy CUDA Memory from GPU to CPU
        outputImage = (char *)malloc(pixelCount);
        cudaMemcpy(outputImage, devOutput, pixelCount, cudaMemcpyDeviceToHost);

        // Cleaning
        cudaFree(devInput);
        cudaFree(devOutput);

        printf("Entry %d\n GridSize=%d\n BlockSize=%d\n Ellapsed %.1fms\n",
               i,
               blockSizes[i],
               timer.getElapsedTimeInMilliSec());
    }
}

float gaussianBlur[7][7] =
 	{{ 0, 0, 1, 2, 1, 0, 0 },
	{ 0, 3, 13, 22, 13, 3, 0 },
	{ 1, 13, 59, 97, 59, 13, 1 },
	{ 2, 22, 97, 159, 97, 22, 2 },
	{ 1, 13, 59, 97, 59, 13, 1 },
	{ 0, 3, 13, 22, 13, 3, 0 },
	{ 0, 0, 1, 2, 1, 0, 0 }};

__device__ float gaussianBlurGPU[7][7] =
 	{{ 0, 0, 1, 2, 1, 0, 0 },
	{ 0, 3, 13, 22, 13, 3, 0 },
	{ 1, 13, 59, 97, 59, 13, 1 },
	{ 2, 22, 97, 159, 97, 22, 2 },
	{ 1, 13, 59, 97, 59, 13, 1 },
	{ 0, 3, 13, 22, 13, 3, 0 },
	{ 0, 0, 1, 2, 1, 0, 0 }};

void Labwork::labwork5_CPU() {
        int pixelCount = inputImage->width * inputImage->height;
        outputImage = static_cast<char *>(malloc(pixelCount * 3));
	for (int row = 3; row < inputImage->height-3; row++) {
		for (int col = 3; col < inputImage->width-3; col++) {
			int sumR = 0;
			int sumG = 0;
			int sumB = 0;
			for (int j = 0; j < 7; j++) {
				for (int i = 0; i < 7; i++) {
					int pos = (col + (i - 3)) + (row + (j - 3)) * inputImage->width;
					sumR += inputImage->buffer[pos * 3] * gaussianBlur[j][i];
					sumG += inputImage->buffer[pos * 3+1] * gaussianBlur[j][i];
					sumB += inputImage->buffer[pos * 3+2] * gaussianBlur[j][i];
				}
			}
			sumR /= 1003;
			sumG /= 1003;
			sumB /= 1003;
			int pos = col+ row * inputImage->width;
			outputImage[pos * 3] = sumR;
			outputImage[pos * 3 + 1] = sumG;
			outputImage[pos * 3 + 2] = sumB;
		}
        }
	printf("finished labwork5 CPU\n");
}

__global__ void gaussianCUDABlock(uchar3 *input, uchar3 *output)
{
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = tid_x + blockDim.x * gridDim.x * tid_y;

    int sumR = 0, sumG = 0, sumB = 0;
    for (int i = -3; i <= 3; ++i)
        for (int j = -3; j <= 3; ++j)
        {
            int cell_tid = tid + i * blockDim.x * gridDim.x + j;
            sumR += input[cell_tid].x * gaussianBlurGPU[i + 3][j + 3];
            sumG += input[cell_tid].y * gaussianBlurGPU[i + 3][j + 3];
            sumB += input[cell_tid].z * gaussianBlurGPU[i + 3][j + 3];
        }

    output[tid].x = sumR / 1003;
    output[tid].y = sumG / 1003;
    output[tid].z = sumB / 1003;
}

void Labwork::labwork5_GPU()
{
    int size = 8;
    dim3 blockSize = dim3(size, size);
    dim3 gridSize = dim3(round(inputImage->width / size), round(inputImage->height / size));
    // dim3 gridSize = dim3(120,68);
    printf("gridSize: %d - size: %d\n", gridSize, size);

    // Calculate number of pixels
    // inputImage struct: width, height, buffer
    int pixelCount = inputImage->width * inputImage->height * 3;

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));

    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount, cudaMemcpyHostToDevice);

    // Processing
    gaussianCUDABlock<<<gridSize, blockSize>>>(devInput, devOutput);

    // Copy CUDA Memory from GPU to CPU
    outputImage = (char *)malloc(pixelCount * 3);
    cudaMemcpy(outputImage, devOutput, pixelCount, cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
    printf("finished labwork5 GPU\n");
}

void Labwork::labwork6_GPU()
{
}

void Labwork::labwork7_GPU()
{
}

void Labwork::labwork8_GPU()
{
}

void Labwork::labwork9_GPU()
{
}

void Labwork::labwork10_GPU()
{
}
