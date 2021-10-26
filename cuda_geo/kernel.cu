#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <gdal_priv.h>
#include <array>
#include <cpl_conv.h>
#include "device_launch_parameters.h"

using namespace std;

__global__ void ndvi(
	const float* a, const float* b, float* result, int n) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		if (a[i] + b[i] == 0) result[i] = 0;
		else {
			result[i] = (a[i] - b[i] ) / (a[i]+b[i]);
		}
	}
}


int main(int argc, char* argv[]) {

	array<GDALDataset*, 2> input;
	GDALDataset* output;

	GDALAllRegister();
	//const char* inputFileNames[2] = { "band4.tif", "band5.tif" };

	cout << "Start ..." << endl;
	for (int i = 0; i < 2; i++)
	{
		input[i] = (GDALDataset*)GDALOpen(argv[i+1], GA_ReadOnly);
		if (input[i] == NULL)
		{
			cout << "File read error: " << argv[i] << "\n";
			return EXIT_FAILURE;
		}
	}

	int X = input[0]->GetRasterXSize();
	int Y = input[0]->GetRasterYSize();

	const char* imageFormat = "GTiff";
	GDALDriver* gdalDriver = GetGDALDriverManager()->GetDriverByName(imageFormat);
	if (gdalDriver == NULL)
	{
		cout << "Can't create output file!\n";
		return EXIT_FAILURE;
	}

	output = gdalDriver->Create("ndvi.tif", X, Y, 1, GDT_Float32, NULL);

	double info[6];
	input[0]->GetGeoTransform(info);


	const char* gdalProjection = input[0]->GetProjectionRef();
	output->SetGeoTransform(info);
	output->SetProjection(gdalProjection);

	GDALRasterBand* redBand = input[0]->GetRasterBand(1);
	GDALRasterBand* nirBand = input[1]->GetRasterBand(1);
	GDALRasterBand* resultBand = output->GetRasterBand(1);

	float* redBuffer = (float*)CPLMalloc(sizeof(float) * X);
	float* nirBuffer = (float*)CPLMalloc(sizeof(float) * X);

	float* result = new float[X], * result_gpu;
	cudaMalloc((void**)&result_gpu, X * sizeof(float));

	int n = X;
	const int block_size = 256;
	int num_blocks = (n + block_size - 1) / block_size;

	//Calculate NDWI 
	for (int i = 0; i < Y; i++)
	{
		redBand->RasterIO(GF_Read, 0, i, X, 1, redBuffer, X, 1, GDT_UInt32, 0, 0);
		nirBand->RasterIO(GF_Read, 0, i, X, 1, nirBuffer, X, 1, GDT_UInt32, 0, 0);

		float* a = new float[X], * a_gpu; 
		cudaMalloc((void**)&a_gpu, X * sizeof(float)); 

		float* b = new float[X], * b_gpu; //
		cudaMalloc((void**)&b_gpu, X * sizeof(float)); //

		float* tmp = new float[X], * result_gpu;
		cudaMalloc((void**)&result_gpu, X * sizeof(float));

		// поменяли ссылку для gpu
		a = redBuffer;
		b = nirBuffer;

		cudaMemcpy(a_gpu, a, X * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(b_gpu, b, X * sizeof(float), cudaMemcpyHostToDevice);

		// run cuda
		ndvi << < num_blocks, block_size >> > (b_gpu, a_gpu, result_gpu, X);

		// get data from gpu
		cudaMemcpy(tmp, result_gpu,
			X * sizeof(float), cudaMemcpyDeviceToHost);

		resultBand->RasterIO(GF_Write, 0, i, X, 1, tmp, X, 1, GDT_Float32, 0, 0); //Write data
		cudaFree(a_gpu);
		cudaFree(b_gpu);
	}

	cudaFree(result_gpu);

	CPLFree(redBuffer);
	CPLFree(nirBuffer);
	GDALClose(input[0]);
	GDALClose(input[1]);
	GDALClose(output);

	cout << "Finish ..." << endl;
	return 0;

}