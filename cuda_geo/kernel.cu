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
#include <windows.h>
#include <stdio.h>
#include <algorithm>
#include <string>

using namespace std;

// we use landsat 8

__global__ void ndvi(
	const float* band5, const float* band4, float* result, int n) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		if (band5[i] + band4[i] == 0) result[i] = 0;
		else {
			result[i] = (band5[i] - band4[i] ) / (band5[i]+ band4[i]);
		}
	}
}

__global__ void savi(
	const float* band5, const float* band4, float* result, int n, float l) {
	//l =0.5
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		if (band5[i] + band4[i] + (l) == 0) result[i] = 0;
		else {
			result[i] = ((band5[i] - band4[i]) / (band5[i] +band4[i] + l  )) *  ( l  +1.0 );

		}
	}
}

__global__ void msavi(
	const float* band5, const float* band4, float* result, int n) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		if (2 * band5[i] + 1 - sqrt((2 * band5[i] + 1) * (2 * band5[i] + 1) - 8 * (band5[i] - band4[i]))) result[i] = 0;
		else {
			result[i] = (2 * band5[i] + 1 - sqrt((2 * band5[i] + 1)* (2 * band5[i] + 1) - 8 * (band5[i] - band4[i]))) / 2;
		}
	}
}

__global__ void ndmi(
	const float* band6, const float* band5, float* result, int n) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		if (band5[i] + band6[i] == 0) result[i] = 0;
		else {
			result[i] = (band5[i] - band6[i]) / (band5[i] + band6[i]);
		}
	}
}

__global__ void evi(
	const float* band2, const float* band4, const float* band5, float* result, int n) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		if (band5[i] + 6.0f * band4[i] - 7.5f * band2[i] + 1.0f == 0.0f) result[i] = 0.0f;
		else {
			result[i] = 2.5f * ((band5[i] - band4[i]) / (band5[i] + 6.0f * band4[i] - 7.5f * band2[i] + 1.0f));
		}
	}
}


int main(int argc, char* argv[]) {


	printf("What do you want to calculate:\n");
	printf("1-ndvi\n");
	printf("2-savi\n");
	printf("3-msavi\n");
	printf("4-ndmi\n");
	printf("5-evi\n");

	int what_to_do = 0;
	cin >> what_to_do;

	float l2;
	char* inputFileNames[3] = { "band4.tif", "band5.tif", "band6.tif" };
	if (what_to_do != 0) {
		if (what_to_do == 2) {
			cout << "Enter L:" << endl;
			cin >>l2;
		}
	
		OPENFILENAME ofn = { sizeof(OPENFILENAME) };
		char szFile[_MAX_PATH] = "name";
		//const char szExt[] = "ext\0"; // extra '\0' for lpstrFilter

		ofn.hwndOwner = GetConsoleWindow();
		ofn.lpstrFile = szFile; // <--------------------- initial file name
		ofn.nMaxFile = sizeof(szFile) / sizeof(szFile[0]);
		//ofn.lpstrFilter = ofn.lpstrDefExt = szExt;
		ofn.Flags = OFN_SHOWHELP | OFN_OVERWRITEPROMPT;
		if (what_to_do == 1 || what_to_do == 2 || what_to_do == 3 ) {
			printf("landsat8:\n");
			printf("Select band4:\n");
			if (GetOpenFileName(&ofn))
			{
				printf("band4 '%s'\n", ofn.lpstrFile);
				std::string str = ofn.lpstrFile;
				std::replace( str.begin(), str.end(), '\\', '/');
				cout << str << endl;
				char* cstr = strcpy(new char[str.length() + 1], str.c_str());
				cout << cstr << endl;
				inputFileNames[0] = cstr;
			}
			
			printf("Select band5:\n");
			if (GetOpenFileName(&ofn))
			{
				printf("band5 '%s'\n", ofn.lpstrFile);
				std::string str = ofn.lpstrFile;
				std::replace(str.begin(), str.end(), '\\', '/');
				cout << str << endl;
				char* cstr = strcpy(new char[str.length() + 1], str.c_str());
				cout << cstr << endl;
				inputFileNames[1] = cstr;
			}
		}

		if (what_to_do == 4) {
			printf("NDMI landsat8:\n");
			printf("Select band5:\n");
			if (GetOpenFileName(&ofn))
			{
				printf("band5 '%s'\n", ofn.lpstrFile);
				std::string str = ofn.lpstrFile;
				std::replace(str.begin(), str.end(), '\\', '/');
				cout << str << endl;
				char* cstr = strcpy(new char[str.length() + 1], str.c_str());
				cout << cstr << endl;
				inputFileNames[0] = cstr;
			}

			printf("Select band6:\n");
			if (GetOpenFileName(&ofn))
			{
				printf("band6 '%s'\n", ofn.lpstrFile);
				std::string str = ofn.lpstrFile;
				std::replace(str.begin(), str.end(), '\\', '/');
				cout << str << endl;
				char* cstr = strcpy(new char[str.length() + 1], str.c_str());
				cout << cstr << endl;
				inputFileNames[1] = cstr;
			}
		}

		if (what_to_do == 5) {
			printf("EVI landsat8:\n");
			printf("Select band2:\n");
			if (GetOpenFileName(&ofn))
			{
				printf("band5 '%s'\n", ofn.lpstrFile);
				std::string str = ofn.lpstrFile;
				std::replace(str.begin(), str.end(), '\\', '/');
				cout << str << endl;
				char* cstr = strcpy(new char[str.length() + 1], str.c_str());
				cout << cstr << endl;
				inputFileNames[0] = cstr;
			}

			printf("Select band4:\n");
			if (GetOpenFileName(&ofn))
			{
				printf("band6 '%s'\n", ofn.lpstrFile);
				std::string str = ofn.lpstrFile;
				std::replace(str.begin(), str.end(), '\\', '/');
				cout << str << endl;
				char* cstr = strcpy(new char[str.length() + 1], str.c_str());
				cout << cstr << endl;
				inputFileNames[1] = cstr;
			}
			printf("Select band5:\n");
			if (GetOpenFileName(&ofn))
			{
				printf("band6 '%s'\n", ofn.lpstrFile);
				std::string str = ofn.lpstrFile;
				std::replace(str.begin(), str.end(), '\\', '/');
				cout << str << endl;
				char* cstr = strcpy(new char[str.length() + 1], str.c_str());
				cout << cstr << endl;
				inputFileNames[2] = cstr;
			}
		}
	}

	cout << endl;
	cout << endl;
	cout << endl;
	//cout << l << endl;

	cout << inputFileNames[0]<<endl;
	cout << inputFileNames[1]<<endl;
	cout << inputFileNames[2] << endl;
	//std::replace(inputFileNames[1].begin(), inputFileNames[1].end(), 'x', 'y');

	array<GDALDataset*, 3> input;

	if (what_to_do == 5) {
		array<GDALDataset*, 3> input;
	}

	GDALDataset* output;

	

	GDALAllRegister();
	//inputFileNames[0] = "C:\\Users\\stamacake\\source\\repos\\cuda_geo\\cuda_geo\\band4.tif";
	//inputFileNames[1] = "C:\\Users\\stamacake\\source\\repos\\cuda_geo\\cuda_geo\\band5.tif";

	cout << "Start ..." << endl;


	if (what_to_do != 5) {
		for (int i = 0; i < 2; i++)
		{
			input[i] = (GDALDataset*)GDALOpen(inputFileNames[i], GA_ReadOnly);
			if (input[i] == NULL)
			{
				cout << "File read error: " << inputFileNames[i] << "\n";
				return EXIT_FAILURE;
			}
		}
	}
	

	if (what_to_do==5) {
		for (int i = 0; i < 3; i++)
		{
			cout << "Start3333 ..." << endl;
			cout << i << endl;
			input[i] = (GDALDataset*)GDALOpen(inputFileNames[i], GA_ReadOnly);
			if (input[i] == NULL)
			{
				cout << "File read error: " << inputFileNames[i] << "\n";
				return EXIT_FAILURE;
			}
		}
	}

	cout << "BOOOOB ..." << endl;

	int X = input[0]->GetRasterXSize();
	int Y = input[0]->GetRasterYSize();




	const char* imageFormat = "GTiff";
	GDALDriver* gdalDriver = GetGDALDriverManager()->GetDriverByName(imageFormat);
	if (gdalDriver == NULL)
	{
		cout << "Can't create output file!\n";
		return EXIT_FAILURE;
	}
	cout << "11111 ..." << endl;
	output = gdalDriver->Create("output.tif", X, Y, 1, GDT_Float32, NULL);

	double info[6];
	input[0]->GetGeoTransform(info);

	cout << "22222 ..." << endl;
	const char* gdalProjection = input[0]->GetProjectionRef();
	output->SetGeoTransform(info);
	output->SetProjection(gdalProjection);
	cout << gdalProjection << endl;

	GDALRasterBand* band1 = input[0]->GetRasterBand(1);
	GDALRasterBand* band2 = input[1]->GetRasterBand(1);
	GDALRasterBand* band3 = NULL;
	if (what_to_do == 5) {
		band3 = input[2]->GetRasterBand(1);
	}
	GDALRasterBand* resultBand = output->GetRasterBand(1);

	float* band1_buffer = (float*)CPLMalloc(sizeof(float) * X);
	float* band2_buffer = (float*)CPLMalloc(sizeof(float) * X);


	float* l_buffer = (float*)CPLMalloc(sizeof(float) * X);
	float* one_buffer = (float*)CPLMalloc(sizeof(float) * X);

	float* band3_buffer = NULL;
	if (what_to_do==5) {
		band3_buffer = (float*)CPLMalloc(sizeof(float) * X);
	}

	float* result = new float[X], * result_gpu;
	cudaMalloc((void**)&result_gpu, X * sizeof(float));





	int n = X;
	const int block_size = 256; // can be optimized
	int num_blocks = (n + block_size - 1) / block_size;
	cout << "333333 ..." << endl;







	for (int i = 0; i < Y; i++)
	{
		band1->RasterIO(GF_Read, 0, i, X, 1, band1_buffer, X, 1, GDT_Float32, 0, 0);
		band2->RasterIO(GF_Read, 0, i, X, 1, band2_buffer, X, 1, GDT_Float32, 0, 0);
		if (what_to_do == 5) {
			band3->RasterIO(GF_Read, 0, i, X, 1, band3_buffer, X, 1, GDT_Float32, 0, 0);
		}

		float* a = new float[X], * a_gpu; 
		cudaMalloc((void**)&a_gpu, X * sizeof(float)); 

		float* b = new float[X], * b_gpu; //
		cudaMalloc((void**)&b_gpu, X * sizeof(float)); //

		float* ll = new float[X], * l_gpu; //
		cudaMalloc((void**)&l_gpu, X * sizeof(float)); //
		float* one1 = new float[X], * one_gpu; //
		cudaMalloc((void**)&one_gpu, X * sizeof(float)); //

		float* c = NULL;
		float* c_gpu = NULL;
		if (what_to_do == 5) {
			c = new float[X], * c_gpu;
			cudaMalloc((void**)&c_gpu, X * sizeof(float));
		}

		float* tmp = new float[X], * result_gpu;
		cudaMalloc((void**)&result_gpu, X * sizeof(float));

		// поменяли ссылку для gpu
		a = band1_buffer;
		b = band2_buffer;

		ll = l_buffer;
		one1 = one_buffer;

		if (what_to_do == 5) {
			c = band3_buffer;
		}
		
		//cout << "555555 ..." << endl;

		cudaMemcpy(a_gpu, a, X * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(b_gpu, b, X * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(l_gpu, ll, X * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(one_gpu, one1, X * sizeof(float), cudaMemcpyHostToDevice);
		if (what_to_do == 5) {
			cudaMemcpy(c_gpu, c, X * sizeof(float), cudaMemcpyHostToDevice);
		}

		// run cuda
		if (what_to_do == 1) {
			//cout << "check" << endl;
			ndvi << < num_blocks, block_size >> > (b_gpu, a_gpu, result_gpu, X);
		}
		if (what_to_do == 2) {
			savi << < num_blocks, block_size >> > (b_gpu, a_gpu, result_gpu, X, l2);
		}
		if (what_to_do == 3) {
			msavi << < num_blocks, block_size >> > (b_gpu, a_gpu, result_gpu, X);
		}
		if (what_to_do == 4) {
			ndmi << < num_blocks, block_size >> > (b_gpu, a_gpu, result_gpu, X);
		}
		if (what_to_do == 5) {
			evi << < num_blocks, block_size >> > (b_gpu, a_gpu, c_gpu, result_gpu, X);
		}
		

		// get data from gpu
		cudaMemcpy(tmp, result_gpu,
			X * sizeof(float), cudaMemcpyDeviceToHost);

		//cout << tmp << endl;

		resultBand->RasterIO(GF_Write, 0, i, X, 1, tmp, X, 1, GDT_Float32, 0, 0); //Write data
		cudaFree(a_gpu);
		cudaFree(b_gpu);
		if (what_to_do==5) {
			cudaFree(c_gpu);
		}
	}

	cout << "77777777 ..." << endl;

	cudaFree(result_gpu);

	CPLFree(band1_buffer);
	CPLFree(band2_buffer);
	if (what_to_do==5) {
		CPLFree(band3_buffer);
		GDALClose(input[2]);
	}


	cout << "88888888888 ..." << endl;

	GDALClose(input[0]);
	GDALClose(input[1]);
	GDALClose(output);

	cout << "Finish ..." << endl;
	return 0;

}