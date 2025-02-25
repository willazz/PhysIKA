/*
 * @file cuda_utilities.h
 * @Brief cuda utilities
 * @author Wei Chen
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_UTILITIES_CUDA_UTILITIES_H_
#define PHYSIKA_CORE_UTILITIES_CUDA_UTILITIES_H_

#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <iostream>
#include "cuda_helper_math.h"

namespace Physika{

#define INVALID -1
#define EPSILON   1e-6
#define M_PI 3.14159265358979323846
#define M_E 2.71828182845904523536

	#define BLOCK_SIZE 64

	using cuint = unsigned int;

	static cuint iDivUp(cuint a, cuint b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	// compute grid and thread block size for a given number of elements
	static cuint cudaGridSize(cuint totalSize, cuint blockSize)
	{
		return iDivUp(totalSize, blockSize);
	}

	static dim3 cudaGridSize3D(uint3 totalSize, uint3 blockSize)
	{
		dim3 gridDims;
		gridDims.x = iDivUp(totalSize.x, blockSize.x);
		gridDims.y = iDivUp(totalSize.y, blockSize.y);
		gridDims.z = iDivUp(totalSize.z, blockSize.z);

		return gridDims;
	}

	/** check whether cuda thinks there was an error and fail with msg, if this is the case
	* @ingroup tools
	*/
	static inline void checkCudaError(const char *msg) {
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err) {
			throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
		}
	}

	// use this macro to make sure no error occurs when cuda functions are called
#ifdef NDEBUG
#define cuSafeCall(X)  X
#else
#define cuSafeCall(X) X; Physika::checkCudaError(#X);
#endif

	// use this macro to make sure no error occurs when cuda kernels functions are launched
#ifdef NDEBUG
#define cuSynchronize() {}
#else
#define cuSynchronize()	{						\
		char str[200];							\
		cudaDeviceSynchronize();				\
		cudaError_t err = cudaGetLastError();	\
		if (err != cudaSuccess)					\
		{										\
			sprintf(str, "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__);		\
			throw std::runtime_error(std::string(str));																\
		}																											\
	}
#endif

}// end of namespace Physika

#endif //PHYSIKA_CORE_UTILITIES_CUDA_UTILITIES_H_