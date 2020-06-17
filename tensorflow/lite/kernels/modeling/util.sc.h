// Created by Nicolas Agostini
/* Copyright 2020 The TFLITE-SOC Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_MODELING_UTIL_SC_H_
#define TENSORFLOW_LITE_KERNELS_MODELING_UTIL_SC_H_

#include <systemc/systemc.h>
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

namespace tflite_soc {

using namespace tflite::cpu_backend_gemm;

template <typename LhsScalar, typename RhsScalar, typename DstScalar>
void PrintMatricesInfo(
      const MatrixParams<LhsScalar>& lhs_params,// const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params,// const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params//, DstScalar* dst_data
) {
  if (&lhs_params) {
    printf("\nlhs (%d,%d)\n", lhs_params.rows, lhs_params.cols);
  }
  if (&rhs_params) {
    printf("\nrhs (%d,%d)\n", rhs_params.rows, rhs_params.cols);
  }

  if (&dst_params) {
    printf("\nout (%d,%d)\n", dst_params.rows, dst_params.cols);
  }
}

template <typename typeScalar>
void PrintMatrix(const MatrixParams<typeScalar>& matrix_params,
                 const typeScalar* matrix_data) {
#define MAX_COLS 32

  if (&matrix_params && matrix_data) {
    printf("\nMatrix (%d,%d)\n", matrix_params.rows, matrix_params.cols);
    for (unsigned i = 0; i < matrix_params.rows; ++i) {
      printf("\n");
      for (unsigned j = 0; j < matrix_params.cols; ++j) {
        printf("%d,", matrix_data[i * matrix_params.cols + j]);
        if (j > MAX_COLS) {
          printf("...");
          break;
        }
      }
    }
    printf("\n ~~~~~~");
    printf("\n ~~~~~~");
  }
#undef MAX_COLS
}

template <typename LhsScalar, typename RhsScalar, typename DstScalar>
void PrintMatrices(
      const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data
) {
  #define MAX_COLS 32

  if (&lhs_params && lhs_data) {
    printf("\nlhs (%d,%d)\n", lhs_params.rows, lhs_params.cols);
    for (unsigned i = 0; i < lhs_params.rows; ++i) {
      printf("\n");
      for (unsigned j = 0; j < lhs_params.cols; ++j) {
        printf("%d,", lhs_data[i * lhs_params.cols + j]);
        if (j > MAX_COLS) {
          printf("...");
          break;
        }
      }
    }
    printf("\n ~~~~~~");
    printf("\n ~~~~~~");
  }

  if (&rhs_params && rhs_data) {
    printf("\nrhs (%d,%d)\n", rhs_params.rows, rhs_params.cols);
    for (unsigned i = 0; i < rhs_params.rows; ++i) {
      printf("\n");
      for (unsigned j = 0; j < rhs_params.cols; ++j) {
        printf("%d,", rhs_data[i * rhs_params.cols + j]);
        if (j > MAX_COLS) {
          printf("...");
          break;
        }
      }
    }
    printf("\n ~~~~~~");
    printf("\n ~~~~~~");
  }

  if (&dst_params && dst_data) {
    printf("\nout (%d,%d)\n", dst_params.rows, dst_params.cols);
    for (unsigned i = 0; i < dst_params.rows; ++i) {
      printf("\n");
      for (unsigned j = 0; j < dst_params.cols; ++j) {
        printf("%d,", dst_data[i * dst_params.cols + j]);
        if (j > MAX_COLS) {
          printf("...");
          break;
        }
      }
    }
    printf("\n ~~~~~~");
    printf("\n ~~~~~~");
  #undef MAX_COLS
  }
}

// Simple Hello World module
SC_MODULE(hello_world){
    SC_CTOR(hello_world){} 
    void say_hello(){cout << "Hello World.\n";
  }
};

void say_hello();

// Ram module
// Extracted from here: https://www.doulos.com/knowhow/systemc/faq/#q2
SC_MODULE(ram) {

  sc_in<bool> clock;
  sc_in<bool> RnW;   // ReadNotWrite
  sc_in<int> address;
  sc_inout<int> data;

  void ram_proc();

  SC_HAS_PROCESS(ram);

  ram(sc_module_name name_, int size_=64, bool debug_ = false) :
    sc_module(name_), size(size_), debug(debug_)
  {
    SC_THREAD(ram_proc);
    sensitive << clock.pos();

    buffer = new int[size];
    if (debug) {
      cout << "Running constructor of " << name() << endl;
    }
  }

  private:
    int * buffer;
    const int size;
    const bool debug;
};
}
#endif // TENSORFLOW_LITE_KERNELS_MODELING_UTIL_SC_H_