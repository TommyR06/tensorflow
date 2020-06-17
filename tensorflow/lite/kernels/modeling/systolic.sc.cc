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

#include "tensorflow/lite/kernels/modeling/systolic.sc.h"

namespace tflite_soc {

template <typename LhsScalar, typename RhsScalar, typename DstScalar>
void SystolicDut::SetupGemm(int lhs_width_, int depth_, int rhs_width_,
               LhsScalar const* lhs_data_, RhsScalar const* rhs_data_,
               DstScalar* out_data_) {

  lhs_width = lhs_width_;
  depth = depth_;
  rhs_width = rhs_width_;
  lhs_data = lhs_data_;
  rhs_data = rhs_data_;
  out_data = out_data_;
}

template <typename DstScalar>
void SystolicDut::Test(DstScalar* data) {}

inline void ReLu(int & v) {
  if (v < 0)
    v = 0;
}

void SystolicDut::Gemm() {
  while (1) {
    for (int i = 0; i < lhs_width*rhs_width; i++)
      out_data[i]=0;

    for (int iA = 0; iA < lhs_width; iA++) {
      int tj = 0.0;
      for (int jA = 0; jA < depth; jA++) {
        int pA2 = iA * depth + jA;
        int tk = 0.0;
        for (int kB = 0; kB < rhs_width; kB++) {
          int pB2 = jA * rhs_width + kB;
          tk += rhs_data[pB2];
        }
        tj += lhs_data[pA2] * tk;
      }
      out_data[iA] = tj;
    }

    // Apply fused ReLu
    for (int i = 0; i < lhs_width*rhs_width; i++)
      ReLu(out_data[i]);
    wait();
  }
}

template void SystolicDut::Test<int>(int*);
template void SystolicDut::SetupGemm<int, int, int>(int, int, int, int const*,
                                                    int const*, int*);
}