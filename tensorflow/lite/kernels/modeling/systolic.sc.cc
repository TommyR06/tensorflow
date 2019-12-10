// Created by Nicolas Agostini

#include "tensorflow/lite/kernels/modeling/systolic.sc.h"

namespace tflite_soc {

template <typename LhsScalar, typename RhsScalar, typename DstScalar>
void SystolicDut::SetupGemm(int lhs_width_, int depth_, int rhs_width_,
               LhsScalar const* lhs_data_, RhsScalar const* rhs_data_,
               DstScalar* out_data_) {}

// void SetupGemm(int lhs_width_, int depth_, int rhs_width_,
//                int const* lhs_data_, int const* rhs_data_,
//                int* out_data_) {
//                }

template <typename DstScalar>
void SystolicDut::Test(DstScalar* data) {}

void SystolicDut::Gemm() {

  *out_data =1;
}

template void SystolicDut::Test<int>(int*);
template void SystolicDut::SetupGemm<int, int, int>(int, int, int, int const*,
                                                    int const*, int*);
}