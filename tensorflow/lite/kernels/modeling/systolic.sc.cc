// Created by Nicolas Agostini

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

void SystolicDut::Gemm() {

  *out_data =1;
}

template void SystolicDut::Test<int>(int*);
template void SystolicDut::SetupGemm<int, int, int>(int, int, int, int const*,
                                                    int const*, int*);
}