// Created by Nicolas Agostini

#include "tensorflow/lite/kernels/modeling/systolic.sc.h"

namespace tflite_soc {

void SystolicDut::SetupGemm(int lhs_width_, int depth_, int rhs_width_,
                            int* lhs_data_, int* rhs_data_, int* out_data_) {

}

void SystolicDut::Gemm() {

  *out_data =1;
}
}