// Created by Nicolas Agostini

#ifndef TENSORFLOW_LITE_KERNELS_MODELING_SYSTOLIC_H_
#define TENSORFLOW_LITE_KERNELS_MODELING_SYSTOLIC_H_

#include <systemc/systemc.h>

namespace tflite_soc {

SC_MODULE(SystolicDut) {

  sc_in<bool> clock;
  sc_in<bool> run_gemm;

  //  So in the LHS MxK matrix, the depth is K and the width in M.
  // And in the RHS KxN matrix, the depth is K and the width in N.
  //
  // This is illustrated in this picture:
  //
  //                             RHS width
  //                        <----------------->
  //                        +-----------------+ ^
  //                        |       RHS       | | Depth
  //                        +-----------------+ v
  //                 ^ +--+ +-----------------+
  //                 | |L | |                 |
  //       LHS width | |H | |      Result     |
  //                 | |S | |                 |
  //                 v +--+ +-----------------+
  //                   <-->
  //                   Depth
  void Gemm();

  void SetupGemm(int lhs_width_, int depth_, int rhs_width_, int* lhs_data_,
                 int* rhs_data_, int* out_data_);

  SC_HAS_PROCESS(SystolicDut);

  // Parameters for the DUT
  SystolicDut(sc_module_name name_, int size_=64, bool debug_ = false) :
    sc_module(name_), size(size_), debug(debug_)
  {
    SC_THREAD(Gemm);
    sensitive << run_gemm.pos();

    buffer = new int[size];
    if (debug) {
      cout << "Running constructor of " << name() << endl;
    }
  }

  private:
    int * buffer;
    const int size;
    const bool debug;

    int lhs_width;
    int depth;
    int rhs_width;
    int* lhs_data;
    int* rhs_data;
    int* out_data;
};
}
#endif