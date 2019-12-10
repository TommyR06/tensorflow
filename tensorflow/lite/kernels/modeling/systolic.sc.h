// Created by Nicolas Agostini

#ifndef TENSORFLOW_LITE_KERNELS_MODELING_SYSTOLIC_H_
#define TENSORFLOW_LITE_KERNELS_MODELING_SYSTOLIC_H_

#include <systemc/systemc.h>

namespace tflite_soc {

SC_MODULE(SystolicDut) {

  sc_in<bool> clock;

  void Gemm();

  SC_HAS_PROCESS(SystolicDut);

  // Parameters for the DUT
  SystolicDut(sc_module_name name_, int size_=64, bool debug_ = false) :
    sc_module(name_), size(size_), debug(debug_)
  {
    SC_THREAD(Gemm);
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
#endif