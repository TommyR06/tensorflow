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
#include "tensorflow/lite/kernels/modeling/systolic_stim.sc.h"

namespace tflite_soc {

void SystolicStim::Method() {}

SystolicStim::SystolicStim(sc_module_name name_, bool debug_)
    : sc_module(name_), debug(debug_) {

  if (debug) {
    std::cout << "Running constructor of " << name() << std::endl;
  }
}

}  // namespace tflite_soc