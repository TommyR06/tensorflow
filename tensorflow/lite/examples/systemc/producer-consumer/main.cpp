// Created by Nicolas Agostini

#include <systemc.h>
#include "stack.h"
#include "producer.h"
#include "consumer.h"


int sc_main(int argc, char* argv[]) {

  sc_clock ClkSlow("ClkSlow", 100.0, SC_NS);
  sc_clock ClkFast("ClkFast", 50.0, SC_NS);


  stack Stack1("S1");

  producer P1("P1");
  P1.out(Stack1);
  P1.Clock(ClkSlow);

  // Consumer consumes faster then producer
  consumer C1("C1");
  C1.in(Stack1);
  C1.Clock(ClkFast);

  sc_start(5000, SC_NS);

  return 0;
}
