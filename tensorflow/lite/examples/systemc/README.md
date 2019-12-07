# ATTENTION

Bazel bug is preventing this to compile with multiple jobs.

See `https://github.com/bazelbuild/bazel/issues/10384`

Please compile and run with:

```
bazel build --jobs 1 //tensorflow/lite/tools/systemc:systemc_model
bazel run //tensorflow/lite/examples/systemc:systemc_model

```

# Example on how to use SystemC with tflite libraries

This is a simple example on how to use SystemC from within tflite build
