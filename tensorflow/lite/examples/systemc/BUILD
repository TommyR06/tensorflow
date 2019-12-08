load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["logging.h"])

common_copts = ["-Wall"] + tflite_copts()

cc_library(
    name = "logging",
    hdrs = ["logging.h"],
    copts = common_copts,
)

cc_binary(
    name = "hello_systemc",
    srcs = [
        "hello_main.sc.cc",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        #":logging",
        "@systemc//:systemc",
    ],
)

cc_binary(
    name = "hello_channel",
    srcs = [
        "channel_main.sc.cc",
        "channel/stack_if.sc.h",
        "channel/stack.sc.h",
        "channel/producer.sc.h",
        "channel/consumer.sc.h",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "@systemc//:systemc",
    ],
)

# cc_library(
#     name = "benchmark_tflite_model_lib",
#     srcs = ["benchmark_tflite_model.cc"],
#     hdrs = ["benchmark_tflite_model.h"],
#     copts = common_copts + select({
#         "//tensorflow:ios": [
#             "-xobjective-c++",
#         ],
#         "//conditions:default": [],
#     }),
#     deps = [
#         ":benchmark_model_lib",
#         ":benchmark_utils",
#         ":logging",
#         "@com_google_absl//absl/base:core_headers",
#         "@com_google_absl//absl/strings",
#         "@gemmlowp",
#         "//tensorflow/lite:framework",
#         "//tensorflow/lite:string_util",
#         "//tensorflow/lite/kernels:builtin_ops",
#         "//tensorflow/lite/nnapi:nnapi_util",
#         "//tensorflow/lite/profiling:profile_summarizer",
#         "//tensorflow/lite/profiling:profiler",
#         "//tensorflow/lite/tools/evaluation:utils",
#     ] + select({
#         "//tensorflow:android": [
#             "//tensorflow/lite/delegates/gpu:delegate",
#         ],
#         "//tensorflow:ios": [
#             "//tensorflow/lite/delegates/gpu:metal_delegate",
#         ],
#         "//conditions:default": [],
#     }),
# )
# 
# cc_library(
#     name = "benchmark_model_lib",
#     srcs = [
#         "benchmark_model.cc",
#     ],
#     hdrs = ["benchmark_model.h"],
#     copts = common_copts,
#     deps = [
#         ":benchmark_params",
#         ":benchmark_utils",
#         ":logging",
#         "//tensorflow/core:stats_calculator_portable",
#         "//tensorflow/lite:framework",
#         "//tensorflow/lite/c:common",
#         "//tensorflow/lite/profiling:memory_info",
#         "//tensorflow/lite/profiling:time",
#         "//tensorflow/lite/tools:command_line_flags",
#     ],
# )
# 
# cc_library(
#     name = "benchmark_utils",
#     srcs = [
#         "benchmark_utils.cc",
#     ],
#     hdrs = ["benchmark_utils.h"],
#     copts = common_copts,
#     deps = ["//tensorflow/lite/profiling:time"],
# )
