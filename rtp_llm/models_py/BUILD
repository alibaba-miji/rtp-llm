load("//bazel:arch_select.bzl", "requirement")

py_library(
    name = "utils",
    srcs = glob([
        "utils/*.py",
    ]),
)

py_library(
    name = "distributed",
    srcs = glob([
        "distributed/*.py",
    ]),
    deps = [
        "//rtp_llm/models_py/distributed:deepep_wrapper",
    ],
    visibility = ["//visibility:public"],
)

# flashinfer-python is only available for CUDA12
flashinfer = ["flashinfer-python"]
requirement(flashinfer)

py_library(
    name = "modules",
    srcs = glob([
        "modules/*.py",
        "modules/**/*.py",
    ]),
    deps = [
        ":utils",
        "//rtp_llm/models_py/triton_kernels:triton_kernels",
        ":distributed",
    ] + select({
        "@//:using_cuda12": flashinfer,
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_library(
    name = "kernels",
    srcs = glob([
        "kernels/*.py",
        "kernels/**/*.py",
    ]),
    deps = [
        ":utils",
    ],
)

py_library(
    name = "models",
    srcs = glob([
        "model_desc/*.py",
        "model_desc/cuda_graph/*.py"
    ]),
    deps = [
        ":utils",
        ":modules",
        ":kernels",
        "//rtp_llm/model_loader:loader",
    ] + select({
        "@//:using_cuda12": [
            "@deep_gemm_ext//:deep_gemm"
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
