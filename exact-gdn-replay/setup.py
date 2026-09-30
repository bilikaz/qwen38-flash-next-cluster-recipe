from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mbx-gdn-replay",
    version="0.1.0",
    description="Exact native FP32 GDN replay for the pinned two-GB10 v6 recipe",
    license="Apache-2.0",
    packages=["mbx_gdn_replay"],
    package_data={"mbx_gdn_replay": ["source-manifest.json"]},
    ext_modules=[
        CUDAExtension(
            "mbx_gdn_replay._C",
            sources=["csrc/ops.cpp", "csrc/verify.cu", "csrc/commit.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17", "-arch=sm_120f"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    python_requires=">=3.12",
)
