# The installed extension must use the same Torch/CUDA ABI as the runtime.
ARG BASE_IMAGE=myllmbox/qwen38-flash-next-cluster-vllm@sha256:861ac752164e0d723c5eff3f876586c6678c26ad4a516112c48745f6a101ff4d
FROM ${BASE_IMAGE} AS builder
WORKDIR /opt/mbx-exact-gdn-build
COPY . .
# No JIT compiler or network download at serving startup. One compiler process
# bounds build memory on a Spark that is already hosting a model.
RUN MAX_JOBS=1 python3 setup.py bdist_wheel

FROM ${BASE_IMAGE}
COPY --from=builder /opt/mbx-exact-gdn-build/dist/ /opt/mbx/exact-gdn/wheels/
COPY install.py source-manifest.json /opt/mbx/exact-gdn/
COPY tests /opt/mbx/exact-gdn/tests
RUN python3 -m pip install --no-deps --no-index /opt/mbx/exact-gdn/wheels/*.whl \
    && python3 /opt/mbx/exact-gdn/install.py
ENV MBX_EXACT_GDN_REPLAY=1
LABEL org.opencontainers.image.description="Pinned v6 recipe with exact native FP32 GDN replay"
