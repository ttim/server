#!/bin/bash
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

source ../common/util.sh
RET=0
BASE_DIR=$(pwd)
NUM_GPUS=${NUM_GPUS:=1}
TENSORRTLLM_BACKEND_REPO_TAG=main

MODEL_NAME="ensemble"
NAME="tensorrt_llm_benchmarking_test"
MODEL_REPOSITORY="$(pwd)/triton_model_repository"
TENSORRTLLM_BACKEND_DIR="/opt/tritonserver/tensorrtllm_backend"
GPT_DIR="$TENSORRTLLM_BACKEND_DIR/tensorrt_llm/examples/gpt"
TOKENIZER_DIR="$GPT_DIR/gpt2"
ENGINES_DIR="${BASE_DIR}/engines/inflight_batcher_llm/${NUM_GPUS}_gpu"

TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
SERVER_LOG="${NAME}_server.log"
SERVER_TIMEOUT=${SERVER_TIMEOUT:=120}
SERVER_ARGS="--model-repository=${MODEL_REPOSITORY} --disable-auto-complete-config --backend-directory=${BACKEND_DIR} \
            --backend-config=python,shm-region-prefix-name=prefix0_"

# Select the GPUs that will be available to the inference server.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=0}

function build_tensorrt_engine_inflight_batcher {
    cd ${GPT_DIR}
    python3 build.py --model_dir="./c-model/gpt2/${NUM_GPUS}-gpu/" \
        --world_size="${NUM_GPUS}" \
        --dtype float16 \
        --use_inflight_batching \
        --use_gpt_attention_plugin float16 \
        --paged_kv_cache \
        --use_gemm_plugin float16 \
        --remove_input_padding \
        --use_layernorm_plugin float16 \
        --hidden_act gelu \
        --parallel_build \
        --output_dir="${ENGINES_DIR}"
    cd ${BASE_DIR}
}

function build_base_model {
    cd ${GPT_DIR}

    # Download weights from HuggingFace Transformers
    rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
    pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd

    # Convert weights from HF Tranformers to FT format
    python3 hf_gpt_convert.py -p 8 -i gpt2 -o ./c-model/gpt2 --tensor-parallelism ${NUM_GPUS} --storage-type float16
    cd ${BASE_DIR}
}

function install_trt_llm {
    # Install CMake
    bash ${TENSORRTLLM_BACKEND_DIR}/tensorrt_llm/docker/common/install_cmake.sh
    export PATH="/usr/local/cmake/bin:${PATH}"

    # PyTorch needs to be built from source for aarch64
    ARCH="$(uname -i)"
    if [ "${ARCH}" = "aarch64" ]; then
        TORCH_INSTALL_TYPE="src_non_cxx11_abi"
    else
        TORCH_INSTALL_TYPE="pypi"
    fi
    (cd $TENSORRTLLM_BACKEND_DIR/tensorrt_llm &&
        bash docker/common/install_pytorch.sh $TORCH_INSTALL_TYPE &&
        python3 ./scripts/build_wheel.py --trt_root="${TRT_ROOT}" &&
        pip3 install ./build/tensorrt_llm*.whl)
}

function replace_config_tags {
    tag_to_replace="${1}"
    new_value="${2}"
    config_file_path="${3}"
    sed -i "s|${tag_to_replace}|${new_value}|g" ${config_file_path}
}

rm -rf $TENSORRTLLM_BACKEND_DIR && mkdir $TENSORRTLLM_BACKEND_DIR
apt-get update && apt-get install git-lfs -y --no-install-recommends
git clone --single-branch --depth=1 -b ${TENSORRTLLM_BACKEND_REPO_TAG} https://github.com/triton-inference-server/tensorrtllm_backend.git $TENSORRTLLM_BACKEND_DIR
cd $TENSORRTLLM_BACKEND_DIR && git lfs install && git submodule update --init --recursive

install_trt_llm
build_base_model
build_tensorrt_engine_inflight_batcher
pip3 install tritonclient

rm -rf ${MODEL_REPOSITORY} && mkdir ${MODEL_REPOSITORY}
cp -r ${TENSORRTLLM_BACKEND_DIR}/all_models/inflight_batcher_llm/* ${MODEL_REPOSITORY}
cp -r ${ENGINES_DIR}/* ${MODEL_REPOSITORY}/tensorrt_llm/1

replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_REPOSITORY}/ensemble/config.pbtxt"
replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_REPOSITORY}/preprocessing/config.pbtxt"
replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_REPOSITORY}/postprocessing/config.pbtxt"
replace_config_tags '${tokenizer_dir}' "${TOKENIZER_DIR}/" "${MODEL_REPOSITORY}/preprocessing/config.pbtxt"
replace_config_tags '${tokenizer_dir}' "${TOKENIZER_DIR}/" "${MODEL_REPOSITORY}/postprocessing/config.pbtxt"
replace_config_tags '${tokenizer_type}' 'auto' "${MODEL_REPOSITORY}/preprocessing/config.pbtxt"
replace_config_tags '${tokenizer_type}' 'auto' "${MODEL_REPOSITORY}/postprocessing/config.pbtxt"
replace_config_tags '${decoupled_mode}' 'True' "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
replace_config_tags '${batching_strategy}' 'inflight_fused_batching' "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
replace_config_tags '${max_queue_delay_microseconds}' "0" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
replace_config_tags '${engine_dir}' "${MODEL_REPOSITORY}/tensorrt_llm/1/" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
replace_config_tags "model_version: -1" "model_version: 1" "${MODEL_REPOSITORY}/ensemble/config.pbtxt"

run_server
if (($SERVER_PID == 0)); then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

ARCH="amd64"
STATIC_BATCH=1
INSTANCE_CNT=1
CONCURRENCY=1
MODEL_FRAMEWORK="tensorrt"
PERF_CLIENT="perf_analyzer"
REPORTER=../common/reporter.py
INPUT_DATA="./input_data.json"
PERF_CLIENT_PROTOCOL="grpc"
EXPORT_FILE=profile-export-tensorrt-llm-model.json
rm -rf $EXPORT_FILE *.tjson *.json *.csv

echo '{
  "data": [
    {
      "text_input": ["Hello, my name is"],
      "stream": [true],
      "max_tokens": [16],
      "bad_words": [""],
      "stop_words": [""]
    }
  ]
}' >$INPUT_DATA

PERF_CLIENT_ARGS="-v -m $MODEL_NAME -i $PERF_CLIENT_PROTOCOL --async --streaming --input-data=$INPUT_DATA --profile-export-file=$EXPORT_FILE \
                  --shape=text_input:1 --shape=max_tokens:1 --shape=bad_words:1 --shape=stop_words:1 --measurement-mode=count_windows \
                  --concurrency-range=$CONCURRENCY --measurement-request-count=10 --stability-percentage=999"

set +e
$PERF_CLIENT $PERF_CLIENT_ARGS -f ${NAME}.csv 2>&1 | tee ${NAME}_perf_analyzer.log
set +o pipefail
set -e

kill $SERVER_PID
wait $SERVER_PID
rm -rf $MODEL_REPO $INPUT_DATA

echo -e "[{\"s_benchmark_kind\":\"benchmark_perf\"," >>${NAME}.tjson
echo -e "\"s_benchmark_repo_branch\":\"${BENCHMARK_REPO_BRANCH}\"," >>${NAME}.tjson
echo -e "\"s_benchmark_name\":\"${NAME}\"," >>${NAME}.tjson
echo -e "\"s_server\":\"triton\"," >>${NAME}.tjson
echo -e "\"s_protocol\":\"${PERF_CLIENT_PROTOCOL}\"," >>${NAME}.tjson
echo -e "\"s_framework\":\"${MODEL_FRAMEWORK}\"," >>${NAME}.tjson
echo -e "\"s_model\":\"${MODEL_NAME}\"," >>${NAME}.tjson
echo -e "\"s_concurrency\":\"${CONCURRENCY}\"," >>${NAME}.tjson
echo -e "\"l_batch_size\":${STATIC_BATCH}," >>${NAME}.tjson
echo -e "\"l_instance_count\":${INSTANCE_CNT}," >>${NAME}.tjson
echo -e "\"s_architecture\":\"${ARCH}\"}]" >>${NAME}.tjson

if [ -f $REPORTER ]; then
    set +e

    URL_FLAG=
    if [ ! -z ${BENCHMARK_REPORTER_URL} ]; then
        URL_FLAG="-u ${BENCHMARK_REPORTER_URL}"
    fi

    python3 $REPORTER -v -o ${NAME}.json --csv ${NAME}.csv ${URL_FLAG} ${NAME}.tjson
    if (($? != 0)); then
        RET=1
    fi

    set -e
fi

if (($RET == 0)); then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
