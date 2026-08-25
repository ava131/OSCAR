export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket

MODEL_PATH=/home/z50058184/qwen3.5-4b \
ROTATION_ROOT=rotzoo/Qwen3.5-4B \
CUDA_VISIBLE_DEVICES=0,1 \
TP_SIZE=1 \
PORT=6068 \
bash scripts/start_qwen35_4b_oscar_sglang.sh
