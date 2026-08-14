export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1


torchrun --standalone --nproc_per_node=4 train.py \
    --outdir=training-runs/cifar10/260425_151340_fm-cifar10-trig \
    --data=../datasets/cifar10.zip \
    --preset=fm-cifar10-trig \
    --max-batch-gpu=256 \
    --no-fp16 \
    --status=1000 \
    --snapshot=25000 \
    --checkpoint=50000 \
    --metrics=25000 \
    --metric-names=fid,fd_dinov2,mind,mind_dinov2 \
    --metric-num-samples=20000 \
    --mind-num-samples=5000 \
    --metric-ref=../fid-refs/cifar10.pkl \
    --metric-batch-size=512 \
    "$@"
