from dataclasses import dataclass
from typing import Literal

import typer

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-4B" # model wo experts, model w experts, big model like qwen235b
MODEL_TYPE = "qwen3-4B"

# For h100 80g * 8:
# training gpu cannot be only 1 because of oom


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["nccl", "rdma"] = "nccl"
    # enable single node
    # tuning training/rollout gpus: --num-train-gpus 2 --training-tp-size 2 --num-rollout-gpus 4 --rollout-tp-size 4
    # enable different Protocol By: PROTOCOL=NCCL
    # docker: xinji1/slime_rdma:rdma in condor

    # TODO: Right now ep=pp=1
    
    num_train_gpus: int = 1 # 1, 2, 4
    num_rollout_gpus: int = 1 # 1, 2, 4
    # training/rollout parallel
    training_tp_size: int = 1 #  1, 2, 4
    rollout_tp_size: int = 1 #  1, 2, 4
    

    # TODO:
    # parallelism: ep, pp
    # multi-nodes
    # check actor num nodes > 1
    # better performance



def prepare(args: ScriptArgs):
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command("hf download Qwen/Qwen3-4B --local-dir /root/models/Qwen3-4B")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    num_gpus = args.num_train_gpus + args.num_rollout_gpus
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=num_gpus)


def execute(args: ScriptArgs):
    num_gpus = args.num_train_gpus + args.num_rollout_gpus
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 100 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
        "--balance-data "
    )
    # Training parallellism settings
    perf_args = (
        f"--tensor-model-parallel-size {args.training_tp_size} "
        # "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        # "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_tp_size} " # basically equal to tp_size in sglang
        f"--rollout-num-gpus {args.num_rollout_gpus} "
        "--sglang-mem-fraction-static 0.8 "
    )
    if args.mode == "rdma":
        sglang_args += "--sglang-remote-instance-weight-loader-support-transfer-engine "

    # ci_args = "--ci-test "

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 " 
        f"--actor-num-gpus-per-node {args.num_train_gpus} "
        # 1GB buffer for weight update
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        f"--check-weight-update-equal "
    )
    if args.mode == "rdma":
        misc_args += "--update-weight-transfer-mode rdma "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        # f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
        extra_env_vars={"RAY_DEBUG": "1"},
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
