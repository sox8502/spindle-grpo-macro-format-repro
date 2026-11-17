from copy import deepcopy

from spindle.configs.qwen35_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "star-sft-qwen35-4b-65k-4h200"
    max_context_length = 65536
    trainer_gpu = "H200"
    inference_gpu = "H200"
    inference_max_replicas = 6
    # Own frontend and checkpoints keep the decision-model-4b service untouched; base weights are shared read-only.
    platform = deepcopy(Parent.platform)
    platform["frontend"] = "star-sft-65k"
    platform["modal"]["environment"] = "main"
    platform["secrets"].update(api="star-sft-65k-api", sampler_proxy="decision-model-spindle-proxy", huggingface="")
    platform["storage"].update(assets="decision-model-spindle-assets", checkpoints="star-sft-65k-checkpoints",
                               bulletin="star-sft-65k-bulletin")


config = Config()
