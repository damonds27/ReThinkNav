#!/usr/bin/env python3
import os
import yaml
import torch
import random
import argparse
import numpy as np
from habitat import logger
import habitat_extensions  # noqa: F401
import vlnce_baselines     # noqa: F401
from vlnce_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_extensions.measures import TopDownMapVLNCE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        required=True,
        help="experiment id that matches to exp-id in Notion log",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument('--local_rank', type=int, default=0, help="local gpu id")
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        help="The LLM model to be used (e.g., gpt-4o-2024-08-06, Qwen/Qwen2.5-1.5B)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for accessing the LLM service",
    )
    parser.add_argument(
        "--target_episode_ids", 
        type=int,
        default=0,
        help="target episode ids to load",
    )
    args = parser.parse_args()
    run_exp(**vars(args))




def modify_config(config_path, output_path, new_target_episode_id):
    # 读取配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 修改目标字段
    if 'DATASET' in config and 'TARGET_EPISODE_IDS' in config['DATASET']:
        config['DATASET']['TARGET_EPISODE_IDS'] = new_target_episode_id
        print(f"TARGET_EPISODE_IDS 已更新为 {new_target_episode_id}")
    else:
        print("未找到 DATASET.TARGET_EPISODE_IDS 字段，未作修改。")

    # 写回文件（也可以覆盖原文件）
    with open(output_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
def run_exp(exp_name: str, exp_config: str, 
            opts=None, local_rank=None,
            llm: str = None, api_key: str = None, target_episode_ids: int = 181) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
        llm: The LLM model to be used (e.g., gpt-4o-2024-08-06).
        api_key: API key for accessing the LLM service.
    Returns:
        None.
    """
    config_path = "/home/liaolin/project/open-nav/Open-Nav/habitat_extensions/config/vlnce_task.yaml"           # 原始配置文件路径
    output_path = "/home/liaolin/project/open-nav/Open-Nav/habitat_extensions/config/vlnce_task.yaml"  # 修改后保存路径
    new_target_episode_id = target_episode_ids           # 要修改的目标值
    modify_config(config_path, output_path, new_target_episode_id)
    
    config = get_config(exp_config, opts)
    # print("config: ", config)
    config.defrost()

    config.CHECKPOINT_FOLDER += exp_name
    if os.path.isdir(config.EVAL_CKPT_PATH_DIR):
        config.EVAL_CKPT_PATH_DIR += exp_name
    config.RESULTS_DIR += exp_name
    config.LOG_FILE = exp_name + '_' + config.LOG_FILE

    config.TASK_CONFIG.SEED = 0

    config.local_rank = local_rank

    config.TARGET_EPISODE_IDS = target_episode_ids




    if llm is not None:
        config.LLM = llm
    if api_key is not None:
        config.API_KEY = api_key

    config.freeze()
    
    # Check if the 'logs/running_log' directory exists; if not, create it
    log_dir = 'logs/running_log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Add the file handler for logging
    logger.add_filehandler(os.path.join(log_dir, config.LOG_FILE))

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    trainer.eval()
    
if __name__ == "__main__":
    __spec__ = None 
    main()