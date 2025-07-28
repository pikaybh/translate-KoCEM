import os

from fire import Fire

from modules.translation_chain import translate_and_evaluate_chain
from modules.HitL import human_review_from_parquet
from schemas import KOCEM_CONFIGS, T_KoCEM_Config, T_Split
from utils import get_kocem_dataset, setup_logger


logger = setup_logger(__name__)


def translate_only(
    cache_dir: str = None, 
    max_retries: int = 5, 
    subdataset: T_KoCEM_Config = None, 
    split: T_Split = None
) -> None:
    """
    Runs the translation and review pipeline for a selected KoCEM config(subdataset) and split.

    Args:
        cache_dir (str): Directory to cache the dataset.
        max_retries (int): Maximum number of feedback-based retries per sample.
        subdataset (str): KoCEM config name to process (required).
        split (str): Dataset split to process (e.g., 'train', 'test', 'validation'). If falsy, process all splits.
    """
    output_dir = os.getenv("OUTPUT_DIR", "output")
    cache_dir = cache_dir or os.getenv("CACHE_DIR", ".cache")
    eval_result_dir = os.getenv("EVAL_DIR", "eval_results")

    configs_to_run = [subdataset] if subdataset else KOCEM_CONFIGS
    for config in configs_to_run:
        logger.debug(f"[CONFIG] {config} 시작...")
        ds = get_kocem_dataset(config, cache_dir=cache_dir)
        
        os.makedirs(f"{output_dir}/{eval_result_dir}/{config}", exist_ok=True)
        
        splits_to_run = [split] if split else list(ds.keys())
        for split_name in splits_to_run:
            logger.info(f"[START] {config}/{split_name} 번역+평가")
            output_path = f"{output_dir}/{eval_result_dir}/{config}/{split_name}.parquet"
            translate_and_evaluate_chain(ds, split=split_name, cache_dir=cache_dir, output_path=output_path, max_retries=max_retries)


def run_pipeline(
    cache_dir: str = None, 
    max_retries: int = 5, 
    subdataset: T_KoCEM_Config = None, 
    split: T_Split = None
) -> None:
    """
    Runs the translation and review pipeline for a selected KoCEM config(subdataset) and split.

    Args:
        cache_dir (str): Directory to cache the dataset.
        max_retries (int): Maximum number of feedback-based retries per sample.
        subdataset (str): KoCEM config name to process (required).
        split (str): Dataset split to process (e.g., 'train', 'test', 'validation'). If falsy, process all splits.
    """
    output_dir = os.getenv("OUTPUT_DIR", "output")
    cache_dir = cache_dir or os.getenv("CACHE_DIR", ".cache")
    eval_result_dir = os.getenv("EVAL_DIR", "eval_results")
    human_reviewed_dir = os.getenv("REVIEWED_DIR", "human_reviewed")

    configs_to_run = [subdataset] if subdataset else KOCEM_CONFIGS
    for config in configs_to_run:
        logger.debug(f"[CONFIG] {config} 시작...")
        ds = get_kocem_dataset(config, cache_dir=cache_dir)
        
        os.makedirs(f"{output_dir}/{eval_result_dir}/{config}", exist_ok=True)
        os.makedirs(f"{output_dir}/{human_reviewed_dir}/{config}", exist_ok=True)
        
        splits_to_run = [split] if split else list(ds.keys())
        for split_name in splits_to_run:
            logger.info(f"[START] {config}/{split_name} 번역+평가")
            output_path = f"{output_dir}/{eval_result_dir}/{config}/{split_name}.parquet"
            translated_path = translate_and_evaluate_chain(ds, split=split_name, cache_dir=cache_dir, output_path=output_path, max_retries=max_retries)
            
            logger.info(f"[START] {config}/{split_name} human review")
            human_review_path = f"{output_dir}/{human_reviewed_dir}/{config}/{split_name}.parquet"
            human_review_from_parquet(translated_path, output_path=human_review_path)


def main(*args, **kwargs):
    """
    Main entry point for running the pipeline.
    """
    if not args:
        # No command, show help or process all configs
        logger.info("No command provided. Use one of: ls, configs, translate, run, pipeline.")
        return
    
    cmd = args[0]
    if cmd in ["ls", "configs"]:
        logger.info(f"Available KoCEM configs:\n- {'\n- '.join(KOCEM_CONFIGS)}")
    elif cmd in ["translate"]:
        translate_only(**kwargs)
    elif cmd in ["run", "pipeline"]:
        run_pipeline(**kwargs)
    else:
        logger.error(f"Unknown command: {cmd}")


if __name__ == "__main__":
    Fire(main)
