import os

import mlflow
from fire import Fire

from modules import translate_and_evaluate_chain, human_review_from_parquet
from schemas import KOCEM_CONFIGS, KoCEMConfigType, KoCEMDataSplitType
from utils import get_kocem_dataset, refresh_terminal, setup_logger


def translate_only(
    cache_dir: str = None, 
    max_retries: int = 5, 
    subdataset: KoCEMConfigType = None, 
    split: KoCEMDataSplitType = None,
    ignore: KoCEMDataSplitType = None,
    overwrite: bool = False
):
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
        if ignore:
            splits_to_run = [s for s in splits_to_run if s != ignore]
        for split_name in splits_to_run:
            refresh_terminal()
            logger.info(f"[START] {config}/{split_name} 번역+평가")
            output_path = f"{output_dir}/{eval_result_dir}/{config}/{split_name}.parquet"
            if not overwrite and os.path.exists(output_path):
                logger.info(f"[SKIP] {output_path} already exists and {overwrite=}. Skipping.")
                continue
            try:
                translate_and_evaluate_chain(ds, split=split_name, cache_dir=cache_dir, output_path=output_path, max_retries=max_retries)
            except Exception as e:
                logger.error(f"Error during translation and evaluation for {config}/{split_name}: {e}")
                logger.info(f"Skipping {config}/{split_name} due to error.")
                continue


def from_translation_chain(
    subdataset: KoCEMConfigType = None, 
    split: KoCEMDataSplitType = None
) -> None:
    """
    Runs the translation and review pipeline for a selected KoCEM config(subdataset) and split.
    Args:
        subdataset (str): KoCEM config name to process (required).
        split (str): Dataset split to process (e.g., 'train', 'test', 'validation'). If falsy, process all splits.
    """
    output_dir = os.getenv("OUTPUT_DIR", "output")
    eval_result_dir = os.getenv("EVAL_DIR", "eval_results")
    human_reviewed_dir = os.getenv("REVIEWED_DIR", "human_reviewed")

    configs_to_run = [subdataset] if subdataset else KOCEM_CONFIGS
    for config in configs_to_run:
        logger.debug(f"[CONFIG] {config} 시작...")
        ds = get_kocem_dataset(config)
        
        os.makedirs(f"{output_dir}/{eval_result_dir}/{config}", exist_ok=True)
        os.makedirs(f"{output_dir}/{human_reviewed_dir}/{config}", exist_ok=True)
        
        splits_to_run = [split] if split else list(ds.keys())
        for split_name in splits_to_run:
            logger.info(f"[START] {config}/{split_name} human review")
            
            translated_path = f"{output_dir}/{eval_result_dir}/{config}/{split_name}.parquet"
            human_review_path = f"{output_dir}/{human_reviewed_dir}/{config}/{split_name}.parquet"
            human_review_from_parquet(translated_path, output_path=human_review_path)


def from_HitL_output(
    subdataset: KoCEMConfigType = None, 
    split: KoCEMDataSplitType = None,
    again: bool = False
) -> None:
    """
    Runs the translation and review pipeline for a selected KoCEM config(subdataset) and split.
    Args:
        subdataset (str): KoCEM config name to process (required).
        split (str): Dataset split to process (e.g., 'train', 'test', 'validation'). If falsy, process all splits.
    """
    output_dir = os.getenv("OUTPUT_DIR", "output")
    human_reviewed_dir = os.getenv("REVIEWED_DIR", "human_reviewed")

    configs_to_run = [subdataset] if subdataset else KOCEM_CONFIGS
    for config in configs_to_run:
        logger.debug(f"[CONFIG] {config} 시작...")
        ds = get_kocem_dataset(config)
        os.makedirs(f"{output_dir}/{human_reviewed_dir}/{config}", exist_ok=True)
        splits_to_run = [split] if split else list(ds.keys())
        for split_name in splits_to_run:
            logger.info(f"[START] {config}/{split_name} human review")
            human_review_path = f"{output_dir}/{human_reviewed_dir}/{config}/{split_name}.parquet"
            if again:
                # 모든 row의 human_feedback을 True로 강제하여 다시 리뷰
                import pandas as pd
                try:
                    df = pd.read_parquet(human_review_path)
                except Exception:
                    logger.error(f"파일을 읽을 수 없습니다: {human_review_path}")
                    continue
                df["human_feedback"] = True
                df.to_parquet(human_review_path, index=False)
            human_review_from_parquet(human_review_path, output_path=human_review_path)


def run_pipeline(
    cache_dir: str = None, 
    max_retries: int = 5, 
    subdataset: KoCEMConfigType = None, 
    split: KoCEMDataSplitType = None
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
    elif cmd in ["anno", "review", "human_review", "annotation"]:
        from_translation_chain(**kwargs)
    elif cmd in ["resume"]:
        again = kwargs.pop("again", False)
        from_HitL_output(**kwargs, again=again)
    else:
        logger.error(f"Unknown command: {cmd}")


if __name__ == "__main__":
    logger = setup_logger(__name__)
    mlflow.openai.autolog()
    Fire(main)
