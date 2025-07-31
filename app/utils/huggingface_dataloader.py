from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset, DatasetDict

from .logs import setup_logger


logger = globals()['logger'] \
    if 'logger' in globals() \
    else setup_logger(__name__)


def get_kocem_dataset(config: str, cache_dir: str = '.cache') -> DatasetDict:
    """
    Downloads and caches the KoCEM dataset from Huggingface for a specific config.

    Args:
        config (str): KoCEM subdataset/config name (e.g. 'Architectural_Planning').
        cache_dir (str): Directory to cache the dataset.

    Returns:
        DatasetDict: Huggingface datasets object for KoCEM.
    """
    if not config:
        raise ValueError("Config name is required. Available configs: ['Architectural_Planning', 'Building_System', 'Comprehensive_Understanding', 'Construction_Management', 'Drawing_Interpretation', 'Domain_Reasoning', 'Interior', 'Industry_Jargon', 'Materials', 'Safety_Management', 'Standard_Nomenclature', 'Structural_Engineering']")
    return load_dataset('pikaybh/KoCEM', config, cache_dir=cache_dir)


def dict2list(data: dict) -> list:
    """
    Converts a dictionary to a list of its values.

    Args:
        data (dict): Input dictionary.

    Returns:
        list: List of values from the dictionary.
    """
    return list(data.values())


def main() -> None:
    """
    Example usage for KoCEM dataset download and logging.
    """
    with ThreadPoolExecutor() as executor:
        future = executor.submit(get_kocem_dataset)
        ds = future.result()
    logger.info(ds)

if __name__ == "__main__":
    main()

__all__ = ['get_kocem_dataset', 'dict2list']