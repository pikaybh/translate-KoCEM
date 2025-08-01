import os
import yaml

from dotenv import load_dotenv

from schemas import Quiz, Option

load_dotenv()
ROOT_ALIAS = os.getenv('SRC', 'app')
PROMPT_DIR = os.getenv('PROMPT_DIR', 'prompts')


def _get_rule_prompt(
    data: dict,
    rule: str | None = None
) -> str | dict[str, str]:
    """
    Loads a rule-based prompt from a dictionary.

    Args:
        data (dict): Dictionary containing prompts.
        rule (str): Rule to filter the prompts.

    Returns:
        str | dict[str, str]: The prompt template as a string or a dictionary of prompts.
    """
    if rule:
        if rule not in data:
            raise ValueError(f"Rule '{rule}' not found in the prompt data.")
        return data[rule]
    return data


def _get_versioned_prompt(
    data: dict,
    version: str | None = None
) -> str | dict[str, str]:
    """
    Loads a versioned prompt from a dictionary.

    Args:
        data (dict): Dictionary containing prompts.
        version (str): Version of the prompt to load.

    Returns:
        str | dict[str, str]: The prompt template as a string or a dictionary of prompts.
    """
    if version:
        # 재귀적으로 version key를 찾음
        def find_version(d):
            if isinstance(d, dict):
                if version in d:
                    return d[version]
                for v in d.values():
                    found = find_version(v)
                    if found is not None:
                        return found
            return None
        result = find_version(data)
        if result is None:
            raise ValueError(f"Version '{version}' not found in the prompt data.")
        return result
    return data


def load_prompt(
    template: str, 
    rule: str | None = None, 
    version: str | None = None
) -> str | dict[str, str]:
    """
    Loads a prompt template from the specified directory.

    Args:
        template (str): Name of the template file (without extension).
        version (str): Version of the prompt to load.

    Returns:
        str | dict[str, str]: The prompt template as a string or a dictionary of prompts.
    """

    file_path = os.path.join(ROOT_ALIAS, PROMPT_DIR, f"{template}.yaml")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt template '{template}' not found in {PROMPT_DIR}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        prompts = yaml.safe_load(file)

    ruled_prompts = _get_rule_prompt(data=prompts, rule=rule)
    versioned_prompts = _get_versioned_prompt(data=ruled_prompts, version=version)

    # dict가 반환되면 가장 안쪽 string만 반환
    prompts_final = versioned_prompts
    while isinstance(prompts_final, dict):
        prompts_final = next(iter(prompts_final.values()))
    return prompts_final


def label_alpha_numeric(idx: int) -> str:
    """
    Converts an index to an alphanumeric label (A, B, C, ...).
    
    Args:
        idx (int): Index to convert.
    
    Returns:
        str: Alphanumeric label corresponding to the index.
    """
    return chr(ord('A') + idx) if idx < 26 else f"AA{idx - 26}"


def construct_option(
    option: Option
) -> str:
    """
    Constructs a string representation of a single option.

    Args:
        option (Option): The Option object containing label and value.

    Returns:
        str: Formatted string of the option.
    """
    return f"{option.label}: {option.value.strip()}"


def construct_options(
    options: list[Option]
) -> str:
    """
    Constructs a string representation of options from a Quiz object.

    Args:
        quiz (Quiz): The Quiz object containing options.

    Returns:
        str: Formatted string of options.
    """
    # options_str = []
    # for option in options:
        # label = label_alpha_numeric(idx)
        # description = option.description.strip()
        # options_str.append(f"{label}: {description}")
        # options_str.append(construct_option(option))
    
    return "\n    ".join([construct_option(option) for option in options])


__all__ = ['load_prompt', 'label_alpha_numeric', 'construct_option', 'construct_options']