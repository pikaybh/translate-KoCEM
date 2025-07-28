import os, json
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate , HumanMessagePromptTemplate

from modules.evaluator import evaluate_translation
from utils import get_kocem_dataset, load_prompt, save_parquet, setup_logger


logger = globals()['logger'] \
    if 'logger' in globals() \
    else setup_logger(__name__)
load_dotenv()

# 번역 프롬프트 템플릿 및 체인 생성 (ChatPromptTemplate for chat model)
translation_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        load_prompt(
            template="translation", 
            rule="system", 
            version="v2"
        )
    ),
    HumanMessagePromptTemplate.from_template(
        load_prompt(
            template="translation", 
            rule="human", 
            version="v2"
        )
    )
])
translation_chain = translation_prompt | ChatOpenAI(model="gpt-4.1")


def extract_feedback(eval_result):
    """
    Robustly extract feedback from evaluator result dict.
    """
    feedbacks = []
    for k, v in eval_result.items():
        if isinstance(v, dict):
            for key in ['reason', 'feedback', 'explanation', 'reasoning']:
                if key in v and v[key]:
                    feedbacks.append(str(v[key]))
        elif isinstance(v, str) and v and k in ['reason', 'feedback', 'explanation', 'reasoning']:
            feedbacks.append(v)
    # If nothing found, try top-level 'reasoning'
    if not feedbacks and 'reasoning' in eval_result:
        feedbacks.append(str(eval_result['reasoning']))
    return '\n'.join(feedbacks).strip()


def translate_sample_with_evaluation(
    ko_text: str,
    max_retries: int = 2,
) -> dict:
    """
    Translates a single Korean sentence to English with evaluator feedback loop.

    Args:
        ko_text (str): Korean text to translate.
        max_retries (int): Maximum number of feedback-based retries.
    
    Returns:
        dict: Dictionary containing original Korean text, translated English text, evaluation results, and number of retries.
    """

    attempt = 0
    en_text = None
    feedback = None
    while attempt <= max_retries:
        # Only pass variables defined in the prompt
        if feedback:
            en_text = translation_chain.invoke({"text": ko_text, "feedback": feedback}).content
        else:
            en_text = translation_chain.invoke({"text": ko_text, "feedback": ""}).content

        eval_result = evaluate_translation(ko_text, en_text)
        # Unify pass/fail logic: accept Y, YES, y, yes
        value = str(eval_result.get('value', '')).strip().lower()
        score = float(eval_result.get('score', 0))
        passed = value in ['y', 'yes'] or score >= 0.8
        logger.debug(f"Try {attempt}: {ko_text} -> {en_text} | Eval: {eval_result}")

        if passed:
            break

        feedback = extract_feedback(eval_result)
        attempt += 1

    return {"ko": ko_text, "en": en_text, "eval": eval_result, "retries": attempt}


def translate_text(text: str) -> str:
    """ 
    Translates a single text string using the translation chain.
    
    Args:
        text (str): Korean text to translate.
    
    Returns:
        str: Translated English text.
    """
    return translation_chain.invoke({"text": text, "feedback": ""}).content if text else ""


def translate_fields_with_evaluation(item: dict, max_retries: int = 2) -> dict:
    """
    Translates fields of a dataset item with evaluation loop.

    Args:
        item (dict): Dataset item containing fields to translate.
    
    Returns:
        dict: Translated item with evaluation results.
    """

    # question
    ko_question = item.get("question", "")
    # Feedback/evaluation loop for question
    eval_loop = []
    feedback = None
    attempt = 0
    en_question = None
    while attempt <= max_retries:
        if feedback:
            en_question_try = translation_chain.invoke({"text": ko_question, "feedback": feedback}).content
        else:
            en_question_try = translation_chain.invoke({"text": ko_question, "feedback": ""}).content
        eval_result = evaluate_translation(ko_question, en_question_try)
        passed = str(eval_result.get('value', '')).strip().upper() == 'Y'
        eval_loop.append({
            "attempt": attempt,
            "en_question": en_question_try,
            "feedback": feedback or "",
            **eval_result
        })
        if passed:
            en_question = en_question_try
            break
        feedback = extract_feedback(eval_result)
        attempt += 1
    if en_question is None:
        en_question = en_question_try

    # options robust parsing
    ko_options_raw = item.get("options", "")
    options_list = []
    if ko_options_raw:
        try:
            options_list = json.loads(ko_options_raw)
            if not isinstance(options_list, list):
                options_list = [ko_options_raw]
        except Exception:
            try:
                options_list = eval(ko_options_raw)
                if not isinstance(options_list, list):
                    options_list = [ko_options_raw]
            except Exception:
                for delim in ["|", ",", ";", "/", "\\n"]:
                    if delim in ko_options_raw:
                        options_list = [opt.strip() for opt in ko_options_raw.split(delim) if opt.strip()]
                        break
                if not options_list:
                    options_list = [ko_options_raw]
    if len(options_list) > 1 and all(len(opt) == 1 for opt in options_list):
        options_list = ["".join(options_list)]

    # Option translation with individual feedback loop
    option_labels = [chr(65 + i) for i in range(len(options_list))] if len(options_list) > 1 else []
    ko_options_labeled = [f"{label}. {opt}" for label, opt in zip(option_labels, options_list)] if option_labels else options_list
    en_options_labeled = []
    for idx, opt in enumerate(options_list):
        feedback = None
        attempt = 0
        en_option = None
        while attempt <= max_retries:
            if feedback:
                en_option_try = translation_chain.invoke({"text": opt, "feedback": feedback}).content
            else:
                en_option_try = translation_chain.invoke({"text": opt, "feedback": ""}).content
            eval_result = evaluate_translation(opt, en_option_try)
            passed = str(eval_result.get('value', '')).strip().upper() == 'Y'
            if passed:
                en_option = en_option_try
                break
            feedback = extract_feedback(eval_result)
            attempt += 1
        if en_option is None:
            en_option = en_option_try
        if option_labels:
            en_options_labeled.append(f"{option_labels[idx]}. {en_option}")
        else:
            en_options_labeled.append(en_option)

    # answer
    ko_answer = item.get("answer", "")
    en_answer = translate_text(ko_answer)

    # explanation
    ko_explanation = item.get("explanation", "")
    en_explanation = translate_text(ko_explanation)

    result = dict(item)
    result.update({
        "ko_question": ko_question,
        "en_question": en_question,
        "ko_options": ko_options_labeled,
        "en_options": en_options_labeled,
        "ko_answer": ko_answer,
        "en_answer": en_answer,
        "ko_explanation": ko_explanation,
        "en_explanation": en_explanation,
        "eval": eval_loop[-1] if eval_loop else {},
        "eval_loop": eval_loop
    })
    return result


def process_dataset_with_evaluation(
    dataset: Any,
    split: str = "train",
    max_retries: int = 2,
) -> list:
    """
    Processes the entire dataset with translation and evaluation loop.
    
    Args:
        dataset (Any): Dataset to process, expected to have a 'train' split.
        split (str): Dataset split to process (default: "train").
        max_retries (int): Maximum number of feedback-based retries per sample.

    Returns:
        list: List of translated items with evaluation results.
    """
    results = []
    items = list(dataset[split])
    if items:
        logger.debug(f"[DEBUG] 첫 번째 item keys: {list(items[0].keys())}")
        logger.debug(f"[DEBUG] 첫 번째 item 전체: {items[0]}")
    
    for item in tqdm(items, desc=f"Translating {split} items", unit="item"):
        result = translate_fields_with_evaluation(item, max_retries=max_retries)
        # Flatten nested dicts/lists for Parquet compatibility
        for k, v in result.items():
            if isinstance(v, (dict, list)):
                result[k] = json.dumps(v, ensure_ascii=False)
        results.append(result)
    logger.debug(f"[DEBUG] 번역 결과 리스트 길이: {len(results)}")
    if results:
        logger.debug(f"[DEBUG] 번역 결과 샘플: {results[0]}")
    else:
        logger.warning(f"[DEBUG] 번역 결과가 비어 있습니다.")
    return results


def translate_and_evaluate_chain(
    dataset: Any,
    split: str = "train",
    output_path: str = "./translated_kocem.parquet",
    max_retries: int = 2,
    cache_dir: str = None,
) -> str:
    """
    Translates the KoCEM dataset from Korean to English using LangChain, with evaluator feedback loop.
    
    Args:
        dataset (Any): Dataset to process, expected to have a 'train' split.
        split (str): Dataset split to process (default: "train").
        output_path (str): Path to save the translated dataset (parquet).
        max_retries (int): Maximum number of feedback-based retries per sample.
        cache_dir (str): Directory to cache the dataset (default: None).
    
    Returns:
        str: Path to the saved translated dataset.
    """
    
    cache_dir = cache_dir or os.getenv("CACHE_DIR", ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    results = process_dataset_with_evaluation(dataset, split=split, max_retries=max_retries)
    with ThreadPoolExecutor() as executor:
        future = executor.submit(save_parquet, output_path, results)
        future.result()
    
    logger.info(f"번역+평가 데이터 저장 완료: {output_path}")
    return output_path


def main() -> None:
    """
    Example usage for dataset translation and evaluation chain.
    """
    ds = get_kocem_dataset()
    translate_and_evaluate_chain(ds)


if __name__ == "__main__":
    main()

__all__ = ['translate_and_evaluate_chain']