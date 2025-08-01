import os, json
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate , HumanMessagePromptTemplate

from models import evaluate_translation, evallm
from schemas import Quiz, Option
from utils import (get_kocem_dataset, load_prompt, save_parquet, setup_logger, label_alpha_numeric, 
                    construct_option, construct_options, serialize_for_parquet)


logger = globals()['logger'] \
    if 'logger' in globals() \
    else setup_logger(__name__)
load_dotenv()

PROBELM2SOLVE = {
    """
""": ", ",
"'\n '": "', '"
}


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
            version="v3"
        )
    )
])


def print_chain(*args):
    """
    Prints the translation chain input for debugging.
    """
    for _, arg in enumerate(args):
        logger.debug(arg.messages[0].content)
        logger.debug(arg.messages[1].content)
    return args[0] if args else None


engine = ChatOpenAI(
    model="gpt-4.1", 
    store=True,
    metadata={
        "KoCEM": "Translate:2",
        "chain": "translation chain",
    }
)
translation_chain = translation_prompt | print_chain | engine.with_structured_output(Quiz)


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


def translate_chain_and_evaluation_loop(
    item: dict,
    max_retries: int = 2
) -> dict:
    """
    Translates a single item with evaluation loop.

    Args:
        item (dict): Dataset item to translate.
        max_retries (int): Maximum number of feedback-based retries.
    
    Returns:
        dict: Translated item with evaluation results.
    """
    
    """Parse dataset item"""
    # Prepare question from dataset item
    ko_question = item.get("question", "")

    # Prepare options from dataset item
    ko_options_raw = item.get("options", "")
    for problem, solve in PROBELM2SOLVE.items():
        ko_options_raw = ko_options_raw.replace(problem, solve)
    # options_list = eval(ko_options_raw)
    options_list = evallm(messages=ko_options_raw)
    ko_options = [
        Option(
            label=label_alpha_numeric(idx),
            value=option.strip()
        )
        for idx, option in enumerate(options_list)
    ]

    # Prepare answer from dataset item
    ko_answer_content = item.get("answer", "")
    ko_answer = Option(
        label=label_alpha_numeric(options_list.index(ko_answer_content)) if ko_answer_content in options_list else None,
        value=ko_answer_content
    )

    # Prepare explanation from dataset item
    ko_explanation = item.get("explanation", "")

    # Create Quiz object from parsed data
    korean_quiz = Quiz(
        question=ko_question,
        options=ko_options,
        answer=ko_answer,
        explanation=ko_explanation
    )
    logger.debug(f"{korean_quiz = }")

    # Initialize evaluation loop variables
    eval_loop = []
    feedbacks = []
    attempt = 1

    english_quiz = None
    
    while attempt <= max_retries:
        feedback_str = "\n  ".join(feedbacks) if feedbacks else ""
        input_data = {
            "ko_question": ko_question, 
            "ko_options": construct_options(korean_quiz.options),
            "ko_answer": construct_option(korean_quiz.answer),
            "ko_explanation": ko_explanation,
            "feedback": feedback_str
        }
        logger.debug(f"Input data for translation: {input_data}")

        english_quiz = translation_chain.invoke(input_data)
        logger.debug(english_quiz)
        
        eval_result = evaluate_translation(korean_quiz, english_quiz)
        logger.debug(f"Eval result for question [Trial {attempt}]: {eval_result}")

        passed = str(eval_result.get('value', '')).strip().upper() == 'Y'
        eval_loop.append({
            "attempt": attempt,
            "translated_question": english_quiz,
            "feedback": feedback_str,
            **eval_result
        })
        if passed:
            break
        feedbacks.append(extract_feedback(eval_result))
        attempt += 1
        
    result = dict(item)
    result.update({
        "ko_question": ko_question,
        "en_question": english_quiz.question,
        "ko_options": [option.value for option in ko_options],
        "en_options": [option.value for option in english_quiz.options],
        "ko_answer": ko_answer.value,
        "en_answer": english_quiz.answer.value,
        "ko_explanation": ko_explanation,
        "en_explanation": english_quiz.explanation,
        "eval": eval_loop[-1] if eval_loop else {},
        "eval_loop": eval_loop
    })
    logger.debug(f"Final result: {result}")
    
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
    # if items:
    #     logger.debug(f"[DEBUG] 첫 번째 item keys: {list(items[0].keys())}")
    #     logger.debug(f"[DEBUG] 첫 번째 item 전체: {items[0]}")
    
    for item in tqdm(items, desc=f"Translating {split} items", unit="item"):
        try:
            result = translate_chain_and_evaluation_loop(item, max_retries=max_retries)
            for k, v in result.items():
                # If Option or Quiz or their list, convert to dict/list-of-dict
                if isinstance(v, (Option, Quiz)):
                    result[k] = serialize_for_parquet(v)
                elif isinstance(v, list) and v and isinstance(v[0], (Option, Quiz)):
                    result[k] = serialize_for_parquet(v)
                elif isinstance(v, dict):
                    # If dict contains Option/Quiz, convert those values
                    if any(isinstance(val, (Option, Quiz)) for val in v.values()):
                        result[k] = {kk: serialize_for_parquet(val) if isinstance(val, (Option, Quiz)) else val for kk, val in v.items()}
                    else:
                        result[k] = v
                elif isinstance(v, list):
                    # If list contains Option/Quiz, convert those items
                    if any(isinstance(item, (Option, Quiz)) for item in v):
                        result[k] = [serialize_for_parquet(item) if isinstance(item, (Option, Quiz)) else item for item in v]
                    else:
                        result[k] = v
            results.append(result)

        except Exception as e:
            logger.error(f"Error processing item {item.get('id', 'unknown')}: {e}")
            logger.info(f"Skipping item due to error: {item.get('id', 'unknown')}")
            continue

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