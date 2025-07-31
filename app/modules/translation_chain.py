import os, json
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate , HumanMessagePromptTemplate

from modules.evaluator import evaluate_translation
from schemas import Quiz, Option
from utils import (get_kocem_dataset, load_prompt, save_parquet, setup_logger, label_alpha_numeric, 
                    construct_option, construct_options, dict2list, serialize_for_parquet)


logger = globals()['logger'] \
    if 'logger' in globals() \
    else setup_logger(__name__)
load_dotenv()

PROBELM_LIST = [
    """
"""
]

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
translation_chain = translation_prompt | print_chain | ChatOpenAI(model="gpt-4.1", store=True).with_structured_output(Quiz)


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
    feedbacks = []
    while attempt <= max_retries:
        feedback_str = "\n".join(feedbacks) if feedbacks else ""
        translation_response = translation_chain.invoke({"text": ko_text, "feedback": feedback_str})
        logger.debug(translation_response)
        en_text = translation_response.content

        eval_result = evaluate_translation(ko_text, en_text)
        value = str(eval_result.get('value', '')).strip().lower()
        score = float(eval_result.get('score', 0))
        passed = value in ['y', 'yes'] or score >= 0.8
        logger.debug(f"Try {attempt}: {ko_text} -> {en_text} | Eval: {eval_result}")

        if passed:
            break

        feedbacks.append(extract_feedback(eval_result))
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
    if not text:
        logger.debug("Empty text provided for translation.")
        return ""
    translation_response = translation_chain.invoke({"text": text, "feedback": ""})
    logger.debug(translation_response)
    return translation_response.content


# def translate_fields_with_evaluation(item: dict, max_retries: int = 2) -> dict:
#     """
#     Translates fields of a dataset item with evaluation loop.

#     Args:
#         item (dict): Dataset item containing fields to translate.
    
#     Returns:
#         dict: Translated item with evaluation results.
#     """

#     # question
#     ko_question = item.get("question", "")
#     # Feedback/evaluation loop for question
#     eval_loop = []
#     feedbacks = []
#     attempt = 0
#     en_question = None
#     while attempt <= max_retries:
#         feedback_str = "\n".join(feedbacks) if feedbacks else ""
#         en_question_try = translation_chain.invoke({"text": ko_question, "feedback": feedback_str}).content
#         eval_result = evaluate_translation(ko_question, en_question_try)
#         passed = str(eval_result.get('value', '')).strip().upper() == 'Y'
#         eval_loop.append({
#             "attempt": attempt,
#             "en_question": en_question_try,
#             "feedback": feedback_str,
#             **eval_result
#         })
#         if passed:
#             en_question = en_question_try
#             break
#         feedbacks.append(extract_feedback(eval_result))
#         attempt += 1
#     if en_question is None:
#         en_question = en_question_try

#     # options robust parsing
#     ko_options_raw = item.get("options", "")
#     for problem in PROBELM_LIST:
#         ko_options_raw = ko_options_raw.replace(problem, ", ")
#     options_list = eval(ko_options_raw)
#     # options_list = []
#     # if ko_options_raw:
#     #     try:
#     #         options_list = json.loads(ko_options_raw)
#     #         if not isinstance(options_list, list):
#     #             options_list = [ko_options_raw]
#     #     except Exception:
#     #         try:
#     #             options_list = eval(ko_options_raw)
#     #             if not isinstance(options_list, list):
#     #                 options_list = [ko_options_raw]
#     #         except Exception:
#     #             for delim in ["|", ",", ";", "/", "\\n"]:
#     #                 if delim in ko_options_raw:
#     #                     options_list = [opt.strip() for opt in ko_options_raw.split(delim) if opt.strip()]
#     #                     break
#     #             if not options_list:
#     #                 options_list = [ko_options_raw]
#     # if len(options_list) > 1 and all(len(opt) == 1 for opt in options_list):
#     #     options_list = ["".join(options_list)]

#     # Option translation with individual feedback loop
#     ko_options_labeled = options_list
#     en_options_labeled = []
#     for idx, option in enumerate(options_list):
#         feedbacks = []
#         attempt = 1
#         en_option = None
#         while attempt <= max_retries:
#             option_label = chr(ord("A")+idx)
#             feedback_str = "\n".join(feedbacks) if feedbacks else ""
#             en_option_try = translation_chain.invoke({"text": option, "feedback": feedback_str}).content
#             logger.debug(f"Translating option [Trial {attempt}] {option_label}: {option} -> {en_option_try}")
#             eval_result = evaluate_translation(option, en_option_try)
#             logger.debug(f"Eval result for option [Trial {attempt}] {option_label}: {eval_result}")
#             passed = str(eval_result.get('value', '')).strip().upper() == 'Y'
#             if passed:
#                 en_option = en_option_try
#                 break
#             feedbacks.append(extract_feedback(eval_result))
#             logger.debug(f"Feedback for option [Trial {attempt}] {option_label}: {feedbacks[-1]}")
#             attempt += 1
#         if en_option is None:
#             en_option = en_option_try
#         en_options_labeled.append(en_option)

#     # answer
#     ko_answer = item.get("answer", "")
#     answer_idx = ko_options_labeled.index(ko_answer)
#     en_answer = en_options_labeled[answer_idx]  # translate_text(ko_answer)

#     # explanation
#     ko_explanation = item.get("explanation", "")
#     en_explanation = translate_text(ko_explanation)

#     result = dict(item)
#     result.update({
#         "ko_question": ko_question,
#         "en_question": en_question,
#         "ko_options": ko_options_labeled,
#         "en_options": en_options_labeled,
#         "ko_answer": ko_answer,
#         "en_answer": en_answer,
#         "ko_explanation": ko_explanation,
#         "en_explanation": en_explanation,
#         "eval": eval_loop[-1] if eval_loop else {},
#         "eval_loop": eval_loop
#     })
#     return result


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
    for problem in PROBELM_LIST:
        ko_options_raw = ko_options_raw.replace(problem, ", ")
    options_list = eval(ko_options_raw)
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
        label=label_alpha_numeric(options_list.index(ko_answer_content)),
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