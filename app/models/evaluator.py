# evaluator.py

from typing import Dict

from dotenv import load_dotenv
from langchain.evaluation import CriteriaEvalChain
from langchain_openai import ChatOpenAI

from utils import load_prompt, setup_logger


logger = globals()['logger'] \
    if 'logger' in globals() \
    else setup_logger(__name__)
load_dotenv()

criterion = [("accuracy", "v2"), ("fluency", "v2"), ("adquacy", "v1"), ("format", "v2")]
criteria = {
    key: load_prompt(
        template="translation",
        version=version
    ) for (key, version) in criterion
}
core_model = ChatOpenAI(model="gpt-4.1")
eval_chain = CriteriaEvalChain.from_llm(llm=core_model, criteria=criteria)


def evaluate_translation(ko_text: str, en_text: str) -> Dict[str, str]:
    """
    Evaluates the quality of a translation using LangChain's CriteriaEvalChain.

    Args:
        ko_text (str): Original Korean text.
        en_text (str): Translated English text.
    
    Returns:
        Dict[str, str]: Evaluation results for criterion.
    """
    return eval_chain.evaluate_strings(prediction=en_text, input=ko_text)


def main() -> None:
    """
    Example usage for translation evaluation.
    """
    ko = "이 문장은 테스트입니다."
    en = "This sentence is a test."
    logger.info(evaluate_translation(ko, en))


if __name__ == "__main__":
    main()

__all__ = ['evaluate_translation']