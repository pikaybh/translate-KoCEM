from typing import List

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from schemas import OptionParseType
from utils import load_prompt, setup_logger

from .chainio import IdentityChain


load_dotenv()

logger = globals()['logger'] if 'logger' in globals() else setup_logger(__name__)
chainio = IdentityChain(logger=logger, level="DEBUG")

evallm_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        load_prompt(
            template="evallm",
            rule="system",
            version="v1"
        )
    ),
    HumanMessagePromptTemplate.from_template(
        load_prompt(
            template="evallm",
            rule="human",
            version="v1"
        )
    )
])


def evallm(
    messages: str,
    model: str = "openai:gpt-4.1"
) -> List[str]:
    """
    Evaluates a list of messages using a hacker-like evaluation chain.

    Args:
        messages (str): The input messages to evaluate.
        model (str): The model to use for evaluation.
        **kwargs: Additional keyword arguments for the chat model.

    Returns:
        dict: Evaluation results.
    """
    chat_model = init_chat_model(model)
    chain_ = (
        evallm_prompt
        | chat_model.with_structured_output(OptionParseType)
        | chainio
    )
    return chain_.invoke({"input_str": messages})[0].output


if __name__ == "__main__":
    # Example usage
    example = "['고정식 레이아웃은 제품의 대량생산에 적합한 방식이다.', '레이아웃은 장래 공장 규모의 변화에 대응한 융통성이 있어야 한다.'\n'기능이 유사한 기계를 집합시키는 방식은 표준화가 어려운 공장에 채용한다.'\n'제품의 흐름에 따라 기계를 배치하는 방식을 연속작업식 레이아웃이라 한다.']" 
    result = evallm(example)
    print(result)
    for item in result:
        print(f"{item = }")

__all__ = ['evallm']