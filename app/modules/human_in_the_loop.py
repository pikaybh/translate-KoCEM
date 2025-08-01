"""
Human-in-the-loop review module for translation results.
"""


import json
import readchar
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate , HumanMessagePromptTemplate

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

from schemas import Quiz
from utils import load_prompt, save_parquet, setup_logger


logger = globals()['logger'] if 'logger' in globals() else setup_logger(__name__)
load_dotenv()


# 번역 프롬프트 템플릿 및 체인 생성 (ChatPromptTemplate for chat model)
hitl_translation_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        load_prompt(
            template="human-in-the-loop", 
            rule="system", 
            version="v1"
        )
    ),
    HumanMessagePromptTemplate.from_template(
        load_prompt(
            template="human-in-the-loop", 
            rule="human", 
            version="v1"
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
hitl_translation_chain = hitl_translation_prompt | print_chain | init_chat_model("openai:gpt-4.1").with_structured_output(Quiz)


def show_history(console, eval_history, eval_result=None):
    """
    Display evaluator loop history or single result using rich panels.

    Args:
        console (Console): Rich console instance for output.
    """
    if eval_history:
        # If eval_history is a string, try to parse as JSON
        if isinstance(eval_history, str):
            try:
                eval_history = json.loads(eval_history)
            except Exception:
                eval_history = [eval_history]
        # Try to get id from eval_history or eval_result
        eval_id = None
        if isinstance(eval_history, list) and len(eval_history) > 0:
            eval_id = eval_history[0].get("id") if isinstance(eval_history[0], dict) and "id" in eval_history[0] else None
        if not eval_id and eval_result and isinstance(eval_result, dict):
            eval_id = eval_result.get("id")
        panel_title = f"[bold magenta]Evaluator History [white]{eval_id or ''}[/white]"
        # Build tables for each attempt
        tables = []
        if isinstance(eval_history, list):
            for idx, eval_item in enumerate(eval_history):
                # If eval_item is a string, try to parse as JSON
                if isinstance(eval_item, str):
                    try:
                        eval_item = json.loads(eval_item)
                    except Exception:
                        eval_item = {"reasoning": str(eval_item)}
                reasoning = eval_item.get("reasoning", "")
                value = eval_item.get("value", "")
                score = eval_item.get("score", "")
                feedback = eval_item.get("feedback", "")
                attempt = eval_item.get("attempt", idx)
                table = Table(show_header=True, header_style="bold magenta")
                table.title = f"Attempt {attempt+1}"
                table.add_column("Field", style="dim", width=16)
                table.add_column("Content", style="white")
                table.add_row("Feedback", feedback)
                table.add_row("Reasoning", reasoning)
                table.add_row("Pass", str(value))
                table.add_row("Score", str(score))
                tables.append(table)
        # Combine all tables into one Panel
        feedback_panel = Panel(Group(*tables) if tables else "No evaluation history.", title=panel_title, border_style="magenta")
        console.print(feedback_panel)
    elif eval_result is not None:
        # If eval_result is a string, try to parse as JSON
        if isinstance(eval_result, str):
            try:
                eval_result = json.loads(eval_result)
            except Exception:
                eval_result = {"reasoning": str(eval_result)}
        reasoning = eval_result.get("reasoning", "") if isinstance(eval_result, dict) else str(eval_result)
        value = eval_result.get("value", "") if isinstance(eval_result, dict) else ""
        score = eval_result.get("score", "") if isinstance(eval_result, dict) else ""
        table = Table(show_header=True, header_style="bold magenta")
        table.title = "Evaluator Result"
        table.add_column("Field", style="dim", width=16)
        table.add_column("Content", style="white")
        table.add_row("Reasoning", reasoning)
        table.add_row("Pass", str(value))
        table.add_row("Score", str(score))
        feedback_panel = Panel(table, title="[bold magenta]Evaluator Result", border_style="magenta")
        console.print(feedback_panel)


def human_review_from_parquet(input_path: str, output_path: str = "./final_translated_kocem.parquet"):
    """
    Human-in-the-loop review for translation results. Loads from parquet, allows manual correction, then saves as parquet.

    Args:
        input_path (str): Path to the translated dataset (parquet).
        output_path (str): Path to save the final reviewed dataset (parquet).
    """

    logger.info("Human-in-the-loop review 시작...")
    df = pd.read_parquet(input_path)
    console = Console()
    rows = df.to_dict(orient="records")
    reviewed_indices = set()
    session_exit = False
    to_review_indices = [i for i, row in enumerate(rows) if row.get("human_feedback", True)]
    current_item = 0
    total_items = len(to_review_indices)
    while not session_exit:
        if not to_review_indices:
            break
        idx = to_review_indices[current_item]
        row = rows[idx]
        ko_question = row.get("ko_question", "")
        en_question = row.get("en_question", "")
        ko_options = row.get("ko_options", [])
        en_options = row.get("en_options", [])
        if isinstance(ko_options, str):
            try:
                ko_options = json.loads(ko_options)
            except Exception:
                ko_options = [ko_options]
        if not isinstance(ko_options, list):
            ko_options = [str(ko_options)]
        if isinstance(en_options, str):
            try:
                en_options = json.loads(en_options)
            except Exception:
                en_options = [en_options]
        if not isinstance(en_options, list):
            en_options = [str(en_options)]

        ko_answer = row.get("ko_answer", "")
        en_answer = row.get("en_answer", "")
        ko_explanation = row.get("ko_explanation", "")
        en_explanation = row.get("en_explanation", "")
        item_id = row.get("id", "")

        fields = [
            {"name": "Question", "ko": ko_question, "en": en_question},
        ]
        for i, (ko_opt, en_opt) in enumerate(zip(ko_options, en_options)):
            fields.append({"name": f"Option {chr(65+i)}", "ko": ko_opt, "en": en_opt})
        fields.append({"name": "Answer", "ko": ko_answer, "en": en_answer})
        fields.append({"name": "Explanation", "ko": ko_explanation, "en": en_explanation})

        # Show evaluation history before main review table
        eval_history = row.get("eval_loop", None)
        eval_result = row.get("eval", None)
        show_history(console, eval_history, eval_result)

        field_feedback = row.get("field_feedback", {})
        if not isinstance(field_feedback, dict):
            field_feedback = {}
        current_field = 0
        while True:
            # Progress bar
            with Progress(transient=True) as progress:
                task = progress.add_task("Review Progress", total=total_items)
                progress.update(task, completed=current_item)
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Field", style="dim", width=16)
            table.add_column("Korean", style="bold yellow")
            table.add_column("English", style="bold green")
            for i, f in enumerate(fields):
                highlight = "[reverse]" if i == current_field else ""
                table.add_row(
                    str(f.get("name", "")),
                    str(f.get("ko", "")),
                    f"{highlight}{str(f.get('en', ''))}{highlight}"
                )
            panel_title = f"[bold blue]Human Review Item ([white]{item_id}[bold blue]) [{current_item+1}/{total_items}]" if item_id else f"[bold blue]Human Review Item ([white]unknown[bold blue]) [{current_item+1}/{total_items}]"
            console.print(Panel(table, title=panel_title, border_style="blue"))
            console.print(f"[cyan]↑/↓: Move field   ←/→: Move item   A: Add Option   Enter: Give feedback   Esc: Cancel/Back   F2: End review session[/cyan]")
            key = readchar.readkey()
            if key == readchar.key.ESC:
                # 취소/뒤로가기: 아무것도 하지 않고 화면만 다시 그림
                console.print("[red]취소됨. 이전 화면으로 돌아갑니다.[/red]")
                continue
            elif key == readchar.key.F2:
                session_exit = True
                break
            elif key == readchar.key.UP:
                current_field = (current_field - 1) % len(fields)
            elif key == readchar.key.DOWN:
                current_field = (current_field + 1) % len(fields)
            elif key == readchar.key.RIGHT:
                current_item = (current_item + 1) % total_items
                break
            elif key == readchar.key.LEFT:
                current_item = (current_item - 1) % total_items
                break
            elif key.lower() == 'a':

                # Option 추가
                next_idx = len(ko_options) + 1
                console.print(f"[bold yellow]새로운 Option을 몇 번째로 추가할까요? (1~{next_idx}, 기본값: {next_idx}): ")
                idx_input = console.input("").strip()
                try:
                    insert_idx = int(idx_input) - 1 if idx_input else next_idx - 1
                    if insert_idx < 0 or insert_idx > len(ko_options):
                        raise ValueError
                except Exception:
                    insert_idx = next_idx - 1
                console.print("[bold yellow]새로운 Option의 한글 텍스트를 입력하세요 (취소하려면 Enter): ")
                
                new_ko_opt = console.input("").strip()
                if new_ko_opt:
                    try:
                        from modules.translation_chain import translation_chain
                    except ImportError:
                        continue
                    new_en_opt = translation_chain.invoke({"text": new_ko_opt, "feedback": ""}).content
                    ko_options.insert(insert_idx, new_ko_opt)
                    en_options.insert(insert_idx, new_en_opt)
                    fields.insert(insert_idx+1, {"name": f"Option {chr(65+insert_idx)}", "ko": new_ko_opt, "en": new_en_opt})
                    
                    # Option 라벨 재정렬
                    for i in range(len(ko_options)):
                        fields[i+1]["name"] = f"Option {chr(65+i)}"
                    console.print(f"[green]Option 추가됨: {new_ko_opt} → {new_en_opt} (위치: {insert_idx+1})")
                    current_field = insert_idx+1  # 새 옵션으로 이동

            elif key == readchar.key.ENTER or key == '\r':
                console.print(f"[bold yellow]Enter feedback for {fields[current_field]['name']} (leave empty to skip): ")
                feedback = console.input("")
                if feedback.strip():
                    field_feedback[fields[current_field]["name"]] = feedback.strip()
                    
                    # Compose feedback message mentioning the field and its content
                    field_name = fields[current_field]["name"]
                    field_ko = str(fields[current_field]["ko"])
                    field_en = str(fields[current_field]["en"])
                    feedback_message = f"- Field: {field_name}\n  - Content: {field_ko}\n  - Target: {field_en}\n  - Feedback: {feedback.strip()}"
                    
                    # Prepare full item for translation_chain
                    chain_input = {
                        "ko_question": ko_question,
                        "ko_options": ko_options,
                        "ko_answer": ko_answer,
                        "ko_explanation": ko_explanation,
                        "en_question": en_question,
                        "en_options": en_options,
                        "en_answer": en_answer,
                        "en_explanation": en_explanation,
                        "feedback": feedback_message
                    }
                    chain_result = hitl_translation_chain.invoke(chain_input)
                    logger.debug(f"Chain result: {chain_result}")

                    # Update only the selected field's English value
                    if field_name.startswith("Option"):
                        opt_idx = current_field - 1
                        if hasattr(chain_result, "options") and len(chain_result.options) > opt_idx:
                            en_options[opt_idx] = chain_result.options[opt_idx].value if hasattr(chain_result.options[opt_idx], "value") else chain_result.options[opt_idx]
                            fields[current_field]["en"] = en_options[opt_idx]
                    elif field_name == "Question":
                        if hasattr(chain_result, "question"):
                            en_question = chain_result.question
                            fields[current_field]["en"] = en_question
                    elif field_name == "Answer":
                        if hasattr(chain_result, "answer"):
                            en_answer = chain_result.answer.value if hasattr(chain_result.answer, "value") else chain_result.answer
                            fields[current_field]["en"] = en_answer
                    elif field_name == "Explanation":
                        if hasattr(chain_result, "explanation"):
                            en_explanation = chain_result.explanation
                            fields[current_field]["en"] = en_explanation
                else:
                    field_feedback[fields[current_field]["name"]] = ""
            console.clear()
        row["human_feedback"] = True if any(v for v in field_feedback.values()) else False
        row["field_feedback"] = field_feedback
        row["en_question"] = en_question
        row["en_options"] = en_options
        row["en_answer"] = en_answer
        row["en_explanation"] = en_explanation
        reviewed_indices.add(idx)
    with ThreadPoolExecutor() as executor:
        future = executor.submit(save_parquet, output_path, rows)
        future.result()
    logger.info(f"Human-in-the-loop review 완료! 결과 저장: {output_path}")


__all__ = ["human_review_from_parquet"]