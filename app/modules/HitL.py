"""
Human-in-the-loop review module for translation results.
"""


import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from utils import save_parquet, setup_logger


logger = globals()['logger'] \
    if 'logger' in globals() \
    else setup_logger(__name__)
load_dotenv()


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
        from rich.console import Group
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

    import readchar
    logger.info("Human-in-the-loop review 시작...")
    df = pd.read_parquet(input_path)
    console = Console()
    rows = df.to_dict(orient="records")
    reviewed_indices = set()
    session_exit = False
    while not session_exit:
        to_review_indices = [i for i, row in enumerate(rows) if row.get("human_feedback", True)]
        if not to_review_indices:
            break
        for idx in to_review_indices:
            if session_exit:
                break
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
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Field", style="dim", width=16)
                table.add_column("Korean", style="bold yellow")
                table.add_column("English", style="bold green")
                for i, f in enumerate(fields):
                    highlight = "[reverse]" if i == current_field else ""
                    table.add_row(f["name"], f["ko"], f"{highlight}{f['en']}{highlight}")
                panel_title = f"[bold blue]Human Review Item ([white]{item_id}[bold blue])" if item_id else "[bold blue]Human Review Item ([white]unknown[bold blue])"
                console.print(Panel(table, title=panel_title, border_style="blue"))
                console.print("[cyan]←/→: Move field   Enter: Give feedback   Esc: Next item   F2: End review session[/cyan]")
                key = readchar.readkey()
                if key == readchar.key.ESC:
                    break
                elif key == readchar.key.F2:
                    session_exit = True
                    break
                elif key == readchar.key.RIGHT:
                    current_field = (current_field + 1) % len(fields)
                elif key == readchar.key.LEFT:
                    current_field = (current_field - 1) % len(fields)
                elif key == readchar.key.ENTER or key == '\r':
                    console.print(f"[bold yellow]Enter feedback for {fields[current_field]['name']} (leave empty to skip): ")
                    feedback = console.input("")
                    if feedback.strip():
                        field_feedback[fields[current_field]["name"]] = feedback.strip()
                        try:
                            from modules.translation_chain import evaluate_translation, extract_feedback, translation_chain
                        except ImportError:
                            continue
                        max_retries = 2
                        attempt = 0
                        if fields[current_field]["name"].startswith("Option"):
                            opt_idx = current_field - 1
                            ko_opt = ko_options[opt_idx]
                            opt_feedback = feedback.strip()
                            opt_en = None
                            while attempt <= max_retries:
                                opt_en_eval = translation_chain.invoke({"text": ko_opt, "feedback": opt_feedback}).content
                                opt_en = opt_en_eval
                                break
                            en_options[opt_idx] = opt_en
                            fields[current_field]["en"] = opt_en
                        elif fields[current_field]["name"] == "Question":
                            q_feedback = feedback.strip()
                            eval_loop = []
                            while attempt <= max_retries:
                                en_question_eval = translation_chain.invoke({"text": ko_question, "feedback": q_feedback}).content
                                eval_result = evaluate_translation(ko_question, en_question_eval)
                                value = str(eval_result.get('value', '')).strip().lower()
                                score = float(eval_result.get('score', 0))
                                passed = value in ['y', 'yes'] or score >= 0.8
                                eval_loop.append({
                                    "attempt": attempt,
                                    "en_question": en_question_eval,
                                    "feedback": q_feedback or "",
                                    **eval_result
                                })
                                if passed:
                                    break
                                q_feedback = extract_feedback(eval_result)
                                attempt += 1
                            en_question = eval_loop[-1]["en_question"] if eval_loop else en_question
                            fields[current_field]["en"] = en_question
                            row["eval_loop"] = eval_loop
                            row["eval"] = eval_loop[-1] if eval_loop else {}
                        elif fields[current_field]["name"] == "Answer":
                            a_feedback = feedback.strip()
                            en_answer_eval = translation_chain.invoke({"text": ko_answer, "feedback": a_feedback}).content
                            en_answer = en_answer_eval
                            fields[current_field]["en"] = en_answer
                        elif fields[current_field]["name"] == "Explanation":
                            e_feedback = feedback.strip()
                            en_explanation_eval = translation_chain.invoke({"text": ko_explanation, "feedback": e_feedback}).content
                            en_explanation = en_explanation_eval
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