import io, json, os
from collections import defaultdict
import pandas as pd
from PIL import Image
# from pandas_image_methods import PILMethods

from schemas import Option, Quiz

CANDIDATES = ["options", "eval_loop", "history", "feedbacks", "splits"]


# Flatten nested dicts/lists for Parquet compatibility
def serialize_for_parquet(obj):
    if isinstance(obj, Option):
        return obj.model_dump()
    elif isinstance(obj, Quiz):
        return obj.model_dump()
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], Option):
            return [o.model_dump() for o in obj]
        elif obj and isinstance(obj[0], Quiz):
            return [q.model_dump() for q in obj]
    return obj


def extract_bytes(b):
    # dict로 한 번이라도 감싸져 있으면 끝까지 'bytes'만 추출
    while isinstance(b, dict):
        b = b.get('bytes', None)
    return b


def flatten_for_df(obj):
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False)
    return obj


def parse_json_list(val):
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() == "none":
            return []
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return val
    return val


def pandas_type_to_hf_feature(dtype):
    t = str(dtype)
    # list/dict/object 타입이면 value 예시를 받아 재귀적으로 내부 구조를 추론
    if t.startswith('bytes'):
        return "bytes"
    elif t.startswith("int"):
        return "int64"
    elif t.startswith("float"):
        return "float64"
    elif t == "bool":
        return "bool"
    else:
        return "string"

def feature_yaml_recursion(name, value, indent=6):
    pad = ' ' * indent
    if isinstance(value, dict):
        lines = [f"{pad}- name: {name}", f"{pad}  struct:"]
        for k, v in value.items():
            lines.extend(feature_yaml_recursion(k, v, indent+2))
        return lines
    elif isinstance(value, list):
        lines = [f"{pad}- name: {name}", f"{pad}  list:"]
        if value:
            lines.extend(feature_yaml_recursion(f"item", value[0], indent+2))
        else:
            lines.append(f"{pad}    dtype: unknown")
        return lines
    elif isinstance(value, bytes):
        return [f"{pad}- name: {name}", f"{pad}  dtype: bytes"]
    elif isinstance(value, int):
        return [f"{pad}- name: {name}", f"{pad}  dtype: int64"]
    elif isinstance(value, float):
        return [f"{pad}- name: {name}", f"{pad}  dtype: float64"]
    elif isinstance(value, bool):
        return [f"{pad}- name: {name}", f"{pad}  dtype: bool"]
    elif isinstance(value, str):
        return [f"{pad}- name: {name}", f"{pad}  dtype: string"]
    else:
        return [f"{pad}- name: {name}", f"{pad}  dtype: unknown"]


# 모든 컬럼을 dict/list는 json 문자열로, 숫자 타입은 string으로 변환
def safe_json(obj):
    if isinstance(obj, bytes):  # 🔥 이 줄 추가!
        return obj
    
    if isinstance(obj, (Option, Quiz)):
        return obj.model_dump()
    
    elif isinstance(obj, list):
        # 리스트 내부: dict/object는 그대로, 나머지는 str로 변환
        out = []
        for item in obj:
            if isinstance(item, dict):
                out.append(safe_json(item))
            elif isinstance(item, (Option, Quiz)):
                out.append(item.model_dump())
            else:
                out.append(str(item) if item is not None else "None")
        return out
    
    if isinstance(obj, dict):
        # image dict 내부라면, bytes key만 따로 예외 처리
        if set(obj.keys()) == {'bytes', 'path'}:
            return {
                'bytes': obj['bytes'] if isinstance(obj['bytes'], bytes) else extract_bytes(obj['bytes']),
                'path': obj['path']
            }
        return {k: safe_json(v) for k, v in obj.items()}
    
    elif isinstance(obj, str):
        return str(obj)
    
    elif isinstance(obj, int):
        return int(obj)
    elif isinstance(obj, float):
        return float(obj)
    # list가 아닌데 list로 변환 가능한 경우 (예: np.ndarray, tuple 등)
    try:
        as_list = list(obj)
        # str은 iterable이지만 list로 변환하면 문자 단위로 쪼개지므로 제외
        if not isinstance(obj, str):
            return safe_json(as_list)
    except Exception:
        pass
    # 변환 불가면 str로 저장
    # return str(obj) if obj is not None else "None"
    return obj
    # raise ValueError("Unsupported type for safe_json conversion: {}: {}".format(obj, type(obj)))


def _display_img(img):
    import matplotlib.pyplot as plt
    import sys

    # Only display if running interactively
    if hasattr(sys, 'ps1') or sys.flags.interactive:
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show()


def _save_image_to_bytes(
    img, 
    path: str = None, display_image: bool = True
) -> dict[str, str | bytes | None]:
    """
    이미지 객체를 bytes로 변환하고, 필요시 display합니다.
    
    Args:
        img (Image.Image): PIL 이미지 객체.
        path (str): 이미지 저장 경로 (선택적).
    
    Returns:
        dict: {"bytes": bytes, "path": str (optional)} 형태로 반환.
    """
    
    if display_image:
        _display_img(img)
    
    y = {}
    with io.BytesIO() as buf:
        img.save(buf, format=img.format or 'PNG')
        buf.seek(0)
        y.update({
            "bytes": buf.getvalue(),
            "path": path.replace("\\", "/") if path else None  # bytes로 저장된 경우 path는 None
        })
    return y


def _image_to_bytes(x, display_image: bool = True) -> dict[str, str | bytes | None]:
    """
    이미지 dict/bytes를 받아 {"bytes": bytes, "path": str (optional)}로 변환.
    display_image=True일 경우 변환된 이미지를 바로 display(IPython)로 보여줌.
    output_format: 'base64' (default) or 'hex' (16진수 문자열)
    """
    if x is None:
        return {"bytes": None, "path": None}
    elif isinstance(x, Image.Image):
        return _save_image_to_bytes(img=x, display_image=display_image)
    elif isinstance(x, bytes):
        img = Image.open(io.BytesIO(x))
        return _save_image_to_bytes(img=img, display_image=display_image)
    elif isinstance(x, dict):
        bytes_ = x.get('bytes', None)
        path_ = x.get('path', None)
        if bytes_:
            img = Image.open(io.BytesIO(bytes_))
            return _save_image_to_bytes(img=img, path=path_, display_image=display_image)
        else:
            return {"bytes": bytes_, "path": path_}
    else:
        raise ValueError(f"Unsupported image type: ({type(x) = }). Expected Image.Image, bytes, or dict with 'bytes' key.")


def save_parquet(path: str, rows: list):
    """
    Parquet 파일을 pandas로 저장합니다. (pyarrow 완전 제거)
    모든 dict/list 컬럼은 JSON 문자열로 변환, 이미지 bytes는 그대로 저장.
    """

    # Step 1: 리스트 컬럼 JSON 문자열 파싱
    for row in rows:
        for col in CANDIDATES:
            if col in row:
                row[col] = parse_json_list(row[col])

        if "image" in row:
            row["image"] = _image_to_bytes(row["image"])

    # Step 2: 모든 dict/list 컬럼은 JSON 문자열로 변환
    rows = [safe_json(row) for row in rows]
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (dict, list)) and k != "image":
                row[k] = flatten_for_df(v)
            if k == "image" and isinstance(v, dict):
                # 이미지 bytes는 그대로, path는 string
                pass

    # Step 3: DataFrame 생성 및 저장
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"[완료] Parquet 저장: {path}, shape={df.shape}")


def refresh_terminal():
    # Terminal refresh (ctrl+l effect)
    print("\033[2J\033[1;1H", end="")


def write_readme_from_outputs(
    output_dir: str = "output",
    eval_result_dir: str = "eval_results",
    readme_path: str = "output/gpt-eval/README.md"
):
    """
    번역 결과물(Parquet 파일들)로부터 README.md를 자동 생성합니다. (pandas 기반)
    """
    configs = defaultdict(lambda: {"splits": {}, "features": set()})
    all_tags = set()
    total_instances = 0

    eval_dir = os.path.join(output_dir, eval_result_dir)
    if not os.path.exists(eval_dir):
        raise FileNotFoundError(f"{eval_dir} 경로가 존재하지 않습니다.")

    # 1. config별 split별 parquet 파일 탐색 (pandas 기반)
    total_bytes = 0
    for config_name in os.listdir(eval_dir):
        config_path = os.path.join(eval_dir, config_name)
        if not os.path.isdir(config_path):
            continue
        for f in os.listdir(config_path):
            if not f.endswith(".parquet"): continue
            split = f.split(".")[0]
            file_path = os.path.join(config_path, f)
            try:
                df = pd.read_parquet(file_path)
            except Exception:
                continue
            num_examples = len(df)
            file_bytes = os.path.getsize(file_path)
            total_bytes += file_bytes
            configs[config_name]["splits"][split] = {
                "num_examples": num_examples,
                "file": os.path.relpath(file_path, os.path.join(output_dir, eval_result_dir)),
                "num_bytes": file_bytes,
                "download_size": file_bytes
            }
            configs[config_name]["features"].update(df.columns)
            # 태그 추출 (예: question_type, field 등)
            for col in ["question_type", "field", "subfield"]:
                if col in df.columns:
                    tags = set(x for x in df[col].dropna().unique() if x is not None)
                    all_tags.update(tags)

    # 2. YAML 헤더 생성
    yaml_lines = ["---"]
    yaml_lines.append("language:\n- ko\n- en")
    yaml_lines.append("license: cc-by-nc-4.0")
    yaml_lines.append("size_categories:\n- 10K<n<100K")
    yaml_lines.append("task_categories:\n- question-answering\n- multiple-choice")
    yaml_lines.append(f"pretty_name: kocem:{eval_result_dir.lower()}")
    yaml_lines.append("dataset_info:")

    for config, info in configs.items():
        yaml_lines.append(f"  {config}:")
        yaml_lines.append(f"    features:")
        dtype_map = {}
        schema_map = {}
        # split별로 dtype 추출
        for s_info in info["splits"].values():
            split_file = s_info["file"]
            try:
                df = pd.read_parquet(os.path.join(output_dir, eval_result_dir, split_file))
                for col in df.columns:
                    dtype_map[col] = str(df[col].dtype)
                    schema_map[col] = df[col].dtype
            except Exception:
                print(f"[오류] Parquet 파일 dtype 추출 실패: {split_file}")
                pass
        # 들여쓰기: 4칸
        for feat in sorted(info["features"]):
            dtype = dtype_map.get(feat, None)
            # 실제 데이터 예시 추출 (split별 첫 row)
            example_value = None
            for s_info in info["splits"].values():
                split_file = s_info["file"]
                try:
                    df = pd.read_parquet(os.path.join(output_dir, eval_result_dir, split_file))
                    if feat in df.columns and len(df) > 0:
                        example_value = df[feat].iloc[0]
                        # JSON 문자열이면 dict/list로 변환
                        if isinstance(example_value, str):
                            try:
                                parsed = json.loads(example_value)
                                example_value = parsed
                            except Exception:
                                pass
                        break
                except Exception:
                    pass
            if feat == "image":
                # 이미지 컬럼은 bytes로 처리
                yaml_lines.append(f"      - name: {feat}\n        dtype: image")
            elif example_value is not None and isinstance(example_value, (dict, list)):
                # dict/list/object 타입이면 재귀적으로 구조화
                yaml_lines.extend(feature_yaml_recursion(feat, example_value, indent=6))
            elif dtype is not None:
                yaml_lines.append(f"      - name: {feat}\n        dtype: {pandas_type_to_hf_feature(dtype)}")
            else:
                yaml_lines.append(f"      - name: {feat}\n        dtype: null")
        yaml_lines.append(f"    splits:")
        for split, s_info in info["splits"].items():
            yaml_lines.append(f"      - name: {split}")
            yaml_lines.append(f"        num_examples: {s_info['num_examples']}")
            yaml_lines.append(f"        num_bytes: {s_info['num_bytes']}")
            yaml_lines.append(f"        download_size: {s_info['download_size']}")
        yaml_lines.append(f"    dataset_size: {sum(s['num_bytes'] for s in info['splits'].values())}")
        feature_list = ', '.join(sorted(info["features"]))
        split_list = ', '.join(sorted(info["splits"].keys()))
        n = sum(s["num_examples"] for s in info["splits"].values())
        tag_list = ', '.join(sorted(all_tags)) if all_tags else "-"
        yaml_lines.append(f"    description: |\n      이 config는 {split_list} split에 걸쳐 {n}개의 인스턴스를 포함합니다.\n      Features: {feature_list}.\n      주요 태그: {tag_list}.")
    
    # Configs 정보 추가
    yaml_lines.append("configs:")
    for config, info in configs.items():
        yaml_lines.append(f"- config_name: {config}")
        yaml_lines.append("  data_files:")
        for split, s_info in info["splits"].items():
            yaml_lines.append(f"  - split: {split}")
            path = s_info['file'].replace('\\', '/').replace('\\', '/')
            yaml_lines.append(f"    path: {path}")

    # Tags 정보 추가
    yaml_lines.append("tags:")
    for tag in sorted(all_tags):
        yaml_lines.append(f"- {tag}")
    yaml_lines.append("---\n")

    # 3. 본문(표, 설명 등)
    table_lines = [
        f"\n# KoCEM:{eval_result_dir}\n",
        "| Config | Splits | Features | Instances |",
        "|--------|--------|----------|-----------|"
    ]
    for config, info in configs.items():
        splits = ", ".join(sorted(info["splits"].keys()))
        features = ", ".join(sorted(info["features"]))
        n = sum(s["num_examples"] for s in info["splits"].values())
        table_lines.append(f"| {config} | {splits} | {features} | {n} |")
    table_lines.append(f"| **Total** |  |  | **{total_instances}** |")

    disclaimer = (
        "\n<div style=\"border: 2px solid #f87171; background-color: #fef2f2; border-radius: 12px; padding: 20px; margin: 20px 0; color: #991b1b; font-family: 'Segoe UI', 'Apple SD Gothic Neo', Arial, sans-serif; font-size: 1.05em;\">\n"
        "  <strong style=\"font-size:1.18em;\">⚠️ Disclaimer</strong><br>\n"
        "  Please note that this dataset was independently compiled for personal research purposes only.<br>\n"
        "  I regret that I am unable to share access under any circumstances.<br>\n"
        "  Accordingly, I kindly ask that you refrain from submitting access requests, as all such requests will have to be declined.<br>\n"
        "  <u>Thank you very much for your understanding and consideration.</u>\n"
        "</div>\n"
    )

    contact = "\n## 문의\n\n- pikaybh@snu.ac.kr\n"

    # 4. 파일로 저장
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))
        f.write("\n".join(table_lines))
        f.write(disclaimer)
        f.write(contact)


__all__ = ['save_parquet', 'serialize_for_parquet', 'refresh_terminal', 'write_readme_from_outputs']