import base64, json, os, re, yaml
import numpy as np
from collections import defaultdict
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from pandas_image_methods import PILMethods
import io

from schemas import Option, Quiz


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


def _image_to_base64(x, display_image: bool = True) -> dict:
    """
    이미지 dict/bytes를 받아 {"bytes": bytes, "path": str (optional)}로 변환.
    display_image=True일 경우 변환된 이미지를 바로 display(IPython)로 보여줌.
    output_format: 'base64' (default) or 'hex' (16진수 문자열)
    """
    bytes_ = x.get('bytes', None)
    path_ = x.get('path', None)
    
    if bytes_:
        img = Image.open(io.BytesIO(bytes_))
        if display_image:
            _display_img(img)
        with io.BytesIO() as buf:
            img.save(buf, format=img.format or 'PNG')
            buf.seek(0)  # 버퍼의 시작으로 이동
            y = {
                "bytes": bytes(buf.getvalue()),
                "path": path_
            }
    else:
        y = {
            "bytes": bytes(),
            "path": path_
        }
    print("({}) {}...".format(type(y["bytes"]), y["bytes"][:10]))  # Print first 10 bytes for debug
    return y


def save_parquet(path: str, rows: list):
    list_candidates = ["options", "eval_loop", "history", "feedbacks", "splits"]

    # --- Step 1: Pre-parse JSON-stringified lists in list candidate columns ---
    def parse_json_list(val):
        if isinstance(val, str):
            s = val.strip()
            if s.startswith('[') and s.endswith(']'):
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return parsed
                
        return val
    
    for row in rows:
        for col in list_candidates:
            if col in row:
                row[col] = parse_json_list(row[col])

    for row in rows:
        if "image" in row:
            row["image"] = _image_to_base64(row["image"])
            print(f"Image keys: {row['image'].keys()}")  # Print first 10 bytes for debug
            print(f"Image bytes: {row['image']['bytes'][:10]}...")  # Print first 10 bytes for debug
            print(f"Image path: {row['image']['path']}")  # Print image path for debug

    # --- Step 2: image 컬럼을 dash-joined int string dict로 변환 ---
    # for row in rows:
    #     if "image" in row:
    #         row["image"] = _image_to_base64(row["image"])
            # print(f"{row['image']['bytes'][:10] = }...")  # Print first 10 bytes for debug
            # image 컬럼이 dict면 json 문자열로 변환 (string 컬럼 저장용)
            # b = row["image"].get("bytes", None)
            # b = _extract_bytes(b)
            # if not isinstance(b, list):
            #     if isinstance(b, bytes):
            #         # row["image"]["bytes"] = list(np.frombuffer(b, dtype=np.uint8))
            #         row["image"]["bytes"] = np.frombuffer(b, dtype=np.uint8)
            #     else:
            #         row["image"]["bytes"] = []
            # else:
            #     row["image"]["bytes"] = b
            # print(f"{row['image']['bytes'][:10] = }...")  # Print first 10 bytes for debug
            # if isinstance(row["image"], dict):
            #     if isinstance(row["image"].get("bytes", None), list):
            #         row["image"]["bytes"] = [int(x) for x in row["image"]["bytes"]]
            #     row["image"] = json.dumps(row["image"], ensure_ascii=False)

    # --- Step 2.4: image 컬럼이 없는 row에도 string으로 강제 (ArrowTypeError 방지) ---
    # for row in rows:
    #     if "image" not in row or row["image"] is None:
    #         row["image"] = ""
    #     elif not isinstance(row["image"], str):
    #         # 만약 dict 등으로 남아있으면 robust하게 string 변환
    #         try:
    #             row["image"] = json.dumps(row["image"], ensure_ascii=False)
    #         except Exception:
    #             row["image"] = str(row["image"])

    # --- Step 2.5: Parquet 스키마 명시적으로 지정 (image: struct<bytes: list<uint8>, path: string>) ---
    def _type_recursion(sample, column) -> pa.DataType:
        if isinstance(sample, list):
            return pa.list_(_type_recursion(sample[0], column))
        elif isinstance(sample, int):
            return pa.int64()
        elif isinstance(sample, float):
            return pa.float64()
        elif isinstance(sample, str):
            return pa.string()
        elif isinstance(sample, dict):
            # dict는 struct로 처리, 내부 필드 타입은 재귀적으로 추론
            fields = []
            for k, v in sample.items():
                fields.append(pa.field(k, _type_recursion(v, column)))
            return pa.struct(fields)
        elif isinstance(sample, np.ndarray):
            return pa.array(_type_recursion(sample.tolist(), column))
        # elif isinstance(sample, bytes):
        #     return pa.by  # bytes는 binary로 처리
        else:
            return pa.string()  # 기본적으로 string으로 처리
    
    # 모든 컬럼 이름 추출
    all_columns = set()
    for row in rows:
        all_columns.update(row.keys())
    all_columns = list(all_columns)

    # pyarrow schema 생성
    fields = []
    for col in all_columns:
        if col == "image":
            # fields.append(pa.field("image", pa.string()))
            print("#" * 30 + "HERE")
            fields.append(pa.field("image", pa.struct([
                pa.field("bytes", pa.binary()),
                pa.field("path", pa.string())
            ])))
        else:
        # 타입 추론: list->list(string), int/float->int64/double, str->string, dict->string
            sample = next((
                row[col] for row in rows 
                if col in row and row[col] is not None
            ), None)
            fields.append(pa.field(col, _type_recursion(sample, col)))
            # if isinstance(sample, list):
            #     fields.append(pa.field(col, pa.list_(pa.string())))
            # elif isinstance(sample, int):
            #     fields.append(pa.field(col, pa.int64()))
            # elif isinstance(sample, float):
            #     fields.append(pa.field(col, pa.float64()))
            # elif isinstance(sample, str):
            #     fields.append(pa.field(col, pa.string()))
            # elif isinstance(sample, dict):
            #     fields.append(pa.field(col, pa.struct([])))
            # else:
            #     fields.append(pa.field(col, pa.string()))
        print(f"Column '{col}' type inferred as {fields[-1].type}")
    schema = pa.schema(fields)
    print(f"Parquet schema: {schema}")
# 
    # # --- Step 2.7: image['bytes']가 bytes 타입이면 list<uint8>로 변환 ---
    # for row in rows:
    #     if "image" in row and isinstance(row["image"], dict):
    #         img = row["image"]
    #         # 이중 중첩 dict 방지
    #         if isinstance(img.get("bytes", None), dict):
    #             img = img["bytes"]
    #         b = img.get("bytes", None)
    #         if isinstance(b, bytes):
    #             row["image"]["bytes"] = list(np.frombuffer(b, dtype=np.uint8))
    #         elif isinstance(b, dict):
    #             # dict면 dict["bytes"]로 재귀적으로 추출
    #             inner_b = b.get("bytes", None)
    #             if isinstance(inner_b, bytes):
    #                 row["image"]["bytes"] = list(np.frombuffer(inner_b, dtype=np.uint8))
    #             elif isinstance(inner_b, list):
    #                 row["image"]["bytes"] = inner_b
    #             else:
    #                 raise TypeError(f"image['bytes'] is nested dict but not bytes/list: {type(inner_b)}")
    #         elif isinstance(b, list):
    #             pass  # OK
    #         else:
    #             raise TypeError(f"image['bytes'] must be list/bytes, got {type(b)}")
# 
    # # --- Step 2.9: pyarrow Table 생성 및 Parquet 저장 (schema 적용) ---
    # --- Step 2.8: image['bytes']가 list<uint8>이 아닐 경우 한 줄로 강제 변환 ---
    # def _extract_bytes(b):
    #     # Recursively extract innermost non-dict value for image['bytes']
    #     while isinstance(b, dict):
    #         b = b.get("bytes", None)
    #     return b

    # for row in rows:
    #     if "image" in row and isinstance(row["image"], dict):
    #         b = row["image"].get("bytes", None)
    #         # b = _extract_bytes(b)
    #         if not isinstance(b, list):
    #             if isinstance(b, bytes):
    #                 row["image"]["bytes"] = list(np.frombuffer(b, dtype=np.uint8))
    #             else:
    #                 row["image"]["bytes"] = []
    #         else:
    #             row["image"]["bytes"] = b
    #     print(f"{row['image']['bytes'][:10] = }...")  # Print first 10 bytes for debug
    print("#" * 80)
    # for row in rows:
    #     # for k, v in row.items():
    #     #     print(f"{k}: {type(v)}")
    #     row["image"] = eval(row["image"])
        # print(f"{type(row['image']['bytes']) = }\n{type(row['image']['path'])}")

    # rows = serialize_for_parquet(rows)
    # for  row in rows:
    #     row = serialize_for_parquet(row)
# 
    rows = [safe_json(row) for row in rows]
    print("#" * 80)
    print(f"{type(rows) = }")
    for row in rows:
        print(f"{type(row) = }")
        for k, v in row.items():
            print(f"{k}: {type(v)}")
            if isinstance(v, list) and k != "image":
                for item in v:
                    print(f"  {type(item)}")
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    print(f"  {k2}: {type(v2)}")
                    if k2 == "translated_question":
                        for k3, v3 in v2.items():
                            print(f"    {k3}: {type(v3)}")

    #sym:table
    # Schema (Arrow/Parquet):
    # {
    #   "Fields": {
    #     "image": {"DataType": {"TypeId": 25, "Name": "struct", "IsFixedWidth": false}, "Name": "image", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "date": {"DataType": {"TypeId": 9, "Name": "int64", "IsFixedWidth": true}, "Name": "date", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "number": {"DataType": {"TypeId": 9, "Name": "int64", "IsFixedWidth": true}, "Name": "number", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "subfield": {"DataType": {"TypeId": 13, "Name": "utf8", "IsFixedWidth": false}, "Name": "subfield", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "question": {"DataType": {"TypeId": 13, "Name": "utf8", "IsFixedWidth": false}, "Name": "question", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "subject": {"DataType": {"TypeId": 13, "Name": "utf8", "IsFixedWidth": false}, "Name": "subject", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "options": {"DataType": {"TypeId": 13, "Name": "utf8", "IsFixedWidth": false}, "Name": "options", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "answer": {"DataType": {"TypeId": 13, "Name": "utf8", "IsFixedWidth": false}, "Name": "answer", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "answer_key": {"DataType": {"TypeId": 13, "Name": "utf8", "IsFixedWidth": false}, "Name": "answer_key", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "explanation": {"DataType": {"TypeId": 13, "Name": "utf8", "IsFixedWidth": false}, "Name": "explanation", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "question_type": {"DataType": {"TypeId": 13, "Name": "utf8", "IsFixedWidth": false}, "Name": "question_type", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "field": {"DataType": {"TypeId": 13, "Name": "utf8", "IsFixedWidth": false}, "Name": "field", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "korean_national_technical_certification": {"DataType": {"TypeId": 13, "Name": "utf8", "IsFixedWidth": false}, "Name": "korean_national_technical_certification", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "exam": {"DataType": {"TypeId": 13, "Name": "utf8", "IsFixedWidth": false}, "Name": "exam", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "human_acc": {"DataType": {"TypeId": 13, "Name": "utf8", "IsFixedWidth": false}, "Name": "human_acc", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "difficulty": {"DataType": {"TypeId": 13, "Name": "utf8", "IsFixedWidth": false}, "Name": "difficulty", "IsNullable": true, "HasMetadata": false, "Metadata": null},
    #     "id": {"DataType": {"TypeId": 13, "Name": "utf8", "IsFixedWidth": false}, "Name": "id", "IsNullable": true, "HasMetadata": false, "Metadata": null}
    #   },
    #   ... (truncated for brevity)
    # }


    def enforce_list(value):
        if isinstance(value, list):
            return value
        if value is None:
            return []
        # 나머지는 리스트로 감싸기
        return [value]

    # 모든 row, 모든 list 컬럼에 대해!
    list_columns = ["options", "ko_options", "en_options", "eval_loop", "history", "feedbacks", "splits"]
    for row in rows:
        for col in list_columns:
            if col in row:
                row[col] = enforce_list(row[col])

    for row_idx, row in enumerate(rows):
        for col in list_columns:
            if col in row and not isinstance(row[col], list):
                print(f"[FATAL] {col} row {row_idx}: {row[col]} ({type(row[col])})")
            # 리스트 내부에 list가 아닌 값이 있는 경우
            if col in row and isinstance(row[col], list):
                for i, item in enumerate(row[col]):
                    if not isinstance(item, (dict, str)):  # 예: int, float 등
                        print(f"[FATAL] {col} row {row_idx} 내부 {i}: {item} ({type(item)})")


    for row in rows:
        if "image" in row and isinstance(row["image"], dict):
            # row["image"]["bytes"]가 이미 bytes라면 그대로, str이면 bytes로 디코딩 필요
            b = row["image"].get("bytes")
            if isinstance(b, str):
                # base64로 저장했으면 decode 필요, 아니면 바로 bytes(b, ...)로 변환
                row["image"]["bytes"] = b.encode("latin1")  # or .encode("utf-8") if original encoding
            elif isinstance(b, bytes):
                pass
            else:
                row["image"]["bytes"] = b  # or b"" for empty



    table = pa.Table.from_pylist(rows, schema=schema)  # , schema=schema)
    # print(f"{row['image']['bytes'] = }")  # Print first 10 bytes for debug
    pq.write_table(table, path)

    # --- Step 3: 모든 컬럼에 대해 safe_json 재귀 적용 ---
    # for row in rows:
    #     for col in row:
    #         row[col] = safe_json(row[col])

    # --- Step 4: list 후보 컬럼은 반드시 list로 강제 (None/NaN/str/other → []) ---
    for row_idx, row in enumerate(rows):
        for col in list_candidates:
            if col in row:
                v = row[col]
                if isinstance(v, str):
                    v_strip = v.strip()
                    if v_strip.startswith('[') and v_strip.endswith(']'):
                        parsed = json.loads(v_strip)
                        if isinstance(parsed, list):
                            row[col] = parsed
                            continue

                if isinstance(v, list):
                    continue

                if v is None or (isinstance(v, float) and np.isnan(v)):
                    print(f"[FIX][{col}] row {row_idx}: None/NaN detected, converting to []")
                    row[col] = []
                    continue
                
                print(f"[FIX][{col}] row {row_idx}: Non-list value detected ({type(v)}), converting to []")
                row[col] = []

    # --- Step 5: Debug print for all list candidate columns and image column ---
    # print("[DEBUG] List candidate column type check (row type, first element type if list):")
    # def _shorten(val, maxlen=6):
    #     if isinstance(val, list):
    #         l = len(val)
    #         if l == 0:
    #             return val
    #         if l > maxlen:
    #             return val[:2] + ["..."] + val[-2:]
    #         return val
    #     elif isinstance(val, dict):
    #         keys = list(val.keys())
    #         if len(keys) > maxlen:
    #             short = {k: val[k] for k in keys[:2]}
    #             short["..."] = f"...{len(keys)-4} more..."
    #             for k in keys[-2:]:
    #                 short[k] = val[k]
    #             return short
    #         return val
    #     elif isinstance(val, str):
    #         if len(val) > 80:
    #             return val[:30] + " ... " + val[-30:] + f" (len={len(val)})"
    #         return val
    #     return val
    # for col in list_candidates:
    #     for i, row in enumerate(rows):
    #         v = row.get(col, None)
    #         vtype = type(v)
    #         if isinstance(v, list):
    #             if len(v) == 0:
    #                 print(f"  {col} row {i}: type=list, len=0 (empty list)")
    #             else:
    #                 first_type = type(v[0])
    #                 preview = _shorten(v)
    #                 print(f"  {col} row {i}: type=list, first_elem_type={first_type}, value={preview} (len={len(v)})")
    #         else:
    #             print(f"  {col} row {i}: type={vtype}, value={_shorten(v)}")

    # --- Step 10: Parquet 저장 후, Arrow 레벨에서 실제 값을 row별로 확인 ---
    # print("[PARQUET RAW VERIFY] Checking actual Arrow-level values in list candidate columns after Parquet write...")
    # pf = pq.ParquetFile(path)
    # table2 = pf.read()
    # print("[PARQUET RAW VERIFY] Schema after write:")
    # print(table2.schema)
    # for col in list_candidates:
    #     if col in table2.column_names:
    #         coldata = table2.column(col)
    #         for i in range(len(coldata)):
    #             v = coldata[i].as_py()
    #             print(f"[PARQUET RAW VERIFY] {col} row {i}: type={type(v)}, value={v}")
    #             # if not isinstance(v, list):
    #             #     print(f"[PARQUET RAW VERIFY][ERROR] {col} row {i}: type={type(v)}, value={v} (NOT a list! This will break ParquetViewer!)")
    # 
    # # --- Debug print for image column after Parquet write ---
    # if "image" in table2.column_names:
    #     coldata = table2.column("image")
    #     print(f"{coldata = }")
    #     print("[PARQUET RAW VERIFY] image column after Parquet write:")
    #     for i in range(len(coldata)):
    #         v = coldata[i].as_py()
    #         print(f"  image row {i}: type={type(v)}, value={v}")


def refresh_terminal():
    # Terminal refresh (ctrl+l effect)
    print("\033[2J\033[1;1H", end="")


def write_readme_from_outputs(
    output_dir: str = "output",
    eval_result_dir: str = "eval_results",
    readme_path: str = "output/gpt-eval/README.md"
):
    """
    Write a README.md file from the outputs.
    번역된 결과물(Parquet 파일들)로부터 README.md를 자동 생성합니다.
    """

    configs = defaultdict(lambda: {"splits": {}, "features": set()})
    all_tags = set()
    total_instances = 0

    eval_dir = os.path.join(output_dir, eval_result_dir)
    if not os.path.exists(eval_dir):
        raise FileNotFoundError(f"{eval_dir} does not exist.")

    # 1. config별 split별 parquet 파일 탐색 (pandas 없이 pyarrow로 대체)
    for config_name in os.listdir(eval_dir):
        if not config_name == "Drawing_Interpretation":
            continue
        config_path = os.path.join(eval_dir, config_name)
        if not os.path.isdir(config_path):
            continue
        for f in os.listdir(config_path):
            if not f.endswith(".parquet"): continue
            split = f.split(".")[0]
            file_path = os.path.join(config_path, f)
            try:
                pf = pq.ParquetFile(file_path)
                table = pf.read()
            except Exception:
                continue
            num_examples = table.num_rows
            total_instances += num_examples
            configs[config_name]["splits"][split] = {
                "num_examples": num_examples,
                "file": os.path.relpath(file_path, os.path.join(output_dir, eval_result_dir))
            }
            configs[config_name]["features"].update(table.column_names)
            # 태그 추출 (예: question_type, field 등)
            for col in ["question_type", "field", "subfield"]:
                if col in table.column_names:
                    # pyarrow Table에서 null이 아닌 값만 추출
                    coldata = table.column(col)
                    # as_py()로 변환 후 None/null 제거, set으로 unique 추출
                    tags = set(x for x in coldata.to_pylist() if x is not None)
                    all_tags.update(tags)

    # 2. YAML 헤더 생성
    yaml_lines = ["---"]
    yaml_lines.append("language:\n- ko\n- en")
    yaml_lines.append("license: cc-by-nc-4.0")
    yaml_lines.append("size_categories:\n- 10K<n<100K")
    yaml_lines.append("task_categories:\n- question-answering\n- multiple-choice")
    yaml_lines.append(f"pretty_name: kocem:{eval_result_dir.lower()}")
    yaml_lines.append("dataset_info:")

    def arrow_type_to_hf_feature(arrow_type):
        t = str(arrow_type)
        if t == "string":
            return "string"
        elif t == "double":
            return "float64"
        elif t == "int64":
            return "int64"
        elif t == "int32":
            return "int32"
        elif t == "bool":
            return "bool"
        elif t.startswith("list<"):
            m = re.match(r"list<element: (.*)>", t)
            if m:
                elem_type = m.group(1)
                return "list:\n    - dtype: {}".format(arrow_type_to_hf_feature(elem_type))
            else:
                return "list:\n    - dtype: string"
        elif t.startswith("struct<"):
            m = re.match(r"struct<(.+)>", t)
            if m:
                fields = m.group(1)
                if "bytes" in fields:
                    return "image"
                else:
                    return "dict"
        else:
            return "string"  # Fallback to string representation

    # Change dataset_info to a dict: config_name: {features, splits, description}
    for config, info in configs.items():
        yaml_lines.append(f"  {config}:")
        yaml_lines.append(f"    features:")
        dtype_map = {}
        schema_map = {}
        for s_info in info["splits"].values():
            split_file = s_info["file"]
            try:
                pf = pq.ParquetFile(os.path.join(output_dir, eval_result_dir, split_file))
                schema = pf.schema_arrow
                for field in schema:
                    t = str(field.type)
                    dtype_map[field.name] = t
                    schema_map[field.name] = field.type
            except Exception:
                print(f"[ERROR] Failed to read Parquet file for dtype extraction: {split_file}")
                pass
        # Indent: 4 for all nested fields, 2 for top-level
        def arrow_to_yaml(name, typ, level=0):
            pad = ' ' * (4 + level * 2)
            t = str(typ)
            if t.startswith('list<element: struct<'):
                struct_fields = typ.value_type
                lines = [f"{pad}- name: {name}", f"{pad}  list:", f"{pad}  struct:"]
                for f in struct_fields:
                    lines.extend(arrow_to_yaml(f.name, f.type, level+1))
                return lines
            elif t.startswith('list<'):
                elem_type = typ.value_type
                if str(elem_type).startswith('struct<'):
                    lines = [f"{pad}- name: {name}", f"{pad}  list:", f"{pad}  struct:"]
                    for f in elem_type:
                        lines.extend(arrow_to_yaml(f.name, f.type, level+1))
                    return lines
                else:
                    return [f"{pad}- name: {name}", f"{pad}  list: {arrow_type_to_hf_feature(elem_type)}"]
            elif t.startswith('struct<'):
                lines = [f"{pad}- name: {name}", f"{pad}  struct:"]
                for f in typ:
                    lines.extend(arrow_to_yaml(f.name, f.type, level+1))
                return lines
            elif name == "image":
                return [f"{pad}- name: {name}", f"{pad}  dtype: image"]
            else:
                return [f"{pad}- name: {name}", f"{pad}  dtype: {arrow_type_to_hf_feature(typ)}"]

        for feat in sorted(info["features"]):
            arrow_type = schema_map.get(feat)
            if arrow_type is not None:
                lines = arrow_to_yaml(feat, arrow_type, level=0)
                yaml_lines.extend(lines)
            else:
                yaml_lines.append(f"      - name: {feat}\n        dtype: null")
        yaml_lines.append(f"    splits:")
        for split, s_info in info["splits"].items():
            yaml_lines.append(f"      - name: {split}\n        num_examples: {s_info['num_examples']}")
        feature_list = ', '.join(sorted(info["features"]))
        split_list = ', '.join(sorted(info["splits"].keys()))
        n = sum(s["num_examples"] for s in info["splits"].values())
        tag_list = ', '.join(sorted(all_tags)) if all_tags else "-"
        yaml_lines.append(f"    description: |\n      This config contains {n} instances across splits: {split_list}.\n      Features: {feature_list}.\n      Major tags: {tag_list}.")
    yaml_lines.append("configs:")
    for config, info in configs.items():
        yaml_lines.append(f"- config_name: {config}")
        yaml_lines.append("  data_files:")
        for split, s_info in info["splits"].items():
            yaml_lines.append(f"  - split: {split}")
            # 경로를 항상 슬래시(/)로 변환
            path = s_info['file'].replace('\\', '/').replace('\\', '/')
            yaml_lines.append(f"    path: {path}")
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

    contact = "\n## Contact\n\n- pikaybh@snu.ac.kr\n"

    # 4. 파일로 저장
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))
        f.write("\n".join(table_lines))
        f.write(disclaimer)
        f.write(contact)


__all__ = ['save_parquet', 'serialize_for_parquet', 'refresh_terminal', 'write_readme_from_outputs']