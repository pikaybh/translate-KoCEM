import json
import pandas as pd

from schemas import Option, Quiz


# 모든 컬럼을 dict/list는 json 문자열로, 숫자 타입은 string으로 변환
def safe_json(obj):
    if isinstance(obj, (Option, Quiz)):
        return obj.model_dump()
    elif isinstance(obj, list):
        return [safe_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    return obj


def save_parquet(path: str, rows: list):
    df = pd.DataFrame(rows)

    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: json.dumps(safe_json(x), ensure_ascii=False) if isinstance(x, (dict, list, Option, Quiz)) else x
        )
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(str)
    
    logger = globals()['logger'] if 'logger' in globals() else None
    if logger:
        logger.info(f"[DEBUG] 저장 직전 DataFrame shape: {df.shape}")
        logger.info(f"[DEBUG] 저장 직전 DataFrame head: {df.head()}")
    
    if "image" in df.columns:
        # 이미지 객체를 파일 경로나 None 등으로 변환, 또는 제거
        df["image"] = df["image"].apply(lambda x: str(x) if isinstance(x, str) else None)

    df.to_parquet(path, index=False)


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


__all__ = ['save_parquet', 'serialize_for_parquet']