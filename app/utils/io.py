import json
import pandas as pd


def save_parquet(path: str, rows: list):
    df = pd.DataFrame(rows)
    
    # 모든 컬럼을 dict/list는 json 문자열로, 숫자 타입은 string으로 변환
    for col in df.columns:
        df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(str)
    
    logger = globals()['logger'] if 'logger' in globals() else None
    if logger:
        logger.info(f"[DEBUG] 저장 직전 DataFrame shape: {df.shape}")
        logger.info(f"[DEBUG] 저장 직전 DataFrame head: {df.head()}")
    
    df.to_parquet(path, index=False)


__all__ = ['save_parquet']