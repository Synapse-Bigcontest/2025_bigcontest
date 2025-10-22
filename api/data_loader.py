# api/data_loader.py

import pandas as pd
import os

import sys
from pathlib import Path

# 이 파일(data_loader.py)의 상위(api) 상위(프로젝트 루트)를 경로에 추가
project_root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root_path))

try:
    import config
except ImportError:
    print("--- [FATAL] config.py를 찾을 수 없습니다. sys.path를 확인하세요. ---")
    sys.exit(1)

logger = config.get_logger(__name__)

def load_and_preprocess_data():
    """
    미리 가공된 final_df.csv 파일을 안전하게 찾아 로드하고,
    데이터를 처리하는 과정에서 발생할 수 있는 모든 오류를 방어합니다.
    """
    try:
        file_path = config.PATH_FINAL_DF

        if not file_path.exists():
            logger.critical(f"--- [CRITICAL DATA ERROR] 데이터 파일을 찾을 수 없습니다. 예상 경로: {file_path}")
            logger.critical(f"--- 현재 작업 경로: {Path.cwd()} ---")
            return None
            
        df = pd.read_csv(file_path)

    except Exception as e:
        logger.critical(f"--- [CRITICAL DATA ERROR] 데이터 파일 로딩 중 예측하지 못한 오류 발생: {e} ---", exc_info=True)
        return None
        
    logger.info("--- [Preprocess] Streamlit Arrow 변환 오류 방지용 데이터 클리닝 시작 ---")
    for col in df.select_dtypes(include='object').columns:
        temp_series = (
            df[col]
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        numeric_series = pd.to_numeric(temp_series, errors='coerce') 
        df[col] = numeric_series.fillna(temp_series)
        
    logger.info("--- [Preprocess] 데이터 클리닝 완료 ---")

    cols_to_process = ['월매출금액_구간', '월매출건수_구간', '월유니크고객수_구간', '월객단가_구간']
    
    for col in cols_to_process:
        if col in df.columns:
            try:
                series_str = df[col].astype(str).fillna('')
                series_split = series_str.str.split('_').str[0]
                series_numeric = pd.to_numeric(series_split, errors='coerce')
                df[col] = series_numeric.fillna(0).astype(int)
            except Exception as e:
                logger.warning(f"--- [DATA WARNING] '{col}' 컬럼 처리 중 오류 발생: {e}. 해당 컬럼을 건너뜁니다. ---", exc_info=True)
                continue
                
    logger.info(f"--- [Preprocess] 데이터 로드 및 전처리 최종 완료. (Shape: {df.shape}) ---")
    return df