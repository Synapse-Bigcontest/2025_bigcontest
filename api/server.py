# api/server.py

import uvicorn
import json
import numpy as np
import pandas as pd
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import math

from api.data_loader import load_and_preprocess_data 
import config

logger = config.get_logger(__name__)

# --- Data Loading ---
DF_MERCHANT = load_and_preprocess_data()
if DF_MERCHANT is None:
    logger.critical("--- [API Server Error] 데이터 로딩 실패. 서버를 종료합니다. ---")
    exit()

# --- FastAPI App & Models ---
app = FastAPI()

class MerchantRequest(BaseModel):
    merchant_id: str

def replace_nan_with_none(data):
    """
    딕셔셔너리나 리스트 내의 모든 NaN 값을 None으로 재귀적으로 변환합니다.
    """
    if isinstance(data, dict):
        return {k: replace_nan_with_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_nan_with_none(i) for i in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    return data

# --- API Endpoints ---

# Streamlit UI의 가맹점 검색용 엔드포인트 추가
@app.get("/merchants")
def get_merchant_list():
    """
    Streamlit UI에서 가게 검색용으로 사용할
    (가맹점ID, 가맹점명) 리스트를 반환합니다.
    """
    try:
        logger.info(f"✅ [API] '/merchants' 가맹점 목록 요청 수신")
        # 'to_dict('records')'가 JSON으로 직렬화하기 가장 좋음
        merchant_list = DF_MERCHANT[['가맹점ID', '가맹점명']].drop_duplicates().to_dict('records')
        logger.info(f"✅ [API] 가맹점 목록 {len(merchant_list)}개 반환 완료")
        return merchant_list
    except Exception as e:
        logger.critical(f"❌ [API CRITICAL] '/merchants' 처리 중 오류: {e}\n{traceback.format_exc()}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"가게 목록 로딩 실패: {e}")


@app.post("/profile")
def get_merchant_profile(request: MerchantRequest):
    """
    가맹점 ID를 받아 프로파일링된 데이터와 동종/동일 상권 평균 데이터를 반환합니다.
    """
    merchant_id = request.merchant_id
    logger.info(f"✅ [API] '/profile' 가맹점 ID '{merchant_id}' 프로파일링 요청 수신")
    try:
        store_df_multiple = DF_MERCHANT[DF_MERCHANT['가맹점ID'] == merchant_id]

        if store_df_multiple.empty:
            logger.warning(f"⚠️ [API] 404 - '{merchant_id}' 가맹점 ID를 찾을 수 없습니다.")
            raise HTTPException(status_code=404, detail=f"'{merchant_id}' 가맹점 ID를 찾을 수 없습니다.")
        
        if len(store_df_multiple) > 1:
            logger.info(f"   [INFO] '{merchant_id}'에 대해 {len(store_df_multiple)}개의 데이터 발견. 최신 데이터로 필터링합니다.")
            temp_df = store_df_multiple.copy()
            temp_df['기준년월_dt'] = pd.to_datetime(temp_df['기준년월'])
            latest_store_df = temp_df.sort_values(by='기준년월_dt', ascending=False).iloc[[0]]
        else:
            latest_store_df = store_df_multiple

        store_data = latest_store_df.iloc[0].to_dict()

        # (고객 비율 및 자동추출특징 계산 로직은 원본과 동일)
        # 4-1. 고객 성별 비율 계산 및 저장
        store_data['남성고객비율'] = (
            store_data.get('남성20대이하비율', 0) + store_data.get('남성30대비율', 0) + 
            store_data.get('남성40대비율', 0) + store_data.get('남성50대비율', 0) + 
            store_data.get('남성60대이상비율', 0)
        )
        store_data['여성고객비율'] = (
            store_data.get('여성20대이하비율', 0) + store_data.get('여성30대비율', 0) + 
            store_data.get('여성40대비율', 0) + store_data.get('여성50대비율', 0) + 
            store_data.get('여성60대이상비율', 0)
        )
        
        # 4-2. 연령대별 비율 계산 (20대이하, 30대, 40대, 50대이상)
        store_data['연령대20대이하고객비율'] = store_data.get('남성20대이하비율', 0) + store_data.get('여성20대이하비율', 0)
        store_data['연령대30대고객비율'] = store_data.get('남성30대비율', 0) + store_data.get('여성30대비율', 0)
        store_data['연령대40대고객비율'] = store_data.get('남성40대비율', 0) + store_data.get('여성40대비율', 0)
        store_data['연령대50대고객비율'] = (
            store_data.get('남성50대비율', 0) + store_data.get('여성50대비율', 0) + 
            store_data.get('남성60대이상비율', 0) + store_data.get('여성60대이상비율', 0)
        )

        male_ratio = store_data.get('남성고객비율', 0)
        female_ratio = store_data.get('여성고객비율', 0)
        핵심고객_성별 = '남성 중심' if male_ratio > female_ratio else '여성 중심' 

        age_ratios = {
            '20대이하': store_data.get('연령대20대이하고객비율', 0),
            '30대': store_data.get('연령대30대고객비율', 0),
            '40대': store_data.get('연령대40대고객비율', 0),
            '50대이상': store_data.get('연령대50대고객비율', 0),
        }
        핵심연령대_결과 = max(age_ratios, key=age_ratios.get)
        
        store_data['자동추출특징'] = {
            "핵심고객": 핵심고객_성별,
            "핵심연령대": 핵심연령대_결과,
            "매출순위": f"상권 내 상위 {store_data.get('동일상권내매출순위비율', 0):.1f}%, 업종 내 상위 {store_data.get('동일업종내매출순위비율', 0):.1f}%"
        }

        area = store_data.get('상권')
        category = store_data.get('업종')
        
        average_df = DF_MERCHANT[(DF_MERCHANT['상권'] == area) & (DF_MERCHANT['업종'] == category)]

        if average_df.empty:
            average_data = {}
        else:
            numeric_cols = average_df.select_dtypes(include=np.number).columns
            average_data = average_df[numeric_cols].mean().to_dict()

        average_data['가맹점명'] = f"{area} {category} 업종 평균"
        
        final_result = {
            "store_profile": store_data,
            "average_profile": average_data
        }
        
        clean_result = replace_nan_with_none(final_result)
        
        logger.info(f"✅ [API] '{store_data.get('가맹점명')}({merchant_id})' 프로파일링 성공 (기준년월: {store_data.get('기준년월')})")
        return clean_result

    except HTTPException as e:
        # 404 외의 오류는 여기서 별도 로깅
        if e.status_code != 404:
            logger.error(f"❌ [API ERROR] 처리 중 오류: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logger.critical(f"❌ [API CRITICAL] 예측하지 못한 오류: {e}\n{traceback.format_exc()}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)