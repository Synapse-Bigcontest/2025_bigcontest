# streamlit_app.py

import streamlit as st
import os
import pandas as pd
import json 
import traceback
import requests

from langchain_core.messages import HumanMessage, AIMessage 

import config
from orchestrator import AgentOrchestrator
from modules.visualization import display_merchant_profile
from modules.knowledge_base import load_marketing_vectorstore, load_festival_vectorstore

logger = config.get_logger(__name__)

@st.cache_data
def load_data():
    """
    FastAPI 서버로부터 가맹점 목록 데이터를 로드합니다.
    """
    try:
        logger.info(f"API 서버에서 가게 목록 로드 시도: {config.API_MERCHANTS_ENDPOINT}")
        response = requests.get(config.API_MERCHANTS_ENDPOINT)
        response.raise_for_status()
        data = response.json() # [{'가맹점ID': '...', '가맹점명': '...'}, ...]
        
        if not data:
            st.error("API 서버에서 가게 목록을 받았으나 데이터가 비어있습니다.")
            return None
            
        logger.info(f"가게 목록 {len(data)}개 로드 성공.")
        return pd.DataFrame(data)
        
    except requests.exceptions.ConnectionError:
        st.error(f"API 서버({config.API_SERVER_URL})에 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인하세요.")
        return None
    except Exception as e:
        st.error(f"API 서버에서 가게 목록을 불러오는 데 실패했습니다: {e}")
        logger.critical(f"가게 목록 로딩 실패: {e}", exc_info=True)
        return None

st.set_page_config(page_title="소상공인 마케팅 & 축제 AI 컨설턴트", page_icon="🎈", layout="wide")

merchant_df = load_data()
if merchant_df is None:
    st.error("데이터 로딩에 실패했습니다. API 서버가 정상적으로 응답하는지 확인해주세요.")
    st.stop()


def initialize_session():
    """
    세션 초기화 시, Orchestrator와 모든 AI 모듈(RAG, Vector Store)을
    미리 로드하여 캐시합니다.
    """
    if "orchestrator" not in st.session_state:
        # 1. API 키 로드
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
            st.stop()
            
        with st.spinner("AI 모델 및 빅데이터 벡터 DB를 로딩 중입니다..."):
            try:
                load_marketing_vectorstore()
                db = load_festival_vectorstore()
                if db is None:
                    st.error("축제 벡터 스토어 로딩에 실패했습니다. 'build_vector_store.py'를 실행했는지 확인하세요.")
                    st.stop()
                    
                logger.info("--- [Streamlit] 모든 AI 모듈 로딩 완료 ---")

            except Exception as e:
                st.error(f"AI 모듈 초기화 중 오류 발생: {e}")
                logger.critical(f"AI 모듈 초기화 실패: {e}", exc_info=True)
                st.stop()
                
        st.session_state.orchestrator = AgentOrchestrator(google_api_key)
        
    if "step" not in st.session_state:
        st.session_state.step = "get_merchant_name" 
        st.session_state.messages = []
        st.session_state.merchant_id = None
        st.session_state.merchant_name = None
        st.session_state.profile_data = None
        st.session_state.consultation_result = None
        if "last_recommended_festivals" not in st.session_state:
            st.session_state.last_recommended_festivals = []

def restart_consultation():
    """처음으로 돌아가기 (세션 초기화)"""
    keys_to_reset = ["step", "merchant_name", "merchant_id", "profile_data", "messages", "consultation_result", "last_recommended_festivals"]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.markdown("<h3 style='text-align: center;'>Synapse</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>2025 Big Contest</p>", unsafe_allow_html=True)
        st.write("")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button('처음으로 돌아가기', key='restart_button', use_container_width=True):
                restart_consultation()
                st.rerun()

def render_get_merchant_name_step():
    """UI 1단계: 가맹점 검색 및 선택"""
    st.subheader("컨설팅 받을 가게를 검색해주세요.")
    search_query = st.text_input("가게 이름 또는 가맹점 ID 검색", placeholder="예: 메가, 003AC99735 등")

    if search_query:
        mask = (
            merchant_df['가맹점명'].str.contains(search_query, case=False, na=False) |
            merchant_df['가맹점ID'].str.contains(search_query, case=False, na=False)
        )
        search_results = merchant_df[mask].copy()

        if not search_results.empty:
            search_results['display'] = search_results['가맹점명'] + " (" + search_results['가맹점ID'] + ")"
            options = ["선택해주세요..."] + search_results['display'].tolist()
            selected_display_name = st.selectbox("가게를 선택하세요:", options)

            if selected_display_name != "선택해주세요...":
                try:
                    selected_row = search_results[search_results['display'] == selected_display_name].iloc[0]
                    selected_merchant_id = selected_row['가맹점ID']
                    selected_merchant_name = selected_row['가맹점명']
                    button_label = f"'{selected_merchant_name}' 분석 정보 보기"
                    is_selection_valid = True
                except (IndexError, KeyError):
                    st.error("선택한 가게 정보를 찾는 데 실패했습니다. 다시 시도해주세요.")
                    button_label = "분석 정보 보기"
                    is_selection_valid = False

                if st.button(button_label, disabled=not is_selection_valid, type="primary"):
                    with st.spinner(f"'{selected_merchant_name} ({selected_merchant_id})' 가게 정보를 분석 중입니다..."):
                        
                        profile_data = None
                        try:
                            # 1번 제안: config 사용
                            response = requests.post(config.API_PROFILE_ENDPOINT, json={"merchant_id": selected_merchant_id})
                            response.raise_for_status()
                            profile_data = response.json()
                            
                            if "store_profile" not in profile_data or "average_profile" not in profile_data:
                                st.error("API 응답이 유효하지 않습니다. 'store_profile' 또는 'average_profile' 키가 없습니다.")
                                profile_data = None
                        
                        except requests.exceptions.ConnectionError:
                            st.error(f"API 서버({config.API_SERVER_URL})에 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인하세요.")
                        except requests.exceptions.HTTPError as e:
                            st.error(f"가게 프로필 로딩 실패: {e.response.status_code} {e.response.reason}")
                        except Exception as e:
                            st.error(f"가게 프로필 로딩 중 알 수 없는 오류 발생: {e}")
                            logger.critical(f"가게 프로필 API 호출 실패: {e}", exc_info=True)

                        if profile_data:
                            st.session_state.merchant_name = selected_merchant_name
                            st.session_state.merchant_id = selected_merchant_id 
                            st.session_state.profile_data = profile_data 
                            st.session_state.step = "show_profile_and_chat"
                            st.rerun()
        else:
            st.warning("검색 결과가 없습니다. 다른 검색어를 입력해주세요.")

def render_show_profile_and_chat_step():
    """UI 2단계: 프로필 확인 및 AI 채팅"""
    st.subheader(f"✅ '{st.session_state.merchant_name}' 가게 분석 완료")
    with st.expander("📊 상세 데이터 분석 리포트 보기", expanded=True):
        try:
            display_merchant_profile(st.session_state.profile_data)
        except Exception as e:
            st.error(f"프로필 시각화 중 오류 발생: {e}")
            logger.error(f"--- [Visualize ERROR]: {e}\n{traceback.format_exc()}", exc_info=True)

    st.divider()
    st.subheader("💬 AI 컨설턴트와 상담을 시작하세요.")
    st.info("가게 분석 정보를 바탕으로 궁금한 점을 질문해보세요. (예: '20대 여성 고객을 늘리고 싶어요')")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("요청사항을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI 컨설턴트가 답변을 생성 중입니다...(최대 1~2분)"):
                orchestrator = st.session_state.orchestrator
                
                if "store_profile" not in st.session_state.profile_data:
                    st.error("세션에 'store_profile' 데이터가 없습니다. 다시 시작해주세요.")
                    st.stop()
                    
                agent_history = []
                history_to_convert = st.session_state.messages[:-1][-10:]
                
                for msg in history_to_convert:
                    if msg["role"] == "user":
                        agent_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        agent_history.append(AIMessage(content=msg["content"]))
                
                result = orchestrator.invoke_agent(
                    user_query=prompt,
                    store_profile_dict=st.session_state.profile_data["store_profile"],
                    chat_history=agent_history,
                    last_recommended_festivals=st.session_state.last_recommended_festivals,
                )

                response_text = ""
                st.session_state.last_recommended_festivals = []

                if "error" in result:
                    response_text = f"오류 발생: {result['error']}"

                elif "final_response" in result:
                    response_text = result.get("final_response", "응답을 생성하지 못했습니다.")
                    intermediate_steps = result.get("intermediate_steps", [])
                    
                    try:
                        for step in intermediate_steps:
                            action = step[0]
                            tool_output = step[1]
                            
                            if hasattr(action, 'tool') and action.tool == "recommend_festivals":
                                if tool_output and isinstance(tool_output, list) and isinstance(tool_output[0], dict):
                                    recommended_list = [
                                        f.get("축제명") for f in tool_output if f.get("축제명")
                                    ]
                                    
                                    st.session_state.last_recommended_festivals = recommended_list
                                    logger.info(f"--- [Streamlit] 추천 축제 저장됨 (Intermediate Steps): {recommended_list} ---")
                                    break 
                                    
                    except Exception as e:
                        logger.critical(f"--- [Streamlit CRITICAL] Intermediate steps 처리 중 예외 발생: {e} ---", exc_info=True)

                else:
                    response_text = "알 수 없는 오류가 발생했습니다."

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

def main():
    st.title("🎈 소상공인 마케팅 & 축제참여 상담 AI 컨설턴트")
    st.markdown("신한카드 빅데이터와 AI 에이전트를 활용하여, 사장님 가게를 분석하여 꼭 맞는 지역 축제를 추천하고 맞춤형 마케팅 전략을 제시합니다.")
    
    initialize_session()
    render_sidebar()

    if st.session_state.step == "get_merchant_name":
        render_get_merchant_name_step()
    elif st.session_state.step == "show_profile_and_chat":
        render_show_profile_and_chat_step()

if __name__ == "__main__":
    main()