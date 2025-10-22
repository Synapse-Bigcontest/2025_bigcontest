# 🎈 AI 지역축제 참여 솔루션 (Agentic RAG)

### Agentic RAG 기반 소상공인 맞춤형 지역 축제 컨설팅 시스템

신한카드 빅데이터와 전국 축제 정보를 통합 분석하여, **AI 에이전트**가
가게별로 참여할 만한 지역 축제와 최적의 마케팅 전략 보고서를 자동
생성합니다.

------------------------------------------------------------------------

## 🧭 프로젝트 개요

이 프로젝트는 **Streamlit 웹 인터페이스 + FastAPI 데이터 서버 +
LangChain 에이전트**로 구성된 AI 컨설팅 자동화 시스템입니다.\
사용자는 자신의 가게를 선택하여 상세 프로필을 확인한 뒤,\
예를 들어 "20대 여성 고객을 늘리고 싶어" 또는 "근처 축제 추천해줘" 와
같은 자연어 질문을 입력할 수 있습니다.

AI 에이전트(`Orchestrator`)는 이 질문을 이해하고, **가게의 상세
프로필(JSON)**을 컨텍스트로 삼아 가장 적절한 도구(`Tool`)를 스스로
선택하고 실행합니다.

------------------------------------------------------------------------

## 🔧 예시 흐름

  ----------------------------------------------------------------------------------
  기능           실행 도구                                주요 처리
  -------------- ---------------------------------------- --------------------------
  축제 추천      `recommend_festivals`                    FAISS 벡터 검색 + LLM
                                                          재평가(Re-ranking)

  마케팅 전략    `search_contextual_marketing_strategy`   마케팅 RAG

  가게 분석      `analyze_merchant_profile`               LLM 기반 SWOT 분석

  축제 분석      `analyze_festival_profile`               LLM 기반 축제 요약
  -------------- ---------------------------------------- ----------------------------

최종적으로, 에이전트는 도구 실행 결과를 바탕으로 자연스러운 한국어
**컨설팅 답변**을 생성하여 사용자에게 제공합니다.

------------------------------------------------------------------------

## 📂 프로젝트 구조

``` bash
AI_FESTIVAL_CONSULTANT/
├── orchestrator.py           # AI 에이전트 (Tool-Calling Agent)
├── streamlit_app.py          # Streamlit 웹 인터페이스 (UI)
├── config.py                 # 프로젝트 설정 중앙화 (경로, API, 모델명 등)
│
├── api/
│   ├── server.py             # FastAPI 데이터 서버 (가게 프로필, 목록 API)
│   └── data_loader.py        # 데이터 로드 및 전처리
│
├── modules/
│   ├── filtering.py          # 축제 추천 파이프라인 (FestivalRecommender 클래스)
│   ├── knowledge_base.py     # RAG 및 벡터 스토어 로더 (Tool 제거)
│   ├── llm_provider.py       # LLM 인스턴스 관리
│   ├── profile_utils.py      # 가맹점 프로필 JSON 유틸리티
│   └── visualization.py      # Streamlit 시각화 모듈
│
├── tools/                    # LangChain @tool 도구 정의
│   ├── festival_recommender.py # [Tool] recommend_festivals
│   ├── marketing_strategy.py   # [Tool] 마케팅 전략 관련 도구
│   ├── profile_analyzer.py     # [Tool] 가게/축제 분석 도구
│   └── tool_loader.py          # 모든 도구를 Orchestrator로 전달
│
├── utils/                    # 공통 유틸리티
│   └── parser_utils.py       # LLM JSON 응답 파서
│
└── data/
│   ├── final_df.csv          # 가맹점 데이터
│   └── festival_df.csv       # 축제 정보 데이터
│ 
└── vectorstore/
    ├── faiss_festival        # 축제 벡터 DB
    └── faiss_marketing       # 마케팅 벡터 DB
```

------------------------------------------------------------------------

## 🔄 아키텍처 및 데이터 흐름

이 시스템은 **에이전트 중심의 도구 호출 (Tool-Calling)** 아키텍처를
기반으로 작동합니다.\
사용자의 자연어 질문은 `Orchestrator`라는 AI 에이전트에 의해 해석되며,\
에이전트는 `[가게 프로필]` 컨텍스트를 바탕으로 가장 적절한 도구를
선택하고 실행하여 답변을 생성합니다.

------------------------------------------------------------------------

## 🧩 시스템 구성도

``` mermaid
graph TD
    A["Streamlit UI"] --> B["FastAPI Server api/server.py"]
    B --> A["Streamlit UI"]
    
    A --> C["Orchestrator orchestrator.py AgentExecutor (채팅 입력 포함)"]
    
    C --> D{"Tool Routing (LLM 의도 분석 후 도구 선택)"}
    
    %% RAG (축제 추천) 경로
    D --> E["Tool: recommend_festivals"]
    E --> EM["Embedding Model (HuggingFace, Google 등)"]
    E --> VSF["FAISS (축제 벡터 DB)"]
    
    %% RAG (마케팅 전략) 경로
    D --> G["Tool: search_contextual_marketing_strategy"]
    G --> EM
    G --> VSM["FAISS (마케팅 전략 DB)"]
    
    %% 기타 분석 도구
    D --> I["Tool: analyze_merchant_profile"]
    D --> K["Tool: analyze_festival_profile"]
    
    %% 도구 결과 취합
    E --> C
    G --> C
    I --> C
    K --> C
    
    %% 최종 답변
    C --> A
    A --> M["사용자 (최종 AI 컨설팅 답변 출력)"]

    %% --- Styling ---
    style A fill:#4CAF50,color:#fff
    style B fill:#FF9800,color:#fff
    style C fill:#E91E63,color:#fff
    style D fill:#9C27B0,color:#fff
    style E fill:#03A9F4,color:#fff
    style G fill:#03A9F4,color:#fff
    style I fill:#03A9F4,color:#fff
    style K fill:#03A9F4,color:#fff
    style M fill:#607D8B,color:#fff
    
    %% 새로 추가된 노드 스타일
    style EM fill:#4DD0E1,color:#000
    style VSF fill:#FFC107,color:#000
    style VSM fill:#FFC107,color:#000
```

------------------------------------------------------------------------

## 📍 데이터 흐름 상세

1.  **프로필 로드 (UI → API → UI)**\
    사용자가 Streamlit에서 가게 선택/profile 호출 → 가맹점 프로필 데이터
    세션에 저장\
2.  **에이전트 호출 (UI → Orchestrator)**\
    사용자가 채팅 입력 → `invoke_agent()` 호출\
    전달 데이터: Query + Profile + 이전 대화 기록(History)\
3.  **의도 분석 및 도구 라우팅 (Orchestrator → LLM → Tool)**\
    `profile_utils.py`로 API 응답 → 채팅용 JSON 변환\
    LLM 기반 에이전트가 의도 분석 후 적합 도구 선택\
4.  **도구 실행 및 최종 답변 생성 (Tool → Orchestrator → LLM → UI)**\
    선택 도구 실행 → 결과 반환 (예: 축제 Top3)\
    도구 결과를 LLM에 주입 → 최종 자연어 답변 생성\
    Streamlit UI로 출력 → 사용자 확인

------------------------------------------------------------------------

## ⚙️ 주요 특징 요약

  기능                           설명
  ------------------------------ -----------------------------------------------
  **에이전트 기반 도구 호출**    LLM이 스스로 적합한 도구를 선택 실행
  **FAISS 검색**                 지역 축제 및 데이터 기반 유사 항목 검색
  **RAG 통합**                   지식 기반 문서에서 컨텍스트 검색 후 전략 생성
  **SWOT/요약 분석**             LLM을 통한 가게 및 축제 분석 기능
  **Streamlit + FastAPI 연동**   UI와 API 간의 프로필 데이터 교환 구조

------------------------------------------------------------------------

## 💡 기술 스택

-   **Frontend:** Streamlit\
-   **Backend:** FastAPI\
-   **AI Model:** Gemini 2.5 Flash\
-   **AI Framework:** LangChain (Agents, Tool Calling)\
-   **RAG Engine:** FAISS\
-   **Embeddings:** HuggingFace `dragonkue/BGE-m3-ko`\
-   **Data:** Pandas, NumPy, Matplotlib

------------------------------------------------------------------------

## 🚀 실행 방법

### 1️⃣ FastAPI 서버 실행

``` bash
cd C:\projects\AI_FESTIVAL_CONSULTANT
uv venv
call .venv\Scripts\activate.bat
cd AI_FESTIVAL_CONSULTANT
uv pip install -r requirements.txt
python -m api.server
```

### 2️⃣ Streamlit 앱 실행

``` bash
cd C:\projects\AI_FESTIVAL_CONSULTANT
uv venv
call .venv\Scripts\activate.bat
cd AI_FESTIVAL_CONSULTANT
mkdir .streamlit
echo GOOGLE_API_KEY="(발급받은 gemini API key)" > .streamlit\secrets.toml
uv run streamlit run streamlit_app.py
```
------------------------------------------------------------------------

## 📈 예시 시나리오

  -------------------------------------------------------------------------------------------
  사용자 입력                     실행 도구                                     결과
  ------------------------------- --------------------------------------------- -------------
  "우리 가게 분석해줘"            analyze_merchant_profile                      SWOT 분석
                                                                                리포트

  "주말 방문 고객을 늘리고        recommend_festivals                           Top 3 축제
  싶어요"                                                                       추천

  "서울디저트페어 마케팅 전략     create_festival_specific_marketing_strategy   맞춤형 전략
  알려줘"                                                                       제안
  -------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 🧠 핵심 아이디어

> "LLM이 스스로 도구를 선택하고 실행하는 **Agentic RAG**"

-   LangChain의 **Tool-Calling Agent 구조**
-   가게 프로필(JSON)을 컨텍스트로 하는 자연어 질의 기반 의사결정
-   **FAISS + LLM 재평가 기반** 하이브리드 축제 추천 엔진
