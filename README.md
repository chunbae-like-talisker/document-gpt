# 🍔 Chunbae’s DocumentGPT

> 개인 문서(Q&A)용 Streamlit + LangChain + OpenAI 기반 문서 질의응답 앱  
> 파일 업로드 → 문서 임베딩 → 실시간 LLM 스트리밍 응답까지 가능한 미니 RAG(Retrieval-Augmented Generation) 프로젝트

---

## 🚀 주요 기능

- **문서 업로드 지원**: PDF, TXT, DOCX 파일을 업로드하면 자동으로 텍스트 분리 및 임베딩
- **벡터 검색 (RAG)**: 업로드된 문서 내용 기반으로 GPT가 답변
- **실시간 스트리밍 출력**: OpenAI Chat API의 토큰 단위 응답을 Streamlit UI에서 실시간 표시
- **대화 메모리 유지**: `ConversationSummaryBufferMemory` 기반으로 대화 문맥 기억
- **로컬 캐시 저장**: 동일 문서 재업로드 시 임베딩 결과를 캐시에서 빠르게 불러오기

---

## 🧠 사용 기술 스택

| 분야              | 기술                                                             |
| ----------------- | ---------------------------------------------------------------- |
| **Frontend/UI**   | [Streamlit](https://streamlit.io/)                               |
| **LLM Interface** | [LangChain](https://python.langchain.com/)                       |
| **Embeddings**    | `OpenAIEmbeddings`, `CacheBackedEmbeddings`                      |
| **Vector DB**     | [FAISS](https://github.com/facebookresearch/faiss)               |
| **Model**         | `gpt-3.5-turbo` (via [OpenAI API](https://platform.openai.com/)) |
| **Memory**        | `ConversationSummaryBufferMemory`                                |

---

## ⚙️ 실행 방법

```bash
# 🧩 개발 환경
Python 3.11.14

# 1️⃣ 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# 또는
.venv\Scripts\activate      # Windows

# 2️⃣ 패키지 설치
pip install -r requirements.txt

# 3️⃣ 실행
streamlit run app.py
```
