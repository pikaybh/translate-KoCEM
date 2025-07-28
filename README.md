# KoCEM Translation Framework

LangChain 기반의 한국어-영어 번역 프레임워크로 Hugging Face의 "pikaybh/KoCEM" 데이터셋을 영문으로 번역합니다.

![](src\1.png)
![](src\2.png)
![](src\3.png)


## 🎯 주요 기능

- **자동 번역**: GPT 모델을 사용한 고품질 한->영 번역
- **Evaluator Agent**: 번역 품질 자동 평가 (수정됨)
- **Human-in-Loop**: 인간 검토 및 수정 기능
- **캐싱**: 로컬 데이터셋 캐싱
- **진행률 추적**: 실시간 번역 진행 상황 모니터링

## 📦 설치

```powershell
# 패키지 설치
pip install -r requirements.txt
```

## ⚙️ 환경 설정

`.env` 파일에 OpenAI API 키를 설정하세요:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## 🚀 사용법

### 1. 간단한 테스트
```python
python simple_main.py
```

### 2. 개선된 프레임워크 테스트
```python
python improved_framework.py
```

### 3. 통합 테스트
```python
python integration_test.py
```

### 4. Streamlit UI
```python
streamlit run ui.py
```

## 📁 파일 구조

### 핵심 파일
- `simple_main.py`: 간단한 실행 파일 (추천)
- `improved_framework.py`: 개선된 독립 프레임워크
- `translation_framework.py`: 메인 번역 프레임워크 (수정 중)
- `evaluator.py`: 번역 품질 평가 에이전트 (수정됨)
- `human_loop.py`: 인간 검토 인터페이스
- `data_manager.py`: 데이터셋 관리 및 캐싱 (수정됨)
- `ui.py`: Streamlit 사용자 인터페이스

### 테스트 파일
- `minimal_test.py`: 최소 기능 테스트
- `simple_test.py`: 기본 번역 테스트  
- `sync_test.py`: 동기식 번역 테스트
- `integration_test.py`: 통합 테스트

### 데이터
- `cached_datasets/`: 캐시된 데이터셋 저장소

## 🛠️ 주요 개선사항

### v2.0 수정사항
1. **프롬프트 템플릿 수정**: JSON 형식에서 텍스트 기반으로 변경
2. **KoCEM 데이터셋 구성 지원**: 12개 도메인 중 선택 가능
3. **오류 처리 개선**: 더 안정적인 파싱 및 오류 복구
4. **간소화된 인터페이스**: 복잡한 비동기 처리 단순화

### 사용 가능한 KoCEM 구성
- Architectural_Planning
- Building_System  
- Comprehensive_Understanding (기본)
- Construction_Management
- Drawing_Interpretation
- Domain_Reasoning
- Interior
- Industry_Jargon
- Materials
- Safety_Management
- Standard_Nomenclature
- Structural_Engineering

## 🐛 문제 해결

### 일반적인 문제
1. **API 키 오류**: `.env` 파일에 올바른 OpenAI API 키 설정 확인
2. **데이터셋 로드 오류**: 구성명이 올바른지 확인
3. **프롬프트 오류**: 수정된 버전(`improved_framework.py`) 사용

### 권장사항
- 첫 사용시 `simple_main.py` 또는 `improved_framework.py` 사용
- 안정성을 위해 평가 기능은 선택적으로 사용
- 대량 번역 시 캐싱 기능 적극 활용

## 📈 성능

- **번역 속도**: GPT-4o-mini 기준 약 1-2초/문장
- **캐싱**: 재실행 시 즉시 로드
- **메모리 사용량**: 중간 규모 데이터셋 기준 적당함

이제 프레임워크가 더 안정적으로 작동할 것입니다! 🎉
