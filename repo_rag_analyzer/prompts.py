def create_code_rag_prompt(context_for_rag, search_query):
    """코드베이스 RAG 프롬프트 생성"""
    return f"""
        주어진 코드 컨텍스트를 바탕으로 다음 질문에 대해 한국어로 상세히 답변해 주세요.
        코드 예제가 있다면 포함하고, 함수나 클래스의 사용법을 설명해 주세요.
        만약 컨텍스트에 질문과 관련된 코드가 없다면, "컨텍스트에 관련 코드가 없습니다."라고 답변해 주세요.
        내용은 150자 이내로 요약하고, 질문의 핵심을 유지하세요.
        
        코드 컨텍스트:
        {context_for_rag}
        
        질문: {search_query}
        
        답변:
        """


def create_document_rag_prompt(context_for_rag, search_query):
    """일반 문서용 RAG 프롬프트 생성"""
    return f"""
    주어진 컨텍스트 정보를 사용하여 다음 질문에 대해 한국어로 답변해 주세요.
    만약 컨텍스트에 질문과 관련된 정보가 없다면, "컨텍스트에 관련 정보가 없습니다."라고 답변해 주세요.
    
    컨텍스트:
    {context_for_rag}
    
    질문: {search_query}
    
    답변:
    """


def create_code_query_translation_prompt(korean_text):
    """코드 관련 한국어 질의 영어 번역 프롬프트 생성"""
    return f"""
        다음 한국어 코드 관련 질문을 영어로 번역해주세요. 
        프로그래밍 용어, 함수명, 클래스명, 변수명은 정확히 유지하세요.
        코드의 의미와 맥락을 살려서 번역하세요.
        번역된 영어 텍스트만 출력하세요.
        
        한국어 질문: {korean_text}
        
        English question:
        """


def create_general_text_translation_prompt(korean_text):
    """일반 한국어 텍스트 영어 번역 프롬프트 생성"""
    return f"""
        다음 한국어 텍스트를 자연스러운 영어로 번역해주세요. 기술적 용어는 정확히 번역하세요.
        번역된 영어 텍스트만 출력하고 다른 설명은 하지 마세요.
        
        한국어: {korean_text}
        
        영어:
        """
