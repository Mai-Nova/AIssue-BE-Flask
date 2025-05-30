import os
import time
import logging

from config import Config
from .repository_manager import RepositoryManager
from .document_loader import DocumentLoader
from .faiss_manager import FAISSManager
from .embeddings import GeminiAPIEmbeddings
from common.exceptions import (
    RepositoryError,
    IndexingError,
    RepositorySizeError,
    EmbeddingError,
)

logger = logging.getLogger(__name__)


def _initialize_managers_and_embeddings(embedding_model_name):
    """매니저 인스턴스 및 임베딩 모델 초기화"""
    repo_manager = RepositoryManager()
    doc_loader = DocumentLoader()
    faiss_manager = FAISSManager()

    embeddings = GeminiAPIEmbeddings(
        model_name=embedding_model_name,
        document_task_type="RETRIEVAL_DOCUMENT",
        query_task_type="RETRIEVAL_QUERY",
    )

    return repo_manager, doc_loader, faiss_manager, embeddings


def _setup_repository(repo_manager, repo_url, local_repo_path):
    """저장소 정보 확인 및 복제"""
    logger.info("--- 저장소 정보 확인 중 ---")
    primary_language_name, _ = repo_manager.get_repository_info(
        repo_url, Config.GITHUB_API_TOKEN
    )

    repo_manager.clone_or_load_repository(repo_url, local_repo_path)
    repo_name_for_path = os.path.basename(local_repo_path.rstrip("/\\"))

    return primary_language_name, repo_name_for_path


def create_index_from_repo(repo_url, local_repo_path, embedding_model_name):
    """저장소 URL로부터 코드 및 문서 FAISS 인덱스를 생성합니다."""
    overall_start_time = time.time()
    vector_stores = {"code": None, "document": None}

    try:
        # 매니저 인스턴스 및 임베딩 모델 초기화
        repo_manager, doc_loader, faiss_manager, embeddings = (
            _initialize_managers_and_embeddings(embedding_model_name)
        )

        # 저장소 정보 확인 및 복제
        primary_language_name, repo_name_for_path = _setup_repository(
            repo_manager, repo_url, local_repo_path
        )

        # 코드 인덱싱
        vector_stores["code"] = _process_code_indexing(
            local_repo_path,
            repo_name_for_path,
            primary_language_name,
            doc_loader,
            faiss_manager,
            embeddings,
        )

        # 문서 인덱싱
        vector_stores["document"] = _process_document_indexing(
            local_repo_path, repo_name_for_path, doc_loader, faiss_manager, embeddings
        )

    except RepositorySizeError:
        logger.error("저장소 크기 초과로 인덱싱 중단")
        raise
    except (RepositoryError, IndexingError, EmbeddingError) as e:
        logger.error(f"인덱싱 중 오류: {e}")
        raise
    except Exception as e:
        logger.error(f"인덱싱 파이프라인 중 예상치 못한 오류 발생: {e}", exc_info=True)
        raise
    finally:
        overall_end_time = time.time()
        logger.info(
            f"총 인덱싱 실행 시간: {_format_time(overall_end_time - overall_start_time)}."
        )

    return vector_stores


def _process_code_indexing(
    repo_path, repo_name, primary_language_name, doc_loader, faiss_manager, embeddings
):
    """코드 인덱싱 처리"""
    logger.info("--- 코드 인덱싱 시작 ---")

    faiss_code_index_path = os.path.join(
        Config.FAISS_INDEX_CODE_DIR, f"{repo_name}_code"
    )
    os.makedirs(Config.FAISS_INDEX_CODE_DIR, exist_ok=True)

    # 기존 인덱스 로드 시도
    vector_store = faiss_manager.load_index(faiss_code_index_path, embeddings, "code")

    if not vector_store:
        # 새 인덱스 생성
        logger.info(f"'{repo_path}'에서 코드 파일을 로드합니다...")
        split_code_docs = doc_loader.load_code_documents(
            repo_path, primary_language_name
        )

        if split_code_docs:
            vector_store = faiss_manager.create_index(
                split_code_docs, embeddings, faiss_code_index_path, "code"
            )

    logger.info("--- 코드 인덱싱 종료 ---")
    return vector_store


def _process_document_indexing(
    repo_path, repo_name, doc_loader, faiss_manager, embeddings
):
    """문서 인덱싱 처리"""
    logger.info("--- 문서 인덱싱 시작 ---")

    faiss_docs_index_path = os.path.join(
        Config.FAISS_INDEX_DOCS_DIR, f"{repo_name}_docs"
    )
    os.makedirs(Config.FAISS_INDEX_DOCS_DIR, exist_ok=True)

    # 기존 인덱스 로드 시도
    vector_store = faiss_manager.load_index(
        faiss_docs_index_path, embeddings, "document"
    )

    if not vector_store:
        # 새 인덱스 생성
        split_doc_files = doc_loader.load_documentation_documents(repo_path)

        if split_doc_files:
            vector_store = faiss_manager.create_index(
                split_doc_files, embeddings, faiss_docs_index_path, "document"
            )

    logger.info("--- 문서 인덱싱 종료 ---")
    return vector_store


def load_faiss_index(index_path, embeddings, index_type):
    """저장된 FAISS 인덱스 로드 (하위 호환성을 위한 래퍼 함수)"""
    faiss_manager = FAISSManager()
    return faiss_manager.load_index(index_path, embeddings, index_type)


def _format_time(seconds):
    """초를 'X분 Y초' 또는 'Y초' 형식으로 변환"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}분 {remaining_seconds}초"
    else:
        return f"{remaining_seconds}초"
