import os
import logging
from datetime import datetime, timezone
import threading

from config import Config
from .indexer import (
    create_index_from_repo,
    load_faiss_index,
)
from .searcher import search_and_rag
from .embeddings import GeminiAPIEmbeddings
from common.exceptions import (
    RepositoryError,
    IndexingError,
    RepositorySizeError,
    EmbeddingError,
    RAGError,
    ServiceError,
)

logger = logging.getLogger(__name__)


class RepositoryService:
    """저장소 인덱싱/검색 서비스 클래스"""

    def __init__(self):
        # 저장소 상태 (메모리, 운영 시 DB 권장)
        self.repository_status = {}
        self._status_lock = threading.Lock()  # 상태 접근 동기화를 위한 Lock

    def _get_repo_name_from_url(self, repo_url):
        """URL에서 저장소 이름 추출"""
        return repo_url.split("/")[-1].removesuffix(".git")

    def _get_local_repo_path(self, repo_name):
        """저장소 로컬 경로 반환"""
        return os.path.join(Config.BASE_CLONED_DIR, repo_name)

    def _get_index_path(self, repo_name: str, index_type: str) -> str:
        """주어진 저장소 이름과 인덱스 타입에 대한 FAISS 인덱스 경로를 반환합니다."""
        base_dir = (
            Config.FAISS_INDEX_CODE_DIR
            if index_type == "code"
            else Config.FAISS_INDEX_DOCS_DIR
        )
        if index_type not in ["code", "document"]:
            raise ValueError(f"알 수 없는 인덱스 타입입니다: {index_type}")
        return os.path.join(base_dir, f"{repo_name}_{index_type}")

    def prepare_indexing(self, repo_url):
        """인덱싱 준비: 상태 확인, 초기 상태 설정 및 반환."""
        repo_name = self._get_repo_name_from_url(repo_url)

        with self._status_lock:
            if repo_name in self.repository_status:
                current_status_info = self.repository_status[repo_name]
                # 이미 진행 중이거나 완료/실패된 경우 해당 상태 반환
                return {**current_status_info, "is_new_request": False}

            # 신규 인덱싱 요청
            current_time_iso = datetime.now(timezone.utc).isoformat()
            initial_status_info = {
                "status": "pending",  # 인덱싱 시작 대기
                "repo_url": repo_url,
                "repo_name": repo_name,
                "start_time": current_time_iso,
                "last_updated_time": current_time_iso,
                "end_time": None,
                "error": None,
                "error_code": None,
                "code_index_status": "pending",
                "document_index_status": "pending",
                "progress_message": "인덱싱 작업 시작 대기 중...",
                "is_new_request": True,
            }
            self.repository_status[repo_name] = initial_status_info
            return initial_status_info

    def _update_indexing_progress(
        self,
        repo_name,
        status=None,
        message=None,
        code_status=None,
        doc_status=None,
    ):
        """인덱싱 진행 상태 업데이트 (Lock 내부에서 호출되어야 함)"""
        if repo_name not in self.repository_status:
            return

        update_data = {"last_updated_time": datetime.now(timezone.utc).isoformat()}
        if status:
            update_data["status"] = status
        if message:
            update_data["progress_message"] = message
        if code_status:
            update_data["code_index_status"] = code_status
        if doc_status:
            update_data["document_index_status"] = doc_status

        self.repository_status[repo_name].update(update_data)

    def _finalize_indexing_status(
        self,
        repo_name,
        final_status,
        message,
        error_message=None,
        error_code=None,
        code_status_override=None,
        doc_status_override=None,
    ):
        """인덱싱 완료 또는 실패 시 최종 상태 업데이트 (Lock 내부에서 호출되어야 함)"""
        if repo_name not in self.repository_status:
            return

        completed_time_iso = datetime.now(timezone.utc).isoformat()
        status_entry = self.repository_status[repo_name]

        status_entry.update(
            {
                "status": final_status,
                "end_time": completed_time_iso,
                "last_updated_time": completed_time_iso,
                "progress_message": message,
                "error": error_message,
                "error_code": error_code,
            }
        )

        if final_status == "completed":
            status_entry["code_index_status"] = code_status_override or (
                "completed"
                if self._check_index_exists(repo_name, "code")
                else "not_applicable_or_failed"
            )
            status_entry["document_index_status"] = doc_status_override or (
                "completed"
                if self._check_index_exists(repo_name, "document")
                else "not_applicable_or_failed"
            )
        elif final_status == "failed":
            # 실패 시, 이미 완료된 인덱스는 유지하고 나머지는 failed로 설정
            status_entry["code_index_status"] = code_status_override or (
                "completed"
                if status_entry.get("code_index_status") == "completed"
                else "failed"
            )
            status_entry["document_index_status"] = doc_status_override or (
                "completed"
                if status_entry.get("document_index_status") == "completed"
                else "failed"
            )

    def _handle_indexing_success(self, repo_name, vector_stores):
        """인덱싱 성공 처리 (Lock 내부에서 호출되어야 함)"""
        code_status = (
            "completed" if vector_stores.get("code") else "not_applicable_or_failed"
        )
        doc_status = (
            "completed" if vector_stores.get("document") else "not_applicable_or_failed"
        )
        self._finalize_indexing_status(
            repo_name,
            "completed",
            "인덱싱 완료되었습니다.",
            code_status_override=code_status,
            doc_status_override=doc_status,
        )

    def _handle_indexing_error(self, repo_name, e):
        """인덱싱 오류 처리 (Lock은 호출부에서 관리 또는 내부에서 acquire)"""
        error_map = {
            RepositorySizeError: ("저장소 크기 초과", "REPO_SIZE_EXCEEDED"),
            RepositoryError: ("저장소 관련 오류", "REPOSITORY_ERROR"),
            EmbeddingError: ("임베딩 생성 오류", "EMBEDDING_ERROR"),
            IndexingError: ("인덱싱 처리 오류", "INDEXING_FAILED"),
        }
        default_error = ("예상치 못한 오류 발생", "UNEXPECTED_INDEXING_ERROR")

        error_msg_prefix, error_code = default_error
        for error_type, (msg_prefix, code) in error_map.items():
            if isinstance(e, error_type):
                error_msg_prefix = msg_prefix
                error_code = code
                break

        full_error_msg = f"{error_msg_prefix}: {str(e)}"
        logger.error(
            f"인덱싱 오류 ({repo_name}): {full_error_msg}",
            exc_info=(error_code == "UNEXPECTED_INDEXING_ERROR"),
        )

        with self._status_lock:
            self._finalize_indexing_status(
                repo_name,
                "failed",
                f"인덱싱 실패: {full_error_msg}",
                error_message=full_error_msg,
                error_code=error_code,
            )
        # ServiceError를 발생시켜 API 계층에서 처리하도록 함
        raise ServiceError(full_error_msg, error_code=error_code) from e

    def perform_indexing(self, repo_url):
        """실제 인덱싱 작업 수행 (백그라운드 실행용)."""
        repo_name = self._get_repo_name_from_url(repo_url)
        local_repo_path = self._get_local_repo_path(repo_name)

        with self._status_lock:
            # prepare_indexing에서 이미 pending 상태로 설정됨
            # 여기서는 indexing 상태로 변경
            self._update_indexing_progress(
                repo_name, status="indexing", message="저장소 정보 확인 및 복제 중..."
            )

        try:
            logger.info(f"저장소 인덱싱 실제 작업 시작: {repo_url}")
            # 여기서 추가적인 진행 메시지 업데이트 가능
            # 예: self._update_indexing_progress(repo_name, message="코드 파일 분석 중...")

            vector_stores = create_index_from_repo(
                repo_url=repo_url,
                local_repo_path=local_repo_path,
                embedding_model_name=Config.DEFAULT_EMBEDDING_MODEL,
                # 콜백 함수를 통해 세부 진행 상태 업데이트 가능
                # progress_callback=lambda msg, cs=None, ds=None: \
                # self._update_indexing_progress(repo_name, message=msg, code_status=cs, doc_status=ds)
            )

            with self._status_lock:
                self._handle_indexing_success(repo_name, vector_stores)
            logger.info(f"저장소 인덱싱 완료: {repo_url}")

        except Exception as e:
            # _handle_indexing_error 내부에서 이미 로그 기록 및 상태 업데이트 수행
            # ServiceError를 다시 발생시키므로, 호출 스레드에서 처리 가능
            self._handle_indexing_error(repo_name, e)

    def _load_vector_store(self, repo_name, search_type):
        """벡터 저장소 로드"""
        embeddings = GeminiAPIEmbeddings(
            model_name=Config.DEFAULT_EMBEDDING_MODEL,
            document_task_type="RETRIEVAL_DOCUMENT",
            query_task_type="RETRIEVAL_QUERY",
        )

        index_path = self._get_index_path(repo_name, search_type)
        vector_store = load_faiss_index(index_path, embeddings, search_type)

        if not vector_store:
            raise ServiceError(
                f"'{repo_name}' 저장소의 {search_type} 인덱스 로드에 실패했습니다.",
                error_code="INDEX_LOAD_FAILED",
            )

        return vector_store

    def search_repository(self, repo_url, query, search_type="code"):
        """저장소 검색 서비스 로직"""
        repo_name = self._get_repo_name_from_url(repo_url)

        # 인덱스 존재 여부 및 상태 확인
        status_info = self.get_repository_status(
            repo_name
        )  # 상태 확인을 통해 인덱싱 완료 여부 판단
        if status_info.get("status") != "completed":
            # 세부 인덱스 상태 확인
            if (
                search_type == "code"
                and status_info.get("code_index_status") != "completed"
            ):
                raise ServiceError(
                    f"'{repo_name}' 저장소의 코드 인덱스가 준비되지 않았습니다 (상태: {status_info.get('code_index_status', '알 수 없음')}).",
                    error_code="INDEX_NOT_READY",
                )
            if (
                search_type == "document"
                and status_info.get("document_index_status") != "completed"
            ):
                raise ServiceError(
                    f"'{repo_name}' 저장소의 문서 인덱스가 준비되지 않았습니다 (상태: {status_info.get('document_index_status', '알 수 없음')}).",
                    error_code="INDEX_NOT_READY",
                )
            # 위 조건에 걸리지 않았지만 전체 상태가 completed가 아닐 경우 (예: 부분 실패 후 다른 부분 검색 시도)
            if not self._check_index_exists(repo_name, search_type):
                raise ServiceError(
                    f"'{repo_name}' 저장소의 {search_type} 인덱스가 존재하지 않습니다. 먼저 저장소를 인덱싱해주세요.",
                    error_code="INDEX_NOT_FOUND",
                )

        try:
            logger.info(
                f"검색 시작: 저장소 '{repo_name}', 질의: '{query}', 타입: {search_type}"
            )

            # 벡터 저장소 로드
            vector_store = self._load_vector_store(repo_name, search_type)
            vector_stores = {search_type: vector_store}

            # 검색 및 RAG 수행
            rag_response = search_and_rag(
                vector_stores=vector_stores,
                target_index=search_type,
                search_query=query,
                llm_model_name=Config.DEFAULT_LLM_MODEL,
                top_k=Config.DEFAULT_TOP_K,
                similarity_threshold=Config.DEFAULT_SIMILARITY_THRESHOLD,
            )

            logger.info(f"검색 완료: 저장소 '{repo_name}'")

            return {
                "repo_name": repo_name,
                "query": query,
                "search_type": search_type,
                "answer": rag_response,
                "result_generated_at": datetime.now(  # 'timestamp'에서 이름 변경
                    timezone.utc
                ).isoformat(),
            }

        except RAGError as e:
            error_msg = f"검색 실패: {str(e)}"
            logger.error(error_msg)
            raise ServiceError(error_msg) from e
        except Exception as e:
            error_msg = f"검색 중 예상치 못한 오류: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg, error_code="UNEXPECTED_SEARCH_ERROR") from e

    def _infer_status_from_disk(self, repo_name):
        """디스크의 인덱스 파일 존재 여부로 간이 상태 추론 (메모리 상태 없을 시 사용)"""
        code_index_path = self._get_index_path(repo_name, "code")
        doc_index_path = self._get_index_path(repo_name, "document")

        code_exists = os.path.exists(code_index_path)
        doc_exists = os.path.exists(doc_index_path)

        if code_exists or doc_exists:
            return {
                "status": "completed",  # 디스크에 파일이 있다면 일단 완료로 간주
                "repo_name": repo_name,
                "code_index_status": (
                    "completed" if code_exists else "not_found_on_disk"
                ),
                "document_index_status": (
                    "completed" if doc_exists else "not_found_on_disk"
                ),
                "progress_message": "인덱스 파일은 존재하나, 상세 진행 기록은 없습니다 (서버 재시작 가능성).",
                "last_updated_time": datetime.now(timezone.utc).isoformat(),
                # start_time, end_time 등은 알 수 없음
            }
        return {
            "status": "not_indexed",
            "repo_name": repo_name,
            "progress_message": "인덱싱된 정보가 없습니다 (메모리 및 디스크 확인).",
        }

    def get_repository_status(self, repo_name):
        """저장소 상태 조회. repo_name은 URL에서 추출된 순수 이름이어야 함."""
        with self._status_lock:
            if repo_name not in self.repository_status:
                # 메모리에 상태 정보가 없는 경우, 디스크 상태로 추론
                disk_status = self._infer_status_from_disk(repo_name)
                # 추론된 상태를 메모리에 기록할 수도 있으나, 여기서는 단순 반환
                return disk_status

            # 메모리에 상태가 있는 경우, 최신 시간으로 업데이트 후 반환
            # 단, 최종 상태(completed, failed)가 아닐 경우에만 last_updated_time 갱신
            current_repo_status = self.repository_status[repo_name]
            if current_repo_status.get("status") not in ["completed", "failed"]:
                current_repo_status["last_updated_time"] = datetime.now(
                    timezone.utc
                ).isoformat()
            return current_repo_status  # 복사본 반환 고려 (dict(current_repo_status))

    # _update_error_status 메서드는 _finalize_indexing_status 로 통합됨

    def _check_index_exists(self, repo_name, index_type):
        """주어진 저장소 이름과 인덱스 타입에 대한 인덱스 존재 여부를 확인합니다."""
        index_path = self._get_index_path(repo_name, index_type)
        return os.path.exists(index_path)
