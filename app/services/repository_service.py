"""
Repository Service - Orchestrates all repository-related business logic
"""
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from ..core.exceptions import ValidationError, ServiceError
from ..core.validators import validate_repo_url, validate_search_request
from .status_service import StatusService
from .indexing_service import IndexingService
from .search_service import SearchService
from .readme_summarizer import ReadmeSummarizer
from .translator import Translator

logger = logging.getLogger(__name__)


class RepositoryService:
    """
    Repository Service - Orchestrates repository operations
    Handles all business logic for repository indexing, searching, status, and related operations
    """

    def __init__(self):
        """Initialize the repository service with required dependencies"""
        self.status_service = StatusService()
        self.indexing_service = IndexingService(self.status_service)
        self.search_service = SearchService(self.status_service)
        self.readme_summarizer = ReadmeSummarizer()
        logger.info("RepositoryService가 초기화되었습니다.")

    def process_index_request(
        self,
        request_data: Dict[str, Any],
        callback_url: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process repository indexing request
        
        Args:
            request_data: Request data containing repo_url and optional repository_info
            callback_url: Optional callback URL for completion notification
            user_id: Optional user ID for tracking
            
        Returns:
            Dictionary containing status, response data, and appropriate HTTP status code
        """
        try:
            # Validate and extract request data
            repo_url = validate_repo_url(request_data)
            repository_info = request_data.get("repository_info", {})

            # Log request information
            log_payload = {
                "repo_url": repo_url,
                "repo_full_name": repository_info.get("fullName"),
                "callback_url": callback_url,
                "user_id": user_id,
            }
            logger.info(f"Repository indexing request processed: {log_payload}")

            if repository_info:
                logger.debug(f"Repository details received ({repo_url}): {repository_info}")

            # Start indexing process
            initial_status_result = self.indexing_service.prepare_and_start_indexing(
                repo_url, callback_url, user_id
            )
            repo_name = self.status_service._get_repo_name_from_url(repo_url)

            current_status = initial_status_result.get("status")
            is_new_request = initial_status_result.get("is_new_request", False)

            # Prepare response data in format expected by Express
            response_data = {
                "analysis_id": repo_name,
                "repo_name": repo_name,
                "status": current_status,
                "progress": initial_status_result.get("progress", 0),
                "message": initial_status_result.get("progress_message", ""),
                "started_at": initial_status_result.get("start_time"),
                "estimated_completion": initial_status_result.get("end_time"),
            }

            # Determine appropriate response based on status
            if current_status == "completed":
                message = f"저장소 '{repo_name}'은(는) 이미 성공적으로 인덱싱되었습니다."
                return {
                    "status": "success",
                    "data": response_data,
                    "message": message,
                    "status_code": 200
                }

            # Handle new or in-progress requests
            if is_new_request and (current_status in ["pending", "indexing"]):
                message = f"저장소 '{repo_name}' 인덱싱 작업이 시작되었습니다."
            else:
                message = f"저장소 '{repo_name}' 인덱싱 작업이 이미 진행 중입니다."

            return {
                "status": "success",
                "data": response_data,
                "message": message,
                "status_code": 202
            }

        except ValidationError as e:
            logger.warning(f"Repository indexing validation failed: {e}")
            raise
        except ServiceError as e:
            logger.error(f"Repository indexing service error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in repository indexing: {e}", exc_info=True)
            raise ServiceError("Repository indexing failed due to unexpected error")

    def search_repository(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process repository search request
        
        Args:
            request_data: Request data containing repo_name, query, and search_type
            
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            # Validate search request
            repo_name, query, search_type = validate_search_request(request_data)

            # Ensure search_type is supported (only "code" is currently supported)
            if search_type != "code":
                logger.warning(f"Unsupported search_type: {search_type}. Forcing to 'code'.")
                search_type = "code"

            log_payload = {
                "repo_name": repo_name,
                "query": query,
                "search_type": search_type,
            }
            logger.info(f"Repository search request processed: {log_payload}")

            # Perform search
            result = self.search_service.search_repository(
                f"https://github.com/{repo_name}",
                query,
                search_type,
            )

            logger.info(f"Repository search completed successfully: {repo_name}, query: '{query}'")
            return {
                "status": "success",
                "data": result,
                "message": "검색이 완료되었습니다.",
                "status_code": 200
            }

        except ValidationError as e:
            logger.warning(f"Repository search validation failed: {e}")
            raise
        except ServiceError as e:
            logger.error(f"Repository search service error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in repository search: {e}", exc_info=True)
            raise ServiceError("Repository search failed due to unexpected error")

    def get_repository_status(self, repo_name: str) -> Dict[str, Any]:
        """
        Get repository indexing status
        
        Args:
            repo_name: Repository name in format "owner/repo"
            
        Returns:
            Dictionary containing status information and appropriate HTTP status code
        """
        try:
            logger.info(f"Repository status request processed: {repo_name}")

            # Get status from status service
            status_data_result = self.status_service.get_repository_status_data(repo_name)
            current_status = status_data_result.get("status")

            # Prepare response data in format expected by Express
            response_data = {
                "analysis_id": repo_name,
                "repo_name": repo_name,
                "status": current_status,
                "progress": status_data_result.get("progress", 0),
                "message": status_data_result.get("progress_message", ""),
                "current_step": status_data_result.get("progress_message", ""),
                "error": status_data_result.get("error"),
                "error_code": status_data_result.get("error_code"),
                "started_at": status_data_result.get("start_time"),
                "completed_at": status_data_result.get("completion_time"),
                "estimated_completion": status_data_result.get("estimated_completion"),
                "eta_text": status_data_result.get("eta_text", "계산 중..."),
            }

            # Handle different status cases
            if current_status == "not_indexed":
                error_message = f"저장소 '{repo_name}'에 대한 인덱싱 정보를 찾을 수 없습니다."
                return {
                    "status": "error",
                    "message": error_message,
                    "error_code": "NOT_FOUND",
                    "status_code": 404
                }
            elif current_status == "failed":
                error_message = f"저장소 '{repo_name}' 인덱싱에 실패했습니다: {status_data_result.get('error', '알 수 없는 오류')}"
                return {
                    "status": "error",
                    "message": error_message,
                    "error_code": status_data_result.get("error_code", "INDEXING_FAILED"),
                    "status_code": 409
                }
            elif current_status in ["pending", "indexing"]:
                message = f"저장소 '{repo_name}' 인덱싱 진행 중입니다."
                return {
                    "status": "success",
                    "data": response_data,
                    "message": message,
                    "status_code": 202
                }
            elif current_status == "completed":
                message = f"저장소 '{repo_name}' 인덱싱이 완료되었습니다."
                return {
                    "status": "success",
                    "data": response_data,
                    "message": message,
                    "status_code": 200
                }
            else:
                # Unknown status
                error_message = f"알 수 없는 상태값({current_status})입니다."
                logger.error(f"Unknown status value for repository '{repo_name}': '{current_status}'")
                return {
                    "status": "error",
                    "message": error_message,
                    "error_code": "UNKNOWN_STATUS",
                    "status_code": 500
                }

        except ServiceError as e:
            logger.error(f"Repository status service error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in repository status: {e}", exc_info=True)
            raise ServiceError("Repository status check failed due to unexpected error")

    def summarize_readme(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize README content using AI
        
        Args:
            request_data: Request data containing repo_name and readme_content
            
        Returns:
            Dictionary containing summary results
        """
        try:
            # Validate request data
            if not request_data:
                raise ValidationError("요청 데이터가 필요합니다.")

            repo_name = request_data.get("repo_name")
            readme_content = request_data.get("readme_content")

            if not repo_name:
                raise ValidationError("저장소 이름이 필요합니다.")
            if not readme_content:
                raise ValidationError("README 내용이 필요합니다.")

            logger.info(f"README summary request processed: {repo_name}, content length: {len(readme_content)}")

            # Perform README summarization (async function run synchronously)
            summary = asyncio.run(
                self.readme_summarizer.summarize_readme(repo_name, readme_content)
            )

            if summary:
                response_data = {
                    "repo_name": repo_name,
                    "summary": summary,
                    "original_length": len(readme_content),
                    "summary_length": len(summary),
                }
                message = f"README 요약이 완료되었습니다: {repo_name}"
                return {
                    "status": "success",
                    "data": response_data,
                    "message": message,
                    "status_code": 200
                }
            else:
                # Fallback when summarization fails
                fallback_description = self.readme_summarizer.create_fallback_description(repo_name)
                response_data = {
                    "repo_name": repo_name,
                    "summary": fallback_description,
                    "original_length": len(readme_content),
                    "summary_length": len(fallback_description),
                    "is_fallback": True,
                }
                message = f"README 요약에 실패하여 기본 설명을 생성했습니다: {repo_name}"
                logger.warning(f"README summarization failed, using fallback: {repo_name}")
                return {
                    "status": "success",
                    "data": response_data,
                    "message": message,
                    "status_code": 200
                }

        except ValidationError as e:
            logger.warning(f"README summarization validation failed: {e}")
            raise
        except ServiceError as e:
            logger.error(f"README summarization service error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in README summarization: {e}", exc_info=True)
            raise ServiceError("README summarization failed due to unexpected error")

    def translate_text(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate text using AI
        
        Args:
            request_data: Request data containing text and language options
            
        Returns:
            Dictionary containing translation results
        """
        try:
            # Extract and validate request data
            text = request_data.get("text", "").strip()
            source_language = request_data.get("source_language", "auto")
            target_language = request_data.get("target_language", "ko")

            if not text:
                raise ValidationError("번역할 텍스트가 필요합니다.")

            logger.info(f"Translation request processed: text length {len(text)}, {source_language} -> {target_language}")

            # Initialize translator and perform translation
            translator = Translator()
            translated_text = translator.translate_text(
                text=text,
                source_language=source_language,
                target_language=target_language,
            )

            if translated_text:
                response_data = {
                    "original_text": text,
                    "translated_text": translated_text,
                    "source_language": source_language,
                    "target_language": target_language,
                }
                message = "번역이 완료되었습니다."
                return {
                    "status": "success",
                    "data": response_data,
                    "message": message,
                    "status_code": 200
                }
            else:
                error_message = "번역에 실패했습니다. 원본 텍스트를 사용해주세요."
                logger.error(f"Translation failed: Text length {len(text)}, {source_language} -> {target_language}")
                raise ServiceError(error_message, error_code="TRANSLATION_FAILED")

        except ValidationError as e:
            logger.warning(f"Translation validation failed: {e}")
            raise
        except ServiceError as e:
            logger.error(f"Translation service error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in translation: {e}", exc_info=True)
            raise ServiceError("Translation failed due to unexpected error")