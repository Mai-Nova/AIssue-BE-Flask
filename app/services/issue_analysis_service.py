"""
Issue Analysis Service - Handles all issue analysis business logic
"""
import logging
from typing import Dict, Any, Optional

from ..core.config import Config
from ..core.utils import extract_repo_name_from_url, get_faiss_index_path
from ..core.exceptions import ServiceError, RAGError, ValidationError
from .faiss_service import FAISSService
from .embeddings import GeminiAPIEmbeddings
from .issue_analyzer import issue_analyzer

logger = logging.getLogger(__name__)


class IssueAnalysisService:
    """
    Issue Analysis Service - Orchestrates issue analysis operations
    Handles all business logic for analyzing GitHub issues using RAG
    """

    def __init__(self):
        """Initialize the issue analysis service"""
        self.embeddings = None
        self.faiss_service = None
        logger.info("IssueAnalysisService가 초기화되었습니다.")

    def _initialize_services(self):
        """Initialize FAISS service and embeddings lazily"""
        if self.embeddings is None or self.faiss_service is None:
            try:
                self.embeddings = GeminiAPIEmbeddings(
                    model_name=Config.DEFAULT_EMBEDDING_MODEL,
                    document_task_type="RETRIEVAL_DOCUMENT",
                    query_task_type="RETRIEVAL_QUERY",
                )
                self.faiss_service = FAISSService(embeddings=self.embeddings)
                logger.info("FAISSService와 임베딩 서비스가 초기화되었습니다.")
            except Exception as e:
                logger.error(f"FAISSService 초기화 실패: {e}", exc_info=True)
                raise ServiceError("AI 분석 서비스 초기화 중 오류가 발생했습니다.")

    def analyze_issue(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a GitHub issue using RAG
        
        Args:
            request_data: Request data containing issue information
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Validate required fields
            issue_id = request_data.get("issueId")
            repo_url = request_data.get("repoUrl")
            default_branch = request_data.get("defaultBranch") or "main"

            if not repo_url:
                error_message = "저장소 URL이 필요합니다."
                logger.error(f"Repository URL missing for issue ID {issue_id}")
                return {
                    "summary": error_message,
                    "relatedFiles": [],
                    "codeSnippets": [],
                    "solutionSuggestion": "분석을 위해 저장소 URL을 제공해주세요.",
                    "status_code": 400
                }

            logger.info(f"Issue analysis request received: issue ID {issue_id}, repository: {repo_url}")

            # Extract repository name from URL
            try:
                repo_name = extract_repo_name_from_url(repo_url)
                logger.info(f"Repository name extracted: {repo_name}")
            except ValueError as e:
                error_message = "유효하지 않은 저장소 URL입니다."
                logger.error(f"Invalid repository URL: {repo_url}, error: {e}")
                return {
                    "summary": error_message,
                    "relatedFiles": [],
                    "codeSnippets": [],
                    "solutionSuggestion": "올바른 저장소 URL을 제공해주세요.",
                    "status_code": 400
                }

            # Initialize services
            self._initialize_services()

            # Load vector stores for the repository
            vector_stores = self._load_vector_stores(repo_name)
            if not vector_stores:
                return {
                    "summary": f"'{repo_name}' 저장소의 분석 데이터를 찾을 수 없습니다.",
                    "relatedFiles": [],
                    "codeSnippets": [],
                    "solutionSuggestion": "저장소가 아직 인덱싱되지 않았습니다. 먼저 저장소를 분석해주세요.",
                    "status_code": 404
                }

            # Perform issue analysis
            logger.info(f"Starting issue analysis: issue ID {issue_id}, repository: {repo_name}")
            
            analysis_result = issue_analyzer.analyze_issue(
                vector_stores=vector_stores,
                issue_data=request_data,
                faiss_service=self.faiss_service,
                default_branch=default_branch,
            )

            logger.info(f"Issue analysis completed: issue ID {issue_id}")
            
            # Add status code for successful analysis
            analysis_result["status_code"] = 200
            return analysis_result

        except RAGError as e:
            error_message = "AI 분석 중 검색 오류가 발생했습니다."
            logger.error(f"RAG analysis error (issue ID: {issue_id}): {e}", exc_info=True)
            return {
                "summary": error_message,
                "relatedFiles": [],
                "codeSnippets": [],
                "solutionSuggestion": "검색 엔진 오류로 인해 분석을 완료할 수 없습니다.",
                "status_code": 500
            }
        except ServiceError as e:
            error_message = "분석 서비스 오류가 발생했습니다."
            logger.error(f"Service error (issue ID: {issue_id}): {e}", exc_info=True)
            return {
                "summary": error_message,
                "relatedFiles": [],
                "codeSnippets": [],
                "solutionSuggestion": "서비스 오류로 인해 분석을 완료할 수 없습니다.",
                "status_code": 500
            }
        except Exception as e:
            error_message = "이슈 분석 중 예상치 못한 오류가 발생했습니다."
            logger.error(f"Unexpected error during issue analysis (issue ID: {issue_id}): {e}", exc_info=True)
            return {
                "summary": error_message,
                "relatedFiles": [],
                "codeSnippets": [],
                "solutionSuggestion": "예상치 못한 오류로 인해 분석을 완료할 수 없습니다.",
                "status_code": 500
            }

    def _load_vector_stores(self, repo_name: str) -> Optional[Dict[str, Any]]:
        """
        Load vector stores for the given repository
        
        Args:
            repo_name: Repository name
            
        Returns:
            Dictionary of vector stores or None if loading fails
        """
        try:
            # Load code index
            code_index_path = get_faiss_index_path(repo_name, "code")
            logger.info(f"Attempting to load code index: {code_index_path}")

            code_vector_store = self.faiss_service.load_index(code_index_path, "code")

            if not code_vector_store:
                logger.warning(f"Code index not found for repository '{repo_name}'")
                return None

            # Return vector stores dictionary
            vector_stores = {"code": code_vector_store}
            logger.info(f"Vector stores loaded successfully for repository '{repo_name}'")
            return vector_stores

        except Exception as e:
            logger.error(f"Failed to load vector stores for repository {repo_name}: {e}", exc_info=True)
            return None