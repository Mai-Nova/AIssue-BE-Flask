"""
Chatbot Service - Handles all chatbot-related business logic
"""
import logging
from typing import Dict, Any, List, Optional

from ..core.exceptions import ServiceError, ValidationError
from .repository_context_service import repository_context_service

logger = logging.getLogger(__name__)


class ChatbotService:
    """
    Chatbot Service - Orchestrates chatbot operations
    Handles all business logic for repository context-based question answering
    """

    def __init__(self):
        """Initialize the chatbot service"""
        self.repository_context_service = repository_context_service
        logger.info("ChatbotService가 초기화되었습니다.")

    def ask_repository_question(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answer questions based on repository context
        
        Args:
            request_data: Request data containing repo_name, messages, and optional file names
            
        Returns:
            Dictionary containing answer and context information
        """
        try:
            # Validate request data
            repo_name = request_data.get("repo_name")
            messages = request_data.get("messages")
            readme_filename = request_data.get("readme_filename")
            license_filename = request_data.get("license_filename")
            contributing_filename = request_data.get("contributing_filename")

            # Validate required fields
            if not repo_name or not messages or not isinstance(messages, list) or len(messages) == 0:
                raise ValidationError("repo_name과 messages(배열)는 필수 항목이며, 비어있을 수 없습니다.")

            # Extract user question from messages
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if not user_messages:
                raise ValidationError("사용자 메시지가 필요합니다.")

            # Use the most recent user message as the question
            question = user_messages[-1].get("content", "")
            if not question.strip():
                raise ValidationError("질문 내용이 비어있습니다.")

            logger.info(f"Repository context question received: repo '{repo_name}', question length: {len(question)}")

            # Get answer using repository context service
            result = self.repository_context_service.answer_question_with_context(
                repo_name=repo_name,
                question=question,
                readme_filename=readme_filename,
                license_filename=license_filename,
                contributing_filename=contributing_filename,
                messages=messages,
            )

            logger.info(f"Repository context question answered successfully: repo '{repo_name}'")
            
            # Add status code for successful response
            return {
                **result,
                "status_code": 200
            }

        except ValidationError as e:
            logger.warning(f"Chatbot request validation failed: {e}")
            raise
        except FileNotFoundError as e:
            logger.warning(f"File not found in chatbot service: {e}")
            raise ServiceError(str(e), error_code="FILE_NOT_FOUND")
        except ServiceError as e:
            logger.error(f"Chatbot service error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in chatbot service: {e}", exc_info=True)
            raise ServiceError("저장소 컨텍스트 질문 답변 중 예상치 못한 오류가 발생했습니다.")

    def validate_messages(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Validate message format
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            True if valid, raises ValidationError if invalid
        """
        if not isinstance(messages, list):
            raise ValidationError("messages는 배열이어야 합니다.")
        
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValidationError(f"Message {i}는 객체여야 합니다.")
            
            if "role" not in message:
                raise ValidationError(f"Message {i}에 'role' 필드가 필요합니다.")
            
            if "content" not in message:
                raise ValidationError(f"Message {i}에 'content' 필드가 필요합니다.")
            
            if message.get("role") not in ["user", "assistant"]:
                raise ValidationError(f"Message {i}의 role은 'user' 또는 'assistant'여야 합니다.")
        
        return True