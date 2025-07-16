from flask import request, current_app
from flask_restx import Namespace, Resource, fields

from app.services.chatbot_service import ChatbotService
from app.core.exceptions import ServiceError, ValidationError

# 네임스페이스 생성
chatbot_ns = Namespace("chatbot", description="챗봇 관련 API")

# 서비스 인스턴스 생성
chatbot_service = ChatbotService()

# 저장소 컨텍스트 기반 질문 응답 모델
repository_context_request_model = chatbot_ns.model(
    "RepositoryContextRequest",
    {
        "repo_name": fields.String(
            required=True, description="저장소 이름 (예: owner/repo_name)"
        ),
        "messages": fields.List(
            fields.Nested(
                chatbot_ns.model(
                    "Message",
                    {
                        "role": fields.String(
                            required=True, description="메시지 역할 (user/assistant)"
                        ),
                        "content": fields.String(
                            required=True, description="메시지 내용"
                        ),
                    },
                )
            ),
            required=True,
            description="대화 메시지 목록",
        ),
        "readme_filename": fields.String(description="README 파일명"),
        "license_filename": fields.String(description="LICENSE 파일명"),
        "contributing_filename": fields.String(description="CONTRIBUTING 파일명"),
    },
)

repository_context_response_model = chatbot_ns.model(
    "RepositoryContextResponse",
    {
        "answer": fields.String(description="AI 답변"),
        "context_files": fields.List(
            fields.String, description="컨텍스트로 사용된 파일 목록"
        ),
        "repo_info": fields.Raw(description="저장소 정보"),
    },
)


@chatbot_ns.route("/ask-repository")
class AskRepositoryQuestion(Resource):
    @chatbot_ns.doc("ask_repository_question")
    @chatbot_ns.expect(repository_context_request_model, validate=True)
    @chatbot_ns.marshal_with(repository_context_response_model)
    @chatbot_ns.response(400, "잘못된 요청")
    @chatbot_ns.response(404, "저장소 또는 파일을 찾을 수 없음")
    @chatbot_ns.response(500, "서버 내부 오류")
    def post(self):
        """저장소 컨텍스트를 기반으로 질문에 답변"""
        try:
            data = request.get_json()
            
            current_app.logger.info(f"Repository context question received for repo: {data.get('repo_name')}")

            # 비즈니스 로직을 서비스 레이어로 위임
            result = chatbot_service.ask_repository_question(data)
            
            # 응답 상태 코드에 따라 HTTP 응답 처리
            status_code = result.pop("status_code", 200)
            
            current_app.logger.info(f"Repository context question answered successfully")
            
            return result

        except ValidationError as e:
            current_app.logger.warning(f"Chatbot request validation failed: {e}")
            chatbot_ns.abort(400, str(e))
        except FileNotFoundError as e:
            current_app.logger.warning(f"File not found in chatbot API: {e}")
            chatbot_ns.abort(404, str(e))
        except ServiceError as e:
            current_app.logger.error(f"Chatbot service error: {e}", exc_info=True)
            chatbot_ns.abort(500, str(e))
        except Exception as e:
            current_app.logger.error(f"Unexpected error in chatbot API: {e}", exc_info=True)
            chatbot_ns.abort(500, "서버 내부 오류가 발생했습니다.")
