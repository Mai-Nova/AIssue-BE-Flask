from flask import request
from flask_restx import Resource, fields
import logging

from . import api

from ..services.repository_service import RepositoryService
from ..core.exceptions import ServiceError, ValidationError
from ..core.response_utils import (
    success_response,
    error_response,
)

# --- swagger_config.py 에서 가져온 모델 정의 시작 ---
# 공통 응답 모델들
base_response = api.model(
    "BaseResponse",
    {
        "status": fields.String(
            required=True, description="응답 상태", example="success"
        ),
        "message": fields.String(required=True, description="응답 메시지"),
        "timestamp": fields.String(
            required=True,
            description="응답 시간 (ISO 8601)",
            example="2024-01-01T00:00:00Z",
        ),
    },
)

error_model = api.model(
    "ErrorResponse",
    {
        "status": fields.String(
            required=True, description="응답 상태", example="error"
        ),
        "message": fields.String(required=True, description="오류 메시지"),
        "error_code": fields.String(description="오류 코드"),
        "timestamp": fields.String(required=True, description="응답 시간 (ISO 8601)"),
    },
)

# 저장소 관련 모델들
repo_index_request = api.model(
    "RepositoryIndexRequest",
    {
        "repo_url": fields.String(
            required=True,
            description="GitHub 저장소 URL",
            example="https://github.com/pallets/flask",
        ),
        "repository_info": fields.Raw(
            required=False,
            description="Express에서 전달한 저장소 정보",
            example={
                "githubRepoId": 123456,
                "fullName": "pallets/flask",
                "name": "flask",
                "description": "A lightweight WSGI web application framework",
                "programmingLanguage": "Python",
                "star": 65000,
                "fork": 16000,
            },
        ),
        "callback_url": fields.String(
            required=False,
            description="분석 완료 시 콜백할 Express API URL",
            example="http://localhost:3001/internal/analysis-complete",
        ),
        "user_id": fields.Integer(
            required=False, description="분석을 요청한 사용자 ID", example=1
        ),
    },
)

repo_search_request = api.model(
    "RepositorySearchRequest",
    {
        "repo_name": fields.String(
            required=True,
            description="GitHub 저장소 이름 (owner/repository 형식)",
            example="pallets/flask",
        ),
        "query": fields.String(
            required=True,
            description="검색 질의문",
            example="Flask에서 request 객체를 어떻게 사용하나요?",
        ),
        "search_type": fields.String(
            required=False,
            description="검색 유형 (code만 지원)",
            enum=["code"],
            default="code",
            example="code",
        ),
    },
)

status_data = api.model(
    "StatusData",
    {
        "status": fields.String(description="인덱싱 상태", example="completed"),
        "repo_name": fields.String(description="저장소 이름", example="pallets/flask"),
        "repo_url": fields.String(
            description="저장소 URL", example="https://github.com/pallets/flask"
        ),
        "progress": fields.Float(description="진행률 (0-100)", example=100.0),
        "current_step": fields.String(
            description="현재 단계", example="Flask 저장소 인덱싱 완료"
        ),
        "total_files": fields.Integer(description="전체 파일 수", example=850),
        "processed_files": fields.Integer(description="처리된 파일 수", example=850),
        "start_time": fields.String(
            description="시작 시간", example="2024-07-22T11:00:00Z"
        ),
        "completion_time": fields.String(
            description="완료 시간", example="2024-07-22T11:15:00Z"
        ),
        "error": fields.String(description="오류 메시지", example=None),
    },
)

search_result_item = api.model(
    "SearchResultItem",
    {
        "content": fields.String(
            description="검색된 내용",
            example="from flask import request\\\\n\\\\n@app.route('/login', methods=['POST'])\\\\ndef login():\\\\n    username = request.form['username']\\\\n    # ...",
        ),
        "file_path": fields.String(
            description="파일 경로", example="examples/flaskr/flaskr/auth.py"
        ),
        "score": fields.Float(description="유사도 점수", example=0.91),
        "metadata": fields.Raw(
            description="추가 메타데이터",
            example={"line_number": 42, "type": "function_definition"},
        ),
    },
)

search_results_model = api.model(
    "SearchResults",
    {
        "query": fields.String(
            description="검색 질의문", example="Flask에서 request 객체 사용법"
        ),
        "search_type": fields.String(description="검색 유형", example="code"),
        "repo_name": fields.String(description="저장소 이름", example="pallets/flask"),
        "results": fields.List(
            fields.Nested(search_result_item), description="검색 결과 목록"
        ),
        "total_results": fields.Integer(description="총 결과 수", example=2),
        "search_time": fields.Float(description="검색 소요 시간 (초)", example=0.250),
        "result_generated_at": fields.String(
            description="검색 결과 생성 시간 (ISO 8601)",
            example="2024-07-22T11:18:00Z",
        ),
    },
)

success_response_with_data = api.model(
    "SuccessResponseWithData",
    {
        "status": fields.String(
            required=True, description="응답 상태", example="success"
        ),
        "message": fields.String(
            required=True,
            description="응답 메시지",
            example="pallets/flask 저장소 검색 결과입니다.",
        ),
        "data": fields.Nested(search_results_model, description="응답 데이터"),
        "timestamp": fields.String(
            required=True,
            description="응답 시간 (ISO 8601)",
            example="2024-07-22T11:20:00Z",
        ),
    },
)

progress_response = api.model(
    "ProgressResponse",
    {
        "status": fields.String(
            required=True, description="응답 상태", example="in_progress"
        ),
        "message": fields.String(
            required=True,
            description="응답 메시지",
            example="pallets/flask 저장소 인덱싱이 진행 중입니다.",
        ),
        "progress": fields.Nested(status_data, description="진행 상황 데이터"),
        "timestamp": fields.String(
            required=True,
            description="응답 시간 (ISO 8601)",
            example="2024-07-22T11:05:00Z",
        ),
    },
)
# --- swagger_config.py 에서 가져온 모델 정의 끝 ---


# Namespace 생성 (Blueprint 아래에 위치)
# api 객체는 app.api.__init__ 에서 가져온 것을 사용
repository_ns = api.namespace("repository", description="저장소 인덱싱 및 검색 API")

# 로거 설정
logger = logging.getLogger(__name__)

# 서비스 인스턴스 생성
repository_service = RepositoryService()


@repository_ns.route("/index")
class RepositoryIndex(Resource):
    @repository_ns.doc("index_repository")
    @repository_ns.expect(repo_index_request, validate=True)
    @repository_ns.response(202, "인덱싱 작업 시작됨/진행중", progress_response)
    @repository_ns.response(200, "이미 인덱싱 완료됨", success_response_with_data)
    @repository_ns.response(400, "잘못된 요청", error_model)
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def post(self):
        """저장소 인덱싱 요청. Express에서 호출되는 API."""
        data = {}
        try:
            data = request.get_json(force=True)
            callback_url = data.get("callback_url")
            user_id = data.get("user_id")

            logger.info(
                f"API 요청 시작: [{request.method}] {request.path} (클라이언트 IP: {request.remote_addr})"
            )

            # 비즈니스 로직을 서비스 레이어로 위임
            result = repository_service.process_index_request(data, callback_url, user_id)
            
            logger.info(
                f"API 요청 성공: [{request.method}] {request.path}. 응답 코드: {result['status_code']}. 메시지: {result['message']}"
            )
            
            return success_response(
                data=result.get("data"), 
                message=result["message"], 
                status_code=result["status_code"]
            )

        except ValidationError as e:
            error_message = str(e)
            error_code = "VALIDATION_ERROR"
            status_code = 400
            logger.warning(
                f"API 요청 유효성 검사 실패 [{request.method} {request.path}]: {error_message}. 요청 데이터: {data}"
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except ServiceError as e:
            error_message = str(e)
            error_code = e.error_code or "SERVICE_ERROR"
            status_code = (
                400 if e.error_code else 500
            )
            logger.error(
                f"API 요청 처리 중 서비스 오류 발생 [{request.method} {request.path}]: {error_message}. ErrorCode: {error_code}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except Exception as e:
            error_message = "서버 내부 오류가 발생했습니다."
            error_code = "INTERNAL_SERVER_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 예상치 못한 오류 발생 [{request.method} {request.path}]: {e}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )


@repository_ns.route("/search")
class RepositorySearch(Resource):
    @repository_ns.doc("search_repository")
    @repository_ns.expect(repo_search_request, validate=True)
    @repository_ns.response(200, "검색 완료", success_response_with_data)
    @repository_ns.response(400, "잘못된 요청", error_model)
    @repository_ns.response(404, "인덱스 없음", error_model)  # INDEX_NOT_FOUND 경우
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def post(self):
        """저장소에서 코드 검색"""
        data = {}
        try:
            data = request.get_json()
            
            logger.info(
                f"API 요청 시작: [{request.method}] {request.path} (클라이언트 IP: {request.remote_addr})"
            )

            # 비즈니스 로직을 서비스 레이어로 위임
            result = repository_service.search_repository(data)
            
            logger.info(
                f"API 요청 성공: [{request.method}] {request.path}. 응답 코드: {result['status_code']}. 메시지: {result['message']}"
            )
            
            return success_response(
                data=result["data"], 
                message=result["message"]
            )

        except ValidationError as e:
            error_message = str(e)
            error_code = "VALIDATION_ERROR"
            status_code = 400
            logger.warning(
                f"API 요청 유효성 검사 실패 [{request.method} {request.path}]: {error_message}. 요청 데이터: {data}"
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except ServiceError as e:
            error_message = str(e)
            error_code = e.error_code or "SERVICE_ERROR"
            status_code = 500
            if e.error_code == "INDEX_NOT_FOUND":
                status_code = 404
            logger.error(
                f"API 요청 처리 중 서비스 오류 발생 [{request.method} {request.path}]: {error_message}. ErrorCode: {error_code}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except Exception as e:
            error_message = "서버 내부 오류가 발생했습니다."
            error_code = "INTERNAL_SERVER_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 예상치 못한 오류 발생 [{request.method} {request.path}]: {e}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )


@repository_ns.route("/status/<path:repo_name>")
class RepositoryStatus(Resource):
    @repository_ns.doc("get_repository_status")
    @repository_ns.param(
        "repo_name", "저장소 이름 (owner/repository 형식)", required=True
    )
    @repository_ns.response(200, "상태 조회 성공 (완료됨)", success_response_with_data)
    @repository_ns.response(202, "인덱싱 진행 중", progress_response)
    @repository_ns.response(404, "저장소 정보 없음", error_model)
    @repository_ns.response(409, "인덱싱 실패", error_model)
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def get(self, repo_name):
        """저장소 인덱싱 상태 확인. Express에서 호출되는 API."""
        try:
            logger.info(
                f"API 요청 시작: [{request.method}] {request.path} (클라이언트 IP: {request.remote_addr}). 저장소: {repo_name}"
            )

            # 비즈니스 로직을 서비스 레이어로 위임
            result = repository_service.get_repository_status(repo_name)
            
            if result["status"] == "error":
                logger.error(
                    f"API 요청 처리 중 오류 [{request.method} {request.path}]: {result['message']}. ErrorCode: {result.get('error_code')}"
                )
                return error_response(
                    message=result["message"],
                    error_code=result.get("error_code"),
                    status_code=result["status_code"],
                )
            
            logger.info(
                f"API 요청 성공: [{request.method}] {request.path}. 응답 코드: {result['status_code']}. 메시지: {result['message']}"
            )
            
            return success_response(
                data=result.get("data"), 
                message=result["message"], 
                status_code=result["status_code"]
            )

        except ServiceError as e:
            error_message = str(e)
            error_code = e.error_code or "SERVICE_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 서비스 오류 발생 [{request.method} {request.path}]: {error_message}. ErrorCode: {error_code}. 저장소: {repo_name}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except Exception as e:
            error_message = "상태 조회 중 서버 내부 오류가 발생했습니다."
            error_code = "INTERNAL_SERVER_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 예상치 못한 오류 발생 [{request.method} {request.path}]: {e}. 저장소: {repo_name}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )


# README 요약 관련 모델들
readme_summary_request = api.model(
    "ReadmeSummaryRequest",
    {
        "repo_name": fields.String(
            required=True,
            description="GitHub 저장소 이름 (owner/repository 형식)",
            example="pallets/flask",
        ),
        "readme_content": fields.String(
            required=True,
            description="README 파일 내용",
            example="# Flask\\n\\nFlask is a lightweight WSGI web application framework...",
        ),
    },
)

readme_summary_response = api.model(
    "ReadmeSummaryResponse",
    {
        "status": fields.String(
            required=True, description="응답 상태", example="success"
        ),
        "message": fields.String(
            required=True,
            description="응답 메시지",
            example="README 요약이 완료되었습니다.",
        ),
        "data": fields.Raw(
            description="요약 결과",
            example={
                "repo_name": "pallets/flask",
                "summary": "Flask는 Python으로 작성된 경량 웹 애플리케이션 프레임워크입니다...",
                "original_length": 1500,
                "summary_length": 120,
            },
        ),
        "timestamp": fields.String(
            required=True,
            description="응답 시간 (ISO 8601)",
            example="2024-07-22T11:20:00Z",
        ),
    },
)

# 번역 관련 모델들
translation_request = api.model(
    "TranslationRequest",
    {
        "text": fields.String(
            required=True,
            description="번역할 텍스트",
            example="A lightweight WSGI web application framework",
        ),
        "source_language": fields.String(
            required=False,
            description="원본 언어 (기본값: auto)",
            default="auto",
            example="en",
        ),
        "target_language": fields.String(
            required=False,
            description="대상 언어 (기본값: ko)",
            default="ko",
            example="ko",
        ),
    },
)

translation_response = api.model(
    "TranslationResponse",
    {
        "status": fields.String(
            required=True, description="응답 상태", example="success"
        ),
        "message": fields.String(
            required=True,
            description="응답 메시지",
            example="번역이 완료되었습니다.",
        ),
        "data": fields.Raw(
            description="번역 결과",
            example={
                "original_text": "A lightweight WSGI web application framework",
                "translated_text": "경량 WSGI 웹 애플리케이션 프레임워크",
                "source_language": "en",
                "target_language": "ko",
            },
        ),
        "timestamp": fields.String(
            required=True,
            description="응답 시간 (ISO 8601)",
            example="2024-07-22T11:20:00Z",
        ),
    },
)


@repository_ns.route("/summarize-readme")
class ReadmeSummary(Resource):
    @repository_ns.doc("summarize_readme")
    @repository_ns.expect(readme_summary_request, validate=True)
    @repository_ns.response(200, "README 요약 완료", readme_summary_response)
    @repository_ns.response(400, "잘못된 요청", error_model)
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def post(self):
        """README 내용을 AI로 요약합니다. Express에서 호출되는 API."""
        data = {}
        try:
            data = request.get_json()
            readme_content_length = len(data.get("readme_content", ""))

            logger.info(
                f"API 요청 시작: [{request.method}] {request.path} (클라이언트 IP: {request.remote_addr}). README 길이: {readme_content_length}"
            )

            # 비즈니스 로직을 서비스 레이어로 위임
            result = repository_service.summarize_readme(data)
            
            logger.info(
                f"API 요청 성공: [{request.method}] {request.path}. 응답 코드: {result['status_code']}. 메시지: {result['message']}"
            )
            
            return success_response(
                data=result["data"], 
                message=result["message"], 
                status_code=result["status_code"]
            )

        except ValidationError as e:
            error_message = str(e)
            error_code = "VALIDATION_ERROR"
            status_code = 400
            logger.warning(
                f"API 요청 유효성 검사 실패 [{request.method} {request.path}]: {error_message}. 요청 데이터: {data}"
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except ServiceError as e:
            error_message = str(e)
            error_code = e.error_code or "SERVICE_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 서비스 오류 발생 [{request.method} {request.path}]: {error_message}. ErrorCode: {error_code}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except Exception as e:
            error_message = "README 요약 중 서버 내부 오류가 발생했습니다."
            error_code = "INTERNAL_SERVER_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 예상치 못한 오류 발생 [{request.method} {request.path}]: {e}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )


@repository_ns.route("/translate")
class Translation(Resource):
    """텍스트 번역 API"""

    @repository_ns.doc("translate_text")
    @repository_ns.expect(translation_request, validate=True)
    @repository_ns.response(200, "번역 완료", translation_response)
    @repository_ns.response(400, "잘못된 요청", error_model)
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def post(self):
        """텍스트를 번역합니다."""
        data = {}
        try:
            data = request.get_json()
            text_length = len(data.get("text", ""))

            logger.info(
                f"API 요청 시작: [{request.method}] {request.path} (클라이언트 IP: {request.remote_addr}). 텍스트 길이: {text_length}"
            )

            # 비즈니스 로직을 서비스 레이어로 위임
            result = repository_service.translate_text(data)
            
            logger.info(
                f"API 요청 성공: [{request.method}] {request.path}. 응답 코드: {result['status_code']}. 메시지: {result['message']}"
            )
            
            return success_response(
                message=result["message"], 
                data=result["data"]
            )

        except ValidationError as e:
            error_message = str(e)
            error_code = "VALIDATION_ERROR"
            status_code = 400
            logger.warning(
                f"API 요청 유효성 검사 실패 [{request.method} {request.path}]: {error_message}. 요청 데이터: {data}"
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except ServiceError as e:
            error_message = str(e)
            error_code = e.error_code or "SERVICE_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 서비스 오류 발생 [{request.method} {request.path}]: {error_message}. ErrorCode: {error_code}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except Exception as e:
            error_message = "번역 중 서버 내부 오류가 발생했습니다."
            error_code = "INTERNAL_SERVER_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 예상치 못한 오류 발생 [{request.method} {request.path}]: {e}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
