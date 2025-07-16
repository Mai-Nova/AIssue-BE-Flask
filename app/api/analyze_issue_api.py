from flask import request, current_app
from flask_restx import Namespace, Resource, fields

from app.services.issue_analysis_service import IssueAnalysisService
from app.core.exceptions import ServiceError

# 네임스페이스 생성
issue_ns = Namespace("issue", description="이슈 분석 관련 API")

# 서비스 인스턴스 생성
issue_analysis_service = IssueAnalysisService()

# API 모델 정의
issue_analysis_request_model = issue_ns.model(
    "IssueAnalysisRequest",
    {
        "title": fields.String(required=True, description="이슈 제목"),
        "body": fields.String(description="이슈 본문"),
        "issueId": fields.Integer(required=True, description="이슈 ID"),
        "repoUrl": fields.String(required=True, description="저장소 URL"),
        "defaultBranch": fields.String(
            required=False, description="저장소 기본 브랜치명", default="main"
        ),
    },
)

issue_analysis_response_model = issue_ns.model(
    "IssueAnalysisResponse",
    {
        "summary": fields.String(description="AI 요약 정보"),
        "relatedFiles": fields.List(
            fields.Nested(
                issue_ns.model(
                    "RelatedFile",
                    {
                        "path": fields.String(description="파일 경로"),
                        "relevance": fields.Float(description="관련도 점수"),
                        "githubUrl": fields.String(description="GitHub 파일 URL"),
                    },
                )
            )
        ),
        "codeSnippets": fields.List(
            fields.Nested(
                issue_ns.model(
                    "CodeSnippet",
                    {
                        "file": fields.String(description="코드 스니펫이 포함된 파일"),
                        "code": fields.String(description="코드 스니펫"),
                        "relevance": fields.Float(description="관련도 점수"),
                        "explanation": fields.String(description="코드 스니펫 설명"),
                    },
                )
            )
        ),
        "solutionSuggestion": fields.String(description="AI 해결 제안"),
    },
)


@issue_ns.route("/analyze-issue")
class AnalyzeIssue(Resource):
    @issue_ns.expect(issue_analysis_request_model)
    @issue_ns.marshal_with(issue_analysis_response_model)
    def post(self):
        """이슈 정보를 받아 AI 분석을 수행합니다."""
        data = request.json
        issue_id = data.get("issueId")

        current_app.logger.info(f"이슈 분석 요청 수신: 이슈 ID {issue_id}")

        try:
            # 비즈니스 로직을 서비스 레이어로 위임
            result = issue_analysis_service.analyze_issue(data)
            
            # 응답 상태 코드에 따라 HTTP 응답 처리
            status_code = result.pop("status_code", 200)
            
            current_app.logger.info(f"이슈 분석 완료: 이슈 ID {issue_id}, 상태 코드: {status_code}")
            
            return result, status_code

        except ServiceError as e:
            current_app.logger.error(f"이슈 분석 서비스 오류 (이슈 ID: {issue_id}): {e}", exc_info=True)
            return {
                "summary": "분석 서비스에서 오류가 발생했습니다.",
                "relatedFiles": [],
                "codeSnippets": [],
                "solutionSuggestion": "서비스 오류로 인해 분석을 완료할 수 없습니다.",
            }, 500
        except Exception as e:
            current_app.logger.error(f"이슈 분석 중 예상치 못한 오류 (이슈 ID: {issue_id}): {e}", exc_info=True)
            return {
                "summary": "예상치 못한 오류가 발생했습니다.",
                "relatedFiles": [],
                "codeSnippets": [],
                "solutionSuggestion": "예상치 못한 오류로 인해 분석을 완료할 수 없습니다.",
            }, 500
