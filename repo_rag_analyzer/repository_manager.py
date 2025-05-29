import os
import shutil
import logging

import requests
from git import Repo, GitCommandError

from config import Config
from common.exceptions import RepositoryError, RepositorySizeError

logger = logging.getLogger(__name__)


class RepositoryManager:
    """저장소 복제 및 메타데이터 관리 클래스"""

    def __init__(self):
        pass

    def get_repository_info(self, repo_url, token=None):
        """GitHub API로 저장소 주 사용 언어 및 크기 조회"""
        try:
            owner, repo_name = self._parse_github_url(repo_url)
            languages = self._fetch_languages_from_api(owner, repo_name, token)

            if not languages:
                logger.warning(
                    f"{repo_url}에 대한 언어 데이터를 찾을 수 없습니다. 기본값으로 'unknown'을 사용합니다."
                )
                return "unknown", 0

            primary_language, primary_language_bytes = self._get_primary_language(
                languages
            )

            logger.info(f"{repo_url}의 주 사용 언어 감지: {primary_language}")
            logger.info(
                f"주 사용 언어 ({primary_language}) 코드 바이트 수: {primary_language_bytes:,} bytes "
                f"({primary_language_bytes / (1024*1024):.1f} MB)"
            )

            self._validate_repository_size(primary_language_bytes)

            return primary_language.lower(), primary_language_bytes

        except RepositorySizeError:
            raise
        except requests.exceptions.RequestException as e:
            raise RepositoryError(f"{repo_url}의 저장소 언어 가져오기 오류: {e}") from e
        except ValueError as e:
            raise RepositoryError(f"입력 값 오류: {e}") from e
        except Exception as e:
            raise RepositoryError(
                f"{repo_url}의 언어 가져오기 중 예상치 못한 오류 발생: {e}"
            ) from e

    def clone_or_load_repository(self, repo_url, local_path):
        """Git 저장소 복제 또는 로드"""
        if os.path.exists(local_path):
            logger.info(f"기존 저장소를 {local_path} 에서 로드합니다.")
            try:
                return Repo(local_path)
            except GitCommandError as e:
                logger.warning(f"기존 저장소 로드 실패: {e}. 저장소를 새로 복제합니다.")
                shutil.rmtree(local_path)

        return self._clone_fresh_repository(repo_url, local_path)

    def _parse_github_url(self, repo_url):
        """GitHub URL에서 소유자와 저장소 이름 추출"""
        if not repo_url.startswith("https://github.com/"):
            raise ValueError(f"잘못된 GitHub URL입니다: {repo_url}")

        parts = repo_url.split("/")
        if len(parts) < 5:
            raise ValueError(
                f"URL에서 소유자/저장소 이름을 파싱할 수 없습니다: {repo_url}"
            )

        owner = parts[-2]
        repo_name = parts[-1].removesuffix(".git")
        return owner, repo_name

    def _fetch_languages_from_api(self, owner, repo_name, token):
        """GitHub API에서 언어 정보 가져오기"""
        api_url = f"https://api.github.com/repos/{owner}/{repo_name}/languages"
        headers = {"Accept": "application/vnd.github.v3+json"}

        if token:
            headers["Authorization"] = f"token {token}"

        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def _get_primary_language(self, languages):
        """언어 데이터에서 주 사용 언어와 바이트 수 추출"""
        primary_language = max(languages, key=languages.get)
        primary_language_bytes = languages[primary_language]
        return primary_language, primary_language_bytes

    def _validate_repository_size(self, primary_language_bytes):
        """저장소 크기 검증"""
        max_bytes = Config.MAX_REPO_SIZE_MB * 1024 * 1024
        if primary_language_bytes > max_bytes:
            error_msg = (
                f"저장소의 주 사용 언어 코드 크기가 너무 큽니다. "
                f"크기: {primary_language_bytes / (1024*1024):.1f} MB, "
                f"최대 허용: {Config.MAX_REPO_SIZE_MB:.1f} MB"
            )
            logger.error(error_msg)
            raise RepositorySizeError(error_msg)

    def _clone_fresh_repository(self, repo_url, local_path):
        """새로운 저장소 복제"""
        logger.info(f"{repo_url}을(를) {local_path}에 복제합니다...")
        try:
            return Repo.clone_from(repo_url, local_path)
        except GitCommandError as e:
            raise RepositoryError(f"저장소 복제 실패: {e}") from e
