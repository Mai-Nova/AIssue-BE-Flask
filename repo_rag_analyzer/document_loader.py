import os
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config, LANGUAGE_TO_DETAILS

logger = logging.getLogger(__name__)


class DocumentLoader:
    """문서 로딩 및 분할 기능을 담당하는 클래스"""

    def __init__(self):
        pass

    def load_code_documents(self, repo_path, primary_language_name):
        """코드 문서 로드 및 분할"""
        lang_details = LANGUAGE_TO_DETAILS.get(primary_language_name)

        if not lang_details:
            logger.error(
                f"지원되지 않는 언어: {primary_language_name}. 코드 인덱싱을 건너뜁니다."
            )
            return []

        code_file_extension = lang_details["ext"]
        langchain_language_enum = lang_details["lang_enum"]

        logger.info(
            f"{primary_language_name} ({code_file_extension}) 코드 파일을 처리합니다."
        )

        code_docs = self._load_documents_from_path(repo_path, (code_file_extension,))

        if not code_docs:
            logger.warning("인덱싱할 코드 파일을 찾지 못했습니다.")
            return []

        return self._split_code_documents(code_docs, langchain_language_enum)

    def load_documentation_documents(self, repo_path):
        """문서 파일 로드 및 분할"""
        logger.info(
            f"'{repo_path}'에서 문서 파일을 로드합니다 "
            f"(확장자: {Config.DOCUMENT_FILE_EXTENSIONS}, 최대 깊이: 1)..."
        )

        doc_files = self._load_documents_from_path(
            repo_path, Config.DOCUMENT_FILE_EXTENSIONS, max_depth=1
        )

        if not doc_files:
            logger.warning("인덱싱할 문서 파일을 찾지 못했습니다.")
            return []

        return self._split_documentation_documents(doc_files)

    def _load_documents_from_path(
        self, path, file_extensions, encoding="utf-8", max_depth=None
    ):
        """경로에서 특정 확장자 파일 로드"""
        docs = []
        root_path = os.path.abspath(path)

        for root, dirs, files in os.walk(path):
            current_depth = root.replace(root_path, "").count(os.sep)

            if max_depth is not None and current_depth > max_depth:
                dirs.clear()
                continue

            for file in files:
                if file.endswith(file_extensions):
                    file_path = os.path.join(root, file)
                    doc = self._load_single_file(file_path, encoding)
                    if doc:
                        docs.append(doc)

        depth_info = f" (최대 깊이: {max_depth})" if max_depth is not None else ""
        logger.info(
            f"{path} 경로에서 {len(docs)}개의 문서를 로드했습니다 "
            f"({file_extensions}){depth_info}."
        )
        return docs

    def _load_single_file(self, file_path, encoding):
        """단일 파일 로드"""
        try:
            with open(file_path, "r", encoding=encoding, errors="ignore") as f:
                content = f.read()
            return Document(page_content=content, metadata={"source": file_path})
        except Exception as e:
            logger.warning(f"{file_path} 파일 로드 중 오류 발생: {e}")
            return None

    def _split_code_documents(self, code_docs, langchain_language_enum):
        """코드 문서 분할"""
        text_splitter_code = RecursiveCharacterTextSplitter.from_language(
            language=langchain_language_enum,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        split_code_docs = text_splitter_code.split_documents(code_docs)
        logger.info(
            f"{len(code_docs)}개의 코드 파일을 {len(split_code_docs)}개의 청크로 분할했습니다."
        )
        return split_code_docs

    def _split_documentation_documents(self, doc_files):
        """문서 파일 분할"""
        text_splitter_docs = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
        )
        split_doc_files = text_splitter_docs.split_documents(doc_files)
        logger.info(
            f"{len(doc_files)}개의 문서 파일을 {len(split_doc_files)}개의 청크로 분할했습니다."
        )
        return split_doc_files
