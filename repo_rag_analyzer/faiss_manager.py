import os
import logging

from langchain_community.vectorstores import FAISS
import faiss

from common.exceptions import IndexingError, EmbeddingError

faiss.omp_set_num_threads(1)

logger = logging.getLogger(__name__)


class FAISSManager:
    """FAISS 인덱스 생성 및 관리 클래스"""

    def __init__(self):
        pass

    def create_index(self, docs, embeddings, index_path, index_type):
        """문서 리스트와 임베딩 모델을 사용하여 FAISS 인덱스 생성"""
        if not docs:
            logger.warning(
                f"{index_type} 인덱싱을 위한 문서가 없습니다. 인덱스 생성을 건너뜁니다."
            )
            return None

        logger.info(f"{index_type}에 대한 FAISS 인덱스를 {index_path}에 생성합니다...")

        # 임베딩 생성
        successful_embeddings, successful_docs = self._generate_embeddings(
            docs, embeddings, index_type
        )

        if not successful_docs:
            logger.warning(
                f"유효한 임베딩이 없어 {index_type} FAISS 인덱스를 생성할 수 없습니다."
            )
            return None

        # FAISS 인덱스 생성 및 저장
        return self._create_and_save_index(
            successful_docs, successful_embeddings, embeddings, index_path, index_type
        )

    def load_index(self, index_path, embeddings, index_type):
        """저장된 FAISS 인덱스 로드"""
        if not os.path.exists(index_path):
            return None

        logger.info(f"기존 {index_type} FAISS 인덱스를 {index_path} 에서 로드합니다...")
        try:
            return FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(
                f"{index_type} FAISS 인덱스 로드 중 오류 발생: {e}. 새로 생성합니다."
            )
            return None

    def _generate_embeddings(self, docs, embeddings, index_type):
        """문서들의 임베딩 생성 및 성공한 문서 필터링"""
        doc_contents = [doc.page_content for doc in docs]

        try:
            successful_raw_embeddings, failed_original_indices = (
                embeddings.embed_documents(doc_contents)
            )
        except EmbeddingError as e:
            logger.error(f"{index_type} 문서 임베딩 중 심각한 오류 발생: {e}")
            return [], []
        except Exception as e:
            logger.error(
                f"{index_type} 문서 임베딩 중 예기치 않은 오류: {e}",
                exc_info=True,
            )
            return [], []

        # 성공한 임베딩에 해당하는 문서 필터링
        successful_docs = self._filter_successful_documents(
            docs, failed_original_indices
        )

        self._validate_embedding_document_match(
            successful_raw_embeddings, successful_docs, index_type
        )

        return successful_raw_embeddings, successful_docs

    def _filter_successful_documents(self, docs, failed_indices):
        """실패한 인덱스를 제외한 성공한 문서들만 필터링"""
        return [doc for i, doc in enumerate(docs) if i not in failed_indices]

    def _validate_embedding_document_match(self, embeddings, docs, index_type):
        """임베딩과 문서 수량 일치 검증"""
        if len(embeddings) != len(docs):
            logger.error(
                f"임베딩 수({len(embeddings)})와 필터링된 성공 문서 수({len(docs)}) 불일치!"
            )
            raise IndexingError(
                f"{index_type} 인덱스 생성 실패: 임베딩과 문서 매칭 오류"
            )

    def _create_and_save_index(
        self, docs, embeddings, embedding_model, index_path, index_type
    ):
        """FAISS 인덱스 생성 및 저장"""
        logger.info(
            f"{len(docs)}개의 성공한 문서로 {index_type} FAISS 인덱스를 생성합니다."
        )

        try:
            # 텍스트-임베딩 쌍 생성
            text_embedding_pairs = [
                (doc.page_content, embedding)
                for doc, embedding in zip(docs, embeddings)
            ]

            # 메타데이터 추출
            metadatas = [doc.metadata for doc in docs]

            # FAISS 인덱스 생성
            vector_store = FAISS.from_embeddings(
                text_embeddings=text_embedding_pairs,
                embedding=embedding_model,
                metadatas=metadatas,
            )

            # 인덱스 저장
            vector_store.save_local(index_path)
            logger.info(
                f"{index_type} FAISS 인덱스를 {index_path}에 성공적으로 저장했습니다."
            )
            return vector_store

        except EmbeddingError as e:
            logger.error(f"{index_type} 인덱스 생성 중 임베딩 오류 발생: {e}")
            raise IndexingError(f"{index_type} 인덱스 생성 실패 (임베딩 오류)") from e
        except Exception as e:
            logger.error(
                f"{index_type} FAISS 인덱스 생성 또는 저장 중 오류 발생: {e}",
                exc_info=True,
            )
            raise IndexingError(f"{index_type} 인덱스 생성 실패") from e
