from __future__ import annotations

import os
import threading
from pathlib import Path


from fastapi import Depends, HTTPException, status, FastAPI
from fastapi.security import APIKeyHeader

from mistralai.client import Mistral
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.services.EmbeddingService import EmbeddingService
from src.services.FaissRepository import FaissRepository
from src.services.OpenAgendaClient import OpenAgendaClient
from src.services.EventProcessor import EventProcessor
from src.services.RAGService import RagService
from src.services.RebuildService import RebuildService

load_dotenv()

# ============================================================
# Initialisation des services
# ============================================================

app = FastAPI(
    title="Puls-Events RAG API",
    version="1.0.0",
)

rebuild_api_key = APIKeyHeader(
    name="X-Rebuild-Key",
    auto_error=False,
)

mistral_client = Mistral()
llm = ChatOpenAI(
    model="qwen3.8-max",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    temperature=0.2,
)

openagenda_client = OpenAgendaClient(
    api_key=os.environ["OPENAGENDA_API_KEY"],
)

event_processor = EventProcessor()

embedding_service = EmbeddingService(
    client=mistral_client,
    batch_size=100,
)

faiss_repository = FaissRepository(
    index_path=Path("data/index_faiss/evenements.faiss"),
    metadata_path=Path("data/index_faiss/metadata.parquet"),
)

faiss_repository.load()


rag_service = RagService(
    faiss_repository=faiss_repository,
    mistral_client=mistral_client,
    llm=llm,
)

rebuild_service = RebuildService(
    openagenda_client=openagenda_client,
    event_processor=event_processor,
    embedding_service=embedding_service,
    faiss_repository=faiss_repository,
)

rebuild_lock = threading.Lock()

# ============================================================
# Modèles API
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question à poser au chatbot.",
    )


class AskResponse(BaseModel):
    question: str
    answer: str


class RebuildResponse(BaseModel):
    message: str
    nb_chunks: int
    dimension: int

# ============================================================
# Endpoints
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Puls-Events RAG API",
        "status": "ok",
    }

# ============================================================
# Endpoint /ask
# ============================================================

@app.post("/ask", response_model=AskResponse,
)
def ask(request: AskRequest):

    try:

        with rebuild_lock:

            answer = rag_service.ask(request.question)

        return AskResponse(
            question=request.question,
            answer=answer,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# Endpoint /rebuild
# ============================================================

# Fonction de vérification de la clé API pour l'accès à l'endpoint /rebuild
async def verify_rebuild_key(
    api_key: str | None = Depends(rebuild_api_key),
):
    expected_key = os.environ["REBUILD_API_KEY"]

    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès interdit",
        )

@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    dependencies=[Depends(verify_rebuild_key)],
)
def rebuild():

    if not rebuild_lock.acquire(
        blocking=False
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Une reconstruction est "
                "déjà en cours."
            ),
        )

    try:

        nb_chunks, dimension = (
            rebuild_service.rebuild()
        )

        return RebuildResponse(
            message=(
                "Base FAISS reconstruite "
                "avec succès."
            ),
            nb_chunks=nb_chunks,
            dimension=dimension,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:

        rebuild_lock.release()


