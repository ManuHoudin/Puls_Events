from __future__ import annotations

import pandas as pd

from services.OpenAgendaClient import OpenAgendaClient
from services.EventProcessor import EventProcessor
from services.EmbeddingService import EmbeddingService
from services.FaissRepository import FaissRepository


class RebuildService:
    """
    Reconstruction complète de la base RAG.

    OpenAgenda
        ↓
    nettoyage
        ↓
    documents / chunks
        ↓
    embeddings Mistral
        ↓
    FAISS
    """

    def __init__(
        self,
        openagenda_client: OpenAgendaClient,
        event_processor: EventProcessor,
        embedding_service: EmbeddingService,
        faiss_repository: FaissRepository,
    ):
        self.openagenda = openagenda_client
        self.processor = event_processor
        self.embedding_service = embedding_service
        self.faiss_repository = faiss_repository

    def rebuild(self) -> tuple[int, int]:

        # ----------------------------------------------------
        # 1. Nouveaux événements depuis OpenAgenda
        # ----------------------------------------------------

        df_raw = (
            self.openagenda
            .fetch_bretagne_events()
        )

        if df_raw.empty:
            raise ValueError(
                "OpenAgenda n'a retourné aucun événement."
            )

        # L'identifiant utilisé pour relier un événement
        # à ses chunks.
        df_raw = df_raw.reset_index(
            names="index_evenement"
        )

        # ----------------------------------------------------
        # 2. Nettoyage
        # ----------------------------------------------------

        df_clean = (
            self.processor
            .prepare(df_raw)
        )

        # ----------------------------------------------------
        # 3. Création des chunks
        # ----------------------------------------------------

        df_chunks = (
            self.processor
            .create_chunks(df_clean)
        )

        if df_chunks.empty:
            raise ValueError(
                "Aucun chunk n'a été généré."
            )

        # ----------------------------------------------------
        # 4. Embeddings Mistral
        # ----------------------------------------------------

        df_embeddings = (
            self.embedding_service
            .embed_dataframe(df_chunks)
        )

        # ----------------------------------------------------
        # 5. Construction de l'index FAISS
        # ----------------------------------------------------

        nb_chunks, dimension = (
            self.faiss_repository
            .build(
                df_embeddings,
                df_clean,
            )
        )

        return (
            nb_chunks,
            dimension,
        )