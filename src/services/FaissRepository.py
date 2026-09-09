from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from langchain_core.documents import Document


class FaissRepository:
    """
    Gestion de l'index FAISS et de ses métadonnées.
    """

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index: Any = None
        self.metadata: pd.DataFrame | None = None

    def build(
        self,
        df_embeddings: pd.DataFrame,
        metadata_evenements: pd.DataFrame,
    ) -> tuple[int, int]:
        """
        Construit l'index FAISS à partir des embeddings
        et des métadonnées des événements.
        """

        metadata_faiss = (
            df_embeddings[
                [
                    "index_evenement",
                    "numero_chunk",
                    "texte_chunk",
                ]
            ]
            .merge(
                metadata_evenements,
                on="index_evenement",
                how="left",
                validate="many_to_one",
            )
            .reset_index(drop=True)
        )

        metadata_faiss.insert(
            0,
            "faiss_id",
            np.arange(
                len(metadata_faiss),
                dtype=np.int64,
            ),
        )

        vecteurs = np.asarray(
            df_embeddings["embedding"].tolist(),
            dtype=np.float32,
        )

        if (
            vecteurs.ndim != 2
            or len(vecteurs) != len(metadata_faiss)
        ):
            raise ValueError(
                "Incohérence entre les embeddings "
                "et les métadonnées."
            )

        # Similarité cosinus
        faiss.normalize_L2(vecteurs)

        dimension = vecteurs.shape[1]

        nouvel_index = faiss.IndexIDMap2(
            faiss.IndexFlatIP(dimension)
        )

        nouvel_index.add_with_ids(
            vecteurs,
            metadata_faiss[
                "faiss_id"
            ].to_numpy(dtype=np.int64),
        )

        self.index = nouvel_index
        self.metadata = metadata_faiss

        self.save()

        return (
            len(metadata_faiss),
            dimension,
        )

    def save(self) -> None:

        if self.index is None:
            raise RuntimeError(
                "Aucun index FAISS à sauvegarder."
            )

        if self.metadata is None:
            raise RuntimeError(
                "Aucune métadonnée à sauvegarder."
            )

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        self.metadata.to_parquet(
            self.metadata_path,
            index=False,
        )

    def load(self) -> None:

        self.index = faiss.read_index(
            str(self.index_path)
        )

        self.metadata = pd.read_parquet(
            self.metadata_path
        )

    def search(
        self,
        vecteur_question: np.ndarray,
        k: int = 5,
    ) -> list[Document]:
        """
        Recherche les k chunks les plus proches.
        """

        if self.index is None:
            raise RuntimeError(
                "Index FAISS non chargé."
            )

        if self.metadata is None:
            raise RuntimeError(
                "Métadonnées non chargées."
            )

        vecteur = vecteur_question.copy()

        faiss.normalize_L2(vecteur)

        scores, ids = self.index.search(
            vecteur,
            k,
        )

        documents = []

        for score, faiss_id in zip(
            scores[0],
            ids[0],
        ):

            if faiss_id == -1:
                continue

            ligne = (
                self.metadata[
                    self.metadata["faiss_id"]
                    == faiss_id
                ]
                .iloc[0]
            )

            contenu = f"""
Titre : {ligne["title"]}

Description :
{ligne["texte_chunk"]}

Horaires :
{ligne["timings"]}

Lieu :
{ligne["location.name"]}

Ville :
{ligne["location.city"]}

Région :
{ligne["location.region"]}

Adresse :
{ligne["location.address"]}
""".strip()

            documents.append(
                Document(
                    page_content=contenu,
                    metadata={
                        "uid": ligne["uid"],
                        "title": ligne["title"],
                        "faiss_id": int(faiss_id),
                        "score_similarite": float(score),
                        "agenda_titre_source": (
                            ligne["agenda_titre_source"]
                        ),
                    },
                )
            )

        return documents