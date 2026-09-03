from __future__ import annotations

import numpy as np
import pandas as pd

from mistralai.client import Mistral


class EmbeddingService:
    """
    Génère les embeddings Mistral des chunks.
    """

    def __init__(
        self,
        client: Mistral,
        batch_size: int = 100,
    ):
        self.client = client
        self.batch_size = batch_size

    def embed_dataframe(
        self,
        df_chunks: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Ajoute une colonne 'embedding' contenant
        les vecteurs Mistral.
        """

        df = df_chunks.copy()

        documents = df[
            "texte_chunk"
        ].tolist()

        embeddings = []

        for debut in range(
            0,
            len(documents),
            self.batch_size,
        ):
            lot = documents[
                debut:debut + self.batch_size
            ]

            response = (
                self.client
                .embeddings
                .create(
                    model="mistral-embed",
                    inputs=lot,
                )
            )

            embeddings.extend(
                item.embedding
                for item in sorted(
                    response.data,
                    key=lambda item: (
                        item.index
                        if item.index is not None
                        else -1
                    ),
                )
            )

        if len(embeddings) != len(df):
            raise ValueError(
                "Le nombre d'embeddings ne correspond "
                "pas au nombre de chunks."
            )

        df["embedding"] = embeddings

        return df