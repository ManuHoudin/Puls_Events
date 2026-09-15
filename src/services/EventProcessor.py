from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EventProcessor:
    """
    Nettoyage des événements et préparation des chunks
    destinés à la vectorisation.
    """

    def __init__(
        self,
        seuil_remplissage: float = 0.10,
        chunk_size: int = 1_000,
        chunk_overlap: int = 150,
    ):
        self.seuil_remplissage = seuil_remplissage

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.colonnes_embedding = [
            "title",
            "description",
            "longDescription",
            "keywords",
            "location.city",
            "location.region",
        ]

    def supprimer_colonnes_peu_remplies(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Supprime les colonnes dont moins de 10 % des lignes
        sont renseignées.
        """

        df = df.copy()

        taux_remplissage = df.notna().mean()

        colonnes_a_conserver = taux_remplissage[
            taux_remplissage >= self.seuil_remplissage
        ].index

        return df.loc[:, colonnes_a_conserver].copy()
    

    def normalize_types(self,df: pd.DataFrame,) -> pd.DataFrame:

        df = df.copy()

        def est_booleen_compatible(valeur: Any,) -> bool:

            if isinstance(
                valeur,
                (bool, np.bool_),
            ):
                return True

            if isinstance(
                valeur,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):
                return valeur in (0, 1)

            if isinstance(valeur, str):
                return valeur.strip().lower() in {
                    "true",
                    "false",
                    "1",
                    "0",
                    "oui",
                    "non",
                }

            return False

        def convertir_en_booleen(valeur: Any,) -> Any:

            if valeur is None:
                return pd.NA

            if (
                isinstance(valeur, float)
                and np.isnan(valeur)
            ):
                return pd.NA

            if isinstance(valeur, str):
                valeur = valeur.strip().lower()

            return valeur in (
                True,
                1,
                1.0,
                "true",
                "1",
                "oui",
            )

        for colonne in df.columns:

            valeurs = df[colonne].dropna()

            if valeurs.empty:
                continue

            if valeurs.map(
                est_booleen_compatible
            ).all():

                df[colonne] = (
                    df[colonne]
                    .map(convertir_en_booleen)
                    .astype("boolean")
                )

        if "registration" in df.columns:
            df["registration"] = (
                df["registration"]
                .astype("string")
            )

        return df

    def serialize_json_columns(self,df: pd.DataFrame,) -> pd.DataFrame:

        df = df.copy()

        def serialiser(
            valeur: Any,
        ) -> Any:

            if valeur is None:
                return pd.NA

            if (
                isinstance(valeur, float)
                and np.isnan(valeur)
            ):
                return pd.NA

            if isinstance(
                valeur,
                (dict, list, tuple),
            ):
                return json.dumps(
                    valeur,
                    ensure_ascii=False,
                    default=str,
                )

            return str(valeur)

        for colonne in df.columns:

            valeurs = df[colonne].dropna()

            if valeurs.empty:
                continue

            contient_structure = valeurs.map(
                lambda valeur: isinstance(
                    valeur,
                    (dict, list, tuple),
                )
            ).any()

            if contient_structure:

                df[colonne] = (
                    df[colonne]
                    .map(serialiser)
                    .astype("string")
                )

        return df

    def prepare(self,df: pd.DataFrame,) -> pd.DataFrame:
        """
        Prépare les données avant création des documents.
        """
        df = self.supprimer_colonnes_peu_remplies(df)

        df = self.normalize_types(df)

        df = self.serialize_json_columns(df)

        return df
    

    def create_chunks(self,df: pd.DataFrame,) -> pd.DataFrame:
        """
        Construit les documents textuels utilisés pour les embeddings,
        puis les découpe en chunks.

        Un événement peut produire plusieurs chunks.
        Chaque chunk est rattaché à un seul événement.
        """

        # On ne conserve que les colonnes disponibles.
        colonnes = [
            colonne
            for colonne in self.colonnes_embedding
            if colonne in df.columns
        ]

        documents = (
            df[colonnes]
            .fillna("")
            .astype(str)
            .apply(
                lambda ligne: "\n".join(
                    f"{colonne}: {valeur.strip()}"
                    for colonne, valeur in ligne.items()
                    if valeur.strip()
                ),
                axis=1,
            )
        )

        # Suppression des événements dont le document est vide.
        mask = documents.str.strip().ne("")
        documents = documents.loc[mask]

        chunks: list[dict[str, Any]] = []

        for index_evenement, texte in documents.items():

            textes_chunks = self.splitter.split_text(texte)

            for numero_chunk, texte_chunk in enumerate(
                textes_chunks
            ):
                chunks.append(
                    {
                        "index_evenement": index_evenement,
                        "numero_chunk": numero_chunk,
                        "texte_chunk": texte_chunk,
                    }
                )

        return pd.DataFrame(chunks)