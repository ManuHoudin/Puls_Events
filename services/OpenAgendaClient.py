from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import requests
import time


class OpenAgendaClient:

    def __init__(
        self,
        api_key: str,
        base_url: str = (
            "https://api.openagenda.com/v2"
        ),
        date_min: str = (
            "2025-08-01T00:00:00+02:00"
        ),
        region_cible: str = "Bretagne",
    ):
        self.base_url = base_url
        self.date_min = date_min
        self.region_cible = region_cible

        self.headers = {
            "key": api_key,
        }

    def get_bretagne_agendas(
        self,
    ) -> list[dict[str, Any]]:

        url = f"{self.base_url}/agendas"

        params = {
            "search": self.region_cible,
            "official": 1,
            "size": 100,
            "if[]": [
                "uid",
                "title",
                "description",
                "slug",
            ],
        }

        response = requests.get(
            url,
            params=params,
            headers=self.headers,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()["agendas"]

    def get_agenda_events(
        self,
        agenda_uid: int,
    ) -> list[dict[str, Any]]:

        url = (
            f"{self.base_url}/agendas/"
            f"{agenda_uid}/events"
        )

        params_base = {
            "relative[]": [
                "current",
                "upcoming",
            ],
            "timings[gte]": self.date_min,
            "monolingual": "fr",
            "detailed": 1,
            "size": 300,
            "sort": "timings.asc",
        }

        evenements = []
        after = None

        while True:

            params = params_base.copy()

            if after is not None:
                params["after[]"] = after

            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=30,
            )

            response.raise_for_status()

            resultat = response.json()

            evenements.extend(
                resultat["events"]
            )

            after = resultat.get("after")

            if after is None:
                break

        return evenements

    def fetch_bretagne_events(
        self,
    ) -> pd.DataFrame:

        agendas = self.get_bretagne_agendas()

        tous_evenements = []

        for agenda in agendas:

            agenda_uid = agenda["uid"]

            evenements = (
                self.get_agenda_events(
                    agenda_uid
                )
            )

            for evenement in evenements:

                evenement[
                    "agenda_uid_source"
                ] = agenda_uid

                evenement[
                    "agenda_titre_source"
                ] = agenda["title"]

            tous_evenements.extend(
                evenements
            )

            # Même temporisation que dans ton notebook.
            time.sleep(0.1)

        df = pd.json_normalize(
            tous_evenements
        )

        colonne_region = "location.region"

        if colonne_region not in df.columns:
            raise KeyError(
                f"Colonne région absente : "
                f"{colonne_region}"
            )

        masque_region = (
            df[colonne_region]
            .astype("string")
            .str.strip()
            .str.casefold()
            .eq(
                self.region_cible.casefold()
            )
        )

        return (
            df.loc[masque_region]
            .reset_index(drop=True)
            .copy()
        )