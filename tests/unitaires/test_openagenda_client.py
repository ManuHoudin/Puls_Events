from unittest.mock import Mock, patch, call
import pytest
import pandas as pd

from src.services.OpenAgendaClient import OpenAgendaClient

def test_get_bretagne_agendas():
    client = OpenAgendaClient(
        api_key="test-key"
    )

    donnees = {
        "agendas": [
            {
                "uid": 123,
                "title": "Agenda test",
            },
            {
                "uid": 456,
                "title": "Agenda test 2",
            },
        ]
    }

    response = Mock()
    response.json.return_value = donnees

    with patch(
        "src.services.OpenAgendaClient.requests.get",
        return_value=response,
    ) as mock_get:

        result = client.get_bretagne_agendas()

    response.raise_for_status.assert_called_once()

    assert result == donnees["agendas"]

    mock_get.assert_called_once_with(
        "https://api.openagenda.com/v2/agendas",
        params={
            "search": "Bretagne",
            "official": 1,
            "size": 100,
            "if[]": [
                "uid",
                "title",
                "description",
                "slug",
            ],
        },
        headers={"key": "test-key"},
        timeout=30,
    )

def test_get_bretagne_agendas_http_error():
    client = OpenAgendaClient(
        api_key="test-key"
    )

    response = Mock()
    response.raise_for_status.side_effect = (Exception("HTTP error"))

    with patch(
        "src.services.OpenAgendaClient.requests.get",
        return_value=response,
    ):

        with pytest.raises(Exception, match="HTTP error"):
            client.get_bretagne_agendas()

    response.raise_for_status.assert_called_once()


def test_get_agenda_events_single_page():
    client = OpenAgendaClient(
        api_key="test-key"
    )

    donnees = {
        "events": [
            {"uid": 111, "title": "Concert"},
            {"uid": 222, "title": "Festival"},
        ],
        "after": None,
    }

    response = Mock()
    response.json.return_value = donnees

    with patch(
        "src.services.OpenAgendaClient.requests.get",
        return_value=response,
    ) as mock_get:

        result = client.get_agenda_events(
            agenda_uid=123
        )

    assert result == donnees["events"]

    response.raise_for_status.assert_called_once()

    mock_get.assert_called_once()

    assert mock_get.call_args.kwargs["params"] == {
        "relative[]": [
            "current",
            "upcoming",
        ],
        "timings[gte]": (
            "2025-08-01T00:00:00+02:00"
        ),
        "monolingual": "fr",
        "detailed": 1,
        "size": 300,
        "sort": "timings.asc",
    }


def test_get_agenda_events_pagination():
    client = OpenAgendaClient(
        api_key="test-key"
    )

    response_page_1 = Mock()
    response_page_1.json.return_value = {
        "events": [
            {"uid": 111},
            {"uid": 222},
        ],
        "after": "cursor-123",
    }

    response_page_2 = Mock()
    response_page_2.json.return_value = {
        "events": [
            {"uid": 333},
        ],
        "after": None,
    }

    with patch(
        "src.services.OpenAgendaClient.requests.get",
        side_effect=[
            response_page_1,
            response_page_2,
        ],
    ) as mock_get:

        result = client.get_agenda_events(
            agenda_uid=123
        )

    assert result == [
        {"uid": 111},
        {"uid": 222},
        {"uid": 333},
    ]

    assert mock_get.call_count == 2

    first_params = (
        mock_get.call_args_list[0]
        .kwargs["params"]
    )

    second_params = (
        mock_get.call_args_list[1]
        .kwargs["params"]
    )

    assert "after[]" not in first_params
    assert second_params["after[]"] == "cursor-123"


@patch(
    "src.services.OpenAgendaClient.time.sleep"
)
def test_fetch_bretagne_events(mock_sleep):
    client = OpenAgendaClient(
        api_key="test-key"
    )

    agendas = [
        {
            "uid": 123,
            "title": "Agenda Vannes",
        },
        {
            "uid": 456,
            "title": "Agenda Rennes",
        },
    ]

    evenements_vannes = [
        {
            "uid": 1001,
            "title": "Concert Vannes",
            "location": {
                "region": "Bretagne",
            },
        },
        {
            "uid": 1002,
            "title": "Théâtre Vannes",
            "location": {
                "region": "Bretagne",
            },
        },
    ]

    evenements_rennes = [
        {
            "uid": 2001,
            "title": "Concert Rennes",
            "location": {
                "region": "Pays de la Loire",
            },
        },
    ]

    with patch.object(
        client,
        "get_bretagne_agendas",
        return_value=agendas,
    ) as mock_agendas, patch.object(
        client,
        "get_agenda_events",
        side_effect=[
            evenements_vannes,
            evenements_rennes,
        ],
    ) as mock_events:

        result = client.fetch_bretagne_events()

    assert isinstance(result, pd.DataFrame)

    assert len(result) == 2

    assert result["uid"].tolist() == [
        1001,
        1002,
    ]

    assert (
        result["agenda_uid_source"].tolist()
        == [123, 123]
    )

    assert (
        result["agenda_titre_source"].tolist()
        == [
            "Agenda Vannes",
            "Agenda Vannes",
        ]
    )

    assert (
        result["location.region"].tolist()
        == [
            "Bretagne",
            "Bretagne",
        ]
    )

    mock_agendas.assert_called_once()

    assert mock_events.call_count == 2

    mock_sleep.assert_has_calls(
        [
            call(0.1),
            call(0.1),
        ]
    )