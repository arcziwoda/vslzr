"""Tests for bridge discovery and pairing utilities."""

from unittest.mock import patch, MagicMock

import pytest

from hue_visualizer.bridge.discovery import (
    discover_bridge,
    create_user,
    create_entertainment_user,
    verify_connection,
    list_entertainment_areas,
)
from hue_visualizer.core.exceptions import BridgeDiscoveryError, BridgeConnectionError


class TestDiscoverBridge:
    """Test bridge discovery via meethue.com."""

    @patch("hue_visualizer.bridge.discovery.requests.get")
    def test_returns_first_bridge_ip(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"internalipaddress": "192.168.1.100"}]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        ip = discover_bridge()
        assert ip == "192.168.1.100"

    @patch("hue_visualizer.bridge.discovery.requests.get")
    def test_raises_on_empty_list(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(BridgeDiscoveryError, match="No Hue Bridge"):
            discover_bridge()

    @patch("hue_visualizer.bridge.discovery.requests.get")
    def test_raises_on_network_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("timeout")

        with pytest.raises(BridgeDiscoveryError, match="Failed to discover"):
            discover_bridge()


class TestCreateUser:
    """Test user creation (button-press pairing)."""

    @patch("hue_visualizer.bridge.discovery.requests.post")
    def test_success_returns_username(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"success": {"username": "abc123"}}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        username = create_user("192.168.1.1")
        assert username == "abc123"

    @patch("hue_visualizer.bridge.discovery.requests.post")
    def test_uses_https(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"success": {"username": "u"}}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        create_user("10.0.0.1")
        url = mock_post.call_args[0][0]
        assert url.startswith("https://")

    @patch("hue_visualizer.bridge.discovery.requests.post")
    def test_raises_on_button_not_pressed(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"error": {"type": 101, "description": "link button not pressed"}}
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        with pytest.raises(BridgeConnectionError, match="button not pressed"):
            create_user("10.0.0.1")


class TestCreateEntertainmentUser:
    """Test entertainment user creation (username + clientkey)."""

    @patch("hue_visualizer.bridge.discovery.requests.post")
    def test_success_returns_username_and_clientkey(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"success": {"username": "user1", "clientkey": "key1"}}
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        username, clientkey = create_entertainment_user("192.168.1.1")
        assert username == "user1"
        assert clientkey == "key1"

    @patch("hue_visualizer.bridge.discovery.requests.post")
    def test_sends_generateclientkey_flag(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"success": {"username": "u", "clientkey": "k"}}
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        create_entertainment_user("10.0.0.1")
        payload = mock_post.call_args[1]["json"]
        assert payload["generateclientkey"] is True

    @patch("hue_visualizer.bridge.discovery.requests.post")
    def test_uses_https(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"success": {"username": "u", "clientkey": "k"}}
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        create_entertainment_user("10.0.0.1")
        url = mock_post.call_args[0][0]
        assert url.startswith("https://")

    @patch("hue_visualizer.bridge.discovery.requests.post")
    def test_raises_on_button_not_pressed(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"error": {"type": 101, "description": "link button not pressed"}}
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        with pytest.raises(BridgeConnectionError, match="button not pressed"):
            create_entertainment_user("10.0.0.1")

    @patch("hue_visualizer.bridge.discovery.requests.post")
    def test_raises_when_clientkey_missing(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"success": {"username": "u"}}  # No clientkey
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        with pytest.raises(BridgeConnectionError, match="clientkey"):
            create_entertainment_user("10.0.0.1")

    @patch("hue_visualizer.bridge.discovery.requests.post")
    def test_raises_on_network_error(self, mock_post):
        import requests
        mock_post.side_effect = requests.RequestException("connection refused")

        with pytest.raises(BridgeConnectionError, match="Failed to create"):
            create_entertainment_user("10.0.0.1")


class TestVerifyConnection:
    """Test connection verification."""

    @patch("hue_visualizer.bridge.discovery.requests.get")
    def test_valid_connection_returns_true(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"1": {"name": "Light 1"}}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        assert verify_connection("192.168.1.1", "user1") is True

    @patch("hue_visualizer.bridge.discovery.requests.get")
    def test_uses_https(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"1": {"name": "Light 1"}}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        verify_connection("10.0.0.1", "u")
        url = mock_get.call_args[0][0]
        assert url.startswith("https://")

    @patch("hue_visualizer.bridge.discovery.requests.get")
    def test_invalid_credentials_raises(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"error": {"type": 1, "description": "unauthorized user"}}
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(BridgeConnectionError, match="Invalid credentials"):
            verify_connection("10.0.0.1", "bad_user")


class TestListEntertainmentAreas:
    """Test listing entertainment areas."""

    @patch("hue_visualizer.bridge.discovery.requests.get")
    def test_returns_all_entertainment_configs(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "id": "uuid-living-room",
                    "metadata": {"name": "Living Room"},
                    "channels": [{"id": 0}, {"id": 1}, {"id": 2}],
                },
                {
                    "id": "uuid-studio",
                    "metadata": {"name": "Studio"},
                    "channels": [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}],
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        areas = list_entertainment_areas("192.168.1.1", "user1")

        assert "uuid-living-room" in areas
        assert "uuid-studio" in areas

        assert areas["uuid-living-room"]["name"] == "Living Room"
        assert areas["uuid-living-room"]["num_lights"] == 3
        assert areas["uuid-studio"]["name"] == "Studio"
        assert areas["uuid-studio"]["num_lights"] == 4

    @patch("hue_visualizer.bridge.discovery.requests.get")
    def test_returns_empty_dict_when_no_entertainment_areas(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        areas = list_entertainment_areas("192.168.1.1", "user1")
        assert areas == {}

    @patch("hue_visualizer.bridge.discovery.requests.get")
    def test_uses_v2_api_url(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        list_entertainment_areas("10.0.0.1", "u")
        url = mock_get.call_args[0][0]
        assert url.startswith("https://")
        assert "/clip/v2/resource/entertainment_configuration" in url

    @patch("hue_visualizer.bridge.discovery.requests.get")
    def test_sends_application_key_header(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        list_entertainment_areas("10.0.0.1", "myuser")
        headers = mock_get.call_args[1].get("headers", {})
        assert headers.get("hue-application-key") == "myuser"

    @patch("hue_visualizer.bridge.discovery.requests.get")
    def test_raises_on_network_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("timeout")

        with pytest.raises(BridgeConnectionError, match="Failed to list"):
            list_entertainment_areas("10.0.0.1", "u")
