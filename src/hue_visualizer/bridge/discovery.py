"""Hue Bridge discovery and pairing utilities."""

import logging
import warnings

import requests
import urllib3

from ..core.exceptions import BridgeDiscoveryError, BridgeConnectionError

# Hue bridges use self-signed certificates
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


def discover_bridge() -> str:
    """
    Discover Hue Bridge on the local network using Philips discovery service.

    Returns:
        str: IP address of the discovered bridge

    Raises:
        BridgeDiscoveryError: If no bridge is found or discovery fails
    """
    try:
        response = requests.get("https://discovery.meethue.com/", timeout=5)
        response.raise_for_status()
        bridges = response.json()

        if not bridges:
            raise BridgeDiscoveryError("No Hue Bridge found on the network")

        # Return first bridge IP
        bridge_ip = bridges[0]["internalipaddress"]
        return bridge_ip

    except requests.RequestException as e:
        raise BridgeDiscoveryError(f"Failed to discover bridge: {e}")
    except (KeyError, IndexError) as e:
        raise BridgeDiscoveryError(f"Invalid discovery response format: {e}")


def create_user(bridge_ip: str, app_name: str = "hue-visualizer") -> str:
    """
    Create a new user on the Hue Bridge (requires physical button press).

    Args:
        bridge_ip: IP address of the bridge
        app_name: Application name for the username

    Returns:
        str: The created username/API token

    Raises:
        BridgeConnectionError: If user creation fails
    """
    url = f"https://{bridge_ip}/api"
    payload = {"devicetype": f"{app_name}#python"}

    try:
        response = requests.post(url, json=payload, timeout=5, verify=False)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            result = data[0]

            # Check for error (button not pressed)
            if "error" in result:
                error_type = result["error"].get("type")
                if error_type == 101:
                    raise BridgeConnectionError(
                        "Link button not pressed. Please press the button on the bridge and try again."
                    )
                raise BridgeConnectionError(f"Bridge error: {result['error'].get('description')}")

            # Success
            if "success" in result:
                username = result["success"]["username"]
                return username

        raise BridgeConnectionError(f"Unexpected response format: {data}")

    except requests.RequestException as e:
        raise BridgeConnectionError(f"Failed to create user: {e}")


def create_entertainment_user(
    bridge_ip: str, app_name: str = "hue_visualizer"
) -> tuple[str, str]:
    """
    Create a new user with Entertainment API access (requires physical button press).

    This combines username + clientkey generation in a single call using the
    ``generateclientkey: True`` flag.

    Args:
        bridge_ip: IP address of the bridge
        app_name: Application name for the username

    Returns:
        Tuple of (username, clientkey)

    Raises:
        BridgeConnectionError: If user creation fails or button not pressed
    """
    url = f"https://{bridge_ip}/api"
    payload = {
        "devicetype": f"{app_name}#python",
        "generateclientkey": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            result = data[0]

            if "error" in result:
                error_type = result["error"].get("type")
                if error_type == 101:
                    raise BridgeConnectionError(
                        "Link button not pressed. Press the button on the bridge and try again."
                    )
                raise BridgeConnectionError(
                    f"Bridge error: {result['error'].get('description')}"
                )

            if "success" in result:
                username = result["success"]["username"]
                clientkey = result["success"].get("clientkey")
                if not clientkey:
                    raise BridgeConnectionError(
                        "Bridge did not return a clientkey. "
                        "Ensure your bridge firmware supports the Entertainment API."
                    )
                return username, clientkey

        raise BridgeConnectionError(f"Unexpected response format: {data}")

    except BridgeConnectionError:
        raise
    except requests.RequestException as e:
        raise BridgeConnectionError(f"Failed to create entertainment user: {e}")


def verify_connection(bridge_ip: str, username: str) -> bool:
    """
    Verify that the connection to the bridge works with the given credentials.

    Args:
        bridge_ip: IP address of the bridge
        username: API username/token

    Returns:
        bool: True if connection is valid

    Raises:
        BridgeConnectionError: If verification fails
    """
    url = f"https://{bridge_ip}/api/{username}/lights"

    try:
        response = requests.get(url, timeout=5, verify=False)
        response.raise_for_status()
        data = response.json()

        # Check for error response
        if isinstance(data, list) and len(data) > 0 and "error" in data[0]:
            raise BridgeConnectionError(f"Invalid credentials: {data[0]['error'].get('description')}")

        return True

    except requests.RequestException as e:
        raise BridgeConnectionError(f"Failed to verify connection: {e}")


def list_entertainment_areas(
    bridge_ip: str, username: str
) -> dict[str, dict]:
    """
    List entertainment areas configured on the bridge.

    Args:
        bridge_ip: IP address of the bridge
        username: API username/token

    Returns:
        Dict mapping area_id -> {"name": str, "num_lights": int}

    Raises:
        BridgeConnectionError: If the request fails
    """
    url = f"https://{bridge_ip}/clip/v2/resource/entertainment_configuration"
    headers = {
        "hue-application-key": username,
    }

    try:
        response = requests.get(url, headers=headers, timeout=5, verify=False)
        response.raise_for_status()
        data = response.json()

        items = data.get("data", [])
        if not isinstance(items, list):
            return {}

        areas: dict[str, dict] = {}
        for item in items:
            area_id = item.get("id", "")
            metadata = item.get("metadata", {})
            name = metadata.get("name", f"Area {area_id[:8]}")
            num_lights = len(item.get("channels", []))
            areas[area_id] = {
                "name": name,
                "num_lights": num_lights,
            }

        return areas

    except requests.RequestException as e:
        raise BridgeConnectionError(f"Failed to list entertainment areas: {e}")
