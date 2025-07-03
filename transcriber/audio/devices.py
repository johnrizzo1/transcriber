"""
Audio device management.
"""

import logging
from typing import Optional

import sounddevice as sd

logger = logging.getLogger(__name__)


def list_audio_devices() -> list[dict]:
    """
    List all available audio devices.
    
    Returns:
        List of device information dictionaries
    """
    try:
        devices = sd.query_devices()
        device_list = []
        
        for i, device in enumerate(devices):
            device_info = {
                'id': i,
                'name': device['name'],
                'max_input_channels': device['max_input_channels'],
                'max_output_channels': device['max_output_channels'], 
                'default_samplerate': device['default_samplerate'],
                'is_input': device['max_input_channels'] > 0,
                'is_output': device['max_output_channels'] > 0,
            }
            device_list.append(device_info)
            
        return device_list
        
    except Exception as e:
        logger.error(f"Failed to list audio devices: {e}")
        return []


def get_default_devices() -> tuple[Optional[int], Optional[int]]:
    """
    Get default input and output device IDs.
    
    Returns:
        Tuple of (input_device_id, output_device_id)
    """
    try:
        input_device = sd.default.device[0] if sd.default.device[0] != -1 else None
        output_device = sd.default.device[1] if sd.default.device[1] != -1 else None
        return input_device, output_device
    except Exception as e:
        logger.error(f"Failed to get default devices: {e}")
        return None, None


def find_device_by_name(name: str, input_only: bool = True) -> Optional[int]:
    """
    Find device ID by partial name match.
    
    Args:
        name: Partial device name to search for
        input_only: Whether to search only input devices
        
    Returns:
        Device ID if found, None otherwise
    """
    devices = list_audio_devices()
    
    for device in devices:
        if input_only and not device['is_input']:
            continue
            
        if name.lower() in device['name'].lower():
            return device['id']
            
    return None


def get_device_info(device_id: int) -> Optional[dict]:
    """
    Get detailed information about a specific device.
    
    Args:
        device_id: Device ID to query
        
    Returns:
        Device information dictionary or None if not found
    """
    try:
        device = sd.query_devices(device_id)
        return {
            'id': device_id,
            'name': device['name'],
            'max_input_channels': device['max_input_channels'],
            'max_output_channels': device['max_output_channels'],
            'default_samplerate': device['default_samplerate'],
            'is_input': device['max_input_channels'] > 0,
            'is_output': device['max_output_channels'] > 0,
        }
    except Exception as e:
        logger.error(f"Failed to get device {device_id} info: {e}")
        return None