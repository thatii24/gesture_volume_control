from pycaw.pycaw import AudioUtilities

def set_system_volume(level: float):
    """
    Sets the system master volume.
    :param level: A float between 0.0 and 1.0
    """
    if not 0.0 <= level <= 1.0:
        raise ValueError("Volume level must be between 0.0 and 1.0")

    device = AudioUtilities.GetSpeakers()
    device.EndpointVolume.SetMasterVolumeLevelScalar(level, None)
