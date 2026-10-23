"""Process-local CPU selection and read-only Windows power evidence."""
import ctypes
import os
import re
import subprocess


def parse_cpu_list(text):
    values = [int(part) for part in text.split(",")]
    if not values or min(values) < 0 or len(set(values)) != len(values):
        raise ValueError("CPU list must contain distinct nonnegative processor IDs")
    return values


def windows_api():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return kernel


def current_cpu_list():
    if os.name == "nt":
        if (os.cpu_count() or 0) > 64:
            raise ValueError("This runner supports one Windows processor group only")
        kernel = windows_api()
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        return [i for i in range(64) if process.value & (1 << i)]
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return None


def apply_cpu_affinity(cpu_list=None):
    """Changes only this runner and inherited child processes, never the OS policy."""
    if cpu_list is None:
        return
    cpus = parse_cpu_list(cpu_list)
    if os.name == "nt":
        if max(cpus) >= 64 or (os.cpu_count() or 0) > 64:
            raise ValueError("Unsupported Windows processor group or CPU ID")
        kernel = windows_api()
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), sum(1 << c for c in cpus)):
            raise ctypes.WinError(ctypes.get_last_error())
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    else:
        raise RuntimeError("CPU selection is unsupported on this platform")
    if current_cpu_list() != sorted(cpus):
        raise RuntimeError("Observed process affinity does not match requested CPUs")


def capture_conditions():
    result = {"allowed_cpu_ids": current_cpu_list(),
              "scope": "process eligibility; not per-worker placement or isolated cores",
              "frequency_temperature": "not measured or fixed"}
    if os.name == "nt":
        class PowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", ctypes.c_uint32), ("BatteryFullLifeTime", ctypes.c_uint32)]
        status = PowerStatus()
        if windows_api().GetSystemPowerStatus(ctypes.byref(status)):
            result["ac_line_status"] = {0: "offline", 1: "online"}.get(status.ACLineStatus, "unknown")
        try:
            power = subprocess.check_output(["powercfg", "/getactivescheme"], timeout=8)
            match = re.search(rb"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", power)
            result["power_scheme_guid"] = match.group().decode("ascii").lower() if match else "unknown"
        except (OSError, subprocess.SubprocessError):
            result["power_scheme_guid"] = "unavailable"
    return result
