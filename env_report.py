"""
Compute-environment report  --  CNSNS revision, Reviewer 2 point 7.
===================================================================

THE OBJECTION. Table 1 states only "CPU (single machine); code is GPU-compatible",
yet wall-clock timings carry a central part of the paper's practicality argument
("6-10 minutes on CPU", "~23 s" for the hybrid). Without the machine spec those
timings cannot be checked or compared against future work.

This writes results/environment.json and prints a table body for the manuscript.

Run:  python env_report.py
"""

import platform
import subprocess
import sys

from resultio import dump


def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                             shell=isinstance(cmd, str))
        return out.stdout.strip()
    except Exception:
        return ""


def cpu_model():
    """Best-effort CPU model string across platforms."""
    if sys.platform == "win32":
        # PowerShell CIM is more reliable than the deprecated `wmic`.
        s = _run(['powershell', '-NoProfile', '-Command',
                  '(Get-CimInstance Win32_Processor).Name'])
        if s:
            return " ".join(s.split())
    elif sys.platform == "darwin":
        s = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if s:
            return s
    else:
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def ram_gb():
    if sys.platform == "win32":
        s = _run(['powershell', '-NoProfile', '-Command',
                  '(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory'])
        try:
            return round(int(s) / 1024 ** 3, 1)
        except Exception:
            return None
    try:
        import os
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                     / 1024 ** 3, 1)
    except Exception:
        return None


def collect():
    import numpy
    import scipy
    import torch

    info = dict(
        cpu_model=cpu_model(),
        cpu_count_logical=__import__("os").cpu_count(),
        cpu_count_physical=torch.get_num_threads(),  # see note below
        ram_gb=ram_gb(),
        os=platform.platform(),
        python=sys.version.split()[0],
        torch=torch.__version__,
        torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        cuda_available=torch.cuda.is_available(),
        cuda_device=(torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        blas=_blas_info(numpy),
        float_precision="float64 (PyTorch default overridden in pinn.py)",
    )
    # torch.get_num_threads() is the thread budget, not a core count; correct
    # the field name rather than reporting a misleading "physical cores".
    info.pop("cpu_count_physical")
    return info


def _blas_info(numpy):
    """Which BLAS numpy is linked against -- matters for reproducing timings."""
    try:
        cfg = numpy.__config__.show(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        return f"{blas.get('name', '?')} {blas.get('version', '')}".strip()
    except Exception:
        try:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                numpy.__config__.show()
            for line in buf.getvalue().splitlines():
                if "name" in line.lower() and any(
                        b in line.lower() for b in ("openblas", "mkl", "blis")):
                    return line.strip()
        except Exception:
            pass
        return "unknown"


LABELS = [
    ("cpu_model", "CPU"),
    ("cpu_count_logical", "Logical cores"),
    ("torch_threads", "PyTorch intra-op threads"),
    ("ram_gb", "RAM (GB)"),
    ("os", "Operating system"),
    ("python", "Python"),
    ("torch", "PyTorch"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("blas", "BLAS backend"),
    ("cuda_available", "CUDA available"),
    ("float_precision", "Floating-point precision"),
]


def latex_rows(info):
    """Rows for the Hardware block of Table 1."""
    out = []
    for key, label in LABELS:
        v = info.get(key)
        if v is None:
            continue
        v = str(v).replace("_", r"\_").replace("&", r"\&")
        out.append(f"{label} & \\texttt{{{v}}} \\\\")
    return "\n".join(out)


if __name__ == "__main__":
    info = collect()
    width = max(len(l) for _, l in LABELS) + 2
    print("Compute environment")
    print("-" * 70)
    for key, label in LABELS:
        if info.get(key) is not None:
            print(f"  {label:<{width}} {info[key]}")
    print("\n--- LaTeX rows for Table 1 (Hardware block) ---\n")
    print(latex_rows(info))
    dump("environment", info)
