from __future__ import annotations

from dataclasses import dataclass
import os
import platform

try:
    import psutil
except Exception:  # pragma: no cover - optional runtime fallback
    psutil = None


@dataclass(frozen=True)
class HardwareProfile:
    cpu_threads: int
    ram_gb: float | None
    system: str
    recommended_draws: int
    recommended_mcmc_chains: int
    recommended_mcmc_cores: int
    label: str


def detect_hardware() -> HardwareProfile:
    cpu = max(int(os.cpu_count() or 1), 1)
    ram_gb = None
    if psutil is not None:
        try:
            ram_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            ram_gb = None

    if ram_gb is not None and ram_gb >= 24 and cpu >= 8:
        label = "Forte"
        draws = 100_000
        chains = 4
    elif ram_gb is not None and ram_gb >= 12 and cpu >= 6:
        label = "Intermediário"
        draws = 50_000
        chains = 4
    else:
        label = "Conservador"
        draws = 20_000
        chains = 2

    cores = max(1, min(chains, cpu - 1 if cpu > 1 else 1))
    return HardwareProfile(
        cpu_threads=cpu,
        ram_gb=ram_gb,
        system=platform.system(),
        recommended_draws=draws,
        recommended_mcmc_chains=chains,
        recommended_mcmc_cores=cores,
        label=label,
    )
