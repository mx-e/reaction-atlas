import torch
from typing import Dict, Union
from pathlib import Path
from copy import copy
import sys

# Try multiple candidate locations for MoreRed
for _candidate in [
    Path(__file__).parent.parent.parent / "MoreRed" / "src",    # dev: packages/MoreRed/src
    Path(__file__).parent.parent.parent / "MoreRed_src",        # dev: packages/MoreRed_src
    Path("/app/MoreRed/src"),                                    # container
]:
    if _candidate.is_dir():
        sys.path.append(str(_candidate))
        break

from morered.noise_schedules import PolynomialSchedule  # noqa: E402
from morered.processes import VPGaussianDDPM  # noqa: E402
from morered.sampling import DDPM, MoreRedJT, MoreRedITP  # noqa: E402
from schnetpack import properties as Props  # noqa: E402
from schnetpack import utils  # noqa: E402


class Diffuser:
    def __init__(
        self,
        noise_schedule: PolynomialSchedule,
        noise_key: str = "eps",
        invariant: bool = True,
        dtype: torch.dtype = torch.float64,
    ):
        self.diff_proc = VPGaussianDDPM(
            noise_schedule, noise_key=noise_key, invariant=invariant, dtype=dtype
        )


class TSDenoiser:
    def __init__(
        self,
        model: MoreRedJT,
        diffuser: Diffuser,
    ):
        self.model = model
        self.diffuser = diffuser


def get_sampler_from_model_type(model_type: str):
    if model_type == "DDPM":
        return DDPM
    elif model_type == "MoreRedJT":
        return MoreRedJT
    elif model_type == "MoreRedITP":
        return MoreRedITP
    else:
        raise ValueError(f"Model type not recognized: {model_type}")


def get_ts_model(
    exp_path: str,
    model_type: str,
    model_kwargs: Dict[str, Union[float, int, bool]],
    T: int = 1000,
    s: float = 1e-5,
    dtype: torch.dtype = torch.float64,
    variance_type: str = "lower_bound",
):
    noise_schedule = PolynomialSchedule(
        T=T, s=s, dtype=dtype, variance_type=variance_type
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffuser = Diffuser(noise_schedule)
    ts_denoiser = utils.load_model(Path(exp_path) / "ts_best_model", device=device)
    if device == "cuda":
        ts_denoiser.cuda()

    sampler = get_sampler_from_model_type(model_type)
    ts_model = sampler(diffuser.diff_proc, denoiser=ts_denoiser, **model_kwargs)
    return TSDenoiser(ts_model, diffuser)
