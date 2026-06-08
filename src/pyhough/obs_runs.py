from dataclasses import dataclass
from pyhough import time_conversions


@dataclass(frozen=True)
class ObsRun:
    name: str
    utc_start: str
    utc_end: str

    @property
    def gps_start(self):
        return time_conversions.utc2gps(self.utc_start)

    @property
    def gps_end(self):
        return time_conversions.utc2gps(self.utc_end)


OBS_RUNS = {
    "O3": ObsRun(
        "O3",
        "2019-04-01T15:00:00",
        "2020-03-27T17:00:00",
    ),
    "O4a": ObsRun(
        "O4a",
        "2023-05-24T22:00:00",
        "2024-01-16T16:00:00",
    ),
    "O4b": ObsRun(
        "O4b",
        "2024-04-10T15:00:00",
        "2025-01-28T00:00:00",
    ),
    "O4ab": ObsRun(
        "O4ab",
        "2023-05-24T22:00:00",
        "2025-01-28T00:00:00",
    ),
}


def get_obs_run(obs_run):
    try:
        return OBS_RUNS[obs_run]
    except KeyError:
        raise ValueError(
            f"Unknown observing run '{obs_run}'. "
            f"Valid choices are {list(OBS_RUNS.keys())}"
        )


def check_science_segments_match_run(sci_times, obs_run):
    """
    Verify that the science-segment file lies entirely
    within the requested observing run.
    """

    run = get_obs_run(obs_run)

    sci_start = sci_times[0][0]
    sci_end = sci_times[-1][1]

    if sci_start < run.gps_start:
        raise ValueError(
            f"Science segments start before {obs_run}: "
            f"{sci_start} < {run.gps_start} "
            f"({run.utc_start})"
        )

    if sci_end > run.gps_end:
        raise ValueError(
            f"Science segments end after {obs_run}: "
            f"{sci_end} > {run.gps_end} "
            f"({run.utc_end})"
        )

    print(
        f"Science segments span GPS [{sci_start}, {sci_end}] "
        f"and are fully contained within {obs_run} "
        f"({run.utc_start} -- {run.utc_end})"
    )