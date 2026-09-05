from quant_trafego.observability import RunTelemetry


def test_runtime_telemetry_tracks_stages_and_developer_snapshot():
    telemetry = RunTelemetry()
    telemetry.emit(
        "load",
        "stage_start",
        message="start",
        progress=0.01,
        rows=10,
    )
    telemetry.finish_stage(
        "load",
        message="done",
        rows=10,
    )
    telemetry.emit(
        "inference",
        "stage_start",
        message="running",
        progress=0.20,
    )

    table = telemetry.dataframe()
    summary = telemetry.stage_summary()
    snapshot = telemetry.developer_snapshot()

    assert len(table) == 3
    assert set(summary["stage"]) == {"load", "inference"}
    assert bool(
        summary.loc[
            summary["stage"] == "load",
            "finished",
        ].iloc[0]
    )
    assert snapshot["event_count"] == 3
    assert snapshot["bottleneck_stage_so_far"] in {
        "load",
        "inference",
    }
