from .test_fixture_video import *


def _make_angle_manifest(
    *,
    n: int,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    angles_deg: np.ndarray,
    true_mask: np.ndarray | None = None,
    result_column: str = "olfaction_object_1",
    resolution: tuple[int, int] = (640, 480),
) -> InspectionManifest:
    coords_x = np.asarray(coords_x, dtype=float)
    coords_y = np.asarray(coords_y, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)
    if true_mask is None:
        true_mask = np.ones(n, dtype=bool)
    else:
        true_mask = np.asarray(true_mask, dtype=bool)

    df = pl.DataFrame(
        {
            "nose_x": coords_x,
            "nose_y": coords_y,
            result_column: true_mask,
            "snout_dx": np.cos(np.deg2rad(angles_deg)),
            "snout_dy": np.sin(np.deg2rad(angles_deg)),
        }
    )

    center_x = float(np.nanmax(coords_x) + 0.05)
    center_y = float(np.nanmedian(coords_y))
    return InspectionManifest(
        video=VideoMeta(
            fps=30.0,
            total_frames=n,
            resolution=resolution,
            meters_per_pixel=0.001,
            path=None,
        ),
        labels=["nose"],
        label_colors={"nose": "#1f77b4"},
        coordinate_columns={"nose": CoordinateColumnSpec(x="nose_x", y="nose_y")},
        perimeters=[
            PerimeterSpec(
                label="object_1",
                shape="circle",
                params={"center_x": center_x, "center_y": center_y, "radius": 0.02},
            )
        ],
        heuristics=[
            HeuristicMeta(
                name=result_column,
                result_column=result_column,
                true_frames=int(true_mask.sum()),
                total_frames=n,
                seconds=float(true_mask.sum()) / 30.0,
            )
        ],
        settings=InspectionSettings(
            minimum_seconds_tolerance=0.5,
            maximum_seconds_distraction=1.0 / 3.0,
            meters_per_pixel=0.001,
        ),
        scale_derivation=None,
        df=df,
        quadrants=[],
        feature_inspections=[],
    )


def _make_whisker_manifest(
    *,
    n: int,
    result_column: str = "whisker_obj",
    resolution: tuple[int, int] = (640, 480),
) -> InspectionManifest:
    truth_mask = np.ones(n, dtype=bool)
    df = pl.DataFrame(
        {
            "center_ear_x": np.full(n, 0.10),
            "center_ear_y": np.full(n, 0.10),
            "left_ear_x": np.full(n, 0.08),
            "left_ear_y": np.full(n, 0.10),
            "right_ear_x": np.full(n, 0.12),
            "right_ear_y": np.full(n, 0.10),
            "left_ear_dx": np.full(n, 1.0),
            "left_ear_dy": np.full(n, 0.0),
            "right_ear_dx": np.full(n, 0.0),
            "right_ear_dy": np.full(n, 1.0),
            result_column: truth_mask,
        }
    )
    return InspectionManifest(
        video=VideoMeta(
            fps=30.0,
            total_frames=n,
            resolution=resolution,
            meters_per_pixel=0.001,
            path=None,
        ),
        labels=["center_ear", "left_ear", "right_ear"],
        label_colors={"center_ear": "#1f77b4"},
        coordinate_columns={
            "center_ear": CoordinateColumnSpec(x="center_ear_x", y="center_ear_y"),
            "left_ear": CoordinateColumnSpec(x="left_ear_x", y="left_ear_y"),
            "right_ear": CoordinateColumnSpec(x="right_ear_x", y="right_ear_y"),
        },
        perimeters=[
            PerimeterSpec(
                label="object_1",
                shape="circle",
                params={"center_x": 0.40, "center_y": 0.10, "radius": 0.02},
            )
        ],
        heuristics=[
            HeuristicMeta(
                name=result_column,
                result_column=result_column,
                true_frames=int(truth_mask.sum()),
                total_frames=n,
                seconds=float(truth_mask.sum()) / 30.0,
            )
        ],
        settings=InspectionSettings(
            minimum_seconds_tolerance=0.5,
            maximum_seconds_distraction=1.0 / 3.0,
            meters_per_pixel=0.001,
        ),
        scale_derivation=None,
        df=df,
        quadrants=[],
        feature_inspections=[],
    )


__all__ = [name for name in globals() if not name.startswith("__")]
