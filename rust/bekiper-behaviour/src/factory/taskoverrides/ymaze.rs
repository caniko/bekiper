fn build_ymaze(
    perimeters: &[PerimeterSpec],
    params: TaskParams,
    overrides: Option<&dyn TaskOverrides>,
) -> Result<Box<dyn BehaviouralTask>> {
    let maze_spec = perimeters
        .iter()
        .find(|p| matches!(p, PerimeterSpec::RadialMaze { .. }));

    let maze = if let Some(PerimeterSpec::RadialMaze {
        label,
        center_vertices,
        arms,
    }) = maze_spec
    {
        Perimeter::new(
            label.clone(),
            RadialMaze::new(
                Polygon::new(center_vertices.clone()),
                arms.iter().map(|a| Polygon::new(a.clone())).collect(),
            ),
        )
    } else {
        return Err(BekiperError::Config(
            "y_maze/radial_arm task requires a RadialMaze perimeter".into(),
        ));
    };

    let mut task = YMaze::new(maze, params.fps);
    task.params = params;

    if let Some(ov) = overrides {
        task.apply_overrides(ov).map_err(BekiperError::Config)?;
    }

    task.validate_temporal_policy()
        .map_err(BekiperError::Config)?;
    Ok(Box::new(task))
}
