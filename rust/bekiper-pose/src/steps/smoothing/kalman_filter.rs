fn observations_for_axis(
    x_values: &[Option<f64>],
    y_values: &[Option<f64>],
    likelihoods: &[Option<f64>],
    use_x: bool,
) -> Vec<Option<(f64, f64)>> {
    x_values
        .iter()
        .zip(y_values)
        .zip(likelihoods)
        .map(|((x, y), likelihood)| {
            let confidence = likelihood.unwrap_or(0.0);
            match (x, y) {
                (Some(x), Some(y)) if confidence > 0.0 => {
                    Some((if use_x { *x } else { *y }, confidence))
                }
                _ => None,
            }
        })
        .collect()
}

fn smooth_axis(
    observations: &[Option<(f64, f64)>],
    process_noise: f64,
    measurement_noise: f64,
    min_likelihood: f64,
    max_gap_frames: usize,
) -> Vec<Option<f64>> {
    if observations.is_empty() {
        return Vec::new();
    }

    let q = process_noise.max(1e-9).powi(2);
    let base_r = measurement_noise.max(1e-9).powi(2);
    let min_likelihood = min_likelihood.clamp(1e-9, 1.0);
    let transition = [[1.0, 1.0], [0.0, 1.0]];
    let process = [[q * 0.25, q * 0.5], [q * 0.5, q]];

    let n = observations.len();
    let mut predicted_state = vec![None; n];
    let mut predicted_cov = vec![None; n];
    let mut filtered_state = vec![None; n];
    let mut filtered_cov = vec![None; n];

    let mut state: Option<Vec2> = None;
    let mut cov: Option<Mat2> = None;

    for (idx, observation) in observations.iter().enumerate() {
        let (pred_state, pred_cov) = match (state, cov) {
            (Some(state), Some(cov)) => (
                mat_vec_mul(transition, state),
                mat_add(
                    mat_mul(mat_mul(transition, cov), mat_transpose(transition)),
                    process,
                ),
            ),
            _ => match observation {
                Some((value, confidence)) => {
                    let r = measurement_variance(base_r, *confidence, min_likelihood);
                    ([*value, 0.0], [[r, 0.0], [0.0, r * 10.0]])
                }
                None => continue,
            },
        };

        predicted_state[idx] = Some(pred_state);
        predicted_cov[idx] = Some(pred_cov);

        let (next_state, next_cov) = match observation {
            Some((value, confidence)) => update_axis(
                pred_state,
                pred_cov,
                *value,
                measurement_variance(base_r, *confidence, min_likelihood),
            ),
            None => (pred_state, pred_cov),
        };

        filtered_state[idx] = Some(next_state);
        filtered_cov[idx] = Some(next_cov);
        state = Some(next_state);
        cov = Some(next_cov);
    }

    let mut smoothed_state = filtered_state.clone();
    let mut smoothed_cov = filtered_cov.clone();
    for idx in (0..n.saturating_sub(1)).rev() {
        let (
            Some(f_state),
            Some(f_cov),
            Some(next_pred_state),
            Some(next_pred_cov),
            Some(next_smooth_state),
            Some(next_smooth_cov),
        ) = (
            filtered_state[idx],
            filtered_cov[idx],
            predicted_state[idx + 1],
            predicted_cov[idx + 1],
            smoothed_state[idx + 1],
            smoothed_cov[idx + 1],
        )
        else {
            continue;
        };
        let gain = mat_mul(
            mat_mul(f_cov, mat_transpose(transition)),
            mat_inv(next_pred_cov),
        );
        let state_delta = vec_sub(next_smooth_state, next_pred_state);
        let cov_delta = mat_sub(next_smooth_cov, next_pred_cov);
        smoothed_state[idx] = Some(vec_add(f_state, mat_vec_mul(gain, state_delta)));
        smoothed_cov[idx] = Some(mat_add(
            f_cov,
            mat_mul(mat_mul(gain, cov_delta), mat_transpose(gain)),
        ));
    }

    let mut output: Vec<Option<f64>> = smoothed_state
        .into_iter()
        .map(|state| state.map(|state| state[0]))
        .collect();
    preserve_untrusted_gaps(observations, &mut output, max_gap_frames);
    output
}

fn update_axis(state: Vec2, cov: Mat2, value: f64, measurement_variance: f64) -> (Vec2, Mat2) {
    let residual = value - state[0];
    let innovation = cov[0][0] + measurement_variance;
    let gain = [cov[0][0] / innovation, cov[1][0] / innovation];
    let next_state = [state[0] + gain[0] * residual, state[1] + gain[1] * residual];
    let identity_minus_kh = [[1.0 - gain[0], 0.0], [-gain[1], 1.0]];
    (next_state, mat_mul(identity_minus_kh, cov))
}

fn measurement_variance(base_r: f64, confidence: f64, min_likelihood: f64) -> f64 {
    base_r / confidence.clamp(min_likelihood, 1.0)
}

fn preserve_untrusted_gaps(
    observations: &[Option<(f64, f64)>],
    output: &mut [Option<f64>],
    max_gap_frames: usize,
) {
    let mut start: Option<usize> = None;
    for idx in 0..=observations.len() {
        let missing = idx < observations.len() && observations[idx].is_none();
        if missing && start.is_none() {
            start = Some(idx);
        }
        if missing {
            continue;
        }
        let Some(s) = start.take() else {
            continue;
        };
        let len = idx - s;
        let touches_edge = s == 0 || idx == observations.len();
        if touches_edge || len > max_gap_frames {
            for value in &mut output[s..idx] {
                *value = None;
            }
        }
    }
}
