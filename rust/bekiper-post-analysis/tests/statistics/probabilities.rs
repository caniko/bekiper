#[test]
fn t_probability_underflow_is_null_with_finite_log_probability() {
    let result = t_summary(100.0, 0.1, 1000, 0.95).unwrap();
    assert!(result["p_raw"].is_null());
    assert!(result["log_p_raw"].as_f64().unwrap().is_finite());
    // Independent 80-digit mpmath regularized-beta oracle at df=1000,t=1000.
    close(&result["log_p_raw"], &json!(-3458.0568098975394), 1e-12);
    assert_eq!(result["reason"], "p_underflow:see_log_p_raw");
    assert!(serde_json::to_string(&result).unwrap().contains("null"));
    let mut tests = vec![result, json!({"p_raw":0.01}), json!({"p_raw":null})];
    adjust_family(&mut tests, "holm", "underflow_family").unwrap();
    assert!(tests[0]["p_adjusted"].is_null());
    assert!(tests[0]["log_p_adjusted"].as_f64().unwrap().is_finite());
    assert_eq!(tests[0]["family_size"], 3);
}

#[test]
fn cauchy_tail_survives_intermediate_square_overflow() {
    let result = t_summary(1e200, 1.0, 1, 0.95).unwrap();
    // df=1 is Cauchy: P(two-sided) = 2/pi * atan(1/t).
    let expected = 2.0 / std::f64::consts::PI * (1e-200_f64).atan();
    let actual = result["p_raw"].as_f64().unwrap();
    assert!((actual / expected - 1.0).abs() < 1e-12);
    assert!(result["reason"].is_null());
}
