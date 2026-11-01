fn vec_add(a: Vec2, b: Vec2) -> Vec2 {
    [a[0] + b[0], a[1] + b[1]]
}

fn vec_sub(a: Vec2, b: Vec2) -> Vec2 {
    [a[0] - b[0], a[1] - b[1]]
}

fn mat_add(a: Mat2, b: Mat2) -> Mat2 {
    [
        [a[0][0] + b[0][0], a[0][1] + b[0][1]],
        [a[1][0] + b[1][0], a[1][1] + b[1][1]],
    ]
}

fn mat_sub(a: Mat2, b: Mat2) -> Mat2 {
    [
        [a[0][0] - b[0][0], a[0][1] - b[0][1]],
        [a[1][0] - b[1][0], a[1][1] - b[1][1]],
    ]
}

fn mat_mul(a: Mat2, b: Mat2) -> Mat2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn mat_vec_mul(a: Mat2, b: Vec2) -> Vec2 {
    [
        a[0][0] * b[0] + a[0][1] * b[1],
        a[1][0] * b[0] + a[1][1] * b[1],
    ]
}

fn mat_transpose(a: Mat2) -> Mat2 {
    [[a[0][0], a[1][0]], [a[0][1], a[1][1]]]
}

fn mat_inv(a: Mat2) -> Mat2 {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-12 {
        return [[0.0, 0.0], [0.0, 0.0]];
    }
    let inv_det = 1.0 / det;
    [
        [a[1][1] * inv_det, -a[0][1] * inv_det],
        [-a[1][0] * inv_det, a[0][0] * inv_det],
    ]
}
