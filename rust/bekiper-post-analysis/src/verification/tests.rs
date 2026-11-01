use super::*;

#[test]
fn png_verification_decodes_pixels_instead_of_trusting_the_header() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("figure.png");
    let mut encoder = png::Encoder::new(File::create(&path).unwrap(), 1, 1);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_pixel_dims(Some(png::PixelDimensions {
        xppu: 11811,
        yppu: 11811,
        unit: png::Unit::Meter,
    }));
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&[1, 2, 3])
        .unwrap();
    verify_png(&path, 300).unwrap();
    assert!(verify_png(&path, 150).is_err());
    let bytes = std::fs::read(&path).unwrap();
    let idat = bytes.windows(4).position(|b| b == b"IDAT").unwrap();
    std::fs::write(&path, &bytes[..idat + 4]).unwrap();
    assert!(verify_png(&path, 300).is_err());
}
