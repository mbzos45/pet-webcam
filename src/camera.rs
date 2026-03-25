use anyhow::{Context, Result, bail, ensure};
use image::DynamicImage;
use nokhwa::{
    Camera, native_api_backend, nokhwa_initialize,
    pixel_format::RgbFormat,
    query,
    utils::{RequestedFormat, RequestedFormatType},
};

pub(crate) fn capture_image(camera_id: usize) -> Result<DynamicImage> {
    nokhwa_initialize(|_| {});
    let backend = native_api_backend().context("Failed to get native API backend")?;
    let cam_info = {
        let mut devices = query(backend)?;
        if devices.is_empty() {
            bail!("No camera devices found");
        }
        devices.sort_by(|a, b| {
            let idx_a = a.index().as_index().expect("Camera index is not usize");
            let idx_b = b.index().as_index().expect("Camera index is not usize");
            idx_a.cmp(&idx_b)
        });
        ensure!(
            camera_id < devices.len(),
            "There are {} available devices",
            devices.len()
        );
        devices.swap_remove(camera_id)
    };
    let req_format =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestResolution);
    let mut camera = Camera::with_backend(cam_info.index().clone(), req_format, backend)?;
    camera.open_stream()?;
    for _ in 0..5 {
        camera.frame()?;
    }
    let frame = camera.frame()?;
    camera.stop_stream()?;
    let img_buf = frame.decode_image::<RgbFormat>()?;
    let dyn_img = DynamicImage::ImageRgb8(img_buf);
    tracing::info!("Captured image from {}", camera.info().human_name());
    Ok(dyn_img)
}
