use anyhow::{Result, bail};
use image::{DynamicImage, imageops::FilterType};
use std::path::Path;

pub(crate) fn save_image(
    snapped_img: DynamicImage,
    save_dir: &Path,
    timestamp: String,
    width: Option<u32>,
    height: Option<u32>,
    quality: u8,
) -> Result<()> {
    let save_img_buf = match (width, height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => {
            tracing::info!("Resizing image to {}x{}", w, h);
            snapped_img.resize_exact(w, h, FilterType::CatmullRom)
        }
        _ => snapped_img,
    };

    let Ok(encoder) = webp::Encoder::from_image(&save_img_buf) else {
        bail!("Failed to create webp encoder from image");
    };
    let webp = encoder.encode(quality as f32);
    let img_path = save_dir.join(timestamp).with_extension("webp");
    std::fs::write(&img_path, &*webp)?;
    tracing::info!("Saved image to {}", img_path.display());
    Ok(())
}
