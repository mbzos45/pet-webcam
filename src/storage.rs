use anyhow::{Result, bail};
use image::{DynamicImage, imageops::FilterType};
use std::path::Path;

pub(crate) fn save_image<P>(
    image: DynamicImage,
    save_dir: &Path,
    timestamp: P,
    width: Option<u32>,
    height: Option<u32>,
    quality: f32,
) -> Result<()>
where
    P: AsRef<Path>,
{
    let image = match (width, height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => {
            tracing::info!("Resizing image to {}x{}", w, h);
            image.resize_exact(w, h, FilterType::CatmullRom)
        }
        _ => image,
    };

    let Ok(encoder) = webp::Encoder::from_image(&image) else {
        bail!("Failed to create webp encoder from image");
    };
    let webp = encoder.encode(quality);
    let img_path = save_dir.join(timestamp).with_extension("webp");
    std::fs::write(&img_path, &*webp)?;
    tracing::info!("Saved image to {}", img_path.display());
    Ok(())
}
