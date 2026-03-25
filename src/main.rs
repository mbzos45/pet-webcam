use anyhow::{Result, bail};
use argh::FromArgs;
use chrono::prelude::*;
use std::{collections::HashMap, path::PathBuf};

use camera::capture_image;
use detector::{YoloV8Class, detect_yolov8};

mod camera;
mod detector;
mod storage;

#[macro_use]
extern crate num_derive;

#[derive(FromArgs, Debug)]
#[argh(description = "Capture webcam image and detect objects using YOLOv8 ONNX model")]
struct Args {
    /// camera id
    #[argh(option, short = 'i', default = "0")]
    camera_id: usize,
    /// yolov8 onnx path
    #[argh(option, short = 'm')]
    onnx_path: PathBuf,
    /// timeformat for saving image
    #[argh(option, short = 't', default = "String::from(\"%Y-%m-%d_%H-%M\")")]
    time_format: String,
    /// image size width
    #[argh(option, short = 'w')]
    width: Option<u32>,
    /// image compression quality (0-100)
    #[argh(option, short = 'q', default = "80")]
    quality: u8,

    /// image size height
    #[argh(option, short = 'h')]
    height: Option<u32>,
    /// save image local path or smb path
    #[argh(option, short = 'p')]
    img_path: String,
}


#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[derive(Debug, Clone, Copy)]
struct DetectedItem {
    bounding_box: BoundingBox,
    class: YoloV8Class,
    probability: f32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().without_time().init();
    // Parse command line arguments
    let args: Args = argh::from_env();
    if !args.onnx_path.exists() || !args.onnx_path.is_file() {
        bail!(
            "yolov8 onnx file does not exist: {}",
            args.onnx_path.display()
        );
    }
    let save_dir = PathBuf::from(&args.img_path);
    if !save_dir.exists() || !save_dir.is_dir() {
        bail!(
            "Image save path does not exist or is not a directory: {}",
            save_dir.display()
        );
    }

    let snapped_img = capture_image(args.camera_id)?;
    let timestamp = Local::now().format(&args.time_format);
    let detected_objs = detect_yolov8(&snapped_img, &args.onnx_path)?;
    if detected_objs.is_empty() {
        tracing::info!("No objects detected");
    } else {
        let mut objs_map: HashMap<YoloV8Class, usize> = HashMap::new();
        detected_objs.iter().for_each(|objs| {
            *objs_map.entry(objs.class).or_insert(0) += 1;
        });
        let mut detected_text = String::new();
        for (class, count) in &objs_map {
            let add_text = format!(" {}: {}", class.as_ref(), count);
            detected_text.push_str(&add_text);
        }
        tracing::info!("Detected objects: {}", detected_text);
        if objs_map.contains_key(&YoloV8Class::Person) {
            tracing::info!("Person detected, not saving image");
            return Ok(());
        }
    }
    storage::save_image(
        snapped_img,
        &save_dir,
        timestamp.to_string(),
        args.width,
        args.height,
        args.quality,
    )?;
    Ok(())
}
