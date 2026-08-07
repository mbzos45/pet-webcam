use anyhow::{Result, bail};
use argh::FromArgs;
use chrono::prelude::*;
use std::{collections::HashMap, env, path::PathBuf};

use camera::capture_image;
use detector::{YoloClass, detect_yolo};
use storage::StorageMode;
use storage::smb::{SmbConfig, SmbSession};

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
    /// yolo onnx path
    #[argh(option, short = 'm')]
    onnx_path: Option<PathBuf>,
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
    /// save image local path (required unless --smb is set)
    #[argh(option, short = 'p')]
    img_path: Option<String>,
    /// save the image to an SMB share instead of local disk
    #[argh(switch, short = 's')]
    smb: bool,
    /// smb username (or SMB_USERNAME env var)
    #[argh(option)]
    smb_username: Option<String>,
    /// smb password (or SMB_PASSWORD env var; prefer the env var / .env over this flag)
    #[argh(option)]
    smb_password: Option<String>,
    /// smb destination, e.g. \\host\share\subdir (or SMB_DEST env var)
    #[argh(option)]
    smb_dest: Option<String>,
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
    class: YoloClass,
    probability: f32,
}

fn env_flag_set(name: &str) -> bool {
    env::var(name).is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true"))
}

/// Resolves the active storage mode from CLI args and environment variables
/// (CLI takes precedence), validating required settings before any camera/SMB work starts.
fn resolve_storage_mode(args: &Args) -> Result<StorageMode> {
    let smb_active = args.smb || env_flag_set("SMB_ENABLED");

    if smb_active {
        let username = args
            .smb_username
            .clone()
            .or_else(|| env::var("SMB_USERNAME").ok());
        let password = args
            .smb_password
            .clone()
            .or_else(|| env::var("SMB_PASSWORD").ok());
        let dest = args.smb_dest.clone().or_else(|| env::var("SMB_DEST").ok());

        let mut missing = Vec::new();
        if username.is_none() {
            missing.push("username (--smb-username / SMB_USERNAME)");
        }
        if password.is_none() {
            missing.push("password (--smb-password / SMB_PASSWORD)");
        }
        if dest.is_none() {
            missing.push("destination (--smb-dest / SMB_DEST)");
        }
        if !missing.is_empty() {
            bail!("Missing required SMB setting(s): {}", missing.join(", "));
        }

        Ok(StorageMode::Smb(SmbConfig {
            username: username.unwrap(),
            password: password.unwrap(),
            dest: dest.unwrap(),
        }))
    } else {
        let Some(img_path) = &args.img_path else {
            bail!("--img-path is required when --smb is not set");
        };
        let save_dir = PathBuf::from(img_path);
        if !save_dir.exists() || !save_dir.is_dir() {
            bail!(
                "Image save path does not exist or is not a directory: {}",
                save_dir.display()
            );
        }
        Ok(StorageMode::Local(save_dir))
    }
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt().without_time().init();
    // Parse command line arguments
    let args: Args = argh::from_env();
    if let Some(onnx_path) = &args.onnx_path {
        if !onnx_path.exists() || !onnx_path.is_file() {
            bail!("yolo onnx file does not exist: {}", onnx_path.display());
        }
    }

    // Resolve and validate the storage destination, and for SMB mode establish the
    // connection now, before capture/inference run (see specs/smb-image-storage/spec.md).
    let storage_mode = resolve_storage_mode(&args)?;
    let smb_session = match &storage_mode {
        StorageMode::Smb(config) => Some(SmbSession::preflight(config)?),
        StorageMode::Local(_) => None,
    };

    let snapped_img = capture_image(args.camera_id)?;
    let timestamp = Local::now().format(&args.time_format);
    if let Some(onnx_path) = &args.onnx_path {
        let detected_objs = detect_yolo(&snapped_img, onnx_path)?;
        if detected_objs.is_empty() {
            tracing::info!("No objects detected");
        } else {
            let mut objs_map: HashMap<YoloClass, usize> = HashMap::new();
            detected_objs.iter().for_each(|objs| {
                *objs_map.entry(objs.class).or_insert(0) += 1;
            });
            let mut detected_text = String::new();
            for (class, count) in &objs_map {
                let add_text = format!(" {}: {}", class.as_ref(), count);
                detected_text.push_str(&add_text);
            }
            tracing::info!("Detected objects: {}", detected_text);
            if objs_map.contains_key(&YoloClass::Person) {
                tracing::info!("Person detected, not saving image");
                return Ok(());
            }
        }
    } else {
        tracing::info!("onnx_path is not specified; skipping inference");
    }

    match storage_mode {
        StorageMode::Local(save_dir) => storage::save_image(
            snapped_img,
            &save_dir,
            timestamp.to_string(),
            args.width,
            args.height,
            args.quality as f32,
        )?,
        StorageMode::Smb(_) => {
            let session = smb_session.expect("SMB session was established during pre-flight");
            session.save(
                snapped_img,
                timestamp.to_string(),
                args.width,
                args.height,
                args.quality as f32,
            )?;
        }
    }
    Ok(())
}