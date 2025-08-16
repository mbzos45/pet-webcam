use anyhow::{Context, Result, bail, ensure};
use argh::FromArgs;
use chrono::prelude::*;
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::{Array, Axis, s};
use nokhwa::{
    Camera, native_api_backend, nokhwa_initialize,
    pixel_format::RgbFormat,
    query,
    utils::{RequestedFormat, RequestedFormatType},
};
use num_traits::FromPrimitive;
use ort::{
    inputs,
    logging::LogLevel,
    session::{Session, SessionOutputs},
    value::TensorRef,
};
use std::{cmp::PartialEq, collections::HashMap, path::PathBuf};
use strum_macros::{AsRefStr, EnumIs, EnumString};

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
    /// image size height
    #[argh(option, short = 'h')]
    height: Option<u32>,
    /// save image local path or smb path
    #[argh(option, short = 'p')]
    img_path: String,
}

#[derive(
    Debug,
    Clone,
    Copy,
    Eq,
    Hash,
    PartialEq,
    FromPrimitive,
    ToPrimitive,
    EnumString,
    AsRefStr,
    EnumIs,
)]
enum YoloV8Class {
    Person,
    Bicycle,
    Car,
    Motorcycle,
    Airplane,
    Bus,
    Train,
    Truck,
    Boat,
    TrafficLight,
    FireHydrant,
    StopSign,
    ParkingMeter,
    Bench,
    Bird,
    Cat,
    Dog,
    Horse,
    Sheep,
    Cow,
    Elephant,
    Bear,
    Zebra,
    Giraffe,
    Backpack,
    Umbrella,
    Handbag,
    Tie,
    Suitcase,
    Frisbee,
    Skis,
    Snowboard,
    SportsBall,
    Kite,
    BaseballBat,
    BaseballGlove,
    Skateboard,
    Surfboard,
    TennisRacket,
    Bottle,
    WineGlass,
    Cup,
    Fork,
    Knife,
    Spoon,
    Bowl,
    Banana,
    Apple,
    Sandwich,
    Orange,
    Broccoli,
    Carrot,
    HotDog,
    Pizza,
    Donut,
    Cake,
    Chair,
    Couch,
    PottedPlant,
    Bed,
    DiningTable,
    Toilet,
    Tv,
    Laptop,
    Mouse,
    Remote,
    Keyboard,
    CellPhone,
    Microwave,
    Oven,
    Toaster,
    Sink,
    Refrigerator,
    Book,
    Clock,
    Vase,
    Scissors,
    TeddyBear,
    HairDrier,
    Toothbrush,
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
    let detected_objs = detect_yolov8(&snapped_img, args.onnx_path)?;
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
    // Resize image if width or height is specified
    let save_img_buf = match (args.width, args.height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => {
            tracing::info!("Resizing image to {}x{}", w, h);
            snapped_img.resize_exact(w, h, FilterType::CatmullRom)
        }
        _ => snapped_img,
    };
    let Ok(encoder) = webp::Encoder::from_image(&save_img_buf) else {
        bail!("Failed to create webp encoder from image");
    };
    let webp = encoder.encode(90f32);
    let img_path = save_dir.join(timestamp.to_string()).with_extension("webp");
    std::fs::write(&img_path, &*webp)?;
    tracing::info!("Saved image to {}", img_path.display());
    Ok(())
}

fn capture_image(camera_id: usize) -> Result<DynamicImage> {
    nokhwa_initialize(|_| {});
    let backend = native_api_backend().context("Failed to get native API backend")?;
    let cam_info = {
        let mut devices = query(backend)?;
        devices.sort_by(|a, b| {
            let idx_a = a.index().as_index().unwrap();
            let idx_b = b.index().as_index().unwrap();
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
    let frame = camera.frame()?;
    camera.stop_stream()?;
    let img_buf = frame.decode_image::<RgbFormat>()?;
    let dyn_img = DynamicImage::ImageRgb8(img_buf);
    tracing::info!("Captured image from {}", camera.info().human_name());
    Ok(dyn_img)
}

fn detect_yolov8<P: AsRef<std::path::Path>>(
    original_img: &DynamicImage,
    onnx_path: P,
) -> Result<Vec<DetectedItem>> {
    let (img_width, img_height) = (original_img.width(), original_img.height());
    let img = original_img.resize_exact(640, 640, FilterType::CatmullRom);
    let mut input = Array::zeros((1, 3, 640, 640));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }
    let mut model = Session::builder()?
        .with_log_level(LogLevel::Fatal)?
        .commit_from_file(onnx_path)?;
    let outputs: SessionOutputs =
        model.run(inputs!["images" => TensorRef::from_array_view(&input)?])?;
    let output = outputs["output0"]
        .try_extract_array::<f32>()?
        .t()
        .into_owned();
    let mut boxes = Vec::new();
    let output = output.slice(s![.., .., 0]);
    for row in output.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().copied().collect();
        let (class_id, prob) = row
            .iter()
            // skip bounding box coordinates
            .skip(4)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .context("Failed to find class with maximum probability")?;
        if prob < 0.5 {
            continue;
        }
        let Some(class_enum) = YoloV8Class::from_usize(class_id) else {
            tracing::debug!("Class ID {} is not a valid YoloV8 class", class_id);
            continue;
        };
        let label = class_enum;
        let xc = row[0] / 640. * (img_width as f32);
        let yc = row[1] / 640. * (img_height as f32);
        let w = row[2] / 640. * (img_width as f32);
        let h = row[3] / 640. * (img_height as f32);
        let bounding_box = BoundingBox {
            x1: xc - w / 2.,
            y1: yc - h / 2.,
            x2: xc + w / 2.,
            y2: yc + h / 2.,
        };
        let detected_item = DetectedItem {
            bounding_box,
            class: label,
            probability: prob,
        };
        boxes.push(detected_item);
    }
    boxes.sort_by(|box1, box2| box2.probability.total_cmp(&box1.probability));
    let mut result = Vec::new();

    while !boxes.is_empty() {
        result.push(boxes[0]);
        boxes = boxes
            .iter()
            .filter(|box1| {
                intersection(&boxes[0].bounding_box, &box1.bounding_box)
                    / union(&boxes[0].bounding_box, &box1.bounding_box)
                    < 0.7
            })
            .copied()
            .collect();
    }
    Ok(result)
}

fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
        - intersection(box1, box2)
}
