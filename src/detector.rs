use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::{Array, Axis, s};
use num_traits::FromPrimitive;
use ort::{
    ep::ExecutionProviderDispatch,
    inputs,
    logging::LogLevel,
    session::{Session, SessionOutputs},
    value::TensorRef,
};
use ort::session::builder::AutoDevicePolicy;
use strum_macros::{AsRefStr, EnumIs, EnumString};

use crate::{BoundingBox, DetectedItem};

macro_rules! build_execution_providers {
    () => {{
        #[allow(unused_mut)]
        let mut execution_providers: Vec<ExecutionProviderDispatch> = Vec::new();

        #[cfg(feature = "ep-tensorrt")]
        execution_providers
            .push(ort::ep::TensorRTExecutionProvider::default().build());

        #[cfg(feature = "ep-cuda")]
        execution_providers.push(ort::ep::CUDAExecutionProvider::default().build());

        #[cfg(feature = "ep-coreml")]
        execution_providers
            .push(ort::ep::CoreMLExecutionProvider::default().build());

        #[cfg(feature = "ep-directml")]
        execution_providers
            .push(ort::ep::DirectMLExecutionProvider::default().build());

        #[cfg(feature = "ep-acl")]
        execution_providers.push(ort::ep::ACLExecutionProvider::default().build());

        execution_providers
    }};
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
pub(crate) enum YoloClass {
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

pub(crate) fn detect_yolo<P>(original_img: &DynamicImage, onnx_path: P) -> Result<Vec<DetectedItem>>
where
    P: AsRef<std::path::Path>,
{
    const INPUT_WIDTH: u32 = 640;
    const INPUT_HEIGHT: u32 = 640;
    const NORMALIZATION_FACTOR: f32 = 255.0;
    let (img_width, img_height) = (original_img.width(), original_img.height());
    let mut input = Array::zeros((1, 3, INPUT_HEIGHT as usize, INPUT_WIDTH as usize));
    {
        let input_img =
            original_img.resize_exact(INPUT_WIDTH, INPUT_HEIGHT, FilterType::CatmullRom);
        for pixel in input_img.pixels() {
            let x = pixel.0 as _;
            let y = pixel.1 as _;
            let [r, g, b, _] = pixel.2.0;
            input[[0, 0, y, x]] = (r as f32) / NORMALIZATION_FACTOR;
            input[[0, 1, y, x]] = (g as f32) / NORMALIZATION_FACTOR;
            input[[0, 2, y, x]] = (b as f32) / NORMALIZATION_FACTOR;
        }
    }
    let mut model = Session::builder()?
        .with_log_level(LogLevel::Fatal)
        .unwrap_or_else(|e| e.recover())
        .with_execution_providers(build_execution_providers!())
        .unwrap_or_else(|e| e.recover())
        .with_auto_device(AutoDevicePolicy::MaxPerformance)
        .unwrap_or_else(|e| e.recover())
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
            // Skip bounding box coordinates.
            .skip(4)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .context("Failed to find class with maximum probability")?;
        if prob < 0.5 {
            continue;
        }
        let Some(class_enum) = YoloClass::from_usize(class_id) else {
            tracing::debug!("Class ID {} is not a valid YoloV8 class", class_id);
            continue;
        };
        let xc = row[0] / INPUT_WIDTH as f32 * (img_width as f32);
        let yc = row[1] / INPUT_HEIGHT as f32 * (img_height as f32);
        let w = row[2] / INPUT_WIDTH as f32 * (img_width as f32);
        let h = row[3] / INPUT_HEIGHT as f32 * (img_height as f32);
        let bounding_box = BoundingBox {
            x1: xc - w / 2.0,
            y1: yc - h / 2.0,
            x2: xc + w / 2.0,
            y2: yc + h / 2.0,
        };
        let detected_item = DetectedItem {
            bounding_box,
            class: class_enum,
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
