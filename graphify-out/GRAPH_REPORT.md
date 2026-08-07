# Graph Report - .  (2026-08-07)

## Corpus Check
- Corpus is ~1,236 words - fits in a single context window. You may not need a graph.

## Summary
- 30 nodes · 38 edges · 6 communities (5 shown, 1 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Image Storage
- Detection Geometry & Types
- Camera Capture & Entry Point
- YOLO Detection
- CLI Argument Parsing
- OpenSpec Config

## God Nodes (most connected - your core abstractions)
1. `detect_yolo()` - 9 edges
2. `save_image()` - 6 edges
3. `capture_image()` - 4 edges
4. `intersection()` - 4 edges
5. `union()` - 4 edges
6. `Args` - 4 edges
7. `BoundingBox` - 4 edges
8. `DetectedItem` - 4 edges
9. `main()` - 4 edges
10. `YoloClass` - 2 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `detect_yolo()`  [INFERRED]
  src/main.rs → src/detector.rs
- `main()` --calls--> `capture_image()`  [INFERRED]
  src/main.rs → src/camera.rs
- `detect_yolo()` --references--> `DetectedItem`  [EXTRACTED]
  src/detector.rs → src/main.rs
- `DetectedItem` --references--> `YoloClass`  [EXTRACTED]
  src/main.rs → src/detector.rs
- `intersection()` --references--> `BoundingBox`  [EXTRACTED]
  src/detector.rs → src/main.rs

## Import Cycles
- None detected.

## Communities (6 total, 1 thin omitted)

### Community 0 - "Image Storage"
Cohesion: 0.33
Nodes (6): Path, DynamicImage, Option, P, Result, save_image()

### Community 1 - "Detection Geometry & Types"
Cohesion: 0.52
Nodes (5): intersection(), union(), YoloClass, BoundingBox, DetectedItem

### Community 2 - "Camera Capture & Entry Point"
Cohesion: 0.40
Nodes (5): capture_image(), DynamicImage, Result, main(), Result

### Community 3 - "YOLO Detection"
Cohesion: 0.40
Nodes (5): detect_yolo(), DynamicImage, P, Result, Vec

### Community 4 - "CLI Argument Parsing"
Cohesion: 0.50
Nodes (4): PathBuf, Args, Option, String

## Knowledge Gaps
- **1 isolated node(s):** `OpenSpec Config`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `detect_yolo()` connect `YOLO Detection` to `Detection Geometry & Types`, `Camera Capture & Entry Point`?**
  _High betweenness centrality (0.237) - this node is a cross-community bridge._
- **Why does `main()` connect `Camera Capture & Entry Point` to `Detection Geometry & Types`, `YOLO Detection`?**
  _High betweenness centrality (0.227) - this node is a cross-community bridge._
- **Why does `Args` connect `CLI Argument Parsing` to `Detection Geometry & Types`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **What connects `OpenSpec Config` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._