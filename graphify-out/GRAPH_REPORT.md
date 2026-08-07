# Graph Report - pet-webcam  (2026-08-07)

## Corpus Check
- 12 files · ~5,800 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 102 nodes · 123 edges · 13 communities (11 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b4ccaad8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- encode_webp
- detect_yolo
- capture_image
- ADDED Requirements
- main.rs
- OpenSpec Config
- .save
- tasks.md
- proposal.md
- design.md
- pet-webcam
- Requirement: Pre-capture destination validation
- CLAUDE.md

## God Nodes (most connected - your core abstractions)
1. `detect_yolo()` - 9 edges
2. `encode_webp()` - 7 edges
3. `save_image()` - 7 edges
4. `resolve_storage_mode()` - 6 edges
5. `ADDED Requirements` - 6 edges
6. `Args` - 5 edges
7. `main()` - 5 edges
8. `StorageMode` - 5 edges
9. `SmbSession` - 5 edges
10. `pet-webcam` - 5 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `capture_image()`  [INFERRED]
  src/main.rs → src/camera.rs
- `main()` --calls--> `detect_yolo()`  [INFERRED]
  src/main.rs → src/detector.rs
- `StorageMode` --references--> `SmbConfig`  [EXTRACTED]
  src/storage/mod.rs → src/storage/smb.rs
- `DetectedItem` --references--> `YoloClass`  [EXTRACTED]
  src/main.rs → src/detector.rs
- `detect_yolo()` --references--> `DetectedItem`  [EXTRACTED]
  src/detector.rs → src/main.rs

## Import Cycles
- None detected.

## Communities (13 total, 2 thin omitted)

### Community 0 - "encode_webp"
Cohesion: 0.36
Nodes (8): Path, encode_webp(), DynamicImage, Option, P, Result, Vec, save_image()

### Community 1 - "detect_yolo"
Cohesion: 0.29
Nodes (10): detect_yolo(), intersection(), DynamicImage, P, Result, Vec, union(), YoloClass (+2 more)

### Community 2 - "capture_image"
Cohesion: 0.67
Nodes (3): capture_image(), DynamicImage, Result

### Community 3 - "ADDED Requirements"
Cohesion: 0.12
Nodes (16): ADDED Requirements, Purpose, Requirement: Development .env loading, Requirement: Save to SMB share, Requirement: SMB connection configuration, Requirement: SMB storage mode selection, Scenario: All settings via CLI, Scenario: All settings via environment (+8 more)

### Community 4 - "main.rs"
Cohesion: 0.27
Nodes (10): Args, env_flag_set(), main(), resolve_storage_mode(), Option, PathBuf, Result, String (+2 more)

### Community 6 - ".save"
Cohesion: 0.24
Nodes (10): AsRef, Self, SmbClient, parse_dest(), DynamicImage, Option, Result, String (+2 more)

### Community 7 - "tasks.md"
Cohesion: 0.20
Nodes (9): 1. Dependencies, 2. Args and config resolution, 3. `.env` loading, 4. Storage module restructure, 5. SMB pre-flight (connect/validate), 6. SMB save, 7. `main.rs` wiring and ordering, 8. Documentation (+1 more)

### Community 8 - "proposal.md"
Cohesion: 0.29
Nodes (6): Capabilities, Impact, Modified Capabilities, New Capabilities, What Changes, Why

### Community 9 - "design.md"
Cohesion: 0.33
Nodes (5): Context, Decisions, Goals / Non-Goals, Migration Plan, Risks / Trade-offs

### Community 10 - "pet-webcam"
Cohesion: 0.33
Nodes (5): CLI Options, Local development: `.env`, pet-webcam, SMB storage mode, Usage

### Community 11 - "Requirement: Pre-capture destination validation"
Cohesion: 0.40
Nodes (5): Requirement: Pre-capture destination validation, Scenario: Local mode destination missing, Scenario: SMB authentication rejected, Scenario: SMB destination reachable, Scenario: SMB destination unreachable

## Knowledge Gaps
- **40 isolated node(s):** `graphify`, `Usage`, `CLI Options`, `SMB storage mode`, `Local development: `.env`` (+35 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `StorageMode` connect `main.rs` to `encode_webp`, `.save`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `main()` connect `main.rs` to `detect_yolo`, `capture_image`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Why does `detect_yolo()` connect `detect_yolo` to `main.rs`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **What connects `graphify`, `Usage`, `CLI Options` to the rest of the system?**
  _40 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `ADDED Requirements` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._