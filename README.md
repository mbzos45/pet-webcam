# pet-webcam

Capture a webcam snapshot, optionally run YOLOv8 (ONNX) object detection on it, and save the result — locally or to an SMB share.

## Building

**Prerequisites:**

- Rust (edition 2024 — a recent stable toolchain; install via [rustup](https://rustup.rs))
- Samba's client library (`libsmbclient`), discoverable via `pkg-config` — required to build, since SMB support ([`pavao`](https://crates.io/crates/pavao)) links it directly:
  - macOS: `brew install samba`
  - Debian/Ubuntu: `sudo apt install libsmbclient-dev`
  - Fedora/RHEL: `sudo dnf install libsmbclient-devel`
- A working native camera backend (`nokhwa`) — no extra install on macOS/Windows; on Linux this typically means `v4l-utils`/video4linux headers

```
cargo build --release
```

The binary is written to `target/release/pet-webcam`.

### Optional: ONNX execution providers

YOLO inference (`ort`) always builds with its default CPU provider. To build with a hardware-accelerated provider instead, enable one of these Cargo features:

```
cargo build --release --features ep-coreml   # Apple CoreML (macOS)
cargo build --release --features ep-cuda     # NVIDIA CUDA
cargo build --release --features ep-directml # DirectML (Windows)
cargo build --release --features ep-tensorrt # NVIDIA TensorRT
cargo build --release --features ep-xnnpack  # XNNPACK (CPU, cross-platform)
cargo build --release --features ep-acl      # Arm Compute Library
```

## Usage

```
pet-webcam --img-path ./snapshots
pet-webcam --img-path ./snapshots --onnx-path ./models/yolov8n.onnx
pet-webcam --smb --smb-username alice --smb-dest '\\nas.local\photos\pet-webcam'
```

## CLI Options

| Flag | Short | Description |
| --- | --- | --- |
| `--camera-id` | `-i` | Camera index to capture from (default `0`) |
| `--onnx-path` | `-m` | Path to a YOLOv8 ONNX model. If omitted, detection is skipped and the image is always saved |
| `--time-format` | `-t` | `chrono` format string used for the saved filename (default `%Y-%m-%d_%H-%M`) |
| `--width` | `-w` | Resize output width (requires `--height` too) |
| `--height` | `-h` | Resize output height (requires `--width` too) |
| `--quality` | `-q` | WebP compression quality, 0-100 (default `80`) |
| `--img-path` | `-p` | Local directory to save into. **Required unless `--smb` is set** |
| `--smb` | `-s` | Save to an SMB share instead of local disk |
| `--smb-username` | | SMB username (or `SMB_USERNAME` env var) |
| `--smb-password` | | SMB password (or `SMB_PASSWORD` env var — prefer this over the flag, see below) |
| `--smb-dest` | | SMB destination UNC path, e.g. `\\host\share\subdir` (or `SMB_DEST` env var) |

When an ONNX model is supplied and a person is detected in the frame, the image is **not** saved (in either mode).

## SMB storage mode

Pass `--smb` (or set `SMB_ENABLED=true`) to save captured images to a network share instead of local disk. Configure it via CLI flags or environment variables — CLI flags win when both are set for the same setting:

- `SMB_ENABLED` — `true`/`1` enables SMB mode (equivalent to `--smb`)
- `SMB_USERNAME` / `--smb-username`
- `SMB_PASSWORD` / `--smb-password`
- `SMB_DEST` / `--smb-dest` — UNC path, e.g. `\\nas.local\photos\pet-webcam`

All three of username, password, and destination are required when SMB mode is active. The connection is validated **before** the camera is opened: a missing setting, an unreachable host, or a rejected login exits the program early instead of after burning a capture/inference cycle.

**Prefer `SMB_PASSWORD` (or a `.env` file) over `--smb-password`.** A password passed on the command line is visible in shell history and to anyone who can list processes (`ps`) on the machine.

## Local development: `.env`

Copy `.env.example` to `.env` and fill in your SMB settings — it's loaded automatically at startup (and ignored by git). A missing `.env` file is not an error.

```
cp .env.example .env
```