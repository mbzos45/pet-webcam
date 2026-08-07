## Why

Captured images currently save only to a local directory (`storage::save_image` writes to `--img-path` via `std::fs::write`). The device that hosts the camera often has no reliable local disk (SD card wear, limited space) and needs to push snapshots straight to a network share instead. Adding SMB as a storage backend lets the same binary run headless and stream images to a NAS without a separate sync step.

## What Changes

- Add an SMB storage backend using the [`pavao`](https://crates.io/crates/pavao) crate (safe Rust bindings over the system `libsmbclient`, synchronous) as an alternative to local-disk save.
- Add a `--smb` / `SMB_ENABLED` flag that switches the save destination from local disk to an SMB share; when unset, behavior is unchanged (local save, current default).
- Add SMB connection settings, settable via CLI args or environment variables (CLI takes precedence): username, password, and destination (host/share/path, expressed as a UNC path e.g. `\\host\share\subdir`).
- Add `dotenvy`-based `.env` loading in development builds so SMB credentials can be supplied via a `.env` file instead of shell exporting or CLI flags (the `dotenvy` dependency is already present in `Cargo.toml` but currently unused).
- Validate args/env and establish the SMB connection **before** camera capture or YOLO inference run, so a bad destination or auth failure exits early rather than after burning a capture/inference cycle. New processing order: parse args → load `.env` (dev) → validate args/env → check destination reachability and establish SMB connection (exit non-zero on failure) → capture image → run YOLO detection → save (locally or via SMB, per the resolved mode).
- **BREAKING**: `main()`'s startup sequence changes — destination validation now happens before camera capture instead of only checking a local path's existence; a local-mode run also keeps this ordering (no behavior change for local mode beyond timing).

## Capabilities

### New Capabilities
- `smb-image-storage`: CLI/env-driven SMB save mode — configuration (flag, username, password, destination via args or env, `.env` support), pre-flight connection validation, and writing the captured/processed image to an SMB share instead of local disk.

### Modified Capabilities
<!-- none: local-disk save path stays as-is; startup ordering change is implementation detail of the new capability's validation step -->

## Impact

- **Code**: `src/main.rs` (arg parsing, `.env` loading, startup validation/connection ordering), new `src/storage/smb.rs` (or similar) for the SMB write path, `src/storage.rs` (local save stays, dispatch by mode).
- **Dependencies**: add `pavao` crate — links the system `libsmbclient` (Samba) at build time via `pkg-config`, so both build and runtime hosts need Samba's client library installed; `dotenvy` already in `Cargo.toml`, now actually wired up.
- **Config surface**: new CLI flags and env vars (`SMB_ENABLED`/`--smb`, `SMB_USERNAME`/`--smb-username`, `SMB_PASSWORD`/`--smb-password`, `SMB_DEST`/`--smb-dest`), new `.env.example` for local dev.
- **Runtime behavior**: destination check now happens before camera capture in both modes; SMB mode fails fast (before capture/inference) if the share is unreachable or credentials are rejected.