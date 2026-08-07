## Context

`pet-webcam` is a synchronous, single-shot CLI: `main()` parses args, validates the local save directory, captures one frame (`camera::capture_image`, blocking `nokhwa` calls), optionally runs YOLO inference (`detector::detect_yolo`, blocking `ort` session), and writes one WebP file (`storage::save_image`, blocking `std::fs::write`). Nothing in the current binary is async. `dotenvy = "0.15.7"` is already a dependency in `Cargo.toml` but is never called. See proposal.md - Why/What Changes for the motivation and requirements; see `specs/smb-image-storage/spec.md` for the behavior contract.

The `pavao` crate (docs.rs/pavao, v0.2.x) is a safe wrapper over the system `libsmbclient` (Samba's client library), linked via `pkg-config` at build time. Its API is entirely synchronous: `SmbClient::new(SmbCredentials, SmbOptions)`, `client.open_with(path, SmbOpenOptions)` returning an `SmbFile` that implements `std::io::{Read, Write, Seek}`. No async runtime is involved. `SmbClient::new` only sets up a local libsmbclient context (registers the auth callback); it performs **no network I/O** — the first real round-trip (auth + connect) happens on the first actual operation such as `stat`/`open`/`list_dir`.

An earlier version of this design used the `smb` crate (a pure-Rust, Tokio-based SMB2/3 client). It was rejected during implementation: `pavao` was chosen instead. `pavao` requires the Samba client library (`libsmbclient`) to be installed on the build and runtime host (`brew install samba` on macOS, `libsmbclient-dev`/`samba-client-devel` on Linux) and discoverable via `pkg-config`; this repo's dev machine already had it.

## Goals / Non-Goals

**Goals:**
- Keep `main()` fully synchronous — `pavao` needs no async runtime, so no new concurrency model enters the binary.
- Keep local-mode behavior and code path untouched (same function, same call site).
- Resolve SMB config once, validate/connect before capture, reuse that resolved state for the final save.
- Make pre-flight validation perform a real network round-trip (not just local setup), since `SmbClient::new` alone does not contact the server.

**Non-Goals:**
- Concurrent/parallel capture-while-uploading, retry/backoff policies, or connection pooling — this is a single-shot CLI, one image per run.
- Streaming/chunked upload — captured images are small (single webcam frame, WebP-compressed); write the whole buffer via `Write::write_all`.
- A generic pluggable storage-backend trait for future backends beyond local/SMB — two variants don't justify the abstraction yet.

## Decisions

**1. Use `pavao`'s synchronous API directly in `main()`/`storage/smb.rs`; no async runtime anywhere in the binary.**
`pavao` wraps blocking `libsmbclient` C calls behind `std::io::{Read, Write, Seek}`, matching the rest of this CLI's synchronous style (`nokhwa`, `ort`, `std::fs`).
Alternative considered: the `smb` crate (pure-Rust, Tokio-based), which was the original plan. Rejected during implementation in favor of `pavao` — avoiding a new async runtime entirely is simpler than the previously-planned "scoped `tokio::Runtime` + `block_on`" approach, at the cost of a system library dependency (see Risks).

**2. Pre-flight calls `SmbClient::new` (local context/auth-callback setup only) followed by `client.stat(<destination>)` to force an actual connect + authenticate round-trip.**
`SmbClient::new` alone does not touch the network, so a `stat` on the resolved destination directory is what actually proves the server is reachable and the credentials are accepted — matching the spec's "establish the SMB connection" / "destination unreachable" / "authentication rejected" scenarios. The same `SmbClient` (and its already-negotiated session state) is then reused for the final `open_with`/write, avoiding a second auth round-trip.
Alternative considered: treat `SmbClient::new` returning `Ok` as sufficient pre-flight validation. Rejected — it would let a wrong host, wrong share, or wrong credentials pass pre-flight silently (no network call made), pushing the failure to after capture, which the spec explicitly requires to happen before capture.

**3. Config resolution: CLI flag > environment variable > `.env`-sourced environment variable; `.env` loaded unconditionally via `dotenvy::dotenv().ok()` at the very top of `main()`.**
`dotenvy::dotenv()` only has an effect if a `.env` file exists in the working directory; a missing file is not an error (`.ok()` swallows `NotFound`). Because it's a no-op without the file, gating it behind `cfg(debug_assertions)` buys nothing and would surprise an operator who intentionally drops a `.env` next to a release binary.
Alternative considered: gate `dotenvy::dotenv()` behind `#[cfg(debug_assertions)]` per the proposal's "development builds" wording. Rejected in favor of the always-safe no-op call — the file's presence, not the build profile, is what should decide whether it's used.

**4. New `Args` fields: `smb: bool`, `smb_username: Option<String>`, `smb_password: Option<String>`, `smb_dest: Option<String>`; `img_path` becomes `Option<String>` (required only in local mode, checked at runtime).**
`argh` has no conditional-required-if-flag support, so required-ness is enforced by hand in `main()` right after parsing, alongside the existing onnx-path-exists check, before any SMB or local validation runs.

**5. `pavao` added as a normal (always-compiled) dependency, not behind a Cargo feature; default (non-`vendored`) feature set, relying on the system `libsmbclient` via `pkg-config`.**
Matches how `nokhwa`/`ort` are already always-on; the existing `ep-*` Cargo features gate optional *execution providers* for one already-included crate, not whole app features. A build/runtime toggle (`--smb`) is enough; no need for a second, compile-time toggle for the same thing. The `vendored` feature (builds `libsmbclient` from source via `pavao-sys`) is not enabled — the dev host already has Samba installed, and defaulting to the system library keeps the build fast; revisit `vendored` if a deployment target can't install Samba's client library.
Alternative considered: `smb-storage` Cargo feature for the app-level toggle. Rejected — adds a second axis (compile-time vs. runtime) for a decision that only needs to be made once per run.

**6. New module `src/storage/smb.rs` (making `storage` a module directory: `storage/mod.rs` + `storage/smb.rs`); `storage::mod` gains a small dispatch used by `main()`: `preflight(...)` and `save(...)` calls routed to either the existing local function or the new SMB functions based on resolved mode.**
Keeps the existing `save_image` local-path function and its signature untouched (per Goals), and keeps SMB-specific types (`SmbClient`, `SmbCredentials`) out of `main.rs`.

**7. `SMB_SAVE_PATH`/`smb_save_path` destination string uses UNC syntax (`\\host\share\subdir`), parsed by hand into host/share/subdir instead of using a URL crate.**
This matches the Windows-style UNC convention this project's own `.env` already uses in practice, and is a trivial 3-way split (`\\`-prefix, `\`-separated segments) — no dependency justified for it. `pavao`'s own addressing (`smb://host` server string + separate share string) is assembled internally from the parsed parts.

## Risks / Trade-offs

- **Password on the command line** (`--smb-password`) is visible in shell history and process listings (`ps`). → Mitigate by documenting `SMB_PASSWORD`/`.env` as the recommended path in the README/`.env.example`; CLI flag stays available since the spec requires it, but isn't the suggested default.
- **New system-library build dependency**: `pavao` requires `libsmbclient` (Samba) installed and discoverable via `pkg-config` on every machine that builds this binary (and, for the default non-`vendored` feature, on every machine that *runs* it, since it's dynamically linked). → Accepted as the cost of avoiding a pure-Rust SMB stack; document the Samba client library as a build/runtime prerequisite in the README. The `vendored` Cargo feature is available as a fallback if a target host can't install it system-wide.
- **`SmbClient::new` doesn't validate reachability by itself**: a naive pre-flight that only checks `SmbClient::new()` succeeds would give a false sense of validation. → Addressed by Decision 2 — pre-flight always performs a real `stat` round-trip against the destination.
- **Long-lived connection held across capture+inference**: if capture/inference takes long enough for the SMB session to time out server-side, the final write fails after the (possibly slow) YOLO pass instead of failing fast. → Accepted per spec's "SMB write failure after capture" scenario (report + non-zero exit); no retry/reconnect-on-write-failure in this change.

## Migration Plan

No data migration. Rollout is additive and opt-in: `--smb`/`SMB_ENABLED` unset preserves today's local-save behavior exactly, so existing deployments need no changes to keep working. Enabling SMB mode is a config-only change (flags/env/`.env`) plus, for hosts that build the binary, installing the Samba client library — no code changes required per deployment.