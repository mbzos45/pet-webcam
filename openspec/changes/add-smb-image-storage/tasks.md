## 1. Dependencies

- [x] 1.1 ~~Add `smb` crate~~ Superseded during implementation: added `pavao` (v0.2.x, safe bindings over the system `libsmbclient`) to `Cargo.toml` instead — see design.md Decision 1. Requires Samba's client library installed and discoverable via `pkg-config`.
- [x] 1.2 ~~Add `tokio`~~ Not needed: `pavao` is fully synchronous, no async runtime required.
- [x] 1.3 Confirm `dotenvy = "0.15.7"` (already present) resolves cleanly alongside the new deps; run `cargo build` to check for version conflicts

## 2. Args and config resolution

- [x] 2.1 In `src/main.rs`, change `Args.img_path` from `String` to `Option<PathBuf>` (or `Option<String>`), used only in local mode
- [x] 2.2 Add `Args` fields: `smb: bool` (`--smb`, no default value needed — presence-based flag), `smb_username: Option<String>` (`--smb-username`), `smb_password: Option<String>` (`--smb-password`), `smb_dest: Option<String>` (`--smb-dest`)
- [x] 2.3 Write a config-resolution step (e.g. `resolve_smb_config(&Args) -> Result<SmbConfig>`) that reads `SMB_ENABLED`/`SMB_USERNAME`/`SMB_PASSWORD`/`SMB_DEST` via `std::env::var`, with CLI values overriding env values per field
- [x] 2.4 In the resolution step, if SMB mode is active and username, password, or destination is missing from both CLI and env, `bail!` naming the missing field(s), before any camera/SMB work starts
- [x] 2.5 If SMB mode is *not* active, require `img_path` to be set (`bail!` if `None`), mirroring today's required-arg behavior

## 3. `.env` loading

- [x] 3.1 At the top of `main()`, call `dotenvy::dotenv().ok()` before `argh::from_env()` so `.env`-defined vars are visible to both env resolution and (if ever needed) argh's own env fallback
- [x] 3.2 Add `.env.example` at the repo root documenting `SMB_ENABLED`, `SMB_USERNAME`, `SMB_PASSWORD`, `SMB_DEST` with comments (no real credentials)
- [x] 3.3 Add `.env` to `.gitignore` if not already covered (already covered by the Python template's "Environments" section)

## 4. Storage module restructure

- [x] 4.1 Convert `src/storage.rs` into `src/storage/mod.rs`, keeping the existing `save_image` function and its signature unchanged
- [x] 4.2 Add `src/storage/smb.rs` with the SMB-specific implementation
- [x] 4.3 In `storage/mod.rs`, define the resolved-mode type (e.g. `enum StorageMode { Local(PathBuf), Smb(SmbConfig) }`) used to dispatch between local and SMB paths

## 5. SMB pre-flight (connect/validate)

- [x] 5.1 In `storage/smb.rs`, implement `SmbSession::preflight(config: &SmbConfig) -> Result<SmbSession>` (superseded `pavao`-based approach): parse `config.dest` (UNC `\\host\share\subdir`) into host/share/sub-path, build `SmbCredentials` + `SmbClient::new(...)`, then call `client.stat(<sub-path>)` to force a real connect+auth round-trip (`SmbClient::new` alone does no network I/O — see design.md Decision 2)
- [x] 5.2 Map connection/auth failures to `anyhow` errors with enough context (host/share, without leaking the password) for the exit message required by the spec's "SMB destination unreachable" / "SMB authentication rejected" scenarios
- [x] 5.3 Return a session/handle value that `main()` can hold and later pass to the save step, avoiding a second connect

## 6. SMB save

- [x] 6.1 In `storage/smb.rs`, implement `SmbSession::save(&self, image: DynamicImage, timestamp: impl AsRef<str>, width: Option<u32>, height: Option<u32>, quality: f32) -> Result<()>`, reusing the existing resize + WebP-encode logic (extract the shared resize/encode steps out of `storage::save_image` into a small helper used by both local and SMB paths, per design.md's "existing `save_image` untouched" goal — helper lives in `storage/mod.rs`, not a change to `save_image`'s own signature)
- [x] 6.2 Inside `save`, use the held session to `open_with(path, SmbOpenOptions::default().write(true).create(true).truncate(true))` and write the WebP bytes via `Write::write_all` (superseded `pavao`-based approach, fully synchronous — no `Runtime`/`.await`)
- [x] 6.3 On open/write failure, return an `anyhow` error (spec: "SMB write failure after capture" → report + non-zero exit)
- [x] 6.4 On success, log the destination path at `tracing::info!`, matching local save's existing log line

## 7. `main.rs` wiring and ordering

- [x] 7.1 Reorder `main()` to: load `.env` → parse args → validate onnx path (existing check) → resolve storage config (§2.3–2.5) → run pre-flight (local dir check, or `storage::smb::preflight` for SMB) → capture image → run YOLO detection → save (local `save_image` or `storage::smb::save`, per resolved mode)
- [x] 7.2 Keep the existing "Person detected → skip save" early-return behavior working for both local and SMB modes
- [x] 7.3 Ensure the SMB session/handle from pre-flight is threaded through to the save call (owned by `main()` or passed through)

## 8. Documentation

- [x] 8.1 Update `README.md` (or add one if absent) documenting the new `--smb`/`--smb-username`/`--smb-password`/`--smb-dest` flags, the matching env vars, precedence rules, and the recommendation to prefer env/`.env` over `--smb-password` for credentials (per design.md Risks)

## 9. Verification

- [x] 9.1 `cargo build` succeeds with the new dependencies
- [x] 9.2 Manual test: local mode (`--img-path` set, no `--smb`) still saves to disk exactly as before — verified: captured + saved a real webcam frame to `/tmp/pet-webcam-local-test`, exit 0
- [x] 9.3 Manual test: SMB mode with a missing required setting exits non-zero before camera capture (check logs/camera is never opened) — verified: `--smb` with no username/password/dest → all three reported missing, exit 1, no camera log line
- [x] 9.4 Manual test: SMB mode against an unreachable host exits non-zero before camera capture — verified: `--smb-dest '\\256.256.256.256\bogus'` → pre-flight `stat` failed, exit 1, before capture
- [x] 9.5 Manual test: SMB mode against a reachable test share (or local Samba container) succeeds end-to-end and the file appears on the share — verified against the real destination configured in `.env` (`\\192.168.252.211\Public\webcam`): connect, capture, save all succeeded, exit 0
- [x] 9.6 Manual test: `.env`-supplied SMB settings are picked up when no CLI flags are given, and a CLI flag overrides a conflicting `.env` value — verified: `--smb` alone used `.env`'s destination; `--smb --smb-dest <bogus>` overrode it
