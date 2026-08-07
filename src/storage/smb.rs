use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use pavao::{SmbClient, SmbCredentials, SmbOpenOptions, SmbOptions};
use std::io::Write;

use super::encode_webp;

#[derive(Debug, Clone)]
pub(crate) struct SmbConfig {
    pub(crate) username: String,
    pub(crate) password: String,
    /// UNC-style destination, e.g. `\\host\share\subdir` (forward slashes also accepted).
    pub(crate) dest: String,
}

/// Splits a `\\host\share\subdir` (or `/host/share/subdir`) destination into
/// (host, share, subdir-under-share). `subdir` is empty when the destination is the share root.
fn parse_dest(dest: &str) -> Result<(String, String, String)> {
    let normalized = dest.trim_start_matches(['\\', '/']).replace('\\', "/");
    let mut parts = normalized.splitn(3, '/');
    let host = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("SMB destination '{dest}' is missing a server name (expected \\\\host\\share[\\subdir])"))?;
    let share = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("SMB destination '{dest}' is missing a share name (expected \\\\host\\share[\\subdir])"))?;
    let sub_path = parts.next().unwrap_or("").trim_matches('/').to_string();
    Ok((host.to_string(), share.to_string(), sub_path))
}

/// A held SMB connection, established during pre-flight and reused for the final save
/// (see design.md decision 2: one connection setup per run, not one per operation).
pub(crate) struct SmbSession {
    client: SmbClient,
    sub_path: String,
    base_uri: String,
}

impl SmbSession {
    pub(crate) fn preflight(config: &SmbConfig) -> Result<Self> {
        let (host, share, sub_path) = parse_dest(&config.dest)?;
        let base_uri = format!("smb://{host}/{share}");

        let credentials = SmbCredentials::default()
            .server(format!("smb://{host}"))
            .share(&share)
            .username(&config.username)
            .password(&config.password);

        let client = SmbClient::new(credentials, SmbOptions::default())
            .map_err(|e| anyhow!("Failed to initialize SMB client for {base_uri}: {e}"))?;

        // `SmbClient::new` only sets up a local context; it performs no network I/O.
        // `stat` the destination directory to force an actual connect + auth round-trip,
        // so an unreachable host, rejected login, or missing subdirectory fails here,
        // before camera capture (per specs/smb-image-storage/spec.md).
        let stat_path = if sub_path.is_empty() {
            String::new()
        } else {
            format!("/{sub_path}")
        };
        client.stat(&stat_path).map_err(|e| {
            anyhow!(
                "Failed to reach SMB destination {base_uri}{stat_path}: {e} \
                 (check host, share, credentials, and that the destination directory exists)"
            )
        })?;

        tracing::info!("Connected to SMB destination {base_uri}{stat_path}");
        Ok(SmbSession {
            client,
            sub_path,
            base_uri,
        })
    }

    pub(crate) fn save(
        &self,
        image: DynamicImage,
        timestamp: impl AsRef<str>,
        width: Option<u32>,
        height: Option<u32>,
        quality: f32,
    ) -> Result<()> {
        let webp = encode_webp(image, width, height, quality)?;

        let filename = format!("{}.webp", timestamp.as_ref());
        let file_path = if self.sub_path.is_empty() {
            format!("/{filename}")
        } else {
            format!("/{}/{filename}", self.sub_path)
        };

        let mut file = self
            .client
            .open_with(
                &file_path,
                SmbOpenOptions::default().write(true).create(true).truncate(true),
            )
            .map_err(|e| anyhow!("Failed to open SMB file {}{file_path}: {e}", self.base_uri))?;
        file.write_all(&webp)
            .context("Failed to write image data to SMB share")?;
        drop(file);

        tracing::info!("Saved image to {}{file_path}", self.base_uri);
        Ok(())
    }
}
