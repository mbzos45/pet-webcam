## Purpose

Lets the capture pipeline save snapshots to a network (SMB) share instead of local disk, configured via CLI flags or environment variables, so the device can run without persistent local storage.

## ADDED Requirements

### Requirement: SMB storage mode selection
The system SHALL support a storage mode switch (CLI flag or environment variable) that selects between local-disk save (default, current behavior) and SMB save. Exactly one mode SHALL be active per run.

#### Scenario: Default mode is local
- **WHEN** the SMB mode flag/env var is not set
- **THEN** the system saves the image to the local `--img-path` directory, unchanged from current behavior

#### Scenario: SMB mode explicitly enabled
- **WHEN** the SMB mode flag/env var is set (e.g. `--smb` or `SMB_ENABLED=true`)
- **THEN** the system saves the image to the configured SMB destination instead of local disk

### Requirement: SMB connection configuration
When SMB mode is enabled, the system SHALL accept a username, password, and destination (share path) through CLI arguments or environment variables. CLI arguments SHALL take precedence over environment variables when both are supplied for the same setting.

#### Scenario: All settings via CLI
- **WHEN** SMB mode is enabled and `--smb-username`, `--smb-password`, and `--smb-dest` are all supplied on the command line
- **THEN** the system uses those CLI values to connect

#### Scenario: All settings via environment
- **WHEN** SMB mode is enabled and `SMB_USERNAME`, `SMB_PASSWORD`, and `SMB_DEST` are set as environment variables with no corresponding CLI flags
- **THEN** the system uses those environment values to connect

#### Scenario: CLI overrides environment
- **WHEN** SMB mode is enabled, an environment variable and its corresponding CLI flag are both set to different values
- **THEN** the system uses the CLI-supplied value

#### Scenario: Missing required setting
- **WHEN** SMB mode is enabled and any of username, password, or destination is missing from both CLI and environment
- **THEN** the system reports which setting is missing and exits with a non-zero status before attempting camera capture

### Requirement: Development .env loading
In development builds, the system SHALL load configuration values (including SMB settings) from a `.env` file in the working directory if present, before resolving final config from CLI/environment. A missing `.env` file SHALL NOT be treated as an error.

#### Scenario: .env file present
- **WHEN** a `.env` file in the working directory defines `SMB_USERNAME`, `SMB_PASSWORD`, or `SMB_DEST`
- **THEN** those values are available as if set as environment variables, and are still overridable by explicit CLI flags

#### Scenario: .env file absent
- **WHEN** no `.env` file exists in the working directory
- **THEN** the system proceeds using only real environment variables and CLI flags, without error

### Requirement: Pre-capture destination validation
The system SHALL validate the resolved storage configuration and, for SMB mode, establish the SMB connection before capturing an image from the camera or running detection. If validation or connection fails, the system SHALL exit with a non-zero status without capturing an image.

#### Scenario: Local mode destination missing
- **WHEN** local mode is active and the configured local save directory does not exist
- **THEN** the system exits with a non-zero status before camera capture, as today

#### Scenario: SMB destination unreachable
- **WHEN** SMB mode is active and the configured share cannot be reached (network error, host not found)
- **THEN** the system exits with a non-zero status before camera capture

#### Scenario: SMB authentication rejected
- **WHEN** SMB mode is active and the server rejects the supplied username/password
- **THEN** the system exits with a non-zero status before camera capture

#### Scenario: SMB destination reachable
- **WHEN** SMB mode is active and the share is reachable and credentials are accepted
- **THEN** the system proceeds to camera capture

### Requirement: Save to SMB share
When SMB mode is active and validation succeeds, the system SHALL write the captured (and optionally detection-processed) image to the configured SMB destination, using the same filename/timestamp and encoding conventions as local save.

#### Scenario: Successful SMB save
- **WHEN** capture and detection complete and SMB mode is active
- **THEN** the system writes the resulting image file to the configured SMB share path and logs the destination path on success

#### Scenario: SMB write failure after capture
- **WHEN** the SMB connection is lost or the write is rejected after a successful capture
- **THEN** the system reports the error and exits with a non-zero status