# syntax=docker/dockerfile:1
ARG RUST_VERSION=1.92
ARG APP_NAME=pet-webcam

FROM lukemathwalker/cargo-chef:latest-rust-${RUST_VERSION} AS chef
WORKDIR /app
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
ARG APP_NAME
# 依存関係のビルド（キャッシュ可能）
COPY --from=planner /app/recipe.json recipe.json
RUN apt-get update && apt-get install -y --no-install-recommends  \
    pkg-config \
    libssl-dev \
    libclang-dev \
    libv4l-dev
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    cargo chef cook --release --recipe-path recipe.json

# アプリケーションのビルド
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,target=/app/target,sharing=locked \
    cargo build --release --bin ${APP_NAME} && \
    cp ./target/release/${APP_NAME} /bin/server

# テストステージ（オプション）
#FROM chef AS test
#COPY . .
#RUN --mount=type=cache,target=/usr/local/cargo/registry \
#    --mount=type=cache,target=/usr/local/cargo/git \
#    cargo test

# 本番ステージ：distroless ベースを使う代わりに trixie-slim を使用し、
# ランタイムに必要なユーザー空間ライブラリをインストールする
FROM debian:trixie-slim AS runtime
ARG APP_NAME=pet-webcam
# ランタイムに必要なパッケージを最小限でインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    libv4l-0 \
    v4l-utils \
    libudev1 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ビルド済バイナリをコピー
COPY --from=builder /bin/server /app/
# ONNX モデルをイメージに同梱（運用に応じてホストマウントに変更可）
COPY models/*.onnx /app/models/

RUN mkdir -p /app/models /app/outputs
WORKDIR /app
## セキュリティのため非 root ユーザを作成して実行する（UID/GID 1000）
#RUN groupadd -g 1000 app || true && useradd -r -u 1000 -g app -d /app -s /usr/sbin/nologin app || true \
#    && chown -R app:app /app

#USER app
ENTRYPOINT ["/app/server","-m", "/app/models/yolov8n.onnx", "-p", "/app/outputs"]
