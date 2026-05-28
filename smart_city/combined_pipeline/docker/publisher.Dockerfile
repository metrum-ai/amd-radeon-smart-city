# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

FROM jrottenberg/ffmpeg:6-ubuntu

# Upgrade base-image packages to pick up Ubuntu 24.04 security updates
# (clears 1 HIGH + ~36 MEDIUM CVEs surfaced by Trivy image scan, including
# CVE-2025-68973 in gpgv).
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10002 publisher \
    && useradd --uid 10002 --gid 10002 --home-dir /scripts \
        --shell /usr/sbin/nologin --no-create-home --no-log-init publisher \
    && mkdir -p /scripts \
    && chown -R 10002:10002 /scripts

COPY --chown=10002:10002 publish_input_rtsp.sh /scripts/publish_input_rtsp.sh
RUN chmod +x /scripts/publish_input_rtsp.sh

USER 10002:10002

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["/scripts/publish_input_rtsp.sh", "healthcheck"]

ENTRYPOINT ["/bin/bash", "/scripts/publish_input_rtsp.sh"]
