FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY . .
RUN pip install --no-cache-dir .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG APP_UID=1000
ARG APP_GID=1000

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN groupadd --gid "${APP_GID}" appuser \
 && useradd --uid "${APP_UID}" --gid appuser --create-home appuser \
 && install -d -m 0755 -o appuser -g appuser \
      /home/appuser/.tradingagents \
      /home/appuser/app \
      /home/appuser/app/reports
WORKDIR /home/appuser/app

COPY --from=builder --chown=appuser:appuser /build .
COPY docker/entrypoint.py /usr/local/bin/tradingagents-entrypoint.py

# The entrypoint starts as root only long enough to repair ownership of bind
# mounts/named volumes, then permanently drops to appuser before launching.
USER root
ENTRYPOINT ["python", "/usr/local/bin/tradingagents-entrypoint.py"]
