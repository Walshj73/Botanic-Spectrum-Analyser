FROM python:3.12-slim-bullseye@sha256:411fa4dcfdce7e7a3057c45662beba9dcd4fa36b2e50a2bfcd6c9333e59bf0db

ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1

RUN printf '%s\n' \
    'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20250721T000000Z bullseye main' \
    'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/20250721T000000Z bullseye-security main' \
    > /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    binutils fontconfig fonts-dejavu-core libgl1 libglib2.0-0 libgomp1 \
    libsm6 libxext6 libxrender1 tk \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-gui.txt /tmp/requirements-gui.txt
COPY packaging/linux/requirements-build.txt /tmp/requirements-build.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-build.txt
