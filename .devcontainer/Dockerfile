FROM python:3.10-slim

# System dependencies:
#   git            — version control
#   build-essential — C/C++ compiler for native extensions
#   swig           — corels build dependency
#   libgmp-dev     — corels build dependency (GNU Multiple Precision)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        swig \
        libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
