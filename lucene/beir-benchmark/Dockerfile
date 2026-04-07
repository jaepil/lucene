# =============================================================================
# BEIR Benchmark for Bayesian BM25 Hybrid Search
# =============================================================================
# Build context: repository root (../../ from this file)
#
# Usage (local data, pre-generated embeddings):
#   docker build -t beir-benchmark -f lucene/beir-benchmark/Dockerfile .
#   docker run -v /path/to/beir-prepared:/data/beir-prepared \
#              -v $(pwd)/results:/benchmark/results beir-benchmark
#
# Usage (with embedding generation in Docker):
#   docker build -t beir-benchmark --target with-embeddings \
#     --build-arg BEIR_DATASETS="scifact nfcorpus" \
#     -f lucene/beir-benchmark/Dockerfile .
#   docker run -v $(pwd)/results:/benchmark/results beir-benchmark
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 (optional): Generate dense embeddings with sentence-transformers
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS embeddings

RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY lucene/beir-benchmark/python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lucene/beir-benchmark/python/prepare_data.py .

ARG BEIR_DATASETS="scifact nfcorpus fiqa"
RUN python prepare_data.py \
    --datasets ${BEIR_DATASETS} \
    --data-dir /data/beir-raw \
    --output-dir /data/beir-prepared

# ---------------------------------------------------------------------------
# Stage 2: Build Lucene from source + benchmark module
# ---------------------------------------------------------------------------
FROM eclipse-temurin:25-jdk AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /lucene

COPY gradlew gradlew.bat ./
COPY gradle/ gradle/
COPY build.gradle settings.gradle gradle.properties build-options.properties ./
COPY versions.lock ./
COPY build-tools/ build-tools/
COPY lucene/ lucene/

RUN --mount=type=cache,target=/root/.gradle \
    chmod +x gradlew && \
    ./gradlew :lucene:beir-benchmark:assemble -x test -x javadoc --no-daemon -quiet

# Collect all runtime JARs into a single directory
RUN mkdir -p /jars && \
    find lucene/beir-benchmark/build -name "*.jar" -exec cp {} /jars/ \; && \
    find lucene/core/build -name "lucene-core-*.jar" -exec cp {} /jars/ \; && \
    find lucene/analysis/common/build -name "lucene-analysis-common-*.jar" -exec cp {} /jars/ \; && \
    find lucene/queryparser/build -name "lucene-queryparser-*.jar" -exec cp {} /jars/ \; && \
    find lucene/queries/build -name "lucene-queries-*.jar" -exec cp {} /jars/ \; && \
    find lucene/sandbox/build -name "lucene-sandbox-*.jar" -exec cp {} /jars/ \; && \
    echo "Collected JARs:" && ls -la /jars/

# ---------------------------------------------------------------------------
# Stage 3a: Runtime (local data -- mount beir-prepared as a volume)
# ---------------------------------------------------------------------------
FROM eclipse-temurin:25-jre AS runtime

WORKDIR /benchmark
COPY --from=builder /jars/ /benchmark/lib/
COPY lucene/beir-benchmark/run.sh /benchmark/run.sh
RUN chmod +x /benchmark/run.sh

ENV JAVA_OPTS="-Xmx4g -Xms2g"
ENV BEIR_DATA_DIR="/data/beir-prepared"
ENV BEIR_DATASETS=""
ENV BEIR_OUTPUT="/benchmark/results/results.tsv"

ENTRYPOINT ["/benchmark/run.sh"]

# ---------------------------------------------------------------------------
# Stage 3b: Runtime with embedded data (from Python stage)
# ---------------------------------------------------------------------------
FROM runtime AS with-embeddings
COPY --from=embeddings /data/beir-prepared /data/beir-prepared
