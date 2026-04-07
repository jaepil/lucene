#!/bin/bash
set -euo pipefail

# Collect all JARs into classpath
CLASSPATH=""
for jar in /benchmark/lib/*.jar; do
  if [ -f "$jar" ]; then
    if [ -z "$CLASSPATH" ]; then
      CLASSPATH="$jar"
    else
      CLASSPATH="$CLASSPATH:$jar"
    fi
  fi
done

if [ -z "$CLASSPATH" ]; then
  echo "ERROR: No JAR files found in /benchmark/lib/"
  exit 1
fi

# Build application arguments
APP_ARGS="--data-dir ${BEIR_DATA_DIR:-/data/beir-prepared}"

if [ -n "${BEIR_DATASETS:-}" ]; then
  APP_ARGS="$APP_ARGS --datasets $BEIR_DATASETS"
fi

if [ -n "${BEIR_OUTPUT:-}" ]; then
  mkdir -p "$(dirname "$BEIR_OUTPUT")"
  APP_ARGS="$APP_ARGS --output $BEIR_OUTPUT"
fi

echo "============================================================"
echo "BEIR Benchmark: Bayesian BM25 Hybrid Search Evaluation"
echo "============================================================"
echo "  Data dir: ${BEIR_DATA_DIR:-/data/beir-prepared}"
echo "  Datasets: ${BEIR_DATASETS:-auto-discover}"
echo "  Output:   ${BEIR_OUTPUT:-stdout only}"
echo "  JVM opts: ${JAVA_OPTS:--Xmx4g -Xms2g}"
echo ""

exec java ${JAVA_OPTS:--Xmx4g -Xms2g} \
  --add-opens java.base/java.lang=ALL-UNNAMED \
  -cp "$CLASSPATH" \
  org.apache.lucene.benchmark.beir.BEIRBenchmark \
  $APP_ARGS "$@"
