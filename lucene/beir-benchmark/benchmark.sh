#!/bin/bash
set -euo pipefail

# =============================================================================
# BEIR Benchmark Runner
# =============================================================================
# Usage:
#   ./benchmark.sh                          # Run with defaults (scifact, nfcorpus, fiqa)
#   ./benchmark.sh scifact                  # Run specific dataset
#   ./benchmark.sh scifact,nfcorpus,fiqa    # Run multiple datasets
#
# All 12 BEIR datasets:
#   ./benchmark.sh scifact,nfcorpus,fiqa,arguana,trec-covid,nq,hotpotqa,dbpedia-entity,fever,climate-fever,scidocs,quora
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

# Parse arguments
DATASETS="${1:-scifact nfcorpus fiqa}"
# Convert comma-separated to space-separated for Docker build arg
DATASETS_SPACES="${DATASETS//,/ }"
# Convert space-separated to comma-separated for Java arg
DATASETS_COMMAS="${DATASETS// /,}"

echo "============================================================"
echo "BEIR Benchmark: Bayesian BM25 Hybrid Search Evaluation"
echo "============================================================"
echo "  Datasets: $DATASETS_SPACES"
echo "  Results:  $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# Build and run
cd "$PROJECT_ROOT"

echo "[1/2] Building Docker image (this includes downloading datasets and generating embeddings)..."
docker build \
  -t beir-benchmark \
  --build-arg BEIR_DATASETS="$DATASETS_SPACES" \
  -f lucene/beir-benchmark/Dockerfile \
  .

echo ""
echo "[2/2] Running benchmark..."
docker run --rm \
  -v "$RESULTS_DIR:/benchmark/results" \
  -e JAVA_OPTS="${JAVA_OPTS:--Xmx4g -Xms2g}" \
  -e BEIR_DATASETS="$DATASETS_COMMAS" \
  -e BEIR_OUTPUT="/benchmark/results/results.tsv" \
  beir-benchmark

echo ""
echo "============================================================"
echo "Results saved to: $RESULTS_DIR/results.tsv"
echo "============================================================"
