/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.benchmark.beir;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;

/**
 * BEIR benchmark runner for evaluating Bayesian BM25 and hybrid search strategies.
 *
 * <p>Search strategies evaluated:
 *
 * <ol>
 *   <li>BM25 -- standard Lucene BM25 (baseline)
 *   <li>Bayesian BM25 -- BM25 wrapped in {@link
 *       org.apache.lucene.search.BayesianScoreQuery} with auto-estimated parameters
 *   <li>KNN -- dense vector retrieval via HNSW
 *   <li>RRF(BM25+KNN) -- Reciprocal Rank Fusion
 *   <li>LogOdds(BayesBM25+KNN) -- Log-odds fusion with uniform weights
 *   <li>LogOddsW(w*BayesBM25+(1-w)*KNN) -- Weighted log-odds fusion variants
 * </ol>
 *
 * <p>Usage: {@code java BEIRBenchmark --data-dir /data/beir-prepared [--datasets scifact,nfcorpus]
 * [--top-k 10] [--output results.tsv]}
 */
public final class BEIRBenchmark {

  private static final int TOP_K = 100;
  private static final int NDCG_K = 10;
  private static final int RECALL_K = 100;

  public static void main(String[] args) throws Exception {
    Path dataDir = null;
    List<String> datasets = null;
    int topK = TOP_K;
    Path outputPath = null;

    for (int i = 0; i < args.length; i++) {
      switch (args[i]) {
        case "--data-dir" -> dataDir = Path.of(args[++i]);
        case "--datasets" -> datasets = List.of(args[++i].split(","));
        case "--top-k" -> topK = Integer.parseInt(args[++i]);
        case "--output" -> outputPath = Path.of(args[++i]);
        default -> {
          System.err.println("Unknown argument: " + args[i]);
          printUsage();
          System.exit(1);
        }
      }
    }

    if (dataDir == null) {
      System.err.println("--data-dir is required");
      printUsage();
      System.exit(1);
    }

    if (datasets == null) {
      datasets = discoverDatasets(dataDir);
    }

    if (datasets.isEmpty()) {
      System.err.println("No datasets found in " + dataDir);
      System.exit(1);
    }

    System.out.println("BEIR Benchmark: Bayesian BM25 Hybrid Search Evaluation");
    System.out.println("=".repeat(70));
    System.out.printf("Data directory: %s%n", dataDir);
    System.out.printf("Datasets: %s%n", String.join(", ", datasets));
    System.out.printf("Top-K: %d, NDCG@%d, Recall@%d%n%n", topK, NDCG_K, RECALL_K);

    List<DatasetResult> allResults = new ArrayList<>();

    for (String dataset : datasets) {
      DatasetResult result = runDataset(dataDir, dataset, topK);
      if (result != null) {
        allResults.add(result);
      }
    }

    printSummary(allResults);

    if (outputPath != null) {
      writeResultsTSV(allResults, outputPath);
      System.out.println("\nResults written to: " + outputPath);
    }
  }

  private static DatasetResult runDataset(Path dataDir, String datasetName, int topK)
      throws Exception {
    Path datasetPath = dataDir.resolve(datasetName);
    if (Files.exists(datasetPath) == false) {
      System.err.printf("Dataset directory not found: %s, skipping.%n", datasetPath);
      return null;
    }

    System.out.printf("%n%s%n", "=".repeat(70));
    System.out.printf("Dataset: %s%n", datasetName);
    System.out.printf("%s%n", "=".repeat(70));

    // Load data
    System.out.println("Loading dataset...");
    BEIRDataset dataset = BEIRDataset.load(datasetPath);
    System.out.printf(
        "  Corpus: %,d docs, Queries: %,d, Qrels: %,d judgments%n",
        dataset.corpus().size(),
        dataset.queries().size(),
        dataset.qrels().values().stream().mapToInt(Map::size).sum());

    // Build index
    System.out.println("Building index...");
    Path indexPath = Files.createTempDirectory("beir-index-" + datasetName);
    IndexSearcher searcher = BEIRIndexer.buildIndex(indexPath, dataset.corpus());

    // Estimate Bayesian parameters from actual query score distribution
    System.out.println("Estimating Bayesian BM25 parameters from query score distribution...");
    Analyzer analyzer = BEIRIndexer.createAnalyzer();
    float[] bayesianParams = estimateFromQueries(searcher, analyzer, dataset.queries());
    float bayesianAlpha = bayesianParams[0];
    float bayesianBeta = bayesianParams[1];
    float bayesianBaseRate = bayesianParams[2];
    System.out.printf(
        "  alpha=%.4f, beta=%.4f, baseRate=%.6f%n",
        bayesianAlpha, bayesianBeta, bayesianBaseRate);

    // Train attention weights and run all strategies
    BEIRSearcher benchmarkSearcher =
        new BEIRSearcher(
            searcher, analyzer, bayesianAlpha, bayesianBeta, bayesianBaseRate);

    System.out.println("Training attention weights...");
    benchmarkSearcher.trainAttention(dataset.queries(), dataset.qrels());

    List<BEIRSearcher.SearchStrategy> strategies = benchmarkSearcher.allStrategies();

    DatasetResult datasetResult = new DatasetResult(datasetName);

    System.out.printf(
        "%n  %-40s %10s %10s %10s %8s%n",
        "Strategy", "NDCG@10", "MAP@10", "Recall@100", "Time(ms)");
    System.out.println("  " + "-".repeat(80));

    for (BEIRSearcher.SearchStrategy strategy : strategies) {
      long start = System.currentTimeMillis();
      Map<String, List<String>> results =
          benchmarkSearcher.runStrategy(strategy, dataset.queries(), topK);
      long elapsed = System.currentTimeMillis() - start;

      NDCGEvaluator.Metrics metrics =
          NDCGEvaluator.computeMean(results, dataset.qrels(), NDCG_K, RECALL_K);

      System.out.printf(
          "  %-40s %10.4f %10.4f %10.4f %8d%n",
          strategy.name(), metrics.ndcg(), metrics.map(), metrics.recall(), elapsed);

      datasetResult.addResult(strategy.name(), metrics, elapsed);
    }

    // Cleanup
    searcher.getIndexReader().close();
    deleteDirectory(indexPath);

    return datasetResult;
  }

  /**
   * Estimates Bayesian BM25 parameters (alpha, beta, baseRate) by running a sample of actual
   * queries through BM25 and analyzing the score distribution. This avoids the analyzer mismatch
   * problem of pseudo-query sampling.
   */
  private static float[] estimateFromQueries(
      IndexSearcher searcher, Analyzer analyzer, List<BEIRDataset.QueryEntry> queries)
      throws Exception {
    searcher.setSimilarity(new BM25Similarity());
    QueryParser parser = new QueryParser(BEIRIndexer.FIELD_CONTENTS, analyzer);
    parser.setDefaultOperator(QueryParser.Operator.OR);

    int sampleSize = Math.min(queries.size(), 100);
    List<Float> allScores = new ArrayList<>();
    int totalHits = 0;
    int totalDocs = searcher.getIndexReader().maxDoc();

    for (int i = 0; i < sampleSize; i++) {
      String queryText = queries.get(i).text();
      String escaped = QueryParser.escape(queryText);
      try {
        org.apache.lucene.search.Query q = parser.parse(escaped);
        TopDocs topDocs = searcher.search(q, 1000);
        for (ScoreDoc sd : topDocs.scoreDocs) {
          allScores.add(sd.score);
        }
        totalHits += topDocs.scoreDocs.length;
      } catch (Exception e) {
        // skip unparseable queries
      }
    }

    if (allScores.isEmpty()) {
      System.out.println("  WARNING: No scores collected, using default parameters");
      return new float[] {1.0f, 0.0f, 0.01f};
    }

    float[] scores = new float[allScores.size()];
    for (int i = 0; i < scores.length; i++) {
      scores[i] = allScores.get(i);
    }
    Arrays.sort(scores);

    // beta = median score (the decision boundary)
    float beta = scores[scores.length / 2];

    // alpha = 1 / std(scores) -- controls sigmoid steepness
    double mean = 0;
    for (float s : scores) mean += s;
    mean /= scores.length;
    double variance = 0;
    for (float s : scores) {
      double diff = s - mean;
      variance += diff * diff;
    }
    variance /= scores.length;
    double std = Math.sqrt(variance);
    float alpha = std > 1e-6 ? (float) (1.0 / std) : 1.0f;

    // baseRate = fraction of corpus that gets retrieved per query on average
    float baseRate = (float) totalHits / (sampleSize * totalDocs);
    baseRate = Math.clamp(baseRate, 1e-6f, 0.5f);

    System.out.printf(
        "  Score stats: n=%d, mean=%.4f, std=%.4f, median=%.4f%n",
        scores.length, mean, std, beta);

    return new float[] {alpha, beta, baseRate};
  }

  private static void printSummary(List<DatasetResult> allResults) {
    if (allResults.isEmpty()) return;

    System.out.printf("%n%n%s%n", "=".repeat(90));
    System.out.println("SUMMARY: NDCG@10 across all datasets");
    System.out.printf("%s%n", "=".repeat(90));

    // Collect strategy names
    List<String> strategyNames = allResults.getFirst().strategyNames();

    // Header
    System.out.printf("%-20s", "Dataset");
    for (String sn : strategyNames) {
      String label = sn.length() > 12 ? sn.substring(0, 12) : sn;
      System.out.printf(" %12s", label);
    }
    System.out.println();
    System.out.println("-".repeat(20 + strategyNames.size() * 13));

    // Rows
    double[] sums = new double[strategyNames.size()];
    for (DatasetResult dr : allResults) {
      System.out.printf("%-20s", dr.datasetName);
      for (int i = 0; i < strategyNames.size(); i++) {
        double ndcg = dr.ndcgForStrategy(strategyNames.get(i));
        sums[i] += ndcg;
        System.out.printf(" %12.4f", ndcg);
      }
      System.out.println();
    }

    // Average
    System.out.println("-".repeat(20 + strategyNames.size() * 13));
    System.out.printf("%-20s", "AVERAGE");
    for (int i = 0; i < strategyNames.size(); i++) {
      System.out.printf(" %12.4f", sums[i] / allResults.size());
    }
    System.out.println();
  }

  private static void writeResultsTSV(List<DatasetResult> allResults, Path outputPath)
      throws IOException {
    try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(outputPath))) {
      pw.println("dataset\tstrategy\tndcg@10\tmap@10\trecall@100\ttime_ms");
      for (DatasetResult dr : allResults) {
        for (StrategyResult sr : dr.results) {
          pw.printf(
              "%s\t%s\t%.6f\t%.6f\t%.6f\t%d%n",
              dr.datasetName,
              sr.strategyName,
              sr.metrics.ndcg(),
              sr.metrics.map(),
              sr.metrics.recall(),
              sr.timeMs);
        }
      }
    }
  }

  private static List<String> discoverDatasets(Path dataDir) throws IOException {
    List<String> datasets = new ArrayList<>();
    try (var stream = Files.list(dataDir)) {
      stream
          .filter(Files::isDirectory)
          .filter(p -> Files.exists(p.resolve("corpus.jsonl")))
          .sorted()
          .forEach(p -> datasets.add(p.getFileName().toString()));
    }
    return datasets;
  }

  private static void deleteDirectory(Path dir) throws IOException {
    if (Files.exists(dir) == false) return;
    try (var stream = Files.walk(dir)) {
      stream.sorted(java.util.Comparator.reverseOrder()).forEach(p -> {
        try {
          Files.deleteIfExists(p);
        } catch (IOException e) {
          // best-effort cleanup
        }
      });
    }
  }

  private static void printUsage() {
    System.err.println("Usage: BEIRBenchmark --data-dir <path> [--datasets ds1,ds2,...] "
        + "[--top-k N] [--output results.tsv]");
  }

  // --- Result records ---

  record StrategyResult(String strategyName, NDCGEvaluator.Metrics metrics, long timeMs) {}

  static final class DatasetResult {
    final String datasetName;
    final List<StrategyResult> results = new ArrayList<>();

    DatasetResult(String datasetName) {
      this.datasetName = datasetName;
    }

    void addResult(String strategyName, NDCGEvaluator.Metrics metrics, long timeMs) {
      results.add(new StrategyResult(strategyName, metrics, timeMs));
    }

    List<String> strategyNames() {
      return results.stream().map(StrategyResult::strategyName).toList();
    }

    double ndcgForStrategy(String name) {
      return results.stream()
          .filter(r -> r.strategyName.equals(name))
          .mapToDouble(r -> r.metrics.ndcg())
          .findFirst()
          .orElse(0);
    }
  }
}
