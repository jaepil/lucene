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
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.AttentionLogOddsFusionQuery;
import org.apache.lucene.search.AttentionLogOddsWeightLearner;
import org.apache.lucene.search.BayesianScoreQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.LogOddsFusionQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;

/** Defines and executes different search strategies against a BEIR index. */
final class BEIRSearcher {

  /** A named search strategy. */
  interface SearchStrategy {
    String name();

    TopDocs search(IndexSearcher searcher, String queryText, float[] queryVector, int topK)
        throws IOException, ParseException;
  }

  private final IndexSearcher searcher;
  private final Analyzer analyzer;
  private final float bayesianAlpha;
  private final float bayesianBeta;
  private final float bayesianBaseRate;

  BEIRSearcher(
      IndexSearcher searcher,
      Analyzer analyzer,
      float bayesianAlpha,
      float bayesianBeta,
      float bayesianBaseRate) {
    this.searcher = searcher;
    this.analyzer = analyzer;
    this.bayesianAlpha = bayesianAlpha;
    this.bayesianBeta = bayesianBeta;
    this.bayesianBaseRate = bayesianBaseRate;
  }

  private AttentionLogOddsWeightLearner trainedLearner;

  /** Returns all strategies to benchmark (attention must be trained first via trainAttention). */
  List<SearchStrategy> allStrategies() {
    List<SearchStrategy> strategies = new ArrayList<>();
    strategies.add(bm25());
    strategies.add(bayesianBM25());
    strategies.add(knn());
    strategies.add(rrfHybrid());
    strategies.add(logOddsFusionHybrid());
    strategies.add(logOddsFusionWeightedHybrid(0.7f, 0.3f));
    strategies.add(logOddsFusionWeightedHybrid(0.5f, 0.5f));
    strategies.add(logOddsFusionWeightedHybrid(0.3f, 0.7f));
    if (trainedLearner != null) {
      strategies.add(attentionLogOddsFusionHybrid());
    }
    return strategies;
  }

  /**
   * Trains the attention weight learner using qrels as supervision. For each query, runs both BM25
   * and KNN, collects probability scores for relevant and non-relevant documents, and fits the
   * attention network to minimize BCE loss.
   *
   * <p>Query features: [word_count_normalized, char_length_normalized, 1.0]
   */
  void trainAttention(
      List<BEIRDataset.QueryEntry> queries, Map<String, Map<String, Integer>> qrels)
      throws IOException, ParseException {
    int nSignals = 2;
    // Use query embedding as features (384-dim) + 1 bias
    int embeddingDim = queries.getFirst().embedding().length;
    int nFeatures = embeddingDim + 1;
    AttentionLogOddsWeightLearner learner =
        new AttentionLogOddsWeightLearner(nSignals, nFeatures, 0.5, true, 42);

    List<double[]> allProbs = new ArrayList<>();
    List<Double> allLabels = new ArrayList<>();
    List<double[]> allFeatures = new ArrayList<>();

    searcher.setSimilarity(new BM25Similarity());
    StoredFields storedFields = searcher.storedFields();

    int trainQueries = Math.min(queries.size(), 200);
    System.out.printf("  Training attention on %d queries (features=%d)...%n",
        trainQueries, nFeatures);

    for (int qi = 0; qi < trainQueries; qi++) {
      BEIRDataset.QueryEntry query = queries.get(qi);
      Map<String, Integer> relevance = qrels.getOrDefault(query.id(), Map.of());
      if (relevance.isEmpty()) continue;

      double[] qf = extractQueryFeatures(query.text(), query.embedding());

      // Run BM25 (Bayesian) for this query
      Query textQuery = buildTextQuery(query.text());
      Query bayesianQuery =
          new BayesianScoreQuery(textQuery, bayesianAlpha, bayesianBeta, bayesianBaseRate);
      TopDocs textDocs = searcher.search(bayesianQuery, 50);

      // Run KNN for this query
      TopDocs knnDocs =
          searcher.search(
              new KnnFloatVectorQuery(BEIRIndexer.FIELD_EMBEDDING, query.embedding(), 50), 50);

      // Build a map of docId -> [textScore, knnScore]
      Map<String, double[]> docScores = new LinkedHashMap<>();
      for (ScoreDoc sd : textDocs.scoreDocs) {
        String docId = storedFields.document(sd.doc).get(BEIRIndexer.FIELD_ID);
        docScores.computeIfAbsent(docId, k -> new double[] {0.5, 0.5})[0] = sd.score;
      }
      for (ScoreDoc sd : knnDocs.scoreDocs) {
        String docId = storedFields.document(sd.doc).get(BEIRIndexer.FIELD_ID);
        docScores.computeIfAbsent(docId, k -> new double[] {0.5, 0.5})[1] = sd.score;
      }

      // Create training samples
      for (Map.Entry<String, double[]> entry : docScores.entrySet()) {
        String docId = entry.getKey();
        double[] scores = entry.getValue();
        double label = relevance.getOrDefault(docId, 0) > 0 ? 1.0 : 0.0;

        // Clamp probabilities to (0, 1)
        double p0 = Math.clamp(scores[0], 1e-7, 1.0 - 1e-7);
        double p1 = Math.clamp(scores[1], 1e-7, 1.0 - 1e-7);

        allProbs.add(new double[] {p0, p1});
        allLabels.add(label);
        allFeatures.add(qf);
      }
    }

    if (allProbs.isEmpty()) {
      System.out.println("  WARNING: No training data for attention, skipping.");
      return;
    }

    // Balanced sampling: downsample non-relevant to at most 3x relevant count
    List<Integer> posIdx = new ArrayList<>();
    List<Integer> negIdx = new ArrayList<>();
    for (int i = 0; i < allLabels.size(); i++) {
      if (allLabels.get(i) > 0.5) {
        posIdx.add(i);
      } else {
        negIdx.add(i);
      }
    }

    int maxNeg = posIdx.size() * 3;
    java.util.Random balanceRng = new java.util.Random(42);
    if (negIdx.size() > maxNeg) {
      java.util.Collections.shuffle(negIdx, balanceRng);
      negIdx = negIdx.subList(0, maxNeg);
    }

    List<Integer> sampledIdx = new ArrayList<>(posIdx);
    sampledIdx.addAll(negIdx);
    java.util.Collections.shuffle(sampledIdx, balanceRng);

    int m = sampledIdx.size();
    double[][] probs = new double[m][];
    double[] labels = new double[m];
    double[][] features = new double[m][];
    for (int i = 0; i < m; i++) {
      int idx = sampledIdx.get(i);
      probs[i] = allProbs.get(idx);
      labels[i] = allLabels.get(idx);
      features[i] = allFeatures.get(idx);
    }

    long relevant = posIdx.size();
    System.out.printf("  Raw samples: %d (%d relevant, %d non-relevant)%n",
        allLabels.size(), relevant, allLabels.size() - relevant);
    System.out.printf("  Balanced samples: %d (%d relevant, %d non-relevant)%n",
        m, relevant, m - relevant);

    // Batch training
    learner.fit(probs, labels, features, 0.1, 1000, 1e-7);

    // Report learned weights for sample queries
    if (queries.size() > 0) {
      float[] qf0 = toFloat(extractQueryFeatures(
          queries.getFirst().text(), queries.getFirst().embedding()));
      float[] w0 = learner.computeWeightsAveraged(qf0);
      System.out.printf("  Sample weights (query 0: \"%s\"): text=%.3f, vector=%.3f%n",
          queries.getFirst().text().substring(0, Math.min(40, queries.getFirst().text().length())),
          w0[0], w0[1]);
    }
    if (queries.size() > queries.size() / 2) {
      BEIRDataset.QueryEntry midQ = queries.get(queries.size() / 2);
      float[] qfM = toFloat(extractQueryFeatures(midQ.text(), midQ.embedding()));
      float[] wM = learner.computeWeightsAveraged(qfM);
      System.out.printf("  Sample weights (query %d: \"%s\"): text=%.3f, vector=%.3f%n",
          queries.size() / 2,
          midQ.text().substring(0, Math.min(40, midQ.text().length())),
          wM[0], wM[1]);
    }

    this.trainedLearner = learner;
  }

  static double[] extractQueryFeatures(String queryText, float[] embedding) {
    double[] features = new double[embedding.length + 1];
    for (int i = 0; i < embedding.length; i++) {
      features[i] = embedding[i];
    }
    features[embedding.length] = 1.0;
    return features;
  }

  /**
   * Runs a strategy against all queries and returns results.
   *
   * @return map from query_id to ordered list of retrieved doc_ids
   */
  Map<String, List<String>> runStrategy(
      SearchStrategy strategy, List<BEIRDataset.QueryEntry> queries, int topK)
      throws IOException, ParseException {
    Map<String, List<String>> results = new LinkedHashMap<>();
    StoredFields storedFields = searcher.storedFields();
    for (BEIRDataset.QueryEntry query : queries) {
      TopDocs topDocs = strategy.search(searcher, query.text(), query.embedding(), topK);
      List<String> docIds = new ArrayList<>();
      for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
        String docId = storedFields.document(scoreDoc.doc).get(BEIRIndexer.FIELD_ID);
        docIds.add(docId);
      }
      results.put(query.id(), docIds);
    }
    return results;
  }

  // --- Strategy implementations ---

  private Query buildTextQuery(String queryText) throws ParseException {
    QueryParser parser = new QueryParser(BEIRIndexer.FIELD_CONTENTS, analyzer);
    parser.setDefaultOperator(QueryParser.Operator.OR);
    String escaped = QueryParser.escape(queryText);
    return parser.parse(escaped);
  }

  SearchStrategy bm25() {
    return new SearchStrategy() {
      @Override
      public String name() {
        return "BM25";
      }

      @Override
      public TopDocs search(
          IndexSearcher searcher, String queryText, float[] queryVector, int topK)
          throws IOException, ParseException {
        searcher.setSimilarity(new BM25Similarity());
        return searcher.search(buildTextQuery(queryText), topK);
      }
    };
  }

  SearchStrategy bayesianBM25() {
    return new SearchStrategy() {
      @Override
      public String name() {
        return "BayesianBM25";
      }

      @Override
      public TopDocs search(
          IndexSearcher searcher, String queryText, float[] queryVector, int topK)
          throws IOException, ParseException {
        searcher.setSimilarity(new BM25Similarity());
        Query textQuery = buildTextQuery(queryText);
        Query bayesianQuery =
            new BayesianScoreQuery(textQuery, bayesianAlpha, bayesianBeta, bayesianBaseRate);
        return searcher.search(bayesianQuery, topK);
      }
    };
  }

  SearchStrategy knn() {
    return new SearchStrategy() {
      @Override
      public String name() {
        return "KNN";
      }

      @Override
      public TopDocs search(
          IndexSearcher searcher, String queryText, float[] queryVector, int topK)
          throws IOException {
        return searcher.search(
            new KnnFloatVectorQuery(BEIRIndexer.FIELD_EMBEDDING, queryVector, topK), topK);
      }
    };
  }

  SearchStrategy rrfHybrid() {
    return new SearchStrategy() {
      @Override
      public String name() {
        return "RRF(BM25+KNN)";
      }

      @Override
      public TopDocs search(
          IndexSearcher searcher, String queryText, float[] queryVector, int topK)
          throws IOException, ParseException {
        // Retrieve more candidates from each signal, then fuse
        int candidateK = topK * 3;
        searcher.setSimilarity(new BM25Similarity());
        TopDocs textDocs = searcher.search(buildTextQuery(queryText), candidateK);
        TopDocs vectorDocs =
            searcher.search(
                new KnnFloatVectorQuery(BEIRIndexer.FIELD_EMBEDDING, queryVector, candidateK),
                candidateK);

        return reciprocalRankFusion(textDocs, vectorDocs, topK, 60);
      }
    };
  }

  SearchStrategy logOddsFusionHybrid() {
    return new SearchStrategy() {
      @Override
      public String name() {
        return "LogOdds(BayesBM25+KNN)";
      }

      @Override
      public TopDocs search(
          IndexSearcher searcher, String queryText, float[] queryVector, int topK)
          throws IOException, ParseException {
        searcher.setSimilarity(new BM25Similarity());
        Query textQuery = buildTextQuery(queryText);
        Query bayesianQuery =
            new BayesianScoreQuery(textQuery, bayesianAlpha, bayesianBeta, bayesianBaseRate);

        // KNN scores via cosine similarity are already in [0, 1] for normalized vectors
        // but Lucene KnnFloatVectorQuery scores are (1 + cosine) / 2, already in (0,1)
        Query vectorQuery =
            new KnnFloatVectorQuery(BEIRIndexer.FIELD_EMBEDDING, queryVector, topK * 3);

        List<Query> clauses = List.of(bayesianQuery, vectorQuery);
        Query fusionQuery = new LogOddsFusionQuery(clauses);
        return searcher.search(fusionQuery, topK);
      }
    };
  }

  SearchStrategy logOddsFusionWeightedHybrid(float textWeight, float vectorWeight) {
    String label =
        String.format("LogOddsW(%.1f*BayesBM25+%.1f*KNN)", textWeight, vectorWeight);
    return new SearchStrategy() {
      @Override
      public String name() {
        return label;
      }

      @Override
      public TopDocs search(
          IndexSearcher searcher, String queryText, float[] queryVector, int topK)
          throws IOException, ParseException {
        searcher.setSimilarity(new BM25Similarity());
        Query textQuery = buildTextQuery(queryText);
        Query bayesianQuery =
            new BayesianScoreQuery(textQuery, bayesianAlpha, bayesianBeta, bayesianBaseRate);

        Query vectorQuery =
            new KnnFloatVectorQuery(BEIRIndexer.FIELD_EMBEDDING, queryVector, topK * 3);

        List<Query> clauses = List.of(bayesianQuery, vectorQuery);
        float[] weights = new float[] {textWeight, vectorWeight};
        Query fusionQuery = new LogOddsFusionQuery(clauses, 0.5f, weights);
        return searcher.search(fusionQuery, topK);
      }
    };
  }

  SearchStrategy attentionLogOddsFusionHybrid() {
    return new SearchStrategy() {
      @Override
      public String name() {
        return "Attention(BayesBM25+KNN)";
      }

      @Override
      public TopDocs search(
          IndexSearcher searcher, String queryText, float[] queryVector, int topK)
          throws IOException, ParseException {
        searcher.setSimilarity(new BM25Similarity());
        Query textQuery = buildTextQuery(queryText);
        Query bayesianQuery =
            new BayesianScoreQuery(textQuery, bayesianAlpha, bayesianBeta, bayesianBaseRate);
        Query vectorQuery =
            new KnnFloatVectorQuery(BEIRIndexer.FIELD_EMBEDDING, queryVector, topK * 3);

        List<Query> clauses = List.of(bayesianQuery, vectorQuery);
        // queryVector is the embedding, build feature vector from it
        double[] qfD = new double[queryVector.length + 1];
        for (int i = 0; i < queryVector.length; i++) qfD[i] = queryVector[i];
        qfD[queryVector.length] = 1.0;
        float[] qf = toFloat(qfD);
        float[][] W = trainedLearner.getWeightMatrix();
        float[] bias = trainedLearner.getBias();
        float[] logitMinBounds = trainedLearner.isNormalize() ? trainedLearner.getLogitMin() : null;
        float[] logitMaxBounds = trainedLearner.isNormalize() ? trainedLearner.getLogitMax() : null;

        Query attnQuery = new AttentionLogOddsFusionQuery(
            clauses, 0.5f, W, bias, qf, logitMinBounds, logitMaxBounds);
        return searcher.search(attnQuery, topK);
      }
    };
  }

  private static float[] toFloat(double[] d) {
    float[] f = new float[d.length];
    for (int i = 0; i < d.length; i++) f[i] = (float) d[i];
    return f;
  }

  /** Reciprocal Rank Fusion for two TopDocs results. */
  private static TopDocs reciprocalRankFusion(
      TopDocs docs1, TopDocs docs2, int topK, int rrfK) {
    Map<Integer, Double> scores = new LinkedHashMap<>();
    for (int i = 0; i < docs1.scoreDocs.length; i++) {
      int doc = docs1.scoreDocs[i].doc;
      scores.merge(doc, 1.0 / (rrfK + i + 1), Double::sum);
    }
    for (int i = 0; i < docs2.scoreDocs.length; i++) {
      int doc = docs2.scoreDocs[i].doc;
      scores.merge(doc, 1.0 / (rrfK + i + 1), Double::sum);
    }

    List<Map.Entry<Integer, Double>> sorted = new ArrayList<>(scores.entrySet());
    sorted.sort(Map.Entry.<Integer, Double>comparingByValue().reversed());

    int n = Math.min(topK, sorted.size());
    ScoreDoc[] scoreDocs = new ScoreDoc[n];
    for (int i = 0; i < n; i++) {
      Map.Entry<Integer, Double> e = sorted.get(i);
      scoreDocs[i] = new ScoreDoc(e.getKey(), e.getValue().floatValue());
    }
    return new TopDocs(
        new org.apache.lucene.search.TotalHits(
            scores.size(), org.apache.lucene.search.TotalHits.Relation.EQUAL_TO),
        scoreDocs);
  }
}
