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
package org.apache.lucene.search;

import java.util.Arrays;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.search.QueryUtils;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Tests for {@link AttentionLogOddsFusionQuery}. */
public class TestAttentionLogOddsFusionQuery extends LuceneTestCase {

  private static final float BSQ_ALPHA = 0.5f;
  private static final float BSQ_BETA = 3.0f;

  private static Query bayesian(Query q) {
    return new BayesianScoreQuery(q, BSQ_ALPHA, BSQ_BETA);
  }

  private Directory dir;
  private IndexReader reader;
  private IndexSearcher searcher;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    dir = newDirectory();
    IndexWriter writer = new IndexWriter(dir, new IndexWriterConfig());

    Document doc0 = new Document();
    doc0.add(new TextField("body", "alpha beta gamma", Field.Store.YES));
    writer.addDocument(doc0);

    Document doc1 = new Document();
    doc1.add(new TextField("body", "alpha gamma delta", Field.Store.YES));
    writer.addDocument(doc1);

    Document doc2 = new Document();
    doc2.add(new TextField("body", "beta gamma delta", Field.Store.YES));
    writer.addDocument(doc2);

    Document doc3 = new Document();
    doc3.add(new TextField("body", "gamma delta epsilon", Field.Store.YES));
    writer.addDocument(doc3);

    writer.forceMerge(1);
    reader = DirectoryReader.open(writer);
    writer.close();
    searcher = new IndexSearcher(reader);
    searcher.setSimilarity(new BM25Similarity());
  }

  @Override
  public void tearDown() throws Exception {
    reader.close();
    dir.close();
    super.tearDown();
  }

  /** Attention weights that strongly prefer signal 0. */
  private static float[][] makeSignal0HeavyW() {
    // W[0][0]=2.0 makes signal 0 logit high when feature=1
    // W[1][0]=-2.0 makes signal 1 logit low when feature=1
    return new float[][] {{2.0f}, {-2.0f}};
  }

  /** Attention weights that strongly prefer signal 1. */
  private static float[][] makeSignal1HeavyW() {
    return new float[][] {{-2.0f}, {2.0f}};
  }

  private static float[] zeroBias() {
    return new float[] {0f, 0f};
  }

  public void testRewriteProducesLogOddsFusionQuery() throws Exception {
    Query q1 = bayesian(new TermQuery(new Term("body", "alpha")));
    Query q2 = bayesian(new TermQuery(new Term("body", "beta")));

    AttentionLogOddsFusionQuery attnQ =
        new AttentionLogOddsFusionQuery(
            Arrays.asList(q1, q2), 0.5f, makeSignal0HeavyW(), zeroBias(), new float[] {1.0f});

    Query rewritten = searcher.rewrite(attnQ);
    assertTrue(
        "should rewrite to LogOddsFusionQuery, got " + rewritten.getClass().getName(),
        rewritten instanceof LogOddsFusionQuery);

    LogOddsFusionQuery loq = (LogOddsFusionQuery) rewritten;
    assertNotNull("rewritten query should have weights", loq.getWeights());

    // With feature=1 and W=[[2],[-2]], b=[0,0]:
    // z = [2, -2], softmax([2,-2]) = [exp(2)/(exp(2)+exp(-2)), exp(-2)/(exp(2)+exp(-2))]
    // ~ [0.88, 0.12]
    float[] weights = loq.getWeights();
    assertTrue("signal 0 should dominate, got " + Arrays.toString(weights), weights[0] > 0.8f);
    assertTrue("signal 1 should be small", weights[1] < 0.2f);
  }

  public void testComputeAttentionWeights() {
    Query q1 = new TermQuery(new Term("body", "alpha"));
    Query q2 = new TermQuery(new Term("body", "beta"));

    // Equal features, zero bias -> uniform weights
    float[][] uniformW = {{0f}, {0f}};
    AttentionLogOddsFusionQuery uniformQ =
        new AttentionLogOddsFusionQuery(
            Arrays.asList(q1, q2), 0.5f, uniformW, zeroBias(), new float[] {0f});
    float[] uniformWeights = uniformQ.computeAttentionWeights();
    assertEquals("uniform weights[0]", 0.5f, uniformWeights[0], 1e-5f);
    assertEquals("uniform weights[1]", 0.5f, uniformWeights[1], 1e-5f);
  }

  public void testQueryDependentWeightsAffectRanking() throws Exception {
    Query qAlpha = bayesian(new TermQuery(new Term("body", "alpha")));
    Query qBeta = bayesian(new TermQuery(new Term("body", "beta")));

    // doc0: alpha beta gamma -> matches both
    // doc1: alpha gamma delta -> matches alpha only
    // doc2: beta gamma delta -> matches beta only

    // Attention preferring signal 0 (alpha)
    AttentionLogOddsFusionQuery alphaPreferred =
        new AttentionLogOddsFusionQuery(
            Arrays.asList(qAlpha, qBeta),
            0.5f,
            makeSignal0HeavyW(),
            zeroBias(),
            new float[] {1.0f});

    // Attention preferring signal 1 (beta)
    AttentionLogOddsFusionQuery betaPreferred =
        new AttentionLogOddsFusionQuery(
            Arrays.asList(qAlpha, qBeta),
            0.5f,
            makeSignal1HeavyW(),
            zeroBias(),
            new float[] {1.0f});

    ScoreDoc[] hitsAlpha = searcher.search(alphaPreferred, 10).scoreDocs;
    ScoreDoc[] hitsBeta = searcher.search(betaPreferred, 10).scoreDocs;

    // Find scores of doc1 (alpha-only) and doc2 (beta-only) in each case
    float doc1Alpha = 0, doc2Alpha = 0;
    float doc1Beta = 0, doc2Beta = 0;
    for (ScoreDoc hit : hitsAlpha) {
      if (hit.doc == 1) doc1Alpha = hit.score;
      if (hit.doc == 2) doc2Alpha = hit.score;
    }
    for (ScoreDoc hit : hitsBeta) {
      if (hit.doc == 1) doc1Beta = hit.score;
      if (hit.doc == 2) doc2Beta = hit.score;
    }

    assertTrue("alpha-preferred: doc1 > doc2", doc1Alpha > doc2Alpha);
    assertTrue("beta-preferred: doc2 > doc1", doc2Beta > doc1Beta);
  }

  public void testSearchProducesValidScores() throws Exception {
    Query q1 = bayesian(new TermQuery(new Term("body", "alpha")));
    Query q2 = bayesian(new TermQuery(new Term("body", "beta")));

    AttentionLogOddsFusionQuery attnQ =
        new AttentionLogOddsFusionQuery(
            Arrays.asList(q1, q2), 0.5f, makeSignal0HeavyW(), zeroBias(), new float[] {1.0f});

    ScoreDoc[] hits = searcher.search(attnQ, 10).scoreDocs;
    assertTrue("should have hits", hits.length > 0);
    for (ScoreDoc hit : hits) {
      assertTrue("score in (0,1): " + hit.score, hit.score > 0 && hit.score < 1);
    }
  }

  public void testQueryUtils() throws Exception {
    Query q1 = bayesian(new TermQuery(new Term("body", "alpha")));
    Query q2 = bayesian(new TermQuery(new Term("body", "beta")));

    AttentionLogOddsFusionQuery attnQ =
        new AttentionLogOddsFusionQuery(
            Arrays.asList(q1, q2), 0.5f, makeSignal0HeavyW(), zeroBias(), new float[] {1.0f});
    // After rewrite, QueryUtils validates the resulting LogOddsFusionQuery
    Query rewritten = searcher.rewrite(attnQ);
    QueryUtils.check(random(), rewritten, searcher);
  }

  public void testEqualsAndHashCode() {
    Query q1 = new TermQuery(new Term("body", "alpha"));
    Query q2 = new TermQuery(new Term("body", "beta"));

    AttentionLogOddsFusionQuery a =
        new AttentionLogOddsFusionQuery(
            Arrays.asList(q1, q2), 0.5f, makeSignal0HeavyW(), zeroBias(), new float[] {1.0f});
    AttentionLogOddsFusionQuery b =
        new AttentionLogOddsFusionQuery(
            Arrays.asList(q1, q2), 0.5f, makeSignal0HeavyW(), zeroBias(), new float[] {1.0f});
    AttentionLogOddsFusionQuery c =
        new AttentionLogOddsFusionQuery(
            Arrays.asList(q1, q2), 0.5f, makeSignal1HeavyW(), zeroBias(), new float[] {1.0f});
    AttentionLogOddsFusionQuery d =
        new AttentionLogOddsFusionQuery(
            Arrays.asList(q1, q2), 0.5f, makeSignal0HeavyW(), zeroBias(), new float[] {0.0f});

    assertEquals(a, b);
    assertEquals(a.hashCode(), b.hashCode());
    assertNotEquals(a, c); // different W
    assertNotEquals(a, d); // different queryFeatures
  }

  public void testToString() {
    Query q1 = new TermQuery(new Term("body", "alpha"));
    Query q2 = new TermQuery(new Term("body", "beta"));

    AttentionLogOddsFusionQuery attnQ =
        new AttentionLogOddsFusionQuery(
            Arrays.asList(q1, q2), 0.5f, makeSignal0HeavyW(), zeroBias(), new float[] {1.0f});
    String str = attnQ.toString("body");
    assertTrue("should contain AttentionLogOdds", str.contains("AttentionLogOdds"));
    assertTrue("should contain w=", str.contains("w="));
  }

  public void testIllegalArguments() {
    Query q1 = new TermQuery(new Term("body", "alpha"));
    Query q2 = new TermQuery(new Term("body", "beta"));

    // Wrong W rows
    expectThrows(
        IllegalArgumentException.class,
        () ->
            new AttentionLogOddsFusionQuery(
                Arrays.asList(q1, q2), 0.5f, new float[][] {{1f}}, zeroBias(), new float[] {1f}));

    // Wrong bias length
    expectThrows(
        IllegalArgumentException.class,
        () ->
            new AttentionLogOddsFusionQuery(
                Arrays.asList(q1, q2),
                0.5f,
                makeSignal0HeavyW(),
                new float[] {0f},
                new float[] {1f}));

    // Inconsistent feature dimensions
    expectThrows(
        IllegalArgumentException.class,
        () ->
            new AttentionLogOddsFusionQuery(
                Arrays.asList(q1, q2),
                0.5f,
                makeSignal0HeavyW(),
                zeroBias(),
                new float[] {1f, 2f}));
  }

  /**
   * End-to-end: train learner on synthetic data, then use learned parameters in
   * AttentionLogOddsFusionQuery to verify the full pipeline works.
   */
  public void testEndToEndWithLearnedParameters() throws Exception {
    // Train: signal 0 is good for feature=1, signal 1 is good for feature=0
    int m = 200;
    double[][] probs = new double[m][2];
    double[] labels = new double[m];
    double[][] qf = new double[m][1];
    java.util.Random rng = new java.util.Random(42);

    for (int i = 0; i < m; i++) {
      boolean isKeyword = i < m / 2;
      qf[i][0] = isKeyword ? 1.0 : 0.0;
      boolean relevant = rng.nextBoolean();
      labels[i] = relevant ? 1.0 : 0.0;
      if (isKeyword) {
        probs[i][0] = relevant ? 0.85 : 0.15;
        probs[i][1] = 0.45 + rng.nextDouble() * 0.10;
      } else {
        probs[i][0] = 0.45 + rng.nextDouble() * 0.10;
        probs[i][1] = relevant ? 0.85 : 0.15;
      }
    }

    var learner = new AttentionLogOddsWeightLearner(2, 1, 0.5, 42);
    learner.fit(probs, labels, qf, 0.1, 3000, 1e-8);

    // Get learned W, b
    float[][] W = learner.getWeightMatrix();
    float[] b = learner.getBias();

    Query q1 = bayesian(new TermQuery(new Term("body", "alpha")));
    Query q2 = bayesian(new TermQuery(new Term("body", "beta")));

    // Keyword query (feature=1): should prefer signal 0 (alpha)
    AttentionLogOddsFusionQuery keywordQ =
        new AttentionLogOddsFusionQuery(Arrays.asList(q1, q2), 0.5f, W, b, new float[] {1.0f});
    float[] keywordWeights = keywordQ.computeAttentionWeights();
    assertTrue(
        "keyword: signal0 (" + keywordWeights[0] + ") > signal1 (" + keywordWeights[1] + ")",
        keywordWeights[0] > keywordWeights[1]);

    // Semantic query (feature=0): should prefer signal 1 (beta)
    AttentionLogOddsFusionQuery semanticQ =
        new AttentionLogOddsFusionQuery(Arrays.asList(q1, q2), 0.5f, W, b, new float[] {0.0f});
    float[] semanticWeights = semanticQ.computeAttentionWeights();
    assertTrue(
        "semantic: signal1 (" + semanticWeights[1] + ") > signal0 (" + semanticWeights[0] + ")",
        semanticWeights[1] > semanticWeights[0]);

    // Both queries should produce valid search results
    ScoreDoc[] keywordHits = searcher.search(keywordQ, 10).scoreDocs;
    ScoreDoc[] semanticHits = searcher.search(semanticQ, 10).scoreDocs;
    assertTrue("keyword should have hits", keywordHits.length > 0);
    assertTrue("semantic should have hits", semanticHits.length > 0);
    for (ScoreDoc hit : keywordHits) {
      assertTrue("score in (0,1)", hit.score > 0 && hit.score < 1);
    }
    for (ScoreDoc hit : semanticHits) {
      assertTrue("score in (0,1)", hit.score > 0 && hit.score < 1);
    }
  }
}
