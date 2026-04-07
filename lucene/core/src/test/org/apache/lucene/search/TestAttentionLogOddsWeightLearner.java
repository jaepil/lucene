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

import java.util.Random;
import org.apache.lucene.tests.util.LuceneTestCase;

/**
 * Tests for {@link AttentionLogOddsWeightLearner}. Verifies that the attention mechanism learns
 * query-dependent signal routing from synthetic self-supervised data.
 */
public class TestAttentionLogOddsWeightLearner extends LuceneTestCase {

  public void testComputeWeightsSumToOne() {
    var learner = new AttentionLogOddsWeightLearner(3, 2, 0.5);
    float[] weights = learner.computeWeights(new float[] {1.0f, 0.5f});
    float sum = 0;
    for (float w : weights) {
      assertTrue("weight must be non-negative", w >= 0);
      sum += w;
    }
    assertEquals("weights must sum to 1", 1.0f, sum, 1e-5f);
  }

  public void testComputeWeightsAveragedSumToOne() {
    var learner = new AttentionLogOddsWeightLearner(2, 3, 0.5);
    float[] weights = learner.computeWeightsAveraged(new float[] {0.1f, 0.2f, 0.3f});
    float sum = 0;
    for (float w : weights) {
      sum += w;
    }
    assertEquals("averaged weights must sum to 1", 1.0f, sum, 1e-5f);
  }

  public void testIllegalArguments() {
    expectThrows(
        IllegalArgumentException.class, () -> new AttentionLogOddsWeightLearner(0, 2, 0.5));
    expectThrows(
        IllegalArgumentException.class, () -> new AttentionLogOddsWeightLearner(2, 0, 0.5));
    expectThrows(
        IllegalArgumentException.class, () -> new AttentionLogOddsWeightLearner(2, 2, -0.1));
    expectThrows(
        IllegalArgumentException.class, () -> new AttentionLogOddsWeightLearner(2, 2, 1.1));
  }

  /**
   * Core test: attention learns query-dependent signal routing.
   *
   * <p>Setup (simulating corpus-level self-supervised learning):
   *
   * <ul>
   *   <li>2 signals: signal0 (like BM25 text), signal1 (like vector similarity)
   *   <li>1 query feature: queryType (1.0 = keyword-like, 0.0 = semantic-like)
   *   <li>For keyword queries: signal0 separates relevant/irrelevant well, signal1 is noise
   *   <li>For semantic queries: signal1 separates well, signal0 is noise
   * </ul>
   *
   * <p>After training, the learner should produce:
   *
   * <ul>
   *   <li>Keyword queries (feature=1): higher weight on signal0
   *   <li>Semantic queries (feature=0): higher weight on signal1
   * </ul>
   */
  public void testAttentionLearnsQueryDependentRouting() {
    int m = 400;
    double[][] probs = new double[m][2];
    double[] labels = new double[m];
    double[][] queryFeatures = new double[m][1];
    Random rng = new Random(42);

    for (int i = 0; i < m; i++) {
      boolean isKeyword = i < m / 2;
      queryFeatures[i][0] = isKeyword ? 1.0 : 0.0;
      boolean relevant = rng.nextBoolean();
      labels[i] = relevant ? 1.0 : 0.0;

      if (isKeyword) {
        // Signal 0 is discriminative for keyword queries
        probs[i][0] = relevant ? 0.80 + rng.nextDouble() * 0.15 : 0.15 + rng.nextDouble() * 0.15;
        // Signal 1 is noise (around 0.5)
        probs[i][1] = 0.40 + rng.nextDouble() * 0.20;
      } else {
        // Signal 0 is noise
        probs[i][0] = 0.40 + rng.nextDouble() * 0.20;
        // Signal 1 is discriminative for semantic queries
        probs[i][1] = relevant ? 0.80 + rng.nextDouble() * 0.15 : 0.15 + rng.nextDouble() * 0.15;
      }
    }

    var learner = new AttentionLogOddsWeightLearner(2, 1, 0.5, 42);
    learner.fit(probs, labels, queryFeatures, 0.1, 5000, 1e-8);

    // Keyword query: should prefer signal 0
    float[] keywordWeights = learner.computeWeights(new float[] {1.0f});
    assertTrue(
        "keyword query: signal0 weight ("
            + keywordWeights[0]
            + ") should be > signal1 weight ("
            + keywordWeights[1]
            + ")",
        keywordWeights[0] > keywordWeights[1]);

    // Semantic query: should prefer signal 1
    float[] semanticWeights = learner.computeWeights(new float[] {0.0f});
    assertTrue(
        "semantic query: signal1 weight ("
            + semanticWeights[1]
            + ") should be > signal0 weight ("
            + semanticWeights[0]
            + ")",
        semanticWeights[1] > semanticWeights[0]);
  }

  /** Verify that online update() modifies parameters over time. */
  public void testOnlineUpdateModifiesWeights() {
    var learner = new AttentionLogOddsWeightLearner(2, 1, 0.5, 0);

    float[] before = learner.computeWeights(new float[] {1.0f});

    // Feed observations where signal 0 is always right for feature=1
    Random rng = new Random(123);
    for (int i = 0; i < 100; i++) {
      boolean relevant = rng.nextBoolean();
      double[] probs = relevant ? new double[] {0.9, 0.5} : new double[] {0.1, 0.5};
      learner.update(probs, relevant ? 1.0 : 0.0, new double[] {1.0});
    }

    float[] after = learner.computeWeightsAveraged(new float[] {1.0f});

    // After training, signal 0 should have increased weight
    assertTrue("online update should increase signal0 weight for feature=1", after[0] > before[0]);
  }

  /** Verify getWeightMatrix and getBias return learned parameters. */
  public void testGetLearnedParameters() {
    var learner = new AttentionLogOddsWeightLearner(2, 3, 0.5);

    // After fit, parameters should be retrievable
    double[][] probs = {{0.8, 0.2}, {0.3, 0.7}};
    double[] labels = {1.0, 0.0};
    double[][] qf = {{1.0, 0.0, 0.5}, {0.0, 1.0, 0.5}};
    learner.fit(probs, labels, qf, 0.01, 100, 1e-6);

    float[][] W = learner.getWeightMatrix();
    float[] b = learner.getBias();
    assertEquals(2, W.length);
    assertEquals(3, W[0].length);
    assertEquals(2, b.length);
    assertEquals(2, learner.getNSignals());
    assertEquals(3, learner.getNQueryFeatures());
    assertEquals(0.5, learner.getAlpha(), 1e-10);
  }

  /** Verify fit converges: loss decreases over iterations. */
  public void testFitConverges() {
    int m = 100;
    double[][] probs = new double[m][2];
    double[] labels = new double[m];
    double[][] qf = new double[m][1];
    Random rng = new Random(99);

    for (int i = 0; i < m; i++) {
      boolean relevant = rng.nextBoolean();
      labels[i] = relevant ? 1.0 : 0.0;
      qf[i][0] = 1.0;
      probs[i][0] = relevant ? 0.8 : 0.2;
      probs[i][1] = relevant ? 0.7 : 0.3;
    }

    var learner = new AttentionLogOddsWeightLearner(2, 1, 0.5, 0);
    // Should not throw, should converge without issues
    learner.fit(probs, labels, qf, 0.05, 1000, 1e-8);

    // After fit, predictions should be reasonable
    float[] weights = learner.computeWeights(new float[] {1.0f});
    assertTrue("weight[0] should be valid", weights[0] >= 0 && weights[0] <= 1);
    assertTrue("weight[1] should be valid", weights[1] >= 0 && weights[1] <= 1);
    assertEquals("weights should sum to 1", 1.0f, weights[0] + weights[1], 1e-5f);
  }

  /** Test with 3 signals and 2 features: verify attention can learn multi-dimensional routing. */
  public void testThreeSignalsTwoFeatures() {
    int m = 600;
    double[][] probs = new double[m][3];
    double[] labels = new double[m];
    double[][] qf = new double[m][2];
    Random rng = new Random(77);

    for (int i = 0; i < m; i++) {
      int queryType = i % 3; // 0, 1, 2
      qf[i][0] = queryType == 0 ? 1.0 : 0.0;
      qf[i][1] = queryType == 1 ? 1.0 : 0.0;
      // queryType 2: both features are 0

      boolean relevant = rng.nextBoolean();
      labels[i] = relevant ? 1.0 : 0.0;

      // Each query type has a different best signal
      for (int s = 0; s < 3; s++) {
        if (s == queryType) {
          probs[i][s] = relevant ? 0.85 + rng.nextDouble() * 0.10 : 0.10 + rng.nextDouble() * 0.15;
        } else {
          probs[i][s] = 0.40 + rng.nextDouble() * 0.20;
        }
      }
    }

    var learner = new AttentionLogOddsWeightLearner(3, 2, 0.5, 42);
    learner.fit(probs, labels, qf, 0.1, 5000, 1e-8);

    // Query type 0 (feature=[1,0]): signal 0 should dominate
    float[] w0 = learner.computeWeights(new float[] {1.0f, 0.0f});
    assertTrue("type0: signal0 (" + w0[0] + ") > signal1 (" + w0[1] + ")", w0[0] > w0[1]);
    assertTrue("type0: signal0 (" + w0[0] + ") > signal2 (" + w0[2] + ")", w0[0] > w0[2]);

    // Query type 1 (feature=[0,1]): signal 1 should dominate
    float[] w1 = learner.computeWeights(new float[] {0.0f, 1.0f});
    assertTrue("type1: signal1 (" + w1[1] + ") > signal0 (" + w1[0] + ")", w1[1] > w1[0]);
    assertTrue("type1: signal1 (" + w1[1] + ") > signal2 (" + w1[2] + ")", w1[1] > w1[2]);
  }

  // ---- Normalization tests ----

  /** Verify fit with normalize=true stores valid logit bounds. */
  public void testNormalizeFitStoresBounds() {
    int m = 100;
    double[][] probs = new double[m][2];
    double[] labels = new double[m];
    double[][] qf = new double[m][1];
    Random rng = new Random(42);

    for (int i = 0; i < m; i++) {
      labels[i] = rng.nextBoolean() ? 1.0 : 0.0;
      qf[i][0] = 1.0;
      // Signal 0: wide range [0.05, 0.95]
      probs[i][0] = 0.05 + rng.nextDouble() * 0.90;
      // Signal 1: narrow range [0.4, 0.6]
      probs[i][1] = 0.40 + rng.nextDouble() * 0.20;
    }

    var learner = new AttentionLogOddsWeightLearner(2, 1, 0.5, true, 42);
    assertTrue("normalize should be enabled", learner.isNormalize());

    learner.fit(probs, labels, qf, 0.05, 1000, 1e-6);

    float[] logitMin = learner.getLogitMin();
    float[] logitMax = learner.getLogitMax();
    assertEquals(2, logitMin.length);
    assertEquals(2, logitMax.length);

    // Signal 0 should have wider logit range than signal 1
    float range0 = logitMax[0] - logitMin[0];
    float range1 = logitMax[1] - logitMin[1];
    assertTrue(
        "signal0 logit range (" + range0 + ") > signal1 range (" + range1 + ")", range0 > range1);

    // Bounds should be finite
    assertTrue("logitMin[0] finite", Float.isFinite(logitMin[0]));
    assertTrue("logitMax[0] finite", Float.isFinite(logitMax[0]));
    assertTrue("min < max for signal 0", logitMin[0] < logitMax[0]);
    assertTrue("min < max for signal 1", logitMin[1] < logitMax[1]);
  }

  /** Verify that normalize=true still learns correct query-dependent routing. */
  public void testNormalizePreservesRouting() {
    int m = 400;
    double[][] probs = new double[m][2];
    double[] labels = new double[m];
    double[][] qf = new double[m][1];
    Random rng = new Random(42);

    for (int i = 0; i < m; i++) {
      boolean isKeyword = i < m / 2;
      qf[i][0] = isKeyword ? 1.0 : 0.0;
      boolean relevant = rng.nextBoolean();
      labels[i] = relevant ? 1.0 : 0.0;
      if (isKeyword) {
        probs[i][0] = relevant ? 0.80 + rng.nextDouble() * 0.15 : 0.15 + rng.nextDouble() * 0.15;
        probs[i][1] = 0.40 + rng.nextDouble() * 0.20;
      } else {
        probs[i][0] = 0.40 + rng.nextDouble() * 0.20;
        probs[i][1] = relevant ? 0.80 + rng.nextDouble() * 0.15 : 0.15 + rng.nextDouble() * 0.15;
      }
    }

    var learner = new AttentionLogOddsWeightLearner(2, 1, 0.5, true, 42);
    learner.fit(probs, labels, qf, 0.1, 5000, 1e-8);

    float[] keywordW = learner.computeWeights(new float[] {1.0f});
    float[] semanticW = learner.computeWeights(new float[] {0.0f});

    assertTrue("keyword: signal0 > signal1", keywordW[0] > keywordW[1]);
    assertTrue("semantic: signal1 > signal0", semanticW[1] > semanticW[0]);
  }
}
