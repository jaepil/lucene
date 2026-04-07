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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;

/**
 * A query that computes per-signal fusion weights via learned attention, then delegates to {@link
 * LogOddsFusionQuery} for scoring.
 *
 * <p>At rewrite time, attention weights are computed as {@code softmax(W @ queryFeatures + b)}
 * where W and b are learned parameters (e.g., via {@link AttentionLogOddsWeightLearner}). The
 * result is a standard {@link LogOddsFusionQuery} with those weights, so all scoring, WAND
 * optimization, and explain logic is reused.
 *
 * <p>This enables query-dependent signal weighting: different queries can automatically receive
 * different text-vs-vector weight ratios based on their features.
 *
 * @lucene.experimental
 */
public final class AttentionLogOddsFusionQuery extends Query implements Iterable<Query> {

  private final List<Query> orderedClauses;
  private final float alpha;
  private final float[][] weightMatrix;
  private final float[] bias;
  private final float[] queryFeatures;
  private final float[] logitMin;
  private final float[] logitMax;

  /**
   * Creates a new AttentionLogOddsFusionQuery.
   *
   * @param clauses the sub-queries to combine
   * @param alpha confidence scaling exponent (0.5 = sqrt(n) law)
   * @param weightMatrix learned weight matrix W[nSignals][nQueryFeatures]
   * @param bias learned bias vector b[nSignals]
   * @param queryFeatures query feature vector for computing attention weights
   * @param logitMin per-signal logit lower bounds for normalization, or null to use softplus gating
   * @param logitMax per-signal logit upper bounds for normalization, or null to use softplus gating
   */
  public AttentionLogOddsFusionQuery(
      Collection<? extends Query> clauses,
      float alpha,
      float[][] weightMatrix,
      float[] bias,
      float[] queryFeatures,
      float[] logitMin,
      float[] logitMax) {
    Objects.requireNonNull(clauses, "clauses must not be null");
    Objects.requireNonNull(weightMatrix, "weightMatrix must not be null");
    Objects.requireNonNull(bias, "bias must not be null");
    Objects.requireNonNull(queryFeatures, "queryFeatures must not be null");

    if (Float.isNaN(alpha) || alpha < 0 || alpha > 1) {
      throw new IllegalArgumentException("alpha must be in [0, 1], got " + alpha);
    }
    if (weightMatrix.length != clauses.size()) {
      throw new IllegalArgumentException(
          "weightMatrix rows "
              + weightMatrix.length
              + " must equal clauses size "
              + clauses.size());
    }
    if (bias.length != clauses.size()) {
      throw new IllegalArgumentException(
          "bias length " + bias.length + " must equal clauses size " + clauses.size());
    }
    int nFeatures = queryFeatures.length;
    for (int i = 0; i < weightMatrix.length; i++) {
      if (weightMatrix[i].length != nFeatures) {
        throw new IllegalArgumentException(
            "weightMatrix row "
                + i
                + " has length "
                + weightMatrix[i].length
                + ", expected "
                + nFeatures);
      }
    }

    this.orderedClauses = new ArrayList<>(clauses);
    this.alpha = alpha;
    this.weightMatrix = deepCopy(weightMatrix);
    this.bias = bias.clone();
    this.queryFeatures = queryFeatures.clone();
    this.logitMin = logitMin != null ? logitMin.clone() : null;
    this.logitMax = logitMax != null ? logitMax.clone() : null;
  }

  /**
   * Creates a new AttentionLogOddsFusionQuery without logit normalization (uses softplus gating).
   */
  public AttentionLogOddsFusionQuery(
      Collection<? extends Query> clauses,
      float alpha,
      float[][] weightMatrix,
      float[] bias,
      float[] queryFeatures) {
    this(clauses, alpha, weightMatrix, bias, queryFeatures, null, null);
  }

  @Override
  public Iterator<Query> iterator() {
    return getClauses().iterator();
  }

  /** Returns an unmodifiable view of the clauses in insertion order. */
  public List<Query> getClauses() {
    return List.copyOf(orderedClauses);
  }

  /** Returns the confidence scaling exponent. */
  public float getAlpha() {
    return alpha;
  }

  /** Returns a deep copy of the weight matrix. */
  public float[][] getWeightMatrix() {
    return deepCopy(weightMatrix);
  }

  /** Returns a copy of the bias vector. */
  public float[] getBias() {
    return bias.clone();
  }

  /** Returns a copy of the query features. */
  public float[] getQueryFeatures() {
    return queryFeatures.clone();
  }

  /**
   * Computes the attention weights as softmax(W @ queryFeatures + b).
   *
   * @return per-signal weights summing to 1.0
   */
  public float[] computeAttentionWeights() {
    int nSignals = weightMatrix.length;
    double[] z = new double[nSignals];
    double maxZ = Double.NEGATIVE_INFINITY;
    for (int j = 0; j < nSignals; j++) {
      z[j] = bias[j];
      for (int k = 0; k < queryFeatures.length; k++) {
        z[j] += weightMatrix[j][k] * queryFeatures[k];
      }
      maxZ = Math.max(maxZ, z[j]);
    }
    double sumExp = 0;
    for (int j = 0; j < nSignals; j++) {
      z[j] = Math.exp(z[j] - maxZ);
      sumExp += z[j];
    }
    float[] result = new float[nSignals];
    for (int j = 0; j < nSignals; j++) {
      result[j] = (float) (z[j] / sumExp);
    }
    return result;
  }

  @Override
  public Query rewrite(IndexSearcher indexSearcher) throws IOException {
    float[] weights = computeAttentionWeights();
    List<Query> rewrittenClauses = new ArrayList<>();
    for (Query sub : orderedClauses) {
      rewrittenClauses.add(sub.rewrite(indexSearcher));
    }
    return new LogOddsFusionQuery(rewrittenClauses, alpha, weights, logitMin, logitMax);
  }

  @Override
  public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost)
      throws IOException {
    return rewrite(searcher).createWeight(searcher, scoreMode, boost);
  }

  @Override
  public void visit(QueryVisitor visitor) {
    QueryVisitor v = visitor.getSubVisitor(BooleanClause.Occur.SHOULD, this);
    for (Query q : orderedClauses) {
      q.visit(v);
    }
  }

  @Override
  public String toString(String field) {
    float[] weights = computeAttentionWeights();
    StringBuilder sb = new StringBuilder("AttentionLogOdds(");
    for (int i = 0; i < orderedClauses.size(); i++) {
      if (i > 0) sb.append(" & ");
      Query q = orderedClauses.get(i);
      if (q instanceof BooleanQuery) {
        sb.append("(").append(q.toString(field)).append(")");
      } else {
        sb.append(q.toString(field));
      }
    }
    sb.append(")^").append(alpha);
    sb.append(" w=").append(Arrays.toString(weights));
    return sb.toString();
  }

  @Override
  public boolean equals(Object other) {
    return sameClassAs(other) && equalsTo(getClass().cast(other));
  }

  private boolean equalsTo(AttentionLogOddsFusionQuery other) {
    return alpha == other.alpha
        && orderedClauses.equals(other.orderedClauses)
        && Arrays.deepEquals(weightMatrix, other.weightMatrix)
        && Arrays.equals(bias, other.bias)
        && Arrays.equals(queryFeatures, other.queryFeatures)
        && Arrays.equals(logitMin, other.logitMin)
        && Arrays.equals(logitMax, other.logitMax);
  }

  @Override
  public int hashCode() {
    int h = classHash();
    h = 31 * h + Float.floatToIntBits(alpha);
    h = 31 * h + orderedClauses.hashCode();
    h = 31 * h + Arrays.deepHashCode(weightMatrix);
    h = 31 * h + Arrays.hashCode(bias);
    h = 31 * h + Arrays.hashCode(queryFeatures);
    h = 31 * h + Arrays.hashCode(logitMin);
    h = 31 * h + Arrays.hashCode(logitMax);
    return h;
  }

  private static float[][] deepCopy(float[][] src) {
    float[][] dst = new float[src.length][];
    for (int i = 0; i < src.length; i++) {
      dst[i] = src[i].clone();
    }
    return dst;
  }
}
