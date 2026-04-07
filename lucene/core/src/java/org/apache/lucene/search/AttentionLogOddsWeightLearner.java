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

/**
 * Learns query-dependent signal weights for {@link LogOddsFusionQuery} via attention.
 *
 * <p>Computes per-signal softmax attention weights from query features: {@code w_i(q) = softmax(W @
 * queryFeatures + b)[i]}, then trains W and b to minimize binary cross-entropy loss on labeled
 * relevance data.
 *
 * <p>Supports both batch training ({@link #fit}) and online updates ({@link #update}). Uses
 * Polyak-averaged parameters for stable inference via {@link #computeWeightsAveraged}.
 *
 * <p>Typical usage with corpus-level self-supervised learning:
 *
 * <ol>
 *   <li>Sample pseudo-queries from corpus documents
 *   <li>Run each signal (text, vector) to collect probability scores
 *   <li>Use source document as relevant (label=1), random documents as irrelevant (label=0)
 *   <li>Extract query features (e.g., number of terms, mean IDF)
 *   <li>Call {@link #fit} to learn W and b
 *   <li>At query time, call {@link #computeWeights} and pass result to {@link LogOddsFusionQuery}
 *       or {@link AttentionLogOddsFusionQuery}
 * </ol>
 *
 * @lucene.experimental
 */
public class AttentionLogOddsWeightLearner {

  private static final double CLAMP_MIN = 1e-10;
  private static final double CLAMP_MAX = 1.0 - 1e-10;

  private final int nSignals;
  private final int nQueryFeatures;
  private final double alpha;
  private final boolean normalize;

  private final double[][] W;
  private final double[] b;

  private final double[][] WAvg;
  private final double[] bAvg;

  private final double[] logitMin;
  private final double[] logitMax;

  private final double[][] gradWEma;
  private final double[] gradBEma;
  private int nUpdates;

  /**
   * Creates a new learner with Xavier-style random initialization.
   *
   * @param nSignals number of probability signals to combine (must be >= 1)
   * @param nQueryFeatures dimensionality of the query feature vector (must be >= 1)
   * @param alpha confidence scaling exponent for log-odds fusion (typically 0.5)
   * @param normalize whether to apply per-signal min-max logit normalization during training. When
   *     true, logit values are normalized to [0, 1] per signal before weighted aggregation, and the
   *     normalization bounds are stored for use at inference time. This replaces softplus gating by
   *     ensuring all contributions are non-negative.
   * @param seed random seed for weight initialization
   */
  public AttentionLogOddsWeightLearner(
      int nSignals, int nQueryFeatures, double alpha, boolean normalize, long seed) {
    if (nSignals < 1) {
      throw new IllegalArgumentException("nSignals must be >= 1, got " + nSignals);
    }
    if (nQueryFeatures < 1) {
      throw new IllegalArgumentException("nQueryFeatures must be >= 1, got " + nQueryFeatures);
    }
    if (alpha < 0 || alpha > 1) {
      throw new IllegalArgumentException("alpha must be in [0, 1], got " + alpha);
    }
    this.nSignals = nSignals;
    this.nQueryFeatures = nQueryFeatures;
    this.alpha = alpha;
    this.normalize = normalize;

    double initScale = 1.0 / Math.sqrt(nQueryFeatures);
    Random rng = new Random(seed);
    this.W = new double[nSignals][nQueryFeatures];
    for (int i = 0; i < nSignals; i++) {
      for (int j = 0; j < nQueryFeatures; j++) {
        W[i][j] = rng.nextGaussian() * initScale;
      }
    }
    this.b = new double[nSignals];

    this.WAvg = new double[nSignals][nQueryFeatures];
    for (int i = 0; i < nSignals; i++) {
      System.arraycopy(W[i], 0, WAvg[i], 0, nQueryFeatures);
    }
    this.bAvg = new double[nSignals];

    this.logitMin = new double[nSignals];
    this.logitMax = new double[nSignals];
    java.util.Arrays.fill(logitMin, Double.POSITIVE_INFINITY);
    java.util.Arrays.fill(logitMax, Double.NEGATIVE_INFINITY);

    this.gradWEma = new double[nSignals][nQueryFeatures];
    this.gradBEma = new double[nSignals];
    this.nUpdates = 0;
  }

  /**
   * Creates a new learner with normalize=false and default seed 0.
   *
   * @param nSignals number of probability signals
   * @param nQueryFeatures dimensionality of query features
   * @param alpha confidence scaling exponent
   */
  public AttentionLogOddsWeightLearner(int nSignals, int nQueryFeatures, double alpha) {
    this(nSignals, nQueryFeatures, alpha, false, 0);
  }

  /**
   * Creates a new learner with normalize=false.
   *
   * @param nSignals number of probability signals
   * @param nQueryFeatures dimensionality of query features
   * @param alpha confidence scaling exponent
   * @param seed random seed for weight initialization
   */
  public AttentionLogOddsWeightLearner(int nSignals, int nQueryFeatures, double alpha, long seed) {
    this(nSignals, nQueryFeatures, alpha, false, seed);
  }

  /** Returns the number of signals. */
  public int getNSignals() {
    return nSignals;
  }

  /** Returns the number of query features. */
  public int getNQueryFeatures() {
    return nQueryFeatures;
  }

  /** Returns the confidence scaling exponent. */
  public double getAlpha() {
    return alpha;
  }

  /** Returns whether per-signal logit normalization is enabled. */
  public boolean isNormalize() {
    return normalize;
  }

  /**
   * Returns a copy of the per-signal logit lower bounds used for normalization. Only meaningful
   * after {@link #fit} has been called with normalization enabled.
   */
  public float[] getLogitMin() {
    float[] result = new float[nSignals];
    for (int i = 0; i < nSignals; i++) {
      result[i] = (float) logitMin[i];
    }
    return result;
  }

  /**
   * Returns a copy of the per-signal logit upper bounds used for normalization. Only meaningful
   * after {@link #fit} has been called with normalization enabled.
   */
  public float[] getLogitMax() {
    float[] result = new float[nSignals];
    for (int i = 0; i < nSignals; i++) {
      result[i] = (float) logitMax[i];
    }
    return result;
  }

  /** Returns a copy of the weight matrix W[nSignals][nQueryFeatures]. */
  public float[][] getWeightMatrix() {
    float[][] result = new float[nSignals][nQueryFeatures];
    for (int i = 0; i < nSignals; i++) {
      for (int j = 0; j < nQueryFeatures; j++) {
        result[i][j] = (float) WAvg[i][j];
      }
    }
    return result;
  }

  /** Returns a copy of the bias vector b[nSignals]. */
  public float[] getBias() {
    float[] result = new float[nSignals];
    for (int i = 0; i < nSignals; i++) {
      result[i] = (float) bAvg[i];
    }
    return result;
  }

  /**
   * Batch gradient descent on BCE loss to learn W and b.
   *
   * @param probs probability signals, shape [m][nSignals], values in (0, 1)
   * @param labels binary relevance labels, shape [m], values 0.0 or 1.0
   * @param queryFeatures query feature vectors, shape [m][nQueryFeatures]
   * @param learningRate step size for gradient descent
   * @param maxIterations maximum number of iterations
   * @param tolerance convergence threshold on maximum absolute parameter change
   */
  public void fit(
      double[][] probs,
      double[] labels,
      double[][] queryFeatures,
      double learningRate,
      int maxIterations,
      double tolerance) {
    int m = probs.length;
    double scale = Math.pow(nSignals, alpha);

    double[][] x = new double[m][nSignals];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < nSignals; j++) {
        x[i][j] = logit(clamp(probs[i][j]));
      }
    }

    if (normalize) {
      // Compute per-signal logit bounds from training data
      for (int j = 0; j < nSignals; j++) {
        logitMin[j] = Double.POSITIVE_INFINITY;
        logitMax[j] = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < m; i++) {
          logitMin[j] = Math.min(logitMin[j], x[i][j]);
          logitMax[j] = Math.max(logitMax[j], x[i][j]);
        }
      }
      // Apply min-max normalization to [0, 1]
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < nSignals; j++) {
          double range = logitMax[j] - logitMin[j];
          if (range > 0) {
            x[i][j] = (x[i][j] - logitMin[j]) / range;
          } else {
            x[i][j] = 0.5;
          }
        }
      }
    }

    for (int iter = 0; iter < maxIterations; iter++) {
      // Forward: z = queryFeatures @ W^T + b
      double[][] z = new double[m][nSignals];
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < nSignals; j++) {
          z[i][j] = b[j];
          for (int k = 0; k < nQueryFeatures; k++) {
            z[i][j] += queryFeatures[i][k] * W[j][k];
          }
        }
      }

      // w = softmax(z) per row
      double[][] w = new double[m][];
      for (int i = 0; i < m; i++) {
        w[i] = softmax(z[i]);
      }

      // xBarW = sum(w * x, axis=1)
      double[] xBarW = new double[m];
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < nSignals; j++) {
          xBarW[i] += w[i][j] * x[i][j];
        }
      }

      // p = sigmoid(scale * xBarW)
      double[] p = new double[m];
      for (int i = 0; i < m; i++) {
        p[i] = sigmoid(scale * xBarW[i]);
      }

      // Gradient: gradZ[i][j] = scale * (p[i] - labels[i]) * w[i][j] * (x[i][j] - xBarW[i])
      double[][] gradZ = new double[m][nSignals];
      for (int i = 0; i < m; i++) {
        double error = p[i] - labels[i];
        for (int j = 0; j < nSignals; j++) {
          gradZ[i][j] = scale * error * w[i][j] * (x[i][j] - xBarW[i]);
        }
      }

      // gradW = (1/m) * gradZ^T @ queryFeatures, gradB = mean(gradZ, axis=0)
      double maxChange = 0;
      for (int j = 0; j < nSignals; j++) {
        for (int k = 0; k < nQueryFeatures; k++) {
          double g = 0;
          for (int i = 0; i < m; i++) {
            g += gradZ[i][j] * queryFeatures[i][k];
          }
          g /= m;
          double oldW = W[j][k];
          W[j][k] -= learningRate * g;
          maxChange = Math.max(maxChange, Math.abs(W[j][k] - oldW));
        }
        double gb = 0;
        for (int i = 0; i < m; i++) {
          gb += gradZ[i][j];
        }
        gb /= m;
        double oldB = b[j];
        b[j] -= learningRate * gb;
        maxChange = Math.max(maxChange, Math.abs(b[j] - oldB));
      }

      if (maxChange < tolerance) {
        break;
      }
    }

    // Copy to averaged parameters
    for (int i = 0; i < nSignals; i++) {
      System.arraycopy(W[i], 0, WAvg[i], 0, nQueryFeatures);
    }
    System.arraycopy(b, 0, bAvg, 0, nSignals);

    // Reset online state
    nUpdates = 0;
    for (int i = 0; i < nSignals; i++) {
      java.util.Arrays.fill(gradWEma[i], 0);
    }
    java.util.Arrays.fill(gradBEma, 0);
  }

  /**
   * Online SGD update from a single observation.
   *
   * <p>Uses EMA gradient smoothing with bias correction, L2 gradient clipping, learning rate decay,
   * and Polyak parameter averaging.
   *
   * @param probs probability signals, shape [nSignals], values in (0, 1)
   * @param label binary relevance label (0.0 or 1.0)
   * @param queryFeatures query feature vector, shape [nQueryFeatures]
   * @param learningRate base step size, decayed as lr / (1 + t / decayTau)
   * @param momentum EMA decay factor for gradient smoothing
   * @param decayTau time constant for learning rate decay
   * @param maxGradNorm maximum L2 norm for gradient clipping
   * @param avgDecay decay factor for Polyak parameter averaging
   */
  public void update(
      double[] probs,
      double label,
      double[] queryFeatures,
      double learningRate,
      double momentum,
      double decayTau,
      double maxGradNorm,
      double avgDecay) {
    double scale = Math.pow(nSignals, alpha);

    // Forward
    double[] x = new double[nSignals];
    for (int j = 0; j < nSignals; j++) {
      x[j] = logit(clamp(probs[j]));
    }

    if (normalize) {
      for (int j = 0; j < nSignals; j++) {
        double range = logitMax[j] - logitMin[j];
        if (range > 0) {
          x[j] = Math.clamp((x[j] - logitMin[j]) / range, 0.0, 1.0);
        } else {
          x[j] = 0.5;
        }
      }
    }

    double[] z = new double[nSignals];
    for (int j = 0; j < nSignals; j++) {
      z[j] = b[j];
      for (int k = 0; k < nQueryFeatures; k++) {
        z[j] += queryFeatures[k] * W[j][k];
      }
    }

    double[] w = softmax(z);

    double xBarW = 0;
    for (int j = 0; j < nSignals; j++) {
      xBarW += w[j] * x[j];
    }

    double p = sigmoid(scale * xBarW);
    double error = p - label;

    // Gradient
    double[] gradZ = new double[nSignals];
    for (int j = 0; j < nSignals; j++) {
      gradZ[j] = scale * error * w[j] * (x[j] - xBarW);
    }

    double[][] gradW = new double[nSignals][nQueryFeatures];
    for (int j = 0; j < nSignals; j++) {
      for (int k = 0; k < nQueryFeatures; k++) {
        gradW[j][k] = gradZ[j] * queryFeatures[k];
      }
    }

    // EMA smoothing
    for (int j = 0; j < nSignals; j++) {
      for (int k = 0; k < nQueryFeatures; k++) {
        gradWEma[j][k] = momentum * gradWEma[j][k] + (1.0 - momentum) * gradW[j][k];
      }
      gradBEma[j] = momentum * gradBEma[j] + (1.0 - momentum) * gradZ[j];
    }

    // Bias correction
    nUpdates++;
    double correction = 1.0 - Math.pow(momentum, nUpdates);

    // Corrected gradients + L2 clipping
    double[][] correctedW = new double[nSignals][nQueryFeatures];
    double[] correctedB = new double[nSignals];
    double gradNormSq = 0;
    for (int j = 0; j < nSignals; j++) {
      for (int k = 0; k < nQueryFeatures; k++) {
        correctedW[j][k] = gradWEma[j][k] / correction;
        gradNormSq += correctedW[j][k] * correctedW[j][k];
      }
      correctedB[j] = gradBEma[j] / correction;
      gradNormSq += correctedB[j] * correctedB[j];
    }

    double gradNorm = Math.sqrt(gradNormSq);
    if (gradNorm > maxGradNorm) {
      double clipScale = maxGradNorm / gradNorm;
      for (int j = 0; j < nSignals; j++) {
        for (int k = 0; k < nQueryFeatures; k++) {
          correctedW[j][k] *= clipScale;
        }
        correctedB[j] *= clipScale;
      }
    }

    // Learning rate decay and parameter step
    double effectiveLR = learningRate / (1.0 + nUpdates / decayTau);
    for (int j = 0; j < nSignals; j++) {
      for (int k = 0; k < nQueryFeatures; k++) {
        W[j][k] -= effectiveLR * correctedW[j][k];
      }
      b[j] -= effectiveLR * correctedB[j];
    }

    // Polyak averaging
    for (int j = 0; j < nSignals; j++) {
      for (int k = 0; k < nQueryFeatures; k++) {
        WAvg[j][k] = avgDecay * WAvg[j][k] + (1.0 - avgDecay) * W[j][k];
      }
      bAvg[j] = avgDecay * bAvg[j] + (1.0 - avgDecay) * b[j];
    }
  }

  /**
   * Online SGD update with default hyperparameters (lr=0.01, momentum=0.9, decayTau=1000,
   * maxGradNorm=1.0, avgDecay=0.995).
   */
  public void update(double[] probs, double label, double[] queryFeatures) {
    update(probs, label, queryFeatures, 0.01, 0.9, 1000.0, 1.0, 0.995);
  }

  /**
   * Computes attention weights for a query using raw (non-averaged) parameters.
   *
   * @param queryFeatures query feature vector
   * @return softmax attention weights summing to 1.0
   */
  public float[] computeWeights(float[] queryFeatures) {
    return computeWeightsInternal(queryFeatures, W, b);
  }

  /**
   * Computes attention weights using Polyak-averaged parameters (smoother, recommended for
   * inference after online updates).
   *
   * @param queryFeatures query feature vector
   * @return softmax attention weights summing to 1.0
   */
  public float[] computeWeightsAveraged(float[] queryFeatures) {
    return computeWeightsInternal(queryFeatures, WAvg, bAvg);
  }

  private float[] computeWeightsInternal(float[] queryFeatures, double[][] wMat, double[] bias) {
    double[] z = new double[nSignals];
    for (int j = 0; j < nSignals; j++) {
      z[j] = bias[j];
      for (int k = 0; k < nQueryFeatures; k++) {
        z[j] += wMat[j][k] * queryFeatures[k];
      }
    }
    double[] sw = softmax(z);
    float[] result = new float[nSignals];
    for (int j = 0; j < nSignals; j++) {
      result[j] = (float) sw[j];
    }
    return result;
  }

  private static double clamp(double p) {
    return Math.clamp(p, CLAMP_MIN, CLAMP_MAX);
  }

  private static double logit(double p) {
    return Math.log(p / (1.0 - p));
  }

  private static double sigmoid(double x) {
    if (x >= 0) {
      return 1.0 / (1.0 + Math.exp(-x));
    } else {
      double expX = Math.exp(x);
      return expX / (1.0 + expX);
    }
  }

  private static double[] softmax(double[] z) {
    double max = Double.NEGATIVE_INFINITY;
    for (double v : z) {
      max = Math.max(max, v);
    }
    double[] result = new double[z.length];
    double sum = 0;
    for (int i = 0; i < z.length; i++) {
      result[i] = Math.exp(z[i] - max);
      sum += result[i];
    }
    for (int i = 0; i < z.length; i++) {
      result[i] /= sum;
    }
    return result;
  }
}
