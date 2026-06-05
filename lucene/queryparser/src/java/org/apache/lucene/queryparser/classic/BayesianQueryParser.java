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
package org.apache.lucene.queryparser.classic;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.queries.function.FunctionScoreQuery;
import org.apache.lucene.search.BayesianScoreQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.DoubleValues;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.FullPrecisionFloatVectorSimilarityValuesSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.LogOddsFusionQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;

/**
 * A {@link QueryParser} variant that parses the classic Lucene query syntax, including vector
 * literals, and converts scoring clauses to Bayesian probability signals.
 *
 * <p>Boolean structure is preserved: {@code AND} and required clauses remain {@code MUST}, hard
 * non-vector prohibited clauses remain {@code MUST_NOT}, and pure {@code OR} groups are combined
 * with {@link LogOddsFusionQuery}. Prohibited vector clauses are treated as candidate-local
 * negative evidence: the positive candidate query is rescored by {@code 1 - vectorSimilarity(doc)}
 * using full-precision vector values rather than excluding the global kNN top-k set. Text and
 * vector clauses are both regular Lucene {@link Query} leaves, so a query such as {@code body:cat
 * AND vector:[1, 2, 3] NOT status:deleted} is parsed and executed as one hybrid query.
 *
 * @lucene.experimental
 */
public class BayesianQueryParser extends QueryParser {

  private float scoreAlpha = 1f;
  private float scoreBeta = 0f;
  private float scoreBaseRate = 0f;
  private float fusionAlpha = 0.5f;

  /**
   * Create a Bayesian query parser.
   *
   * @param f the default field for query terms
   * @param a used to find terms in the query text
   */
  public BayesianQueryParser(String f, Analyzer a) {
    super(f, a);
  }

  /**
   * Sets sigmoid calibration parameters used by {@link BayesianScoreQuery}.
   *
   * @param alpha sigmoid steepness
   * @param beta sigmoid midpoint
   */
  public void setBayesianScoreCalibration(float alpha, float beta) {
    setBayesianScoreCalibration(alpha, beta, 0f);
  }

  /**
   * Sets sigmoid calibration parameters used by {@link BayesianScoreQuery}.
   *
   * @param alpha sigmoid steepness
   * @param beta sigmoid midpoint
   * @param baseRate corpus-level relevance prior in {@code [0, 1)}
   */
  public void setBayesianScoreCalibration(float alpha, float beta, float baseRate) {
    validateScoreCalibration(alpha, beta, baseRate);
    this.scoreAlpha = alpha;
    this.scoreBeta = beta;
    this.scoreBaseRate = baseRate;
  }

  /** Returns the sigmoid steepness parameter. */
  public float getBayesianScoreAlpha() {
    return scoreAlpha;
  }

  /** Returns the sigmoid midpoint parameter. */
  public float getBayesianScoreBeta() {
    return scoreBeta;
  }

  /** Returns the corpus-level relevance prior, or 0 if disabled. */
  public float getBayesianScoreBaseRate() {
    return scoreBaseRate;
  }

  /** Sets the confidence scaling exponent used by {@link LogOddsFusionQuery}. */
  public void setFusionAlpha(float fusionAlpha) {
    if (Float.isNaN(fusionAlpha) || fusionAlpha < 0 || fusionAlpha > 1) {
      throw new IllegalArgumentException("fusionAlpha must be in [0, 1], got " + fusionAlpha);
    }
    this.fusionAlpha = fusionAlpha;
  }

  /** Returns the confidence scaling exponent used by {@link LogOddsFusionQuery}. */
  public float getFusionAlpha() {
    return fusionAlpha;
  }

  @Override
  public Query parse(String query) throws ParseException {
    return toBayesianQuery(super.parse(query));
  }

  /** Converts a parsed query tree into a Bayesian probability-scored query tree. */
  protected Query toBayesianQuery(Query query) {
    if (query instanceof MatchNoDocsQuery || query instanceof BayesianScoreQuery) {
      return query;
    }
    if (query instanceof LogOddsFusionQuery) {
      return query;
    }
    if (query instanceof BoostQuery boostQuery) {
      Query boosted = new BoostQuery(boostQuery.getQuery(), boostQuery.getBoost());
      return newBayesianScoreQuery(boosted);
    }
    if (query instanceof BooleanQuery booleanQuery) {
      return toBayesianBooleanQuery(booleanQuery);
    }
    return newBayesianScoreQuery(query);
  }

  /** Factory method for Bayesian score wrappers. */
  protected Query newBayesianScoreQuery(Query query) {
    return new BayesianScoreQuery(query, scoreAlpha, scoreBeta, scoreBaseRate);
  }

  /** Factory method for Bayesian log-odds fusion. */
  protected Query newLogOddsFusionQuery(List<Query> clauses) {
    if (clauses.size() == 1) {
      return clauses.get(0);
    }
    return new LogOddsFusionQuery(clauses, fusionAlpha);
  }

  private Query toBayesianBooleanQuery(BooleanQuery booleanQuery) {
    List<BooleanClause> clauses = booleanQuery.clauses();
    if (clauses.isEmpty()) {
      return booleanQuery;
    }

    boolean hasRequiredOrFilter = false;
    boolean hasProhibited = false;
    for (BooleanClause clause : clauses) {
      if (clause.isRequired() || clause.occur() == BooleanClause.Occur.FILTER) {
        hasRequiredOrFilter = true;
      } else if (clause.isProhibited()) {
        hasProhibited = true;
      }
    }

    if (hasRequiredOrFilter == false
        && hasProhibited == false
        && booleanQuery.getMinimumNumberShouldMatch() <= 1) {
      List<Query> scoringSignals = new ArrayList<>();
      for (BooleanClause clause : clauses) {
        scoringSignals.add(toBayesianQuery(clause.query()));
      }
      return newLogOddsFusionQuery(scoringSignals);
    }

    if (hasProhibited) {
      return toBayesianBooleanQueryWithProhibitedClauses(booleanQuery);
    }

    return newBayesianScoreQuery(buildBayesianBooleanQuery(booleanQuery).build());
  }

  private Query toBayesianBooleanQueryWithProhibitedClauses(BooleanQuery booleanQuery) {
    BooleanQuery.Builder positiveBuilder = new BooleanQuery.Builder();
    positiveBuilder.setMinimumNumberShouldMatch(booleanQuery.getMinimumNumberShouldMatch());
    List<Query> hardProhibited = new ArrayList<>();
    List<KnnFloatVectorQuery> prohibitedVectors = new ArrayList<>();

    for (BooleanClause clause : booleanQuery.clauses()) {
      Query child = clause.query();
      if (clause.isProhibited()) {
        KnnFloatVectorQuery vectorQuery = getKnnFloatVectorQuery(child);
        if (vectorQuery != null) {
          prohibitedVectors.add(vectorQuery);
        } else {
          hardProhibited.add(child);
        }
      } else {
        positiveBuilder.add(child, clause.occur());
      }
    }

    BooleanQuery positiveQuery = positiveBuilder.build();
    if (positiveQuery.clauses().isEmpty()) {
      return newBayesianScoreQuery(booleanQuery);
    }
    Query candidateQuery = toBayesianQuery(positiveQuery);

    for (KnnFloatVectorQuery vectorQuery : prohibitedVectors) {
      candidateQuery =
          FunctionScoreQuery.boostByValue(
              candidateQuery,
              new ComplementDoubleValuesSource(
                  new FullPrecisionFloatVectorSimilarityValuesSource(
                      vectorQuery.getTargetCopy(), vectorQuery.getField())));
    }

    if (hardProhibited.isEmpty()) {
      return candidateQuery;
    }

    BooleanQuery.Builder builder = new BooleanQuery.Builder();
    builder.add(candidateQuery, BooleanClause.Occur.MUST);
    for (Query prohibited : hardProhibited) {
      builder.add(prohibited, BooleanClause.Occur.MUST_NOT);
    }
    return newBayesianScoreQuery(builder.build());
  }

  private BooleanQuery.Builder buildBayesianBooleanQuery(BooleanQuery booleanQuery) {
    BooleanQuery.Builder builder = new BooleanQuery.Builder();
    builder.setMinimumNumberShouldMatch(booleanQuery.getMinimumNumberShouldMatch());
    for (BooleanClause clause : booleanQuery.clauses()) {
      Query child = clause.query();
      if (clause.isProhibited() || clause.occur() == BooleanClause.Occur.FILTER) {
        builder.add(child, clause.occur());
      } else {
        builder.add(toBayesianQuery(child), clause.occur());
      }
    }
    return builder;
  }

  private KnnFloatVectorQuery getKnnFloatVectorQuery(Query query) {
    if (query instanceof KnnFloatVectorQuery vectorQuery) {
      return vectorQuery;
    }
    if (query instanceof BoostQuery boostQuery) {
      return getKnnFloatVectorQuery(boostQuery.getQuery());
    }
    return null;
  }

  private static void validateScoreCalibration(float alpha, float beta, float baseRate) {
    if (Float.isFinite(alpha) == false || alpha <= 0) {
      throw new IllegalArgumentException("alpha must be a positive finite value, got " + alpha);
    }
    if (Float.isFinite(beta) == false) {
      throw new IllegalArgumentException("beta must be a finite value, got " + beta);
    }
    if (baseRate < 0 || baseRate >= 1) {
      throw new IllegalArgumentException("baseRate must be in [0, 1), got " + baseRate);
    }
  }

  private static class ComplementDoubleValuesSource extends DoubleValuesSource {

    private final DoubleValuesSource source;

    ComplementDoubleValuesSource(DoubleValuesSource source) {
      this.source = Objects.requireNonNull(source);
    }

    @Override
    public DoubleValues getValues(LeafReaderContext ctx, DoubleValues scores) throws IOException {
      DoubleValues values = source.getValues(ctx, scores);
      return new DoubleValues() {
        @Override
        public double doubleValue() throws IOException {
          return 1d - Math.clamp(values.doubleValue(), 0d, 1d);
        }

        @Override
        public boolean advanceExact(int doc) throws IOException {
          return values.advanceExact(doc);
        }
      };
    }

    @Override
    public boolean needsScores() {
      return source.needsScores();
    }

    @Override
    public DoubleValuesSource rewrite(IndexSearcher reader) throws IOException {
      DoubleValuesSource rewritten = source.rewrite(reader);
      if (rewritten == source) {
        return this;
      }
      return new ComplementDoubleValuesSource(rewritten);
    }

    @Override
    public Explanation explain(LeafReaderContext ctx, int docId, Explanation scoreExplanation)
        throws IOException {
      DoubleValues values = getValues(ctx, null);
      if (values.advanceExact(docId)) {
        return Explanation.match(
            values.doubleValue(), "complement of vector similarity, computed as 1 - similarity");
      }
      return Explanation.noMatch("no vector similarity value");
    }

    @Override
    public boolean isCacheable(LeafReaderContext ctx) {
      return source.isCacheable(ctx);
    }

    @Override
    public String toString() {
      return "complement(" + source.getClass().getSimpleName() + ")";
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj == null || getClass() != obj.getClass()) {
        return false;
      }
      ComplementDoubleValuesSource other = (ComplementDoubleValuesSource) obj;
      return source.equals(other.source);
    }

    @Override
    public int hashCode() {
      return Objects.hash(getClass(), source);
    }
  }
}
