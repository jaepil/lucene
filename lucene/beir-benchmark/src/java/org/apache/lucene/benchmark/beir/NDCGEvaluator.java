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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/** Computes NDCG@k, MAP@k, and Recall@k -- the standard BEIR evaluation metrics. */
final class NDCGEvaluator {

  record Metrics(double ndcg, double map, double recall) {
    @Override
    public String toString() {
      return String.format("NDCG@10=%.4f  MAP@10=%.4f  Recall@100=%.4f", ndcg, map, recall);
    }
  }

  private NDCGEvaluator() {}

  /**
   * Computes NDCG@k for a single query.
   *
   * @param rankedDocIds ordered list of retrieved document IDs
   * @param relevance map from doc_id to relevance grade
   * @param k cutoff
   */
  static double ndcgAtK(List<String> rankedDocIds, Map<String, Integer> relevance, int k) {
    double dcg = 0.0;
    int limit = Math.min(k, rankedDocIds.size());
    for (int i = 0; i < limit; i++) {
      String docId = rankedDocIds.get(i);
      int rel = relevance.getOrDefault(docId, 0);
      if (rel > 0) {
        dcg += (Math.pow(2, rel) - 1.0) / log2(i + 2);
      }
    }

    double idcg = computeIDCG(relevance, k);
    return idcg > 0 ? dcg / idcg : 0.0;
  }

  /**
   * Computes Average Precision at k for a single query. Uses graded relevance: a document is
   * considered relevant if its grade > 0.
   */
  static double mapAtK(List<String> rankedDocIds, Map<String, Integer> relevance, int k) {
    int limit = Math.min(k, rankedDocIds.size());
    double sumPrecision = 0.0;
    int relevantCount = 0;

    for (int i = 0; i < limit; i++) {
      String docId = rankedDocIds.get(i);
      int rel = relevance.getOrDefault(docId, 0);
      if (rel > 0) {
        relevantCount++;
        sumPrecision += (double) relevantCount / (i + 1);
      }
    }

    long totalRelevant = relevance.values().stream().filter(v -> v > 0).count();
    return totalRelevant > 0 ? sumPrecision / totalRelevant : 0.0;
  }

  /**
   * Computes Recall at k for a single query. A document is considered relevant if its grade > 0.
   */
  static double recallAtK(List<String> rankedDocIds, Map<String, Integer> relevance, int k) {
    int limit = Math.min(k, rankedDocIds.size());
    long totalRelevant = relevance.values().stream().filter(v -> v > 0).count();
    if (totalRelevant == 0) return 0.0;

    int retrieved = 0;
    for (int i = 0; i < limit; i++) {
      String docId = rankedDocIds.get(i);
      if (relevance.getOrDefault(docId, 0) > 0) {
        retrieved++;
      }
    }
    return (double) retrieved / totalRelevant;
  }

  /**
   * Computes mean metrics over all queries.
   *
   * @param results map from query_id to ordered list of retrieved doc_ids
   * @param qrels map from query_id to (doc_id -> relevance)
   * @param ndcgK cutoff for NDCG
   * @param recallK cutoff for Recall
   */
  static Metrics computeMean(
      Map<String, List<String>> results,
      Map<String, Map<String, Integer>> qrels,
      int ndcgK,
      int recallK) {

    double sumNDCG = 0, sumMAP = 0, sumRecall = 0;
    int count = 0;

    for (Map.Entry<String, Map<String, Integer>> entry : qrels.entrySet()) {
      String queryId = entry.getKey();
      Map<String, Integer> relevance = entry.getValue();
      List<String> ranked = results.getOrDefault(queryId, List.of());

      sumNDCG += ndcgAtK(ranked, relevance, ndcgK);
      sumMAP += mapAtK(ranked, relevance, ndcgK);
      sumRecall += recallAtK(ranked, relevance, recallK);
      count++;
    }

    if (count == 0) return new Metrics(0, 0, 0);
    return new Metrics(sumNDCG / count, sumMAP / count, sumRecall / count);
  }

  private static double computeIDCG(Map<String, Integer> relevance, int k) {
    List<Integer> grades = new ArrayList<>(relevance.values());
    grades.sort(Comparator.reverseOrder());
    double idcg = 0.0;
    int limit = Math.min(k, grades.size());
    for (int i = 0; i < limit; i++) {
      int rel = grades.get(i);
      if (rel > 0) {
        idcg += (Math.pow(2, rel) - 1.0) / log2(i + 2);
      }
    }
    return idcg;
  }

  private static double log2(double x) {
    return Math.log(x) / Math.log(2);
  }
}
