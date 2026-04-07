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

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Loads BEIR dataset files produced by the Python preparation script.
 *
 * <p>Expected directory layout:
 *
 * <pre>
 *   datasetDir/
 *     corpus.jsonl   - one JSON object per line: {_id, title, text, embedding}
 *     queries.jsonl  - one JSON object per line: {_id, text, embedding}
 *     qrels.tsv      - tab-separated: query_id  doc_id  score
 * </pre>
 */
final class BEIRDataset {

  record CorpusDoc(String id, String title, String text, float[] embedding) {}

  record QueryEntry(String id, String text, float[] embedding) {}

  private final List<CorpusDoc> corpus;
  private final List<QueryEntry> queries;
  private final Map<String, Map<String, Integer>> qrels;

  BEIRDataset(List<CorpusDoc> corpus, List<QueryEntry> queries, Map<String, Map<String, Integer>> qrels) {
    this.corpus = corpus;
    this.queries = queries;
    this.qrels = qrels;
  }

  List<CorpusDoc> corpus() {
    return corpus;
  }

  List<QueryEntry> queries() {
    return queries;
  }

  Map<String, Map<String, Integer>> qrels() {
    return qrels;
  }

  static BEIRDataset load(Path datasetDir) throws IOException {
    List<CorpusDoc> corpus = loadCorpus(datasetDir.resolve("corpus.jsonl"));
    List<QueryEntry> queries = loadQueries(datasetDir.resolve("queries.jsonl"));
    Map<String, Map<String, Integer>> qrels = loadQrels(datasetDir.resolve("qrels.tsv"));
    return new BEIRDataset(corpus, queries, qrels);
  }

  private static List<CorpusDoc> loadCorpus(Path path) throws IOException {
    List<CorpusDoc> docs = new ArrayList<>();
    try (BufferedReader reader = Files.newBufferedReader(path)) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isBlank()) continue;
        Map<String, Object> obj = SimpleJSON.parse(line);
        String id = (String) obj.get("_id");
        String title = obj.containsKey("title") ? (String) obj.get("title") : "";
        String text = (String) obj.get("text");
        float[] embedding = toFloatArray(obj.get("embedding"));
        docs.add(new CorpusDoc(id, title, text, embedding));
      }
    }
    return docs;
  }

  private static List<QueryEntry> loadQueries(Path path) throws IOException {
    List<QueryEntry> queries = new ArrayList<>();
    try (BufferedReader reader = Files.newBufferedReader(path)) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isBlank()) continue;
        Map<String, Object> obj = SimpleJSON.parse(line);
        String id = (String) obj.get("_id");
        String text = (String) obj.get("text");
        float[] embedding = toFloatArray(obj.get("embedding"));
        queries.add(new QueryEntry(id, text, embedding));
      }
    }
    return queries;
  }

  private static Map<String, Map<String, Integer>> loadQrels(Path path) throws IOException {
    Map<String, Map<String, Integer>> qrels = new LinkedHashMap<>();
    try (BufferedReader reader = Files.newBufferedReader(path)) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isBlank()) continue;
        String[] parts = line.split("\t");
        if (parts.length < 3) continue;
        String queryId = parts[0];
        String docId = parts[1];
        int score = Integer.parseInt(parts[2]);
        qrels.computeIfAbsent(queryId, k -> new LinkedHashMap<>()).put(docId, score);
      }
    }
    return qrels;
  }

  @SuppressWarnings("unchecked")
  private static float[] toFloatArray(Object obj) {
    List<Object> list = (List<Object>) obj;
    float[] arr = new float[list.size()];
    for (int i = 0; i < list.size(); i++) {
      arr[i] = ((Number) list.get(i)).floatValue();
    }
    return arr;
  }
}
