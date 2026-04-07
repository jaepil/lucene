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
import java.nio.file.Path;
import java.util.List;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * Indexes BEIR corpus documents into a Lucene index with both text and vector fields.
 *
 * <p>Index schema:
 *
 * <ul>
 *   <li>{@code id} - StringField (stored, indexed) for document ID lookup
 *   <li>{@code title} - TextField for BM25 search
 *   <li>{@code contents} - TextField for BM25 search (title + text concatenated)
 *   <li>{@code embedding} - KnnFloatVectorField for dense retrieval (cosine similarity)
 * </ul>
 */
final class BEIRIndexer {

  static final String FIELD_ID = "id";
  static final String FIELD_TITLE = "title";
  static final String FIELD_CONTENTS = "contents";
  static final String FIELD_EMBEDDING = "embedding";

  private BEIRIndexer() {}

  static Analyzer createAnalyzer() {
    return new EnglishAnalyzer();
  }

  static IndexSearcher buildIndex(Path indexPath, List<BEIRDataset.CorpusDoc> corpus)
      throws IOException {
    Analyzer analyzer = createAnalyzer();
    Directory dir = FSDirectory.open(indexPath);
    IndexWriterConfig config = new IndexWriterConfig(analyzer);
    config.setRAMBufferSizeMB(512);
    config.setUseCompoundFile(false);

    int embeddingDim = corpus.getFirst().embedding().length;

    try (IndexWriter writer = new IndexWriter(dir, config)) {
      int count = 0;
      for (BEIRDataset.CorpusDoc doc : corpus) {
        Document luceneDoc = new Document();
        luceneDoc.add(new StringField(FIELD_ID, doc.id(), Field.Store.YES));
        luceneDoc.add(new StoredField(FIELD_TITLE, doc.title()));

        String contents = (doc.title() + " " + doc.text()).trim();
        luceneDoc.add(new TextField(FIELD_CONTENTS, contents, Field.Store.NO));

        luceneDoc.add(
            new KnnFloatVectorField(
                FIELD_EMBEDDING, doc.embedding(), VectorSimilarityFunction.COSINE));

        writer.addDocument(luceneDoc);
        count++;
        if (count % 10000 == 0) {
          System.out.printf("  Indexed %,d / %,d documents%n", count, corpus.size());
        }
      }
      writer.forceMerge(1);
    }

    System.out.printf("  Index built: %,d documents, dim=%d%n", corpus.size(), embeddingDim);
    DirectoryReader reader = DirectoryReader.open(dir);
    return new IndexSearcher(reader);
  }
}
