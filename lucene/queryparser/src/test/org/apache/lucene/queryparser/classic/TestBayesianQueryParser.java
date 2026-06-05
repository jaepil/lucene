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

import static org.apache.lucene.index.VectorSimilarityFunction.DOT_PRODUCT;

import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.function.FunctionScoreQuery;
import org.apache.lucene.search.BayesianScoreQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.LogOddsFusionQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.analysis.MockTokenizer;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Tests for {@link BayesianQueryParser}. */
public class TestBayesianQueryParser extends LuceneTestCase {

  public void testOrHybridUsesLogOddsFusion() throws Exception {
    BayesianQueryParser parser =
        new BayesianQueryParser(
            "body", new MockAnalyzer(random(), MockTokenizer.WHITESPACE, false));

    Query query = parser.parse("body:cat OR vector:[1, 2, 3]");
    assertTrue(query instanceof LogOddsFusionQuery);

    LogOddsFusionQuery fusionQuery = (LogOddsFusionQuery) query;
    List<Query> clauses = new ArrayList<>(fusionQuery.getClauses());
    assertEquals(2, clauses.size());
    assertBayesianWrapped(TermQuery.class, clauses.get(0));
    assertBayesianWrapped(KnnFloatVectorQuery.class, clauses.get(1));
  }

  public void testNonVectorNotPreservesHardBooleanSemantics() throws Exception {
    BayesianQueryParser parser =
        new BayesianQueryParser(
            "body", new MockAnalyzer(random(), MockTokenizer.WHITESPACE, false));

    Query query = parser.parse("body:cat AND vector:[1, 2, 3] NOT status:deleted");
    assertTrue(query instanceof BayesianScoreQuery);

    Query inner = ((BayesianScoreQuery) query).getQuery();
    assertTrue(inner instanceof BooleanQuery);

    BooleanQuery booleanQuery = (BooleanQuery) inner;
    assertEquals(2, booleanQuery.clauses().size());
    assertEquals(1, countOccur(booleanQuery, BooleanClause.Occur.MUST));
    assertEquals(1, countOccur(booleanQuery, BooleanClause.Occur.MUST_NOT));

    for (BooleanClause clause : booleanQuery.clauses()) {
      if (clause.occur() == BooleanClause.Occur.MUST) {
        assertTrue(clause.query() instanceof BayesianScoreQuery);
      } else {
        assertEquals(new TermQuery(new Term("status", "deleted")), clause.query());
      }
    }
  }

  public void testVectorNotUsesCandidateLocalNegativeEvidence() throws Exception {
    BayesianQueryParser parser =
        new BayesianQueryParser(
            "body", new MockAnalyzer(random(), MockTokenizer.WHITESPACE, false));

    Query query = parser.parse("body:cat NOT vector:[0, 1, 0]");

    assertTrue(query instanceof FunctionScoreQuery);
    FunctionScoreQuery functionScoreQuery = (FunctionScoreQuery) query;
    assertTrue(functionScoreQuery.getWrappedQuery() instanceof BayesianScoreQuery);
  }

  public void testHybridBooleanQueryExecutesTextVectorAndNotTogether() throws Exception {
    Directory dir = newDirectory();
    RandomIndexWriter writer = new RandomIndexWriter(random(), dir);

    Document doc0 = new Document();
    doc0.add(new StringField("body", "cat", Field.Store.NO));
    doc0.add(new StringField("status", "live", Field.Store.NO));
    doc0.add(new KnnFloatVectorField("vector", new float[] {1, 0, 0}, DOT_PRODUCT));
    writer.addDocument(doc0);

    Document doc1 = new Document();
    doc1.add(new StringField("body", "cat", Field.Store.NO));
    doc1.add(new StringField("status", "deleted", Field.Store.NO));
    doc1.add(new KnnFloatVectorField("vector", new float[] {1, 0, 0}, DOT_PRODUCT));
    writer.addDocument(doc1);

    Document doc2 = new Document();
    doc2.add(new StringField("body", "dog", Field.Store.NO));
    doc2.add(new StringField("status", "live", Field.Store.NO));
    doc2.add(new KnnFloatVectorField("vector", new float[] {1, 0, 0}, DOT_PRODUCT));
    writer.addDocument(doc2);

    Document doc3 = new Document();
    doc3.add(new StringField("body", "cat", Field.Store.NO));
    doc3.add(new StringField("status", "live", Field.Store.NO));
    doc3.add(new KnnFloatVectorField("vector", new float[] {0, 1, 0}, DOT_PRODUCT));
    writer.addDocument(doc3);

    try (IndexReader reader = writer.getReader()) {
      writer.close();
      writer = null;
      IndexSearcher searcher = newSearcher(reader);
      BayesianQueryParser parser =
          new BayesianQueryParser(
              "body", new MockAnalyzer(random(), MockTokenizer.WHITESPACE, false));
      parser.setDefaultVectorTopK(3);
      Query query = parser.parse("body:cat AND vector:[1, 0, 0] NOT status:deleted");

      ScoreDoc[] hits = searcher.search(query, 10).scoreDocs;
      assertEquals(1, hits.length);
      assertEquals(0, hits[0].doc);
    } finally {
      if (writer != null) {
        writer.close();
      }
      dir.close();
    }
  }

  public void testOrHybridSearchRanksDocumentMatchingTextAndVectorFirst() throws Exception {
    Directory dir = newDirectory();
    RandomIndexWriter writer = new RandomIndexWriter(random(), dir);

    Document both = new Document();
    both.add(new StringField("id", "both", Field.Store.YES));
    both.add(new StringField("body", "cat", Field.Store.NO));
    both.add(new KnnFloatVectorField("vector", new float[] {1, 0, 0}, DOT_PRODUCT));
    writer.addDocument(both);

    Document textOnly = new Document();
    textOnly.add(new StringField("id", "textOnly", Field.Store.YES));
    textOnly.add(new StringField("body", "cat", Field.Store.NO));
    writer.addDocument(textOnly);

    Document vectorOnly = new Document();
    vectorOnly.add(new StringField("id", "vectorOnly", Field.Store.YES));
    vectorOnly.add(new StringField("body", "dog", Field.Store.NO));
    vectorOnly.add(new KnnFloatVectorField("vector", new float[] {1, 0, 0}, DOT_PRODUCT));
    writer.addDocument(vectorOnly);

    Document neither = new Document();
    neither.add(new StringField("id", "neither", Field.Store.YES));
    neither.add(new StringField("body", "dog", Field.Store.NO));
    neither.add(new KnnFloatVectorField("vector", new float[] {0, 1, 0}, DOT_PRODUCT));
    writer.addDocument(neither);

    try (IndexReader reader = writer.getReader()) {
      writer.close();
      writer = null;
      IndexSearcher searcher = newSearcher(reader);
      BayesianQueryParser parser =
          new BayesianQueryParser(
              "body", new MockAnalyzer(random(), MockTokenizer.WHITESPACE, false));
      parser.setDefaultVectorTopK(2);
      Query query = parser.parse("body:cat OR vector:[1, 0, 0]");

      ScoreDoc[] hits = searcher.search(query, 10).scoreDocs;
      assertEquals(3, hits.length);
      assertEquals("both", searcher.storedFields().document(hits[0].doc).get("id"));
      assertTrue(hits[0].score > hits[1].score);
      assertTrue(hits[0].score > hits[2].score);
      assertContainsHit(searcher, hits, "textOnly");
      assertContainsHit(searcher, hits, "vectorOnly");
      assertNotContainsHit(searcher, hits, "neither");
    } finally {
      if (writer != null) {
        writer.close();
      }
      dir.close();
    }
  }

  public void testVectorNotDemotesCandidateByVectorSimilarityWithoutTopK() throws Exception {
    Directory dir = newDirectory();
    RandomIndexWriter writer = new RandomIndexWriter(random(), dir);

    Document noVector = new Document();
    noVector.add(new StringField("id", "noVector", Field.Store.YES));
    noVector.add(new StringField("body", "cat", Field.Store.NO));
    writer.addDocument(noVector);

    Document farVector = new Document();
    farVector.add(new StringField("id", "farVector", Field.Store.YES));
    farVector.add(new StringField("body", "cat", Field.Store.NO));
    farVector.add(new KnnFloatVectorField("vector", new float[] {1, 0, 0}, DOT_PRODUCT));
    writer.addDocument(farVector);

    Document nearVector = new Document();
    nearVector.add(new StringField("id", "nearVector", Field.Store.YES));
    nearVector.add(new StringField("body", "cat", Field.Store.NO));
    nearVector.add(new KnnFloatVectorField("vector", new float[] {0, 1, 0}, DOT_PRODUCT));
    writer.addDocument(nearVector);

    Document noTextMatch = new Document();
    noTextMatch.add(new StringField("id", "noTextMatch", Field.Store.YES));
    noTextMatch.add(new StringField("body", "dog", Field.Store.NO));
    noTextMatch.add(new KnnFloatVectorField("vector", new float[] {0, 1, 0}, DOT_PRODUCT));
    writer.addDocument(noTextMatch);

    try (IndexReader reader = writer.getReader()) {
      writer.close();
      writer = null;
      IndexSearcher searcher = newSearcher(reader);
      BayesianQueryParser parser =
          new BayesianQueryParser(
              "body", new MockAnalyzer(random(), MockTokenizer.WHITESPACE, false));
      parser.setDefaultVectorTopK(1);
      Query query = parser.parse("body:cat NOT vector:[0, 1, 0]");

      ScoreDoc[] hits = searcher.search(query, 10).scoreDocs;
      assertEquals(3, hits.length);
      assertContainsHit(searcher, hits, "noVector");
      assertContainsHit(searcher, hits, "farVector");
      assertContainsHit(searcher, hits, "nearVector");
      assertNotContainsHit(searcher, hits, "noTextMatch");
      assertTrue(hitScore(searcher, hits, "noVector") > hitScore(searcher, hits, "nearVector"));
      assertTrue(hitScore(searcher, hits, "farVector") > hitScore(searcher, hits, "nearVector"));
    } finally {
      if (writer != null) {
        writer.close();
      }
      dir.close();
    }
  }

  public void testNestedGroupedHybridQuerySearchesInnerBooleanGroups() throws Exception {
    Directory dir = newDirectory();
    RandomIndexWriter writer = new RandomIndexWriter(random(), dir);

    Document keepByText = new Document();
    keepByText.add(new StringField("id", "keepByText", Field.Store.YES));
    keepByText.add(new StringField("body", "cat", Field.Store.NO));
    keepByText.add(new StringField("section", "news", Field.Store.NO));
    writer.addDocument(keepByText);

    Document keepByVector = new Document();
    keepByVector.add(new StringField("id", "keepByVector", Field.Store.YES));
    keepByVector.add(new StringField("body", "dog", Field.Store.NO));
    keepByVector.add(new StringField("section", "news", Field.Store.NO));
    keepByVector.add(new KnnFloatVectorField("vector", new float[] {1, 0, 0}, DOT_PRODUCT));
    writer.addDocument(keepByVector);

    Document excludedByInnerVectorNot = new Document();
    excludedByInnerVectorNot.add(
        new StringField("id", "excludedByInnerVectorNot", Field.Store.YES));
    excludedByInnerVectorNot.add(new StringField("body", "cat", Field.Store.NO));
    excludedByInnerVectorNot.add(new StringField("section", "news", Field.Store.NO));
    excludedByInnerVectorNot.add(
        new KnnFloatVectorField("blockVector", new float[] {0, 1, 0}, DOT_PRODUCT));
    writer.addDocument(excludedByInnerVectorNot);

    Document missesHybridGroup = new Document();
    missesHybridGroup.add(new StringField("id", "missesHybridGroup", Field.Store.YES));
    missesHybridGroup.add(new StringField("body", "dog", Field.Store.NO));
    missesHybridGroup.add(new StringField("section", "news", Field.Store.NO));
    writer.addDocument(missesHybridGroup);

    Document missesMetadataGroup = new Document();
    missesMetadataGroup.add(new StringField("id", "missesMetadataGroup", Field.Store.YES));
    missesMetadataGroup.add(new StringField("body", "cat", Field.Store.NO));
    missesMetadataGroup.add(new StringField("section", "sports", Field.Store.NO));
    writer.addDocument(missesMetadataGroup);

    try (IndexReader reader = writer.getReader()) {
      writer.close();
      writer = null;
      IndexSearcher searcher = newSearcher(reader);
      BayesianQueryParser parser =
          new BayesianQueryParser(
              "body", new MockAnalyzer(random(), MockTokenizer.WHITESPACE, false));
      parser.setDefaultVectorTopK(10);
      Query query =
          parser.parse(
              "(body:cat OR vector:[1, 0, 0]) AND (section:news NOT blockVector:[0, 1, 0])");

      ScoreDoc[] hits = searcher.search(query, 10).scoreDocs;
      assertEquals(3, hits.length);
      assertContainsHit(searcher, hits, "keepByText");
      assertContainsHit(searcher, hits, "keepByVector");
      assertContainsHit(searcher, hits, "excludedByInnerVectorNot");
      assertTrue(
          hitScore(searcher, hits, "keepByText")
              > hitScore(searcher, hits, "excludedByInnerVectorNot"));
      assertTrue(
          hitScore(searcher, hits, "keepByVector")
              > hitScore(searcher, hits, "excludedByInnerVectorNot"));
      assertNotContainsHit(searcher, hits, "missesHybridGroup");
      assertNotContainsHit(searcher, hits, "missesMetadataGroup");
    } finally {
      if (writer != null) {
        writer.close();
      }
      dir.close();
    }
  }

  public void testCalibrationSettersValidateParameters() {
    BayesianQueryParser parser =
        new BayesianQueryParser(
            "body", new MockAnalyzer(random(), MockTokenizer.WHITESPACE, false));
    parser.setBayesianScoreCalibration(0.7f, 1.2f, 0.01f);
    assertEquals(0.7f, parser.getBayesianScoreAlpha(), 0f);
    assertEquals(1.2f, parser.getBayesianScoreBeta(), 0f);
    assertEquals(0.01f, parser.getBayesianScoreBaseRate(), 0f);

    expectThrows(IllegalArgumentException.class, () -> parser.setBayesianScoreCalibration(0f, 1f));
    expectThrows(
        IllegalArgumentException.class, () -> parser.setBayesianScoreCalibration(1f, Float.NaN));
    expectThrows(
        IllegalArgumentException.class, () -> parser.setBayesianScoreCalibration(1f, 1f, 1f));
    expectThrows(IllegalArgumentException.class, () -> parser.setFusionAlpha(-0.1f));
    expectThrows(IllegalArgumentException.class, () -> parser.setFusionAlpha(1.1f));
  }

  private static void assertBayesianWrapped(
      Class<? extends Query> expectedInnerClass, Query query) {
    assertTrue(query instanceof BayesianScoreQuery);
    assertEquals(expectedInnerClass, ((BayesianScoreQuery) query).getQuery().getClass());
  }

  private static int countOccur(BooleanQuery query, BooleanClause.Occur occur) {
    int count = 0;
    for (BooleanClause clause : query.clauses()) {
      if (clause.occur() == occur) {
        count++;
      }
    }
    return count;
  }

  private static void assertContainsHit(IndexSearcher searcher, ScoreDoc[] hits, String id)
      throws Exception {
    for (ScoreDoc hit : hits) {
      if (id.equals(searcher.storedFields().document(hit.doc).get("id"))) {
        return;
      }
    }
    fail("expected hit " + id + " in " + hitIds(searcher, hits));
  }

  private static void assertNotContainsHit(IndexSearcher searcher, ScoreDoc[] hits, String id)
      throws Exception {
    for (ScoreDoc hit : hits) {
      if (id.equals(searcher.storedFields().document(hit.doc).get("id"))) {
        fail("unexpected hit " + id + " in " + hitIds(searcher, hits));
      }
    }
  }

  private static float hitScore(IndexSearcher searcher, ScoreDoc[] hits, String id)
      throws Exception {
    for (ScoreDoc hit : hits) {
      if (id.equals(searcher.storedFields().document(hit.doc).get("id"))) {
        return hit.score;
      }
    }
    fail("expected hit " + id + " in " + hitIds(searcher, hits));
    return 0f;
  }

  private static List<String> hitIds(IndexSearcher searcher, ScoreDoc[] hits) throws Exception {
    List<String> ids = new ArrayList<>();
    for (ScoreDoc hit : hits) {
      ids.add(searcher.storedFields().document(hit.doc).get("id"));
    }
    return ids;
  }
}
