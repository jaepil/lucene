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
package org.apache.lucene.queryparser.flexible.standard;

import org.apache.lucene.queryparser.flexible.core.QueryNodeException;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermRangeQuery;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Tests vector literal integration into the flexible standard query parser. */
public class TestVectorQueryParser extends LuceneTestCase {

  public void testVectorLiteral() throws Exception {
    StandardQueryParser parser = new StandardQueryParser();
    parser.setDefaultVectorTopK(7);

    Query query = parser.parse("vector:[1, 2.5, -3e-1]", "body");
    assertTrue(query instanceof KnnFloatVectorQuery);

    KnnFloatVectorQuery vectorQuery = (KnnFloatVectorQuery) query;
    assertEquals("vector", vectorQuery.getField());
    assertEquals(7, vectorQuery.getK());
    float[] target = vectorQuery.getTargetCopy();
    assertEquals(3, target.length);
    assertEquals(1f, target[0], 0f);
    assertEquals(2.5f, target[1], 0f);
    assertEquals(-0.3f, target[2], 0f);
  }

  public void testVectorLiteralDoesNotReplaceRangeSyntax() throws Exception {
    StandardQueryParser parser = new StandardQueryParser();
    assertTrue(parser.parse("field:[1 TO 3]", "body") instanceof TermRangeQuery);
    assertTrue(parser.parse("field:{1 TO 3}", "body") instanceof TermRangeQuery);
  }

  public void testInvalidVectorLiteral() {
    StandardQueryParser parser = new StandardQueryParser();
    expectThrows(QueryNodeException.class, () -> parser.parse("vector:[1, , 2]", "body"));
    expectThrows(QueryNodeException.class, () -> parser.parse("vector:[1, 2,]", "body"));
    expectThrows(QueryNodeException.class, () -> parser.parse("vector:[1, abc, 2]", "body"));
    expectThrows(QueryNodeException.class, () -> parser.parse("vector:[1, NaN, 2]", "body"));
  }

  public void testDefaultVectorTopKValidation() {
    StandardQueryParser parser = new StandardQueryParser();
    assertEquals(10, parser.getDefaultVectorTopK());
    parser.setDefaultVectorTopK(3);
    assertEquals(3, parser.getDefaultVectorTopK());
    expectThrows(IllegalArgumentException.class, () -> parser.setDefaultVectorTopK(0));
  }
}
