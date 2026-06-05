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
package org.apache.lucene.queryparser.flexible.standard.nodes;

import java.util.Arrays;
import org.apache.lucene.queryparser.flexible.core.nodes.FieldableNode;
import org.apache.lucene.queryparser.flexible.core.nodes.QueryNodeImpl;
import org.apache.lucene.queryparser.flexible.core.parser.EscapeQuerySyntax;
import org.apache.lucene.search.KnnFloatVectorQuery;

/** A query node for vector literals such as {@code vector:[1, 2, 3]}. */
public class VectorQueryNode extends QueryNodeImpl implements FieldableNode {

  public static final int DEFAULT_TOP_K = 10;

  private CharSequence field;
  private float[] vector;
  private int begin;
  private int end;
  private int k = DEFAULT_TOP_K;

  /**
   * Creates a {@link VectorQueryNode}.
   *
   * @param field vector field
   * @param vector query vector
   * @param begin begin position in the query string
   * @param end end position in the query string
   */
  public VectorQueryNode(CharSequence field, float[] vector, int begin, int end) {
    this.field = field;
    this.vector = vector.clone();
    this.begin = begin;
    this.end = end;
    setLeaf(true);
  }

  @Override
  public CharSequence getField() {
    return field;
  }

  public String getFieldAsString() {
    return field.toString();
  }

  @Override
  public void setField(CharSequence field) {
    this.field = field;
  }

  /** Returns a copy of the parsed query vector. */
  public float[] getVector() {
    return vector.clone();
  }

  /** Sets the parsed query vector. */
  public void setVector(float[] vector) {
    this.vector = vector.clone();
  }

  /** Returns the top-k value used when building a {@link KnnFloatVectorQuery}. */
  public int getK() {
    return k;
  }

  /** Sets the top-k value used when building a {@link KnnFloatVectorQuery}. */
  public void setK(int k) {
    if (k < 1) {
      throw new IllegalArgumentException("k must be at least 1, got " + k);
    }
    this.k = k;
  }

  @Override
  public CharSequence toQueryString(EscapeQuerySyntax escapeSyntaxParser) {
    StringBuilder builder = new StringBuilder();
    if (isDefaultField(field) == false) {
      builder.append(field).append(':');
    }
    builder.append('[');
    for (int i = 0; i < vector.length; i++) {
      if (i > 0) {
        builder.append(", ");
      }
      builder.append(Float.toString(vector[i]));
    }
    builder.append(']');
    return builder;
  }

  @Override
  public String toString() {
    return "<vector field='"
        + field
        + "' vector='"
        + Arrays.toString(vector)
        + "' k='"
        + k
        + "' start='"
        + begin
        + "' end='"
        + end
        + "'/>";
  }

  @Override
  public VectorQueryNode cloneTree() throws CloneNotSupportedException {
    VectorQueryNode clone = (VectorQueryNode) super.cloneTree();
    clone.field = field;
    clone.vector = vector.clone();
    clone.begin = begin;
    clone.end = end;
    clone.k = k;
    return clone;
  }
}
