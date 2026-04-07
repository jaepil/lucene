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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal JSON parser for BEIR data files. Supports objects, arrays, strings, numbers, booleans,
 * and null. No external dependencies.
 */
final class SimpleJSON {

  private final String input;
  private int pos;

  private SimpleJSON(String input) {
    this.input = input;
    this.pos = 0;
  }

  static Map<String, Object> parse(String json) {
    SimpleJSON parser = new SimpleJSON(json.trim());
    Object result = parser.parseValue();
    if (result instanceof Map) {
      @SuppressWarnings("unchecked")
      Map<String, Object> map = (Map<String, Object>) result;
      return map;
    }
    throw new IllegalArgumentException("Expected JSON object at root");
  }

  private Object parseValue() {
    skipWhitespace();
    if (pos >= input.length()) {
      throw new IllegalArgumentException("Unexpected end of input");
    }
    char c = input.charAt(pos);
    return switch (c) {
      case '{' -> parseObject();
      case '[' -> parseArray();
      case '"' -> parseString();
      case 't', 'f' -> parseBoolean();
      case 'n' -> parseNull();
      default -> parseNumber();
    };
  }

  private Map<String, Object> parseObject() {
    expect('{');
    Map<String, Object> map = new LinkedHashMap<>();
    skipWhitespace();
    if (pos < input.length() && input.charAt(pos) == '}') {
      pos++;
      return map;
    }
    while (true) {
      skipWhitespace();
      String key = parseString();
      skipWhitespace();
      expect(':');
      Object value = parseValue();
      map.put(key, value);
      skipWhitespace();
      if (pos < input.length() && input.charAt(pos) == ',') {
        pos++;
      } else {
        break;
      }
    }
    expect('}');
    return map;
  }

  private List<Object> parseArray() {
    expect('[');
    List<Object> list = new ArrayList<>();
    skipWhitespace();
    if (pos < input.length() && input.charAt(pos) == ']') {
      pos++;
      return list;
    }
    while (true) {
      list.add(parseValue());
      skipWhitespace();
      if (pos < input.length() && input.charAt(pos) == ',') {
        pos++;
      } else {
        break;
      }
    }
    expect(']');
    return list;
  }

  private String parseString() {
    expect('"');
    StringBuilder sb = new StringBuilder();
    while (pos < input.length()) {
      char c = input.charAt(pos);
      if (c == '"') {
        pos++;
        return sb.toString();
      }
      if (c == '\\') {
        pos++;
        if (pos >= input.length()) break;
        char esc = input.charAt(pos);
        switch (esc) {
          case '"', '\\', '/' -> sb.append(esc);
          case 'b' -> sb.append('\b');
          case 'f' -> sb.append('\f');
          case 'n' -> sb.append('\n');
          case 'r' -> sb.append('\r');
          case 't' -> sb.append('\t');
          case 'u' -> {
            String hex = input.substring(pos + 1, pos + 5);
            sb.append((char) Integer.parseInt(hex, 16));
            pos += 4;
          }
          default -> {
            sb.append('\\');
            sb.append(esc);
          }
        }
      } else {
        sb.append(c);
      }
      pos++;
    }
    throw new IllegalArgumentException("Unterminated string");
  }

  private Number parseNumber() {
    int start = pos;
    if (pos < input.length() && input.charAt(pos) == '-') pos++;
    while (pos < input.length() && Character.isDigit(input.charAt(pos))) pos++;
    boolean isFloat = false;
    if (pos < input.length() && input.charAt(pos) == '.') {
      isFloat = true;
      pos++;
      while (pos < input.length() && Character.isDigit(input.charAt(pos))) pos++;
    }
    if (pos < input.length() && (input.charAt(pos) == 'e' || input.charAt(pos) == 'E')) {
      isFloat = true;
      pos++;
      if (pos < input.length() && (input.charAt(pos) == '+' || input.charAt(pos) == '-')) pos++;
      while (pos < input.length() && Character.isDigit(input.charAt(pos))) pos++;
    }
    String numStr = input.substring(start, pos);
    if (isFloat) {
      return Double.parseDouble(numStr);
    }
    long val = Long.parseLong(numStr);
    if (val >= Integer.MIN_VALUE && val <= Integer.MAX_VALUE) {
      return (int) val;
    }
    return val;
  }

  private Boolean parseBoolean() {
    if (input.startsWith("true", pos)) {
      pos += 4;
      return Boolean.TRUE;
    }
    if (input.startsWith("false", pos)) {
      pos += 5;
      return Boolean.FALSE;
    }
    throw new IllegalArgumentException("Expected boolean at position " + pos);
  }

  private Object parseNull() {
    if (input.startsWith("null", pos)) {
      pos += 4;
      return null;
    }
    throw new IllegalArgumentException("Expected null at position " + pos);
  }

  private void expect(char c) {
    skipWhitespace();
    if (pos >= input.length() || input.charAt(pos) != c) {
      throw new IllegalArgumentException(
          "Expected '" + c + "' at position " + pos + " but got '"
              + (pos < input.length() ? input.charAt(pos) : "EOF") + "'");
    }
    pos++;
  }

  private void skipWhitespace() {
    while (pos < input.length() && Character.isWhitespace(input.charAt(pos))) {
      pos++;
    }
  }
}
