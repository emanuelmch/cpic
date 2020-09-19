/*
 * Copyright (c) 2020 Emanuel Machado da Silva
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "maths/expressions.h"

#include <gtest/gtest.h>

using namespace Maths;

using std::vector;

TEST(Expressions, Evaluator) {
  EXPECT_EQ(evaluateExpression("1 + 1"), 2);
  EXPECT_EQ(evaluateExpression("2 + 3"), 5);
  EXPECT_EQ(evaluateExpression("2+3 "), 5);
  EXPECT_EQ(evaluateExpression("2+ 3 +4"), 9);
  EXPECT_EQ(evaluateExpression("2 + 3 - 4"), 1);
  EXPECT_EQ(evaluateExpression("2 - 3 + 4"), 3);
  EXPECT_EQ(evaluateExpression("1 * 2 + 3"), 5);
  EXPECT_EQ(evaluateExpression("1 + 2 * 3"), 7);
  EXPECT_EQ(evaluateExpression("1+2*3/2+4/2-1*3"), 3);
  EXPECT_EQ(evaluateExpression("(1)"), 1);
  EXPECT_EQ(evaluateExpression("(1+1)"), 2);
  EXPECT_EQ(evaluateExpression("1*(2+1)"), 3);
  EXPECT_EQ(evaluateExpression("1+(3+1)/2"), 3);
  EXPECT_EQ(evaluateExpression("2*(3*(1+2)+2)"), 22);
  EXPECT_EQ(evaluateExpression("123*456"), 56088);
  EXPECT_EQ(evaluateExpression("3 + 4 * 2 / ( 1 - 5 ) ^ 2 ^ 3 - 1 / 8192"), 3);
  EXPECT_EQ(evaluateExpression("3 + 4 * (2 ^ 2) ^ 3 / ( 1 - 5 ) ^ 2"), 19);
  EXPECT_EQ(evaluateExpression("3 + 4 * 2 ^ 2 ^ 3 / ( 1 - 5 ) ^ 2"), 67);
  EXPECT_EQ(evaluateExpression("3 + (4 * 2) ^ 2 ^ 3 / ( 1 - 5 ) ^ 2"), 1048579);
  EXPECT_EQ(evaluateExpression(" 3 + (4 * 2) ^ 2 ^ 3 ^ 1 ^ 1 / ( 1 - 5 ) ^ 2 ^ 3"), 259);
}

TEST(Expressions, Tokenizer) {
  auto plus = Token::fromChar('+');
  auto divided = Token::fromChar('/');

  EXPECT_EQ(tokenizeExpression("1 +2"), vector<Token>({1, plus, 2}));
  EXPECT_EQ(tokenizeExpression("1000/123"), vector<Token>({1000, divided, 123}));
  EXPECT_EQ(tokenizeExpression("0"), vector({Token(0)}));
}
