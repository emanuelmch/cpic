/*
 * Copyright (c) 2018 Emanuel Machado da Silva
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

#include "board_solver_test.h"

#include "model/board_builder.h"

using namespace CPic;

TYPED_TEST(BoardSolverTest, ShouldSolveSmallTrivialBoards_HomogeneousColumns) {
  Board board = BoardBuilder().column({2, 0})
                            ->column({0, 2})
                            ->row({1, 1})
                            ->row({1, 1})
                            ->build();

  this->solver->solve(&board);

  const int columnCount = 2;
  const int rowCount = 2;

  auto results = board.results;
  ASSERT_EQ(columnCount, results.size());
  for (int col = 0; col < columnCount; ++col) {
    ASSERT_EQ(rowCount, results[col].size());
  }

  for (int row = 0; row < rowCount; ++row) {
    for (int col = 0; col < columnCount; ++col) {
      EXPECT_EQ(col, results[col][row]);
    }
  }
}

//FIXME: ShouldSolveSmallTrivialBoards_HomogeneousRows

//FIXME: ShouldSolveBiggerTrivialBoards
//FIXME: ShouldSolveRectangularTrivialBoards
