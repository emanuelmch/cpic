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

#include "board.h"

#include <cassert>

using namespace CPic;

using std::map;
using std::vector;

typedef unsigned short ushort;

Board::Board(vector<Color> colors, vector<vector<Clue>> columns, vector<vector<Clue>> rows)
        : colors(colors),
          colorCount(colors.size()),
          columnCount(columns.size()),
          rowCount(rows.size()),
          columns(columns),
          rows(rows) {
  for (ushort x = 0; x < columnCount; x++) {
    std::vector<Color> column;
    for (ushort y = 0; y < rowCount; y++) {
      column.push_back(Blank);
    }
    this->results.push_back(column);
  }
}

Board::~Board() {}

bool Board::isValid() {
  for (ushort i = 0; i < columns.size(); ++i) {
    for (auto color : colors) {
      auto count = countColorInColumn(i, color);
      if (count > clueForColumn(i, color).amount) {
        return false;
      }
    }
  }

  for (ushort i = 0; i < rows.size(); ++i) {
    for (auto color : colors) {
      auto count = countColorInRow(i, color);
      if (count > clueForRow(i, color).amount) {
        return false;
      }
    }
  }

  return true;
}

ushort Board::countColorInColumn(ushort column, Color color) const {
  // TODO: Replace this something from algorithms
  ushort count = 0;

  for (ushort i = 0; i < results[column].size(); ++i) {
    if (results[column][i] == color)
      count++;
  }

  return count;
}

ushort Board::countColorInRow(ushort row, Color color) const {
  // TODO: Replace this something from algorithms
  ushort count = 0;

  for (ushort i = 0; i < results.size(); ++i) {
    if (results[i][row] == color)
      count++;
  }

  return count;
}

const Clue Board::clueForColumn(ushort column, Color color) const {
  for (auto clue : columns[column]) {
    if (clue.color == color) {
      return clue;
    }
  }

  assert(!"Asked for a clue for a non-existing color");
}

const Clue Board::clueForRow(ushort row, Color color) const {
  for (auto clue : rows[row]) {
    if (clue.color == color) {
      return clue;
    }
  }

  assert(!"Asked for a clue for a non-existing color");
}
