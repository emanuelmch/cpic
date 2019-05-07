/*
 * Copyright (c) 2019 Emanuel Machado da Silva
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

#pragma once

#include "model.h"

#include <string>
#include <utility>
#include <vector>

namespace Shurikens {

struct ShurikenData {
  ShurikenData(std::string name, Shuriken shuriken, std::vector<Move> moves)
      : name(std::move(name)), shuriken(std::move(shuriken)), moves(std::move(moves)) {}

  const std::string name;
  const Shuriken shuriken;
  const std::vector<Move> moves;
};

inline std::vector<ShurikenData> createAllShurikens() {
  Shuriken shuriken({G, H, I, D, E, F, A, B, C, J, K, L});
  ShurikenData data("trivial0", shuriken, {swap_top});

  return {data};
}
}
