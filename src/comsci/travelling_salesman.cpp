/*
 * Copyright (c) 2021 Emanuel Machado da Silva
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

#include "travelling_salesman.h"
#include "travelling_salesman_data.h"

#include "common/assertions.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <numeric>
#include <random>

static const uint8_t POPULATION_SIZE = 40;
static const uint8_t ELITE_SIZE = 6;
static const uint8_t MUTATION_CHANCE = 5;
// TODO: These next two values should be randomly calculated
static const uint8_t CROSSOVER_CUTPOINT_LEFT = 1;
static const uint8_t CROSSOVER_CUTPOINT_RIGHT = 3;
static_assert(CROSSOVER_CUTPOINT_LEFT < CROSSOVER_CUTPOINT_RIGHT);
static_assert(CROSSOVER_CUTPOINT_RIGHT < CITY_COUNT);

namespace {

struct Chromosome {
  std::array<uint8_t, CITY_COUNT> genes{};
  uint8_t _fitness = 0;

  inline void randomize() {
    for (size_t i = 0; i < genes.size(); ++i) {
      genes[i] = i;
    }

    std::random_device rnd;
    std::shuffle(genes.begin(), genes.end(), rnd);
    ensure(isValid());
  }

  [[nodiscard]] inline bool isValid() const {
    bool valid = true;
    for (size_t i = 0; i < CITY_COUNT; ++i) {
      valid = valid && std::find(genes.cbegin(), genes.cend(), i) != genes.cend();
    }
    return valid;
  }

  uint16_t fitness() {
    if (_fitness == 0) {
      uint8_t distance = 0;
      for (size_t i = 0; i < genes.size() - 1; ++i) {
        auto from = genes[i];
        auto to = genes[i + 1];
        distance += DISTANCE_TABLE[from][to];
      }
      auto last = genes[genes.size() - 1];
      auto first = genes[0];
      distance += DISTANCE_TABLE[last][first];
      _fitness = MAX_DISTANCE - distance;
    }
    ensure(_fitness > 0);
    return _fitness;
  }

  inline void print(const std::string &name) const {
    std::cout << name << ": [" << (int)_fitness << "] ";
    for (size_t i = 0; i < CITY_COUNT; ++i) {
      std::cout << (int)genes[i] << " ";
    }
    std::cout << "\n";
  }
};
}

template <size_t T>
inline std::array<Chromosome, T> selectParents(const std::array<Chromosome, POPULATION_SIZE> &population) {
  std::vector<Chromosome> candidates{population.cbegin(), population.cend()};

  auto next = [&candidates] {
    ensure(!candidates.empty());
    auto sum = std::accumulate(candidates.begin(), candidates.end(), 0,
                               [](const auto &sum, auto &next) { return sum + next.fitness(); });
    auto rnd = std::rand() % sum;
    auto it = candidates.begin();
    while (true) {
      ensure(it != candidates.end());
      rnd -= it->fitness();
      if (rnd <= 0) {
        auto result = *it;
        candidates.erase(it);
        return result;
      }
    }
  };

  std::array<Chromosome, T> result;
  std::for_each(result.begin(), result.end(), [&next](auto &it) { it = next(); });
  return result;
};

inline Chromosome crossover(const Chromosome &left, const Chromosome &right) {
  Chromosome result;
  for (auto &gene : result.genes) {
    gene = std::numeric_limits<typeof(gene)>::max();
  }

  ensure(left.isValid());
  ensure(right.isValid());

  for (size_t i = CROSSOVER_CUTPOINT_LEFT; i < CROSSOVER_CUTPOINT_RIGHT; ++i) {
    result.genes[i] = left.genes[i];
  }
  const auto isAbsentInResult = [&result](const auto &it) {
    return std::find(result.genes.cbegin(), result.genes.cend(), it) == result.genes.cend();
  };

  for (size_t i = 0; i < CITY_COUNT; ++i) {
    if (i >= CROSSOVER_CUTPOINT_LEFT && i < CROSSOVER_CUTPOINT_RIGHT) continue;

    auto it = std::find_if(right.genes.cbegin(), right.genes.cend(), isAbsentInResult);
    result.genes[i] = *it;
  }
  ensure(result.isValid());
  return result;
}

inline void mutate(Chromosome *begin, Chromosome *end) {
  if (begin != nullptr) return;
  for (auto it = begin; it != end; ++it) {
    static_assert(MUTATION_CHANCE <= 100);
    if (static_cast<uint>(std::rand() % 100) < MUTATION_CHANCE) {
      auto index1 = static_cast<unsigned long>(std::rand()) % it->genes.size();
      auto index2 = static_cast<unsigned long>(std::rand()) % it->genes.size();
      std::swap(it->genes[index1], it->genes[index2]);

      it->_fitness = 0;
      ensure(it->isValid());
    }
  }
}

bool ComSci::TravellingSalesman::run() {
  uint64_t generations = 0;

  std::array<Chromosome, POPULATION_SIZE> population{};
  std::for_each(population.begin(), population.end(), [](auto &it) { it.randomize(); });
  std::sort(population.begin(), population.end(),
            [](auto left, auto right) { return left.fitness() > right.fitness(); });

  uint16_t generationsSinceLastImprovement = 0;
  Chromosome solution = population[0];

  do {
    std::array<Chromosome, POPULATION_SIZE> next{};
    for (size_t i = 0; i < ELITE_SIZE; ++i) {
      next[i] = population[i];
    }

    constexpr auto parentCount = POPULATION_SIZE - ELITE_SIZE;
    static_assert(parentCount % 2 == 0);
    auto parents = selectParents<parentCount>(population);
    for (size_t i = 0; i < parentCount; i += 2) {
      next[i + ELITE_SIZE] = crossover(parents[i], parents[i + 1]);
      next[i + ELITE_SIZE + 1] = crossover(parents[i + 1], parents[i]);
    }

    mutate(next.begin(), next.end());

    std::sort(next.begin(), next.end(), [](auto left, auto right) { return left.fitness() > right.fitness(); });
    population = next;

    if (population[0].fitness() > solution.fitness()) {
      solution = population[0];
      generationsSinceLastImprovement = 0;
    }
    ++generations;
  } while (++generationsSinceLastImprovement < 10000);

  std::cout << "Took " << generations << " generations: [" << (int)MAX_DISTANCE - solution.fitness() << "] [ ";
  for (int gene : solution.genes) {
    std::cout << gene << ", ";
  }
  std::cout << "]\n";
  return true;
}
