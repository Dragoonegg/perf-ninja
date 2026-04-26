#include "solution.hpp"
#include <algorithm>
#include <cassert>
#include <immintrin.h>
#include <type_traits>

// The alignment algorithm which computes the alignment of the given sequence
// pairs.

// inline constexpr size_t sequence_size_v = 200;
// inline constexpr size_t sequence_count_v = 16;
// using sequence_t = std::array<uint8_t, sequence_size_v>;
// using result_t = std::array<int16_t, sequence_count_v>;

result_t compute_alignment(std::vector<sequence_t> const &sequences1,
                           std::vector<sequence_t> const &sequences2) {
  static_assert(sequence_count_v == 16,
                "This implementation maps 16 sequence pairs to AVX2 lanes.");

  assert(sequences1.size() == sequence_count_v);
  assert(sequences2.size() == sequence_count_v);

  using score_t = int16_t;
  using score_vec_t = __m256i;

  result_t result{};
  /*
   * 优化点 1：先把输入转置成 SIMD 友好的布局。
   * 每个 AVX2 向量有 16 个 int16 lane，刚好对应 16 对独立序列。
   * DP 在列方向仍然有依赖，但不同序列之间没有依赖，所以可以并行算，
   * 避免标量版本把同一套递推重复跑 16 次。
   */
  score_vec_t sequence1_by_pos[sequence_size_v]{};
  score_vec_t sequence2_by_pos[sequence_size_v]{};

  for (size_t pos = 0; pos < sequence_size_v; ++pos) {
    std::array<score_t, sequence_count_v> lane_values{};

    for (size_t lane = 0; lane < sequence_count_v; ++lane) {
      lane_values[lane] = sequences1[lane][pos];
    }
    sequence1_by_pos[pos] = _mm256_load_si256((__m256i *)(lane_values.data()));

    for (size_t lane = 0; lane < sequence_count_v; ++lane) {
      lane_values[lane] = sequences2[lane][pos];
    }
    sequence2_by_pos[pos] = _mm256_load_si256((__m256i *)(lane_values.data()));
  }

  /*

  
   * Initialise score values.
   */
  // score_t gap_open{-11};
  // score_t gap_extension{-1};
  // score_t match{6};
  // score_t mismatch{-4};

  score_vec_t gap_open_vec = _mm256_set1_epi16(-11);
  score_vec_t gap_extension_vec = _mm256_set1_epi16(-1);
  score_vec_t match_vec = _mm256_set1_epi16(6);
  score_vec_t mismatch_vec = _mm256_set1_epi16(-4);

  /*
   * Setup the matrix.
   * Note we can compute the entire matrix with just one row in memory,
   * since we are only interested in the last value of the last row in the
   * score matrix.
   */
  alignas(32) score_vec_t score_row[sequence_size_v + 1]{};
  alignas(32) score_vec_t vertical_gap_row[sequence_size_v + 1]{};
  score_vec_t last_horizontal_gap{};

  /*
   * Initialise the first row of the matrix.
   */
  vertical_gap_row[0] = gap_open_vec;
  last_horizontal_gap = gap_open_vec;

  for (size_t i = 1; i <= sequence_size_v; ++i) {
    score_row[i] = last_horizontal_gap;
    vertical_gap_row[i] = _mm256_add_epi16(last_horizontal_gap, gap_open_vec);
    last_horizontal_gap =
        _mm256_add_epi16(last_horizontal_gap, gap_extension_vec);
  }

  /*
   * Compute the main recursion to fill the matrix.
   */
  for (unsigned row = 1; row <= sequence_size_v; ++row) {
    score_vec_t last_diagonal_score =
        score_row[0]; // Cache last diagonal score to compute this cell.
    score_vec_t first_vertical_gap = vertical_gap_row[0];
    score_row[0] = first_vertical_gap;
    last_horizontal_gap = _mm256_add_epi16(first_vertical_gap, gap_open_vec);
    vertical_gap_row[0] =
        _mm256_add_epi16(first_vertical_gap, gap_extension_vec);

    score_vec_t sequence1_symbols = sequence1_by_pos[row - 1];

    for (unsigned col = 1; col <= sequence_size_v; ++col) {
      /*
       * 优化点 2：用 SIMD compare + blend 选择 match/mismatch。
       * 标量版本每个 DP 格子都有一次相等判断；随机 DNA 符号会带来较高的
       * Bad_Speculation。这里一次比较 16 对序列，并用 blend 生成替换分数，
       * 让热点循环更接近无分支的数据流。
       */
      score_vec_t is_match =
          _mm256_cmpeq_epi16(sequence1_symbols, sequence2_by_pos[col - 1]);
      score_vec_t substitution_score =
          _mm256_blendv_epi8(mismatch_vec, match_vec, is_match);

      // Compute next score from diagonal direction with match/mismatch.
      score_vec_t best_cell_score =
          _mm256_add_epi16(last_diagonal_score, substitution_score);
      score_vec_t next_diagonal_score = score_row[col];
      score_vec_t vertical_gap = vertical_gap_row[col];

      // Determine best score from diagonal, vertical, or horizontal direction.
      best_cell_score = _mm256_max_epi16(best_cell_score, vertical_gap);
      best_cell_score = _mm256_max_epi16(best_cell_score, last_horizontal_gap);

      // Cache next diagonal value and store the current DP cell.
      last_diagonal_score = next_diagonal_score;
      score_row[col] = best_cell_score;

      /*
       * 优化点 3：gap 状态也用 SIMD add/max 更新。
       * 递推关系保持不变，但 16 条标量依赖链变成 1 条向量依赖链；
       * 这针对的是 perf 里 memory bound 很低、主要是 core/vectorization
       * 受限的情况。
       */
      best_cell_score = _mm256_add_epi16(best_cell_score, gap_open_vec);
      vertical_gap = _mm256_add_epi16(vertical_gap, gap_extension_vec);
      last_horizontal_gap =
          _mm256_add_epi16(last_horizontal_gap, gap_extension_vec);

      /*
       * 优化点 4：根据新的 perf 数据，Retiring 已经超过 90%，说明 CPU
       * 大部分时间都在真正执行指令。这里把同一 DP 格子的 score/vertical gap
       * 先缓存到寄存器，后续 max/add 复用寄存器值，减少内层循环中的重复
       * 数组加载和地址计算，目标是降低 retired 指令量。
       */
      vertical_gap_row[col] = _mm256_max_epi16(vertical_gap, best_cell_score);
      last_horizontal_gap =
          _mm256_max_epi16(last_horizontal_gap, best_cell_score);
    }
  }

  // Report the best score for all 16 lanes.
  _mm256_storeu_si256(reinterpret_cast<score_vec_t *>(result.data()),
                     score_row[sequence_size_v]);

  return result;
}