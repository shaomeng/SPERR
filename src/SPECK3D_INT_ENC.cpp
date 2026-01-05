#include "SPECK3D_INT_ENC.h"

#include <algorithm>
#include <cassert>
#include <cstring>  // std::memcpy()
#include <numeric>

#ifdef __AVX2__
  #include <immintrin.h>
#endif

template <typename T>
void sperr::SPECK3D_INT_ENC<T>::m_deposit_set(Set3D set)
{
  switch (set.num_elem()) {
    case 0:
      return;
    case 1: {
      auto id = set.start_z * m_dims[0] * m_dims[1] + set.start_y * m_dims[0] + set.start_x;
      m_morton_buf[set.get_morton()] = m_coeff_buf[id];
      return;
    }
    case 2: {
      // We directly deposit the 2 elements in `set` instead of performing another partition.
      //
      // Deposit the 1st element.
      auto id = set.start_z * m_dims[0] * m_dims[1] + set.start_y * m_dims[0] + set.start_x;
      auto morton_id = set.get_morton();
      m_morton_buf[morton_id] = m_coeff_buf[id];

      // Deposit the 2nd element.
      if (set.length_x == 2)
        id++;
      else if (set.length_y == 2)
        id += m_dims[0];
      else
        id += m_dims[0] * m_dims[1];
      m_morton_buf[++morton_id] = m_coeff_buf[id];

      return;
    }
    case 4: {
      const auto id = set.start_z * m_dims[0] * m_dims[1] + set.start_y * m_dims[0] + set.start_x;
      auto morton_id = set.get_morton();

      if (set.length_x == 2 && set.length_y == 2) {
        // Element (0, 0, 0)
        m_morton_buf[morton_id] = m_coeff_buf[id];

        // Element (1, 0, 0)
        m_morton_buf[++morton_id] = m_coeff_buf[id + 1];

        // Element (0, 1, 0)
        auto id2 = id + m_dims[0];
        m_morton_buf[++morton_id] = m_coeff_buf[id2];

        // Element (1, 1, 0)
        m_morton_buf[++morton_id] = m_coeff_buf[++id2];

        return;
      }
      else if (set.length_x == 2 && set.length_z == 2) {
        // Element (0, 0, 0)
        m_morton_buf[morton_id] = m_coeff_buf[id];

        // Element (1, 0, 0)
        m_morton_buf[++morton_id] = m_coeff_buf[id + 1];

        // Element (0, 0, 1)
        auto id2 = id + m_dims[0] * m_dims[1];
        m_morton_buf[++morton_id] = m_coeff_buf[id2];

        // Element (1, 0, 1)
        m_morton_buf[++morton_id] = m_coeff_buf[++id2];

        return;
      }
      else if (set.length_y == 2 && set.length_z == 2) {
        // Element (0, 0, 0)
        m_morton_buf[morton_id] = m_coeff_buf[id];

        // Element (0, 1, 0)
        auto id2 = id + m_dims[0];
        m_morton_buf[++morton_id] = m_coeff_buf[id2];

        // Element (0, 0, 1)
        id2 = id + m_dims[0] * m_dims[1];
        m_morton_buf[++morton_id] = m_coeff_buf[id2];

        // Element (0, 1, 1)
        id2 += m_dims[0];
        m_morton_buf[++morton_id] = m_coeff_buf[id2];

        return;
      }
      else
        break;  // Fall back to the recursive case.
    }
    case 8: {
      if (set.length_x == 2 && set.length_y == 2) {
        // Element (0, 0, 0)
        const auto id = set.start_z * m_dims[0] * m_dims[1] + set.start_y * m_dims[0] + set.start_x;
        auto morton_id = set.get_morton();
        m_morton_buf[morton_id] = m_coeff_buf[id];

        // Element (1, 0, 0)
        m_morton_buf[++morton_id] = m_coeff_buf[id + 1];

        // Element (0, 1, 0)
        auto id2 = id + m_dims[0];
        m_morton_buf[++morton_id] = m_coeff_buf[id2];

        // Element (1, 1, 0)
        m_morton_buf[++morton_id] = m_coeff_buf[++id2];

        // Element (0, 0, 1)
        id2 = id + m_dims[0] * m_dims[1];
        m_morton_buf[++morton_id] = m_coeff_buf[id2];

        // Element (1, 0, 1)
        m_morton_buf[++morton_id] = m_coeff_buf[++id2];

        // Element (0, 1, 1)
        id2 = id + m_dims[0] * (m_dims[1] + 1);
        m_morton_buf[++morton_id] = m_coeff_buf[id2];

        // Element (1, 1, 1)
        m_morton_buf[++morton_id] = m_coeff_buf[++id2];

        return;
      }
      else
        break;  // Fall back to the recursive case.
    }
    default:
      break;  // Fall back to the recursive case.
  }

  // The recursive case.
  auto [subsets, lev] = m_partition_S_XYZ(set, 0);
  for (auto& sub : subsets)
    m_deposit_set(sub);
}

template <typename T>
void sperr::SPECK3D_INT_ENC<T>::m_encoder_make_morton()
{
  // For the encoder, this function re-organizes the coefficients in a morton order.
  //
  m_morton_buf.resize(m_dims[0] * m_dims[1] * m_dims[2]);

  // The same traversing order as in `SPECK3D_INT::m_sorting_pass()`
  size_t morton_offset = 0;
  for (size_t tmp = 1; tmp <= m_LIS.size(); tmp++) {
    auto idx1 = m_LIS.size() - tmp;
    for (size_t idx2 = 0; idx2 < m_LIS[idx1].size(); idx2++) {
      auto& set = m_LIS[idx1][idx2];
      set.set_morton(morton_offset);
      m_deposit_set(set);
      morton_offset += set.num_elem();
    }
  }
}

template <typename T>
void sperr::SPECK3D_INT_ENC<T>::m_encoder_make_mmask(size_t idx1, size_t idx2)
{
  // For the encoder only. Populates two data members: `m_mmask` and `m_mmask_offset`
  //
  const auto& set = m_LIS[idx1][idx2];
  const auto len = set.num_elem();

  m_mmask.resize(len);
  m_mmask_offset = set.get_morton();

  size_t processed_bits = 0;
  while (processed_bits + 64 <= len) {
    uint64_t word = 0;
    auto idx_offset = m_mmask_offset + processed_bits;
    bool optimized = false;

#ifdef __AVX2__
    if constexpr (sizeof(T) == 1) {
      const auto* ptr = reinterpret_cast<const uint8_t*>(&m_morton_buf[idx_offset]);
      const __m256i t = _mm256_set1_epi8(static_cast<char>(m_threshold));
      
      // Load 64 bytes (2 vectors)
      __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
      __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 32));
      
      // Compare v >= t using max: (max(v, t) == v)
      __m256i max0 = _mm256_max_epu8(v0, t);
      __m256i cmp0 = _mm256_cmpeq_epi8(max0, v0); // 0xFF if v >= t, 0x00 else
      
      __m256i max1 = _mm256_max_epu8(v1, t);
      __m256i cmp1 = _mm256_cmpeq_epi8(max1, v1);
      
      // Extract mask
      uint32_t mask0 = _mm256_movemask_epi8(cmp0);
      uint32_t mask1 = _mm256_movemask_epi8(cmp1);
      
      word = ((uint64_t)mask1 << 32) | mask0;
      optimized = true;
    }
    else if constexpr (sizeof(T) == 4) {
      const auto* ptr = reinterpret_cast<const uint8_t*>(&m_morton_buf[idx_offset]);
      const __m256i t = _mm256_set1_epi32(static_cast<int>(m_threshold));

      for (int k = 0; k < 8; ++k) {
         __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + k * 32));
         __m256i m = _mm256_max_epu32(v, t);
         __m256i c = _mm256_cmpeq_epi32(m, v);
         uint32_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(c)); // 8 bits
         word |= ((uint64_t)mask << (k * 8));
      }
      optimized = true;
    }
    else if constexpr (sizeof(T) == 8) {
      const auto* ptr = reinterpret_cast<const uint8_t*>(&m_morton_buf[idx_offset]);
      // threshold is power of 2. mask = ~(threshold - 1).
      // If val >= threshold, val & mask != 0.
      uint64_t thresh_mask = ~(m_threshold - 1);
      const __m256i t_mask = _mm256_set1_epi64x(static_cast<long long>(thresh_mask));
      const __m256i zero = _mm256_setzero_si256();

      for (int k = 0; k < 16; ++k) {
         __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + k * 32));
         __m256i masked = _mm256_and_si256(v, t_mask);
         __m256i c = _mm256_cmpeq_epi64(masked, zero); // -1 if zero (insig), 0 if non-zero (sig)
         int mask = _mm256_movemask_pd(_mm256_castsi256_pd(c)); // 4 bits. 1 means insig.
         word |= ((uint64_t)(~mask & 0xF) << (k * 4));
      }
      optimized = true;
    }
#endif

    if (!optimized) {
      for (size_t i = 0; i < 64; i++) {
        uint64_t sig = m_morton_buf[idx_offset + i] >= m_threshold;
        word |= (sig << i);
      }
    }
    m_mmask.wlong(processed_bits, word);
    processed_bits += 64;
  }

  if (processed_bits < len) {
    uint64_t word = 0;
    auto nbits = len - processed_bits;
    auto idx_offset = m_mmask_offset + processed_bits;
    for (size_t i = 0; i < nbits; i++) {
      uint64_t sig = m_morton_buf[idx_offset + i] >= m_threshold;
      word |= (sig << i);
    }
    m_mmask.wlong(processed_bits, word);
  }
}

template <typename T>
void sperr::SPECK3D_INT_ENC<T>::m_process_S(size_t idx1, size_t idx2, size_t& counter, bool output)
{
  auto& set = m_LIS[idx1][idx2];
  auto is_sig = true;

  // If need to output, it means the current set has unknown significance.
  if (output) {
    assert(set.get_morton() >= m_mmask_offset);
    is_sig = m_mmask.has_true(set.get_morton() - m_mmask_offset, set.num_elem());
    m_bit_buffer.wbit(is_sig);
  }

  if (is_sig) {
    counter++;
    m_code_S(idx1, idx2);
    set.make_empty();  // this current set is gonna be discarded.
  }
}

template <typename T>
void sperr::SPECK3D_INT_ENC<T>::m_process_P(size_t idx, size_t morton, size_t& counter, bool output)
{
  bool is_sig = true;

  if (output) {
    assert(morton >= m_mmask_offset);
    is_sig = m_mmask.rbit(morton - m_mmask_offset);
    m_bit_buffer.wbit(is_sig);
  }

  if (is_sig) {
    counter++;  // Let's increment the counter first!
    assert(m_coeff_buf[idx] >= m_threshold);
    m_coeff_buf[idx] -= m_threshold;

    m_bit_buffer.wbit(m_sign_array.rbit(idx));
    m_LSP_new.push_back(idx);
    m_LIP_mask.wfalse(idx);
  }
}

template <typename T>
void sperr::SPECK3D_INT_ENC<T>::m_process_P_lite(size_t idx)
{
  auto is_sig = (m_coeff_buf[idx] >= m_threshold);
  m_bit_buffer.wbit(is_sig);

  if (is_sig) {
    assert(m_coeff_buf[idx] >= m_threshold);
    m_coeff_buf[idx] -= m_threshold;

    m_bit_buffer.wbit(m_sign_array.rbit(idx));
    m_LSP_new.push_back(idx);
    m_LIP_mask.wfalse(idx);
  }
}

template class sperr::SPECK3D_INT_ENC<uint64_t>;
template class sperr::SPECK3D_INT_ENC<uint32_t>;
template class sperr::SPECK3D_INT_ENC<uint16_t>;
template class sperr::SPECK3D_INT_ENC<uint8_t>;
