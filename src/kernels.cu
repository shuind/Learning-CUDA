#include <vector>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

#include "../tester/utils.h"

template <typename T>
__global__ void trace_kernel(const T* input, size_t rows, size_t cols, T* output);

template <typename T>
__global__ void flash_attention_kernel(const T* q, const T* k, const T* v, T* o,
                                       int batch_size, int target_seq_len,
                                       int src_seq_len, int query_heads,
                                       int kv_heads, int head_dim, bool is_causal);

__device__ __forceinline__ float to_float(float v);
__device__ __forceinline__ float to_float(half v);

template <typename T>
__device__ __forceinline__ T from_float(float v);

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  size_t diag = rows < cols ? rows : cols;
  if (diag == 0) {
    return T(0);
  }

  const size_t total = rows * cols;
  T* d_input = nullptr;
  T* d_output = nullptr;
  T h_output = T(0);

  RUNTIME_CHECK(cudaMalloc(&d_input, total * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_output, sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), total * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_output, 0, sizeof(T)));

  const int threads = 256;
  const int blocks = static_cast<int>((diag + threads - 1) / threads);
  trace_kernel<<<blocks, threads>>>(d_input, rows, cols, d_output);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(&h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_output));
  return h_output;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  const int output_elems = batch_size * target_seq_len * query_heads * head_dim;
  if (output_elems <= 0) {
    h_o.clear();
    return;
  }
  h_o.resize(static_cast<size_t>(output_elems));

  const size_t q_elems = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  const size_t kv_elems = static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;

  if (q_elems == 0 || kv_elems == 0 || head_dim == 0) {
    std::fill(h_o.begin(), h_o.end(), T(0));
    return;
  }

  T* d_q = nullptr;
  T* d_k = nullptr;
  T* d_v = nullptr;
  T* d_o = nullptr;

  RUNTIME_CHECK(cudaMalloc(&d_q, q_elems * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, kv_elems * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, kv_elems * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, static_cast<size_t>(output_elems) * sizeof(T)));

  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_elems * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_elems * sizeof(T), cudaMemcpyHostToDevice));

  const int blocks = batch_size * target_seq_len * query_heads;
  const int threads = 1;
  const size_t shared_bytes = static_cast<size_t>(head_dim) * sizeof(float);
  flash_attention_kernel<<<blocks, threads, shared_bytes>>>(
      d_q, d_k, d_v, d_o,
      batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim, is_causal);
  RUNTIME_CHECK(cudaGetLastError());

  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, static_cast<size_t>(output_elems) * sizeof(T),
                           cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
}

template <typename T>
__global__ void trace_kernel(const T* input, size_t rows, size_t cols, T* output) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t diag = rows < cols ? rows : cols;
  if (idx < diag) {
    const size_t offset = idx * cols + idx;
    atomicAdd(output, input[offset]);
  }
}

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ half from_float<half>(float v) { return __float2half_rn(v); }

template <typename T>
__global__ void flash_attention_kernel(const T* q, const T* k, const T* v, T* o,
                                       int batch_size, int target_seq_len,
                                       int src_seq_len, int query_heads,
                                       int kv_heads, int head_dim, bool is_causal) {
  const int idx = blockIdx.x;
  const int total = batch_size * target_seq_len * query_heads;
  if (idx >= total) {
    return;
  }

  const int b = idx / (target_seq_len * query_heads);
  const int rem = idx % (target_seq_len * query_heads);
  const int t = rem / query_heads;
  const int qh = rem % query_heads;

  int kv = 0;
  if (query_heads > 0) {
    kv = static_cast<int>((static_cast<long long>(qh) * kv_heads) / query_heads);
  }
  if (kv < 0) kv = 0;
  if (kv >= kv_heads) kv = kv_heads - 1;

  const float scale = rsqrtf(static_cast<float>(head_dim));
  const T* q_ptr = q + ((b * target_seq_len + t) * query_heads + qh) * head_dim;

  float max_score = -INFINITY;
  for (int s = 0; s < src_seq_len; ++s) {
    if (is_causal && s > t) {
      continue;
    }
    const T* k_ptr = k + ((b * src_seq_len + s) * kv_heads + kv) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += to_float(q_ptr[d]) * to_float(k_ptr[d]);
    }
    const float score = dot * scale;
    if (score > max_score) {
      max_score = score;
    }
  }

  float sum_exp = 0.0f;
  for (int s = 0; s < src_seq_len; ++s) {
    if (is_causal && s > t) {
      continue;
    }
    const T* k_ptr = k + ((b * src_seq_len + s) * kv_heads + kv) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += to_float(q_ptr[d]) * to_float(k_ptr[d]);
    }
    const float score = dot * scale;
    sum_exp += expf(score - max_score);
  }

  const float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;

  extern __shared__ float shared[];
  float* out = shared;
  for (int d = 0; d < head_dim; ++d) {
    out[d] = 0.0f;
  }

  for (int s = 0; s < src_seq_len; ++s) {
    if (is_causal && s > t) {
      continue;
    }
    const T* k_ptr = k + ((b * src_seq_len + s) * kv_heads + kv) * head_dim;
    const T* v_ptr = v + ((b * src_seq_len + s) * kv_heads + kv) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += to_float(q_ptr[d]) * to_float(k_ptr[d]);
    }
    const float score = dot * scale;
    const float weight = expf(score - max_score) * inv_sum;
    for (int d = 0; d < head_dim; ++d) {
      out[d] += weight * to_float(v_ptr[d]);
    }
  }

  T* o_ptr = o + ((b * target_seq_len + t) * query_heads + qh) * head_dim;
  for (int d = 0; d < head_dim; ++d) {
    o_ptr[d] = from_float<T>(out[d]);
  }
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
