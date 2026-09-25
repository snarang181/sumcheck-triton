#include <cuda_runtime.h>
#include <openssl/bn.h>
#include <openssl/evp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


#define CUDA_CHECK(expr)                                                                      \
  do {                                                                                        \
    cudaError_t _err = (expr);                                                                \
    if (_err != cudaSuccess) {                                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err) << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;                                                    \
      std::exit(1);                                                                           \
    }                                                                                         \
  } while (0)

static constexpr int MAX_LIMBS = 8;
static constexpr int MAX_POINTS = 8;

#define WORKLOAD_LIST(X) \
  X(1, 1)   X(2, 2)   X(3, 3)   X(4, 3)   X(5, 3)   X(6, 5)   X(7, 6)  \
  X(8, 4)   X(9, 4)   X(10, 2)  X(11, 3)  X(12, 6)  X(13, 5)  X(14, 5) \
  X(15, 3)  X(16, 3)  X(17, 6)  X(18, 7)  X(19, 6)  X(20, 5)  X(21, 7) \
  X(22, 7)  X(23, 8)  X(24, 5)  X(25, 5)  X(26, 8)  X(27, 8)  X(28, 9) \
  X(29, 11) X(30, 22) X(31, 15) X(32, 12)


struct FieldHost {
  const char* name;
  int bit_width;
  int limbs;
  uint32_t nprime;
  uint32_t modulus[MAX_LIMBS];
  uint32_t point_monts[MAX_POINTS][MAX_LIMBS];
  uint32_t r2[MAX_LIMBS];
  const char* modulus_hex;
};

static const FieldHost FIELDS[] = {
    {"p32_2^32_minus_5",
     32,
     1,
     0xcccccccdu,
     {0xfffffffbu, 0, 0, 0, 0, 0, 0, 0},
     {{0x00000000u},
      {0x00000005u},
      {0x0000000au},
      {0x0000000fu},
      {0x00000014u},
      {0x00000019u},
      {0x0000001eu},
      {0x00000023u}},
     {0x00000019u, 0, 0, 0, 0, 0, 0, 0},
     "0xfffffffb"},
    {"p64_goldilocks",
     64,
     2,
     0xffffffffu,
     {0x00000001u, 0xffffffffu, 0, 0, 0, 0, 0, 0},
     {{0x00000000u, 0x00000000u},
      {0xffffffffu, 0x00000000u},
      {0xfffffffeu, 0x00000001u},
      {0xfffffffdu, 0x00000002u},
      {0xfffffffcu, 0x00000003u},
      {0xfffffffbu, 0x00000004u},
      {0xfffffffau, 0x00000005u},
      {0xfffffff9u, 0x00000006u}},
     {0x00000001u, 0xfffffffeu, 0, 0, 0, 0, 0, 0},
     "0xffffffff00000001"},
    {"p128_mersenne_2^127_minus_1",
     128,
     4,
     0x00000001u,
     {0xffffffffu, 0xffffffffu, 0xffffffffu, 0x7fffffffu, 0, 0, 0, 0},
     {{0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
      {0x00000002u, 0x00000000u, 0x00000000u, 0x00000000u},
      {0x00000004u, 0x00000000u, 0x00000000u, 0x00000000u},
      {0x00000006u, 0x00000000u, 0x00000000u, 0x00000000u},
      {0x00000008u, 0x00000000u, 0x00000000u, 0x00000000u},
      {0x0000000au, 0x00000000u, 0x00000000u, 0x00000000u},
      {0x0000000cu, 0x00000000u, 0x00000000u, 0x00000000u},
      {0x0000000eu, 0x00000000u, 0x00000000u, 0x00000000u}},
     {0x00000004u, 0x00000000u, 0x00000000u, 0x00000000u, 0, 0, 0, 0},
     "0x7fffffffffffffffffffffffffffffff"},
    {"p255_bls12_381_scalar",
     256,
     8,
     0xffffffffu,
     {0x00000001u, 0xffffffffu, 0xfffe5bfeu, 0x53bda402u,
      0x09a1d805u, 0x3339d808u, 0x299d7d48u, 0x73eda753u},
     {{0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
       0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
      {0xfffffffeu, 0x00000001u, 0x00034802u, 0x5884b7fau,
       0xecbc4ff5u, 0x998c4fefu, 0xacc5056fu, 0x1824b159u},
      {0xfffffffcu, 0x00000003u, 0x00069004u, 0xb1096ff4u,
       0xd9789feau, 0x33189fdfu, 0x598a0adfu, 0x304962b3u},
      {0xfffffffau, 0x00000005u, 0x0009d806u, 0x098e27eeu,
       0xc634efe0u, 0xcca4efcfu, 0x064f104eu, 0x486e140du},
      {0xfffffff8u, 0x00000007u, 0x000d2008u, 0x6212dfe8u,
       0xb2f13fd5u, 0x66313fbfu, 0xb31415beu, 0x6092c566u},
      {0xfffffff5u, 0x0000000au, 0x00120c0bu, 0x66d9f3dfu,
       0x960bb7c5u, 0xcc83b7a7u, 0x363b9de5u, 0x04c9cf6du},
      {0xfffffff3u, 0x0000000cu, 0x0015540du, 0xbf5eabd9u,
       0x82c807bau, 0x66100797u, 0xe300a355u, 0x1cee80c6u},
      {0xfffffff1u, 0x0000000eu, 0x00189c0fu, 0x17e363d3u,
       0x6f8457b0u, 0xff9c5787u, 0x8fc5a8c4u, 0x35133220u}},
     {0xf3f29c6du, 0xc999e990u, 0x87925c23u, 0x2b6cedcbu,
      0x7254398fu, 0x05d31496u, 0x9f59ff11u, 0x0748d9d9u},
     "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"},
};

__host__ __device__ inline uint64_t initial_value64(int var, size_t off, int seed) {
  uint64_t x = static_cast<uint64_t>(static_cast<uint32_t>(seed));
  x += 7919ull * static_cast<uint64_t>(var + 1);
  x += 104729ull * static_cast<uint64_t>(off + 1);
  return x;
}

template <int L>
struct FE {
  uint32_t v[L];
};

template <int L>
__device__ __forceinline__ FE<L> zero_fe() {
  FE<L> z{};
#pragma unroll
  for (int i = 0; i < L; ++i) z.v[i] = 0;
  return z;
}

template <int L>
__device__ __forceinline__ FE<L> load_fe(const uint32_t* ptr, size_t idx) {
  FE<L> x;
#pragma unroll
  for (int i = 0; i < L; ++i) x.v[i] = ptr[idx * L + i];
  return x;
}

template <int L>
__device__ __forceinline__ FE<L> load_const_fe(const uint32_t* ptr) {
  FE<L> x;
#pragma unroll
  for (int i = 0; i < L; ++i) x.v[i] = ptr[i];
  return x;
}

template <int L>
__device__ __forceinline__ void store_fe(uint32_t* ptr, size_t idx, const FE<L>& x) {
#pragma unroll
  for (int i = 0; i < L; ++i) ptr[idx * L + i] = x.v[i];
}

template <int L, bool LIMB_MAJOR>
__device__ __forceinline__ FE<L> load_table_fe(const uint32_t* ptr, size_t idx, size_t row_stride) {
  FE<L> x;
#pragma unroll
  for (int i = 0; i < L; ++i) {
    if constexpr (LIMB_MAJOR) {
      x.v[i] = ptr[static_cast<size_t>(i) * row_stride + idx];
    } else {
      x.v[i] = ptr[idx * L + i];
    }
  }
  return x;
}

template <int L, bool LIMB_MAJOR>
__device__ __forceinline__ void store_table_fe(uint32_t* ptr, size_t idx, size_t row_stride,
                                               const FE<L>& x) {
#pragma unroll
  for (int i = 0; i < L; ++i) {
    if constexpr (LIMB_MAJOR) {
      ptr[static_cast<size_t>(i) * row_stride + idx] = x.v[i];
    } else {
      ptr[idx * L + i] = x.v[i];
    }
  }
}

template <int L>
__device__ __forceinline__ int cmp_mod(const FE<L>& a, const uint32_t* mod) {
#pragma unroll
  for (int i = L - 1; i >= 0; --i) {
    uint32_t ai = a.v[i];
    uint32_t mi = mod[i];
    if (ai > mi) return 1;
    if (ai < mi) return -1;
  }
  return 0;
}

template <int L>
__device__ __forceinline__ FE<L> sub_raw(const FE<L>& a, const FE<L>& b, uint32_t* borrow_out) {
  FE<L> out;
  uint64_t borrow = 0;
#pragma unroll
  for (int i = 0; i < L; ++i) {
    uint64_t bi = static_cast<uint64_t>(b.v[i]) + borrow;
    uint64_t ai = static_cast<uint64_t>(a.v[i]);
    out.v[i] = static_cast<uint32_t>(ai - bi);
    borrow = ai < bi;
  }
  *borrow_out = static_cast<uint32_t>(borrow);
  return out;
}

template <int L>
__device__ __forceinline__ FE<L> add_raw(const FE<L>& a, const FE<L>& b, uint32_t* carry_out) {
  FE<L> out;
  uint64_t carry = 0;
#pragma unroll
  for (int i = 0; i < L; ++i) {
    uint64_t uv = static_cast<uint64_t>(a.v[i]) + b.v[i] + carry;
    out.v[i] = static_cast<uint32_t>(uv);
    carry = uv >> 32;
  }
  *carry_out = static_cast<uint32_t>(carry);
  return out;
}

template <int L>
__device__ __forceinline__ FE<L> add_mod(const FE<L>& a, const FE<L>& b, const uint32_t* mod) {
  uint32_t carry;
  FE<L> s = add_raw(a, b, &carry);
  if (carry || cmp_mod(s, mod) >= 0) {
    FE<L> m = load_const_fe<L>(mod);
    uint32_t borrow;
    return sub_raw(s, m, &borrow);
  }
  return s;
}

template <int L>
__device__ __forceinline__ FE<L> sub_mod(const FE<L>& a, const FE<L>& b, const uint32_t* mod) {
  uint32_t borrow;
  FE<L> d = sub_raw(a, b, &borrow);
  if (!borrow) return d;
  FE<L> m = load_const_fe<L>(mod);
  uint32_t carry;
  return add_raw(d, m, &carry);
}

template <int L>
__device__ __forceinline__ FE<L> double_mod(const FE<L>& a, const uint32_t* mod) {
  return add_mod(a, a, mod);
}

template <int L>
__device__ __forceinline__ FE<L> mul5_mod(const FE<L>& a, const uint32_t* mod) {
  FE<L> two = double_mod(a, mod);
  FE<L> four = double_mod(two, mod);
  return add_mod(four, a, mod);
}

template <int L>
__device__ __forceinline__ FE<L> mul3_mod(const FE<L>& a, const uint32_t* mod) {
  return add_mod(double_mod(a, mod), a, mod);
}

template <int L>
__device__ __forceinline__ FE<L> mul7_mod(const FE<L>& a, const uint32_t* mod) {
  return add_mod(mul5_mod(a, mod), double_mod(a, mod), mod);
}

template <int L>
__device__ __forceinline__ FE<L> mont_mul(const FE<L>& a, const FE<L>& b,
                                          const uint32_t* mod, uint32_t nprime) {
  uint32_t t[L + 1];
#pragma unroll
  for (int i = 0; i <= L; ++i) t[i] = 0;

#pragma unroll
  for (int i = 0; i < L; ++i) {
    uint64_t carry = 0;
#pragma unroll
    for (int j = 0; j < L; ++j) {
      uint64_t uv = static_cast<uint64_t>(t[j]) + static_cast<uint64_t>(a.v[j]) * b.v[i] + carry;
      t[j] = static_cast<uint32_t>(uv);
      carry = uv >> 32;
    }
    uint64_t uvn = static_cast<uint64_t>(t[L]) + carry;
    t[L] = static_cast<uint32_t>(uvn);

    uint32_t q = t[0] * nprime;
    carry = 0;
#pragma unroll
    for (int j = 0; j < L; ++j) {
      uint64_t uv = static_cast<uint64_t>(t[j]) + static_cast<uint64_t>(q) * mod[j] + carry;
      uint32_t low = static_cast<uint32_t>(uv);
      carry = uv >> 32;
      if (j > 0) t[j - 1] = low;
    }
    uint64_t last = static_cast<uint64_t>(t[L]) + carry;
    t[L - 1] = static_cast<uint32_t>(last);
    t[L] = static_cast<uint32_t>(last >> 32);
  }

  FE<L> res;
#pragma unroll
  for (int i = 0; i < L; ++i) res.v[i] = t[i];
  if (t[L] || cmp_mod(res, mod) >= 0) {
    FE<L> m = load_const_fe<L>(mod);
    uint32_t borrow;
    res = sub_raw(res, m, &borrow);
  }
  return res;
}

template <int L>
__device__ __forceinline__ FE<L> point_fe(const uint32_t* point_monts, int point_idx) {
  return load_const_fe<L>(point_monts + point_idx * L);
}

template <int L, int POINT_MODE, int POINT_IDX>
__device__ __forceinline__ FE<L> interp_var(const FE<L>& even, const FE<L>& odd,
                                            const FE<L>& point, const uint32_t* mod,
                                            uint32_t nprime) {
  if constexpr (POINT_MODE == 1 && POINT_IDX == 0) {
    (void)odd; (void)point; (void)nprime;
    return even;
  } else if constexpr (POINT_MODE == 1 && POINT_IDX == 1) {
    (void)even; (void)point; (void)nprime;
    return odd;
  } else {
    FE<L> diff = sub_mod(odd, even, mod);
    if constexpr (POINT_MODE == 1 && POINT_IDX >= 2 && POINT_IDX <= 15) {
      (void)point; (void)nprime;
      FE<L> scaled = diff;
#pragma unroll
      for (int k = 1; k < POINT_IDX; ++k) scaled = add_mod(scaled, diff, mod);
      return add_mod(even, scaled, mod);
    } else {
      return add_mod(even, mont_mul(diff, point, mod, nprime), mod);
    }
  }
}

template <int L>
__device__ __forceinline__ FE<L> prod2(const FE<L>& a, const FE<L>& b, const uint32_t* mod,
                                       uint32_t nprime) {
  return mont_mul(a, b, mod, nprime);
}

template <int L>
__device__ __forceinline__ FE<L> prod3(const FE<L>& a, const FE<L>& b, const FE<L>& c,
                                       const uint32_t* mod, uint32_t nprime) {
  return mont_mul(prod2(a, b, mod, nprime), c, mod, nprime);
}

template <int L>
__device__ __forceinline__ FE<L> prod4(const FE<L>& a, const FE<L>& b, const FE<L>& c,
                                       const FE<L>& d, const uint32_t* mod, uint32_t nprime) {
  return mont_mul(prod3(a, b, c, mod, nprime), d, mod, nprime);
}

template <int L>
__device__ __forceinline__ FE<L> prod5(const FE<L>& a, const FE<L>& b, const FE<L>& c,
                                       const FE<L>& d, const FE<L>& e, const uint32_t* mod,
                                       uint32_t nprime) {
  return mont_mul(prod4(a, b, c, d, mod, nprime), e, mod, nprime);
}

template <int L>
__device__ __forceinline__ FE<L> prod6(const FE<L>& a, const FE<L>& b, const FE<L>& c,
                                       const FE<L>& d, const FE<L>& e, const FE<L>& f,
                                       const uint32_t* mod, uint32_t nprime) {
  return mont_mul(prod5(a, b, c, d, e, mod, nprime), f, mod, nprime);
}

template <int L>
__device__ __forceinline__ FE<L> prod7(const FE<L>& a, const FE<L>& b, const FE<L>& c,
                                       const FE<L>& d, const FE<L>& e, const FE<L>& f,
                                       const FE<L>& g, const uint32_t* mod, uint32_t nprime) {
  return mont_mul(prod6(a, b, c, d, e, f, mod, nprime), g, mod, nprime);
}

template <int W, int L>
__device__ __forceinline__ FE<L> eval_workload(const FE<L>* p, const uint32_t* mod,
                                               uint32_t nprime) {
  if constexpr (W == 1) {
    return p[0];
  } else if constexpr (W == 2) {
    return prod2(p[0], p[1], mod, nprime);
  } else if constexpr (W == 3) {
    return add_mod(prod2(p[0], p[1], mod, nprime), p[2], mod);
  } else if constexpr (W == 4) {
    return prod3(p[0], p[1], p[2], mod, nprime);
  } else if constexpr (W == 5) {
    FE<L> aa = prod2(p[0], p[0], mod, nprime);
    FE<L> bb = prod2(p[1], p[1], mod, nprime);
    return mont_mul(mont_mul(aa, bb, mod, nprime), p[2], mod, nprime);
  } else if constexpr (W == 6) {
    FE<L> abc = prod3(p[0], p[1], p[2], mod, nprime);
    FE<L> de = prod2(p[3], p[4], mod, nprime);
    return add_mod(abc, de, mod);
  } else if constexpr (W == 7) {
    FE<L> abcg = prod4(p[0], p[1], p[2], p[3], mod, nprime);
    FE<L> deg = prod3(p[4], p[5], p[3], mod, nprime);
    return add_mod(abcg, deg, mod);
  } else if constexpr (W == 8) {
    FE<L> qadd_a = prod2(p[0], p[1], mod, nprime);
    FE<L> qadd_b = prod2(p[0], p[2], mod, nprime);
    FE<L> qmul_ab = prod3(p[3], p[1], p[2], mod, nprime);
    return add_mod(add_mod(qadd_a, qadd_b, mod), qmul_ab, mod);
  } else if constexpr (W == 9) {
    FE<L> abfz = prod3(p[0], p[1], p[2], mod, nprime);
    FE<L> cfz = prod2(p[3], p[2], mod, nprime);
    return sub_mod(abfz, cfz, mod);
  } else if constexpr (W == 10) {
    return prod2(p[0], p[1], mod, nprime);
  } else if constexpr (W == 11) {
    FE<L> qyy = prod3(p[0], p[1], p[1], mod, nprime);
    FE<L> qxxx = prod4(p[0], p[2], p[2], p[2], mod, nprime);
    return sub_mod(sub_mod(qyy, qxxx, mod), mul5_mod(p[0], mod), mod);
  } else if constexpr (W == 12) {
    FE<L> q_xq_xq_l = prod4(p[0], p[1], p[1], p[3], mod, nprime);
    FE<L> q_xq_xp_l = prod4(p[0], p[1], p[2], p[3], mod, nprime);
    FE<L> q_xp_xp_l = prod4(p[0], p[2], p[2], p[3], mod, nprime);
    FE<L> q_xq_yq = prod3(p[0], p[1], p[4], mod, nprime);
    FE<L> q_xq_yp = prod3(p[0], p[1], p[5], mod, nprime);
    FE<L> q_xp_yq = prod3(p[0], p[2], p[4], mod, nprime);
    FE<L> q_xp_yp = prod3(p[0], p[2], p[5], mod, nprime);
    FE<L> acc = add_mod(q_xq_xq_l, q_xp_xp_l, mod);
    acc = sub_mod(acc, double_mod(q_xq_xp_l, mod), mod);
    acc = sub_mod(acc, q_xq_yq, mod);
    acc = add_mod(acc, q_xq_yp, mod);
    acc = add_mod(acc, q_xp_yq, mod);
    return sub_mod(acc, q_xp_yp, mod);
  } else if constexpr (W == 13) {
    FE<L> q_xr = prod2(p[0], p[1], mod, nprime);
    FE<L> q_xq = prod2(p[0], p[2], mod, nprime);
    FE<L> q_xp_xr_beta = prod4(p[0], p[3], p[1], p[4], mod, nprime);
    FE<L> q_xp_xq_beta = prod4(p[0], p[3], p[2], p[4], mod, nprime);
    return add_mod(sub_mod(sub_mod(q_xr, q_xq, mod), q_xp_xr_beta, mod), q_xp_xq_beta, mod);
  } else if constexpr (W == 14) {
    FE<L> acc = prod2(p[0], p[1], mod, nprime);
    acc = sub_mod(acc, prod2(p[0], p[2], mod, nprime), mod);
    acc = sub_mod(acc, prod4(p[0], p[3], p[1], p[4], mod, nprime), mod);
    return add_mod(acc, prod4(p[0], p[3], p[2], p[4], mod, nprime), mod);
  } else if constexpr (W == 15) {
    FE<L> qxyy = prod4(p[0], p[1], p[2], p[2], mod, nprime);
    FE<L> qxxxx = prod5(p[0], p[1], p[1], p[1], p[1], mod, nprime);
    return sub_mod(sub_mod(qxyy, qxxxx, mod), mul5_mod(prod2(p[0], p[1], mod, nprime), mod), mod);
  } else if constexpr (W == 16) {
    FE<L> qyyy = prod4(p[0], p[1], p[1], p[1], mod, nprime);
    FE<L> qyxxx = prod5(p[0], p[1], p[2], p[2], p[2], mod, nprime);
    return sub_mod(sub_mod(qyyy, qyxxx, mod), mul5_mod(prod2(p[0], p[1], mod, nprime), mod), mod);
  } else if constexpr (W == 17) {
    FE<L> acc = prod4(p[0], p[1], p[2], p[2], mod, nprime);
    acc = sub_mod(acc, double_mod(prod4(p[0], p[1], p[2], p[3], mod, nprime), mod), mod);
    acc = add_mod(acc, prod4(p[0], p[1], p[3], p[3], mod, nprime), mod);
    acc = add_mod(acc, prod4(p[0], p[3], p[2], p[2], mod, nprime), mod);
    acc = sub_mod(acc, double_mod(prod4(p[0], p[3], p[3], p[2], mod, nprime), mod), mod);
    acc = add_mod(acc, prod4(p[0], p[3], p[3], p[3], mod, nprime), mod);
    acc = add_mod(acc, prod4(p[0], p[2], p[2], p[2], mod, nprime), mod);
    acc = sub_mod(acc, double_mod(prod4(p[0], p[2], p[2], p[3], mod, nprime), mod), mod);
    acc = add_mod(acc, prod4(p[0], p[2], p[3], p[3], mod, nprime), mod);
    acc = sub_mod(acc, prod3(p[0], p[4], p[4], mod, nprime), mod);
    acc = add_mod(acc, double_mod(prod3(p[0], p[4], p[5], mod, nprime), mod), mod);
    return sub_mod(acc, prod3(p[0], p[5], p[5], mod, nprime), mod);
  } else if constexpr (W == 18) {
    FE<L> acc = prod3(p[0], p[1], p[2], mod, nprime);
    acc = sub_mod(acc, prod3(p[0], p[1], p[3], mod, nprime), mod);
    acc = add_mod(acc, prod3(p[0], p[4], p[2], mod, nprime), mod);
    acc = sub_mod(acc, prod3(p[0], p[4], p[3], mod, nprime), mod);
    acc = sub_mod(acc, prod3(p[0], p[5], p[3], mod, nprime), mod);
    acc = add_mod(acc, prod3(p[0], p[5], p[6], mod, nprime), mod);
    acc = sub_mod(acc, prod3(p[0], p[4], p[6], mod, nprime), mod);
    return add_mod(acc, prod3(p[0], p[4], p[3], mod, nprime), mod);
  } else if constexpr (W == 19) {
    FE<L> acc = double_mod(prod3(p[0], p[1], p[2], mod, nprime), mod);
    acc = sub_mod(acc, mul3_mod(prod3(p[0], p[3], p[3], mod, nprime), mod), mod);
    acc = sub_mod(acc, double_mod(prod5(p[0], p[4], p[1], p[2], p[5], mod, nprime), mod), mod);
    acc = add_mod(acc, mul3_mod(prod5(p[0], p[4], p[3], p[3], p[5], mod, nprime), mod), mod);
    acc = add_mod(acc, double_mod(prod5(p[0], p[3], p[1], p[2], p[5], mod, nprime), mod), mod);
    return sub_mod(acc, mul3_mod(prod5(p[0], p[3], p[3], p[3], p[5], mod, nprime), mod), mod);
  } else if constexpr (W == 20) {
    FE<L> acc = prod6(p[0], p[1], p[2], p[2], p[3], p[3], mod, nprime);
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[2], p[2], mod, nprime), mod);
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[2], p[4], mod, nprime), mod);
    acc = sub_mod(acc, prod5(p[0], p[1], p[1], p[2], p[2], mod, nprime), mod);
    acc = sub_mod(acc, prod6(p[0], p[1], p[1], p[2], p[3], p[3], mod, nprime), mod);
    acc = add_mod(acc, prod5(p[0], p[1], p[1], p[1], p[2], mod, nprime), mod);
    acc = add_mod(acc, prod5(p[0], p[1], p[1], p[2], p[4], mod, nprime), mod);
    return add_mod(acc, prod5(p[0], p[1], p[1], p[2], p[2], mod, nprime), mod);
  } else if constexpr (W == 21) {
    FE<L> acc = prod6(p[0], p[1], p[1], p[2], p[2], p[3], mod, nprime);
    acc = sub_mod(acc, prod6(p[0], p[1], p[2], p[2], p[4], p[3], mod, nprime), mod);
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[2], p[5], mod, nprime), mod);
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[2], p[6], mod, nprime), mod);
    acc = sub_mod(acc, prod6(p[0], p[1], p[1], p[1], p[2], p[3], mod, nprime), mod);
    acc = add_mod(acc, prod6(p[0], p[1], p[1], p[2], p[4], p[3], mod, nprime), mod);
    acc = add_mod(acc, prod5(p[0], p[1], p[1], p[2], p[5], mod, nprime), mod);
    return add_mod(acc, prod5(p[0], p[1], p[1], p[2], p[6], mod, nprime), mod);
  } else if constexpr (W == 22) {
    FE<L> acc = prod6(p[0], p[1], p[2], p[3], p[4], p[4], mod, nprime);
    acc = add_mod(acc, prod6(p[0], p[1], p[2], p[5], p[4], p[4], mod, nprime), mod);
    acc = sub_mod(acc, prod5(p[0], p[1], p[1], p[2], p[3], mod, nprime), mod);
    acc = sub_mod(acc, prod5(p[0], p[1], p[1], p[2], p[5], mod, nprime), mod);
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[2], p[3], mod, nprime), mod);
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[2], p[5], mod, nprime), mod);
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[3], p[6], mod, nprime), mod);
    return sub_mod(acc, prod5(p[0], p[1], p[2], p[5], p[6], mod, nprime), mod);
  } else if constexpr (W == 23) {
    FE<L> acc = prod6(p[0], p[1], p[1], p[2], p[3], p[4], mod, nprime);
    acc = add_mod(acc, prod6(p[0], p[1], p[1], p[2], p[5], p[4], mod, nprime), mod);
    acc = sub_mod(acc, prod6(p[0], p[1], p[2], p[5], p[6], p[4], mod, nprime), mod);
    acc = sub_mod(acc, prod6(p[0], p[1], p[2], p[3], p[6], p[4], mod, nprime), mod);
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[5], p[5], mod, nprime), mod);
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[5], p[3], mod, nprime), mod);
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[5], p[7], mod, nprime), mod);
    return sub_mod(acc, prod5(p[0], p[1], p[2], p[3], p[7], mod, nprime), mod);
  } else if constexpr (W == 24) {
    FE<L> acc = prod2(p[0], p[1], mod, nprime);
    acc = sub_mod(acc, prod2(p[0], p[2], mod, nprime), mod);
    acc = sub_mod(acc, prod4(p[0], p[3], p[1], p[4], mod, nprime), mod);
    return add_mod(acc, prod4(p[0], p[2], p[3], p[4], mod, nprime), mod);
  } else if constexpr (W == 25) {
    FE<L> acc = prod2(p[0], p[1], mod, nprime);
    acc = sub_mod(acc, prod2(p[0], p[2], mod, nprime), mod);
    acc = sub_mod(acc, prod4(p[0], p[3], p[1], p[4], mod, nprime), mod);
    return add_mod(acc, prod4(p[0], p[3], p[2], p[4], mod, nprime), mod);
  } else if constexpr (W == 26) {
    FE<L> acc = prod2(p[0], p[1], mod, nprime);
    acc = sub_mod(acc, prod4(p[0], p[2], p[1], p[3], mod, nprime), mod);
    acc = add_mod(acc, prod4(p[0], p[4], p[1], p[3], mod, nprime), mod);
    acc = sub_mod(acc, prod4(p[0], p[5], p[1], p[6], mod, nprime), mod);
    return sub_mod(acc, prod4(p[0], p[7], p[1], p[6], mod, nprime), mod);
  } else if constexpr (W == 27) {
    FE<L> acc = prod2(p[0], p[1], mod, nprime);
    acc = sub_mod(acc, prod4(p[0], p[2], p[1], p[3], mod, nprime), mod);
    acc = add_mod(acc, prod4(p[0], p[4], p[1], p[3], mod, nprime), mod);
    acc = sub_mod(acc, prod4(p[0], p[5], p[1], p[6], mod, nprime), mod);
    return sub_mod(acc, prod4(p[0], p[7], p[1], p[6], mod, nprime), mod);
  } else if constexpr (W == 28) {
    FE<L> acc = prod3(p[0], p[1], p[2], mod, nprime);
    acc = add_mod(acc, prod3(p[3], p[4], p[2], mod, nprime), mod);
    acc = add_mod(acc, prod4(p[5], p[1], p[4], p[2], mod, nprime), mod);
    acc = sub_mod(acc, prod3(p[6], p[7], p[2], mod, nprime), mod);
    return add_mod(acc, prod2(p[8], p[2], mod, nprime), mod);
  } else if constexpr (W == 29) {
    FE<L> acc = prod2(p[0], p[1], mod, nprime);
    acc = sub_mod(acc, prod3(p[2], p[3], p[1], mod, nprime), mod);
    acc = add_mod(acc, mul7_mod(prod5(p[4], p[5], p[6], p[7], p[1], mod, nprime), mod), mod);
    return sub_mod(acc, mul7_mod(prod4(p[8], p[9], p[10], p[1], mod, nprime), mod), mod);
  } else if constexpr (W == 30) {
    FE<L> acc = prod3(p[0], p[1], p[2], mod, nprime);
    acc = add_mod(acc, prod3(p[3], p[4], p[2], mod, nprime), mod);
    acc = add_mod(acc, prod4(p[5], p[1], p[4], p[2], mod, nprime), mod);
    acc = sub_mod(acc, prod3(p[6], p[7], p[2], mod, nprime), mod);
    acc = add_mod(acc, prod3(p[8], p[1], p[2], mod, nprime), mod);
    acc = add_mod(acc, prod3(p[9], p[4], p[2], mod, nprime), mod);
    acc = add_mod(acc, prod3(p[10], p[7], p[2], mod, nprime), mod);
    acc = add_mod(acc, prod3(p[11], p[12], p[2], mod, nprime), mod);
    acc = sub_mod(acc, prod3(p[6], p[13], p[2], mod, nprime), mod);
    acc = add_mod(acc, prod4(p[14], p[1], p[4], p[2], mod, nprime), mod);
    acc = add_mod(acc, prod4(p[15], p[7], p[12], p[2], mod, nprime), mod);
    acc = add_mod(acc, prod7(p[16], p[1], p[1], p[1], p[1], p[1], p[2], mod, nprime), mod);
    acc = add_mod(acc, prod7(p[17], p[4], p[4], p[4], p[4], p[4], p[2], mod, nprime), mod);
    acc = add_mod(acc, prod7(p[18], p[7], p[7], p[7], p[7], p[7], p[2], mod, nprime), mod);
    acc = add_mod(acc, prod7(p[19], p[12], p[12], p[12], p[12], p[12], p[2], mod, nprime), mod);
    acc = add_mod(acc, prod6(p[20], p[1], p[4], p[7], p[12], p[2], mod, nprime), mod);
    return add_mod(acc, prod2(p[21], p[2], mod, nprime), mod);
  } else if constexpr (W == 31) {
    FE<L> acc = prod2(p[0], p[1], mod, nprime);
    acc = sub_mod(acc, prod3(p[2], p[3], p[1], mod, nprime), mod);
    acc = add_mod(acc, mul7_mod(prod7(p[4], p[5], p[6], p[7], p[8], p[9], p[1], mod, nprime), mod), mod);
    return sub_mod(acc, mul7_mod(prod6(p[10], p[11], p[12], p[13], p[14], p[1], mod, nprime), mod), mod);
  } else {
    static_assert(W == 32, "unknown workload id");
    FE<L> acc = prod2(p[0], p[1], mod, nprime);
    acc = add_mod(acc, prod2(p[2], p[3], mod, nprime), mod);
    acc = add_mod(acc, prod2(p[4], p[5], mod, nprime), mod);
    acc = add_mod(acc, prod2(p[6], p[7], mod, nprime), mod);
    acc = add_mod(acc, prod2(p[8], p[9], mod, nprime), mod);
    return add_mod(acc, prod2(p[10], p[11], mod, nprime), mod);
  }
}


template <int L, bool LIMB_MAJOR>
__global__ void fill_tables_kernel(uint32_t* tables, int num_vars, size_t n, int seed,
                                   const uint32_t* mod, const uint32_t* r2, uint32_t nprime) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = static_cast<size_t>(num_vars) * n;
  if (idx >= total) return;
  int var = static_cast<int>(idx / n);
  size_t off = idx - static_cast<size_t>(var) * n;
  uint64_t value = initial_value64(var, off, seed);
  if constexpr (L == 1) value %= static_cast<uint64_t>(mod[0]);

  FE<L> raw = zero_fe<L>();
  raw.v[0] = static_cast<uint32_t>(value);
  if constexpr (L > 1) raw.v[1] = static_cast<uint32_t>(value >> 32);
  FE<L> r2_fe = load_const_fe<L>(r2);
  FE<L> enc = mont_mul(raw, r2_fe, mod, nprime);
  uint32_t* base = tables + static_cast<size_t>(var) * n * L;
  store_table_fe<L, LIMB_MAJOR>(base, off, n, enc);
}

template <int W, int NUM_VARS, int L, bool LIMB_MAJOR, int POINT_MODE, int POINT_IDX>
__global__ void eval_kernel(const uint32_t* tables, uint32_t* partials, size_t half_n,
                            const uint32_t* mod, const uint32_t* point_monts, uint32_t nprime) {
  extern __shared__ uint32_t sh[];
  int tid = threadIdx.x;
  size_t off = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
  size_t row_stride = half_n * 2;

  FE<L> total = zero_fe<L>();
  if (off < half_n) {
    FE<L> p[NUM_VARS];
    FE<L> point = point_fe<L>(point_monts, POINT_IDX);
#pragma unroll
    for (int var = 0; var < NUM_VARS; ++var) {
      const uint32_t* base = tables + static_cast<size_t>(var) * row_stride * L;
      FE<L> e = load_table_fe<L, LIMB_MAJOR>(base, 2 * off, row_stride);
      FE<L> o = load_table_fe<L, LIMB_MAJOR>(base, 2 * off + 1, row_stride);
      p[var] = interp_var<L, POINT_MODE, POINT_IDX>(e, o, point, mod, nprime);
    }
    total = eval_workload<W, L>(p, mod, nprime);
  }

#pragma unroll
  for (int limb = 0; limb < L; ++limb) sh[tid * L + limb] = total.v[limb];
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      FE<L> a, b;
#pragma unroll
      for (int limb = 0; limb < L; ++limb) {
        a.v[limb] = sh[tid * L + limb];
        b.v[limb] = sh[(tid + stride) * L + limb];
      }
      FE<L> s = add_mod(a, b, mod);
#pragma unroll
      for (int limb = 0; limb < L; ++limb) sh[tid * L + limb] = s.v[limb];
    }
    __syncthreads();
  }

  if (tid == 0) {
#pragma unroll
    for (int limb = 0; limb < L; ++limb) partials[blockIdx.x * L + limb] = sh[limb];
  }
}

template <int L>
__global__ void reduce_pairs_kernel(const uint32_t* partials_in, uint32_t* partials_out,
                                    size_t n, const uint32_t* mod) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t out_n = (n + 1) / 2;
  if (idx >= out_n) return;
  size_t lhs = 2 * idx;
  size_t rhs = lhs + 1;
  FE<L> a = load_fe<L>(partials_in, lhs);
  FE<L> out = a;
  if (rhs < n) {
    FE<L> b = load_fe<L>(partials_in, rhs);
    out = add_mod(a, b, mod);
  }
  store_fe<L>(partials_out, idx, out);
}

template <int NUM_VARS, int L, bool LIMB_MAJOR>
__global__ void fold_kernel(const uint32_t* tables, uint32_t* out, size_t half_n,
                            const uint32_t* mod, const uint32_t* r_mont, uint32_t nprime) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = static_cast<size_t>(NUM_VARS) * half_n;
  if (idx >= total) return;
  int var = static_cast<int>(idx / half_n);
  size_t off = idx - static_cast<size_t>(var) * half_n;
  size_t row_stride = half_n * 2;
  const uint32_t* base = tables + static_cast<size_t>(var) * row_stride * L;
  FE<L> e = load_table_fe<L, LIMB_MAJOR>(base, 2 * off, row_stride);
  FE<L> o = load_table_fe<L, LIMB_MAJOR>(base, 2 * off + 1, row_stride);
  FE<L> d = sub_mod(o, e, mod);
  FE<L> r = load_const_fe<L>(r_mont);
  FE<L> f = add_mod(e, mont_mul(d, r, mod, nprime), mod);
  uint32_t* out_base = out + static_cast<size_t>(var) * half_n * L;
  store_table_fe<L, LIMB_MAJOR>(out_base, off, half_n, f);
}

struct Workload {
  const char* name;
  const char* poly;
  int id;
  int vars;
  int degree;
  int terms;
};

static const Workload WORKLOADS[] = {
    {"poly_a", "a", 1, 1, 1, 1},
    {"poly_ab", "a*b", 2, 2, 2, 1},
    {"poly_ab_plus_c", "a*b + c", 3, 3, 2, 2},
    {"poly_abc", "a*b*c", 4, 3, 3, 1},
    {"poly_aabbc", "a*a*b*b*c", 5, 3, 5, 1},
    {"poly_abc_plus_de", "a*b*c + d*e", 6, 5, 3, 2},
    {"poly_abcg_plus_deg", "a*b*c*g + d*e*g", 7, 6, 4, 2},
    {"zk_verifiable_asics", "qadd*a + qadd*b + qmul*a*b", 8, 4, 3, 3},
    {"zk_spartan_1", "A*B*fz - C*fz", 9, 4, 3, 2},
    {"zk_spartan_2", "ABC*Z", 10, 2, 2, 1},
    {"zk_witness_non_id", "q*y*y - q*x*x*x - 5*q", 11, 3, 4, 3},
    {"zk_complete_add_1", "q*xq*xq*lambda - 2*q*xq*xp*lambda + q*xp*xp*lambda - q*xq*yq + q*xq*yp + q*xp*yq - q*xp*yp", 12, 6, 4, 7},
    {"zk_complete_add_7", "q*xr - q*xq - q*xp*xr*beta + q*xp*xq*beta", 13, 5, 4, 4},
    {"zk_complete_add_8", "q*yr - q*yq - q*xp*yr*beta + q*xp*yq*beta", 14, 5, 4, 4},
    {"zk_witness_id_point_1", "q*x*y*y - q*x*x*x*x - 5*q*x", 15, 3, 5, 3},
    {"zk_witness_id_point_2", "q*y*y*y - q*y*x*x*x - 5*q*y", 16, 3, 5, 3},
    {"zk_incomplete_add_1", "q*xr*xp*xp - 2*q*xr*xp*xq + q*xr*xq*xq + q*xq*xp*xp - 2*q*xq*xq*xp + q*xq*xq*xq + q*xp*xp*xp - 2*q*xp*xp*xq + q*xp*xq*xq - q*yp*yp + 2*q*yp*yq - q*yq*yq", 17, 6, 4, 12},
    {"zk_incomplete_add_2", "q*yr*xp - q*yr*xq + q*yq*xp - q*yq*xq - q*yp*xq + q*yp*xr - q*yq*xr + q*yq*xq", 18, 7, 3, 8},
    {"zk_complete_add_2", "2*q*yp*lambda - 3*q*xp*xp - 2*q*xq*yp*lambda*alpha + 3*q*xq*xp*xp*alpha + 2*q*xp*yp*lambda*alpha - 3*q*xp*xp*xp*alpha", 19, 6, 5, 6},
    {"zk_complete_add_3", "q*xp*xq*xq*lambda*lambda - q*xp*xq*xq*xq - q*xp*xq*xq*xr - q*xp*xp*xq*xq - q*xp*xp*xq*lambda*lambda + q*xp*xp*xp*xq + q*xp*xp*xq*xr + q*xp*xp*xq*xq", 20, 5, 6, 8},
    {"zk_complete_add_4", "q*xp*xp*xq*xq*lambda - q*xp*xq*xq*xr*lambda - q*xp*xq*xq*yp - q*xp*xq*xq*yr - q*xp*xp*xp*xq*lambda + q*xp*xp*xq*xr*lambda + q*xp*xp*xq*yp + q*xp*xp*xq*yr", 21, 7, 6, 8},
    {"zk_complete_add_5", "q*xp*xq*yq*lambda*lambda + q*xp*xq*yp*lambda*lambda - q*xp*xp*xq*yq - q*xp*xp*xq*yp - q*xp*xq*xq*yq - q*xp*xq*xq*yp - q*xp*xq*yq*xr - q*xp*xq*yp*xr", 22, 7, 6, 8},
    {"zk_complete_add_6", "q*xp*xp*xq*yq*lambda + q*xp*xp*xq*yp*lambda - q*xp*xq*yp*xr*lambda - q*xp*xq*yq*xr*lambda - q*xp*xq*yp*yp - q*xp*xq*yp*yq - q*xp*xq*yp*yr - q*xp*xq*yq*yr", 23, 8, 6, 8},
    {"zk_complete_add_9", "q*xr - q*xp - q*xq*xr*gamma + q*xp*xq*gamma", 24, 5, 4, 4},
    {"zk_complete_add_10", "q*yr - q*yp - q*xq*yr*gamma + q*xq*yp*gamma", 25, 5, 4, 4},
    {"zk_complete_add_11", "q*xr - q*xq*xr*alpha + q*xp*xr*alpha - q*yq*xr*delta - q*yp*xr*delta", 26, 8, 4, 5},
    {"zk_complete_add_12", "q*yr - q*xq*yr*alpha + q*xp*yr*alpha - q*yq*yr*delta - q*yp*yr*delta", 27, 8, 4, 5},
    {"zk_vanilla_zerocheck_hp", "qL*w1*fz1 + qR*w2*fz1 + qM*w1*w2*fz1 - qO*w3*fz1 + qC*fz1", 28, 9, 4, 5},
    {"zk_vanilla_permcheck_hp", "pi*fz2 - p1*p2*fz2 + 7*phi*D1*D2*D3*fz2 - 7*N1*N2*N3*fz2", 29, 11, 5, 4},
    {"zk_jellyfish_zerocheck_hp", "qL*w1*fz + qR*w2*fz + qM*w1*w2*fz - qO*w3*fz + q1*w1*fz + q2*w2*fz + q3*w3*fz + q4*w4*fz - qO*w5*fz + qM1*w1*w2*fz + qM2*w3*w4*fz + qH1*w1*w1*w1*w1*w1*fz + qH2*w2*w2*w2*w2*w2*fz + qH3*w3*w3*w3*w3*w3*fz + qH4*w4*w4*w4*w4*w4*fz + qECC*w1*w2*w3*w4*fz + qc*fz", 30, 22, 7, 17},
    {"zk_jellyfish_permcheck_hp", "pi*fz - p1*p2*fz + 7*phi*D1*D2*D3*D4*D5*fz - 7*N1*N2*N3*N4*N5*fz", 31, 15, 7, 4},
    {"zk_opencheck", "y1*k1 + y2*k2 + y3*k3 + y4*k4 + y5*k5 + y6*k6", 32, 12, 2, 6},
};

static std::string canonical_workload_name(const std::string& x) {
  static const std::pair<const char*, const char*> aliases[] = {
      {"zkphire_0", "zk_verifiable_asics"},
      {"zkphire_1", "zk_spartan_1"},
      {"zkphire_2", "zk_spartan_2"},
      {"zkphire_3", "zk_witness_non_id"},
      {"zkphire_4", "zk_witness_id_point_1"},
      {"zkphire_5", "zk_witness_id_point_2"},
      {"zkphire_6", "zk_incomplete_add_1"},
      {"zkphire_7", "zk_incomplete_add_2"},
      {"zkphire_8", "zk_complete_add_1"},
      {"zkphire_9", "zk_complete_add_2"},
      {"zkphire_10", "zk_complete_add_3"},
      {"zkphire_11", "zk_complete_add_4"},
      {"zkphire_12", "zk_complete_add_5"},
      {"zkphire_13", "zk_complete_add_6"},
      {"zkphire_14", "zk_complete_add_7"},
      {"zkphire_15", "zk_complete_add_8"},
      {"zkphire_16", "zk_complete_add_9"},
      {"zkphire_17", "zk_complete_add_10"},
      {"zkphire_18", "zk_complete_add_11"},
      {"zkphire_19", "zk_complete_add_12"},
      {"zkphire_20", "zk_vanilla_zerocheck_hp"},
      {"zkphire_21", "zk_vanilla_permcheck_hp"},
      {"zkphire_22", "zk_jellyfish_zerocheck_hp"},
      {"zkphire_23", "zk_jellyfish_permcheck_hp"},
      {"zkphire_24", "zk_opencheck"},
  };
  for (const auto& alias : aliases) {
    if (x == alias.first) return alias.second;
  }
  return x;
}

static bool contains(const std::vector<std::string>& xs, const char* x) {
  const std::string canonical = std::string(x);
  for (const std::string& value : xs) {
    if (canonical_workload_name(value) == canonical) return true;
  }
  return false;
}

static bool contains_int(const std::vector<int>& xs, int x) {
  return std::find(xs.begin(), xs.end(), x) != xs.end();
}

static size_t cdiv(size_t x, size_t y) { return (x + y - 1) / y; }


struct RunMetrics {
  uint64_t checksum = 0;
  double eval_ms = 0.0;
  double challenge_ms = 0.0;
  double fold_ms = 0.0;
  double total_ms = 0.0;
};

static double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void require_bn(int ok, const char* message) {
  if (ok != 1) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

static BIGNUM* limbs_to_bn(const uint32_t* limbs, int limb_count) {
  std::vector<unsigned char> bytes(static_cast<size_t>(limb_count) * 4);
  for (int i = 0; i < limb_count; ++i) {
    size_t pos = bytes.size() - static_cast<size_t>(i + 1) * 4;
    uint32_t limb = limbs[i];
    bytes[pos + 0] = static_cast<unsigned char>((limb >> 24) & 0xffu);
    bytes[pos + 1] = static_cast<unsigned char>((limb >> 16) & 0xffu);
    bytes[pos + 2] = static_cast<unsigned char>((limb >> 8) & 0xffu);
    bytes[pos + 3] = static_cast<unsigned char>(limb & 0xffu);
  }
  BIGNUM* out = BN_bin2bn(bytes.data(), static_cast<int>(bytes.size()), nullptr);
  if (out == nullptr) {
    std::cerr << "failed to allocate BIGNUM\n";
    std::exit(1);
  }
  return out;
}

static void bn_to_limbs(const BIGNUM* value, int limb_count, std::vector<uint32_t>& out) {
  std::vector<unsigned char> bytes(static_cast<size_t>(limb_count) * 4);
  int written = BN_bn2binpad(value, bytes.data(), static_cast<int>(bytes.size()));
  if (written != static_cast<int>(bytes.size())) {
    std::cerr << "failed to serialize BIGNUM\n";
    std::exit(1);
  }
  out.assign(limb_count, 0);
  for (int i = 0; i < limb_count; ++i) {
    size_t pos = bytes.size() - static_cast<size_t>(i + 1) * 4;
    out[i] = (static_cast<uint32_t>(bytes[pos + 0]) << 24) |
             (static_cast<uint32_t>(bytes[pos + 1]) << 16) |
             (static_cast<uint32_t>(bytes[pos + 2]) << 8) |
             static_cast<uint32_t>(bytes[pos + 3]);
  }
}

static void update_u32_le(EVP_MD_CTX* ctx, uint32_t value) {
  const unsigned char bytes[4] = {
      static_cast<unsigned char>(value & 0xffu),
      static_cast<unsigned char>((value >> 8) & 0xffu),
      static_cast<unsigned char>((value >> 16) & 0xffu),
      static_cast<unsigned char>((value >> 24) & 0xffu),
  };
  if (EVP_DigestUpdate(ctx, bytes, sizeof(bytes)) != 1) {
    std::cerr << "failed to update SHA3\n";
    std::exit(1);
  }
}

static void sha3_challenge_mont_limbs(int round, const std::vector<uint32_t>& coeffs,
                                      const FieldHost& field, int limb_count,
                                      std::vector<uint32_t>& out) {
  EVP_MD_CTX* ctx = EVP_MD_CTX_new();
  if (ctx == nullptr) {
    std::cerr << "failed to allocate SHA3 context\n";
    std::exit(1);
  }
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int digest_len = 0;
  const char domain[] = "zkduel:field-sweep:v1";
  if (EVP_DigestInit_ex(ctx, EVP_sha3_256(), nullptr) != 1 ||
      EVP_DigestUpdate(ctx, domain, sizeof(domain) - 1) != 1) {
    std::cerr << "failed to initialize SHA3\n";
    std::exit(1);
  }
  update_u32_le(ctx, static_cast<uint32_t>(round));
  for (uint32_t limb : coeffs) update_u32_le(ctx, limb);
  if (EVP_DigestFinal_ex(ctx, digest, &digest_len) != 1) {
    std::cerr << "failed to finalize SHA3\n";
    std::exit(1);
  }
  EVP_MD_CTX_free(ctx);

  unsigned char digest_be[32];
  for (int i = 0; i < 32; ++i) digest_be[i] = digest[31 - i];
  BN_CTX* bn_ctx = BN_CTX_new();
  BIGNUM* modulus = limbs_to_bn(field.modulus, limb_count);
  BIGNUM* digest_bn = BN_bin2bn(digest_be, 32, nullptr);
  BIGNUM* challenge = BN_new();
  BIGNUM* r_mod = BN_new();
  BIGNUM* challenge_mont = BN_new();
  if (bn_ctx == nullptr || digest_bn == nullptr || challenge == nullptr || r_mod == nullptr ||
      challenge_mont == nullptr) {
    std::cerr << "failed to allocate BIGNUM state\n";
    std::exit(1);
  }
  require_bn(BN_mod(challenge, digest_bn, modulus, bn_ctx), "failed to reduce SHA3 digest");
  require_bn(BN_set_bit(r_mod, 32 * limb_count), "failed to build Montgomery radix");
  require_bn(BN_mod(r_mod, r_mod, modulus, bn_ctx), "failed to reduce Montgomery radix");
  require_bn(BN_mod_mul(challenge_mont, challenge, r_mod, modulus, bn_ctx),
             "failed to Montgomery-encode challenge");
  bn_to_limbs(challenge_mont, limb_count, out);

  BN_free(challenge_mont);
  BN_free(r_mod);
  BN_free(challenge);
  BN_free(digest_bn);
  BN_free(modulus);
  BN_CTX_free(bn_ctx);
}

static float median_float(std::vector<float> xs) {
  if (xs.empty()) return 0.0f;
  std::sort(xs.begin(), xs.end());
  size_t m = xs.size() / 2;
  // Same definition as Python's statistics.median: mean of the two middle values for even n.
  return xs.size() % 2 ? xs[m] : 0.5f * (xs[m - 1] + xs[m]);
}

template <int W, int NV, int L, bool LIMB_MAJOR>
static void dispatch_eval_point(unsigned grid, int block, size_t shmem_bytes,
                                const uint32_t* tables, uint32_t* partials, size_t half_n,
                                int point_idx, int point_mode, const uint32_t* d_mod,
                                const uint32_t* d_points, uint32_t nprime) {
#define LAUNCH_P(MODE, IDX)                                                                  \
      eval_kernel<W, NV, L, LIMB_MAJOR, MODE, IDX><<<grid, block, shmem_bytes>>>(            \
          tables, partials, half_n, d_mod, d_points, nprime);                                \
      break;
  if (point_mode == 1) {
    switch (point_idx) {
      case 0: LAUNCH_P(1, 0)
      case 1: LAUNCH_P(1, 1)
      case 2: LAUNCH_P(1, 2)
      case 3: LAUNCH_P(1, 3)
      case 4: LAUNCH_P(1, 4)
      case 5: LAUNCH_P(1, 5)
      case 6: LAUNCH_P(1, 6)
      case 7: LAUNCH_P(1, 7)
      default:
        std::cerr << "point_idx out of range: " << point_idx << "\n";
        std::exit(1);
    }
  } else {
    switch (point_idx) {
      case 0: LAUNCH_P(0, 0)
      case 1: LAUNCH_P(0, 1)
      case 2: LAUNCH_P(0, 2)
      case 3: LAUNCH_P(0, 3)
      case 4: LAUNCH_P(0, 4)
      case 5: LAUNCH_P(0, 5)
      case 6: LAUNCH_P(0, 6)
      case 7: LAUNCH_P(0, 7)
      default:
        std::cerr << "point_idx out of range: " << point_idx << "\n";
        std::exit(1);
    }
  }
#undef LAUNCH_P
}

template <int L, bool LIMB_MAJOR>
static void launch_eval(const Workload& w, unsigned grid, int block, size_t shmem_bytes,
                        const uint32_t* tables, uint32_t* partials, size_t half_n, int point_idx,
                        int point_mode, const uint32_t* d_mod, const uint32_t* d_points,
                        uint32_t nprime) {
  switch (w.id) {
#define LAUNCH_EVAL(W, NV)                                                                  \
    case W:                                                                                  \
      dispatch_eval_point<W, NV, L, LIMB_MAJOR>(grid, block, shmem_bytes, tables, partials,  \
                                                 half_n, point_idx, point_mode, d_mod,       \
                                                 d_points, nprime);                          \
      break;
    WORKLOAD_LIST(LAUNCH_EVAL)
#undef LAUNCH_EVAL
    default:
      std::cerr << "unknown workload id at launch: " << w.id << "\n";
      std::exit(1);
  }
}

template <int L, bool LIMB_MAJOR>
static void launch_fold(const Workload& w, unsigned grid, int block, const uint32_t* tables,
                        uint32_t* out, size_t half_n, const uint32_t* d_mod,
                        const uint32_t* r_mont, uint32_t nprime) {
  switch (w.id) {
#define LAUNCH_FOLD(W, NV)                                                                  \
    case W:                                                                                  \
      fold_kernel<NV, L, LIMB_MAJOR><<<grid, block>>>(                                       \
          tables, out, half_n, d_mod, r_mont, nprime);                                       \
      break;
    WORKLOAD_LIST(LAUNCH_FOLD)
#undef LAUNCH_FOLD
    default:
      std::cerr << "unknown workload id at launch: " << w.id << "\n";
      std::exit(1);
  }
}

template <int L>
static uint32_t* reduce_partials(uint32_t* partials, uint32_t* partials_scratch, size_t n,
                                 int block, const uint32_t* d_mod) {
  uint32_t* src = partials;
  uint32_t* dst = partials_scratch;
  while (n > 1) {
    size_t out_n = (n + 1) / 2;
    reduce_pairs_kernel<L><<<static_cast<unsigned>(cdiv(out_n, block)), block>>>(src, dst, n,
                                                                                 d_mod);
    CUDA_CHECK(cudaGetLastError());
    std::swap(src, dst);
    n = out_n;
  }
  return src;
}

template <int L, bool LIMB_MAJOR>
static RunMetrics run_once(const Workload& w, const uint32_t* initial, uint32_t* buf_a,
                           uint32_t* buf_b, uint32_t* partials, uint32_t* partials_scratch,
                           int rounds, int block, const uint32_t* d_mod, const uint32_t* d_points,
                           uint32_t* d_challenge, const FieldHost& field, uint32_t nprime,
                           const std::string& timing_mode,
                           const std::string& challenge_mode, int point_mode) {
  const uint32_t* cur = initial;
  uint32_t* next_a = buf_a;
  uint32_t* next_b = buf_b;
  size_t cur_n = size_t{1} << rounds;
  RunMetrics metrics;

  for (int round = 0; round < rounds; ++round) {
    size_t half = cur_n / 2;
    size_t num_blocks = cdiv(half, block);
    std::vector<uint32_t> coeffs;
    coeffs.reserve(static_cast<size_t>(w.degree + 1) * L);

    double t0 = 0.0;
    if (timing_mode == "round-sum") {
      CUDA_CHECK(cudaDeviceSynchronize());
      t0 = now_ms();
    }

    for (int point = 0; point <= w.degree; ++point) {
      launch_eval<L, LIMB_MAJOR>(w, static_cast<unsigned>(num_blocks), block,
                                 static_cast<size_t>(block) * L * sizeof(uint32_t), cur, partials,
                                 half, point, point_mode, d_mod, d_points, nprime);
      CUDA_CHECK(cudaGetLastError());
      uint32_t* reduced = reduce_partials<L>(partials, partials_scratch, num_blocks, block, d_mod);
      uint32_t host_one[L];
      CUDA_CHECK(cudaMemcpy(host_one, reduced, L * sizeof(uint32_t), cudaMemcpyDeviceToHost));
#pragma unroll
      for (int limb = 0; limb < L; ++limb) {
        metrics.checksum += static_cast<uint64_t>(limb + 1) * host_one[limb];
        coeffs.push_back(host_one[limb]);
      }
    }

    if (timing_mode == "round-sum") {
      CUDA_CHECK(cudaDeviceSynchronize());
      metrics.eval_ms += now_ms() - t0;
    }

    if (round + 1 < rounds) {
      const uint32_t* r_mont = d_points + L;
      if (timing_mode == "round-sum") t0 = now_ms();
      if (challenge_mode == "sha3") {
        std::vector<uint32_t> challenge_limbs;
        sha3_challenge_mont_limbs(round, coeffs, field, L, challenge_limbs);
        CUDA_CHECK(cudaMemcpy(d_challenge, challenge_limbs.data(), L * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        r_mont = d_challenge;
      }
      if (timing_mode == "round-sum") metrics.challenge_ms += now_ms() - t0;

      size_t total = static_cast<size_t>(w.vars) * half;
      if (timing_mode == "round-sum") {
        CUDA_CHECK(cudaDeviceSynchronize());
        t0 = now_ms();
      }
      launch_fold<L, LIMB_MAJOR>(w, static_cast<unsigned>(cdiv(total, block)), block, cur, next_a,
                                 half, d_mod, r_mont, nprime);
      CUDA_CHECK(cudaGetLastError());
      if (timing_mode == "round-sum") {
        CUDA_CHECK(cudaDeviceSynchronize());
        metrics.fold_ms += now_ms() - t0;
      }
      cur = next_a;
      std::swap(next_a, next_b);
      cur_n = half;
    }
  }
  metrics.total_ms = metrics.eval_ms + metrics.challenge_ms + metrics.fold_ms;
  return metrics;
}


struct Args {
  std::vector<int> rounds{8, 10, 12, 14, 16};
  std::vector<int> bit_widths{32, 64, 128, 256};
  std::vector<std::string> workloads;
  int warmups = 1;
  int repeats = 5;
  int seed = 1;
  int block = 128;
  std::string timing_mode = "whole-run";
  std::string challenge_mode = "fixed";
  std::string timer = "host";
  std::string layout = "element";
  std::string point_mode = "generic";
  std::string out = "results/cuda_field_sweep.csv";
  bool zk_only = false;
};

static Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* flag) {
      if (i + 1 >= argc) {
        std::cerr << flag << " requires a value\n";
        std::exit(2);
      }
      return std::string(argv[++i]);
    };
    if (a == "--rounds") {
      args.rounds.clear();
      while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) args.rounds.push_back(std::stoi(argv[++i]));
    } else if (a == "--bit-widths") {
      args.bit_widths.clear();
      while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) args.bit_widths.push_back(std::stoi(argv[++i]));
    } else if (a == "--workloads") {
      args.workloads.clear();
      while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) args.workloads.emplace_back(argv[++i]);
    } else if (a == "--warmups") {
      args.warmups = std::stoi(need("--warmups"));
    } else if (a == "--repeats") {
      args.repeats = std::stoi(need("--repeats"));
    } else if (a == "--seed") {
      args.seed = std::stoi(need("--seed"));
    } else if (a == "--block") {
      args.block = std::stoi(need("--block"));
    } else if (a == "--timing-mode") {
      args.timing_mode = need("--timing-mode");
      if (args.timing_mode != "whole-run" && args.timing_mode != "round-sum") {
        std::cerr << "--timing-mode must be whole-run or round-sum\n";
        std::exit(2);
      }
    } else if (a == "--challenge-mode") {
      args.challenge_mode = need("--challenge-mode");
      if (args.challenge_mode != "fixed" && args.challenge_mode != "sha3") {
        std::cerr << "--challenge-mode must be fixed or sha3\n";
        std::exit(2);
      }
    } else if (a == "--timer") {
      args.timer = need("--timer");
      if (args.timer != "host" && args.timer != "events") {
        std::cerr << "--timer must be host or events\n";
        std::exit(2);
      }
    } else if (a == "--layout") {
      args.layout = need("--layout");
      if (args.layout != "element" && args.layout != "limb") {
        std::cerr << "--layout must be element or limb\n";
        std::exit(2);
      }
    } else if (a == "--point-mode") {
      args.point_mode = need("--point-mode");
      if (args.point_mode != "generic" && args.point_mode != "specialized") {
        std::cerr << "--point-mode must be generic or specialized\n";
        std::exit(2);
      }
    } else if (a == "--zk-only") {
      args.zk_only = true;
    } else if (a == "--out") {
      args.out = need("--out");
    } else {
      std::cerr << "unknown argument: " << a << "\n";
      std::exit(2);
    }
  }
  return args;
}

template <int L, bool LIMB_MAJOR>
static void run_field_cases(const FieldHost& field, const Args& args, std::ofstream& csv,
                            cudaEvent_t start, cudaEvent_t stop) {
  uint32_t* d_mod = nullptr;
  uint32_t* d_points = nullptr;
  uint32_t* d_r2 = nullptr;
  uint32_t* d_challenge = nullptr;
  CUDA_CHECK(cudaMalloc(&d_mod, L * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_points, MAX_POINTS * L * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_r2, L * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_challenge, L * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_mod, field.modulus, L * sizeof(uint32_t), cudaMemcpyHostToDevice));
  std::vector<uint32_t> packed_points(MAX_POINTS * L);
  for (int p = 0; p < MAX_POINTS; ++p) {
    for (int limb = 0; limb < L; ++limb) packed_points[p * L + limb] = field.point_monts[p][limb];
  }
  CUDA_CHECK(cudaMemcpy(d_points, packed_points.data(), packed_points.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_r2, field.r2, L * sizeof(uint32_t), cudaMemcpyHostToDevice));

  const int point_mode_int = (args.point_mode == "specialized") ? 1 : 0;

  for (const Workload& w : WORKLOADS) {
    if (!args.workloads.empty() && !contains(args.workloads, w.name)) continue;
    if (args.workloads.empty() && args.zk_only && w.id < 8) continue;
    for (int rounds : args.rounds) {
      size_t n = size_t{1} << rounds;
      size_t elems = static_cast<size_t>(w.vars) * n;
      size_t words = elems * L;
      size_t max_partials = cdiv(n / 2, args.block) * L;
      double input_mib = static_cast<double>(words * sizeof(uint32_t)) / (1024.0 * 1024.0);

      uint32_t *initial = nullptr, *buf_a = nullptr, *buf_b = nullptr, *partials = nullptr;
      uint32_t* partials_scratch = nullptr;
      CUDA_CHECK(cudaMalloc(&initial, words * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&buf_a, words * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&buf_b, words * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&partials, max_partials * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&partials_scratch, max_partials * sizeof(uint32_t)));

      fill_tables_kernel<L, LIMB_MAJOR><<<static_cast<unsigned>(cdiv(elems, args.block)), args.block>>>(
          initial, w.vars, n, args.seed + rounds, d_mod, d_r2, field.nprime);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      uint64_t checksum = 0;
      // Whole-run timing uses a synchronized host timer by default, the same measurement as the
      // Triton driver (perf_counter around a synchronized run). --timer events keeps the
      // CUDA-event measurement of earlier runs for comparison.
      auto run_whole_once = [&]() -> float {
        if (args.timer == "host" || args.challenge_mode == "sha3") {
          CUDA_CHECK(cudaDeviceSynchronize());
          double t0 = now_ms();
          RunMetrics run = run_once<L, LIMB_MAJOR>(w, initial, buf_a, buf_b, partials, partials_scratch, rounds, args.block,
                                       d_mod, d_points, d_challenge, field, field.nprime,
                                       args.timing_mode, args.challenge_mode, point_mode_int);
          checksum += run.checksum;
          CUDA_CHECK(cudaDeviceSynchronize());
          return static_cast<float>(now_ms() - t0);
        }
        CUDA_CHECK(cudaEventRecord(start));
        RunMetrics run = run_once<L, LIMB_MAJOR>(w, initial, buf_a, buf_b, partials, partials_scratch, rounds, args.block,
                                     d_mod, d_points, d_challenge, field, field.nprime,
                                     args.timing_mode, args.challenge_mode, point_mode_int);
        checksum += run.checksum;
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
      };

      float first_ms = 0.0f;
      if (args.timing_mode == "whole-run") {
        first_ms = run_whole_once();
      } else {
        RunMetrics first = run_once<L, LIMB_MAJOR>(w, initial, buf_a, buf_b, partials, partials_scratch, rounds, args.block,
                                       d_mod, d_points, d_challenge, field, field.nprime,
                                       args.timing_mode, args.challenge_mode, point_mode_int);
        checksum += first.checksum;
        first_ms = static_cast<float>(first.total_ms);
      }

      for (int i = 0; i < args.warmups; ++i) {
        RunMetrics warm = run_once<L, LIMB_MAJOR>(w, initial, buf_a, buf_b, partials, partials_scratch, rounds, args.block,
                                      d_mod, d_points, d_challenge, field, field.nprime,
                                      args.timing_mode, args.challenge_mode, point_mode_int);
        checksum += warm.checksum;
      }
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<float> samples;
      std::vector<float> eval_samples;
      std::vector<float> challenge_samples;
      std::vector<float> fold_samples;
      samples.reserve(args.repeats);
      for (int rep = 0; rep < args.repeats; ++rep) {
        if (args.timing_mode == "whole-run") {
          samples.push_back(run_whole_once());
        } else {
          RunMetrics run = run_once<L, LIMB_MAJOR>(w, initial, buf_a, buf_b, partials, partials_scratch, rounds, args.block,
                                       d_mod, d_points, d_challenge, field, field.nprime,
                                       args.timing_mode, args.challenge_mode, point_mode_int);
          checksum += run.checksum;
          samples.push_back(static_cast<float>(run.total_ms));
          eval_samples.push_back(static_cast<float>(run.eval_ms));
          challenge_samples.push_back(static_cast<float>(run.challenge_ms));
          fold_samples.push_back(static_cast<float>(run.fold_ms));
        }
      }
      float median = median_float(samples);
      float eval_median = median_float(eval_samples);
      float challenge_median = median_float(challenge_samples);
      float fold_median = median_float(fold_samples);

      csv << w.name << ',' << '"' << w.poly << '"'
          << ",cuda-sumcheck-prime-field-montgomery-" << args.layout << "-layout-"
          << args.point_mode << "-points," << field.name
          << ",montgomery," << args.layout << ',' << args.point_mode << ',' << field.bit_width
          << ',' << L << ',' << field.modulus_hex << ',' << rounds << ',' << n << ',' << w.vars
          << ',' << w.degree << ',' << w.terms << ',' << args.warmups << ',' << args.repeats
          << ',' << args.timing_mode << ',' << args.challenge_mode << ',' << first_ms << ','
          << median << ',' << eval_median << ',' << challenge_median << ',' << fold_median << ','
          << median << ',' << std::fixed << std::setprecision(6) << input_mib << std::defaultfloat
          << ',' << checksum << ",ok\n";
      csv.flush();

      std::cout << w.name << " field=" << field.name << " rounds=" << rounds
                << " first_ms=" << first_ms << " median_ms=" << median
                << " checksum=" << checksum << std::endl;

      CUDA_CHECK(cudaFree(initial));
      CUDA_CHECK(cudaFree(buf_a));
      CUDA_CHECK(cudaFree(buf_b));
      CUDA_CHECK(cudaFree(partials));
      CUDA_CHECK(cudaFree(partials_scratch));
    }
  }

  CUDA_CHECK(cudaFree(d_mod));
  CUDA_CHECK(cudaFree(d_points));
  CUDA_CHECK(cudaFree(d_r2));
  CUDA_CHECK(cudaFree(d_challenge));
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  if (args.block <= 0 || (args.block & (args.block - 1)) != 0) {
    std::cerr << "--block must be a positive power of two\n";
    return 2;
  }

  std::filesystem::path out_path(args.out);
  if (out_path.has_parent_path()) std::filesystem::create_directories(out_path.parent_path());
  std::ofstream csv(args.out);
  if (!csv) {
    std::cerr << "failed to open output: " << args.out << "\n";
    return 1;
  }
  csv << "workload,polynomial,backend,field,arithmetic,layout,point_mode,bit_width,limbs,modulus_hex,rounds,N,vars,degree,terms,warmups,repeats,timing_mode,challenge_mode,first_ms,median_ms,eval_median_ms,challenge_median_ms,fold_median_ms,total_median_ms,input_mib,checksum,status\n";

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (const FieldHost& field : FIELDS) {
    if (!contains_int(args.bit_widths, field.bit_width)) continue;
    bool limb_major = args.layout == "limb";
    if (field.bit_width == 32) {
      if (limb_major) run_field_cases<1, true>(field, args, csv, start, stop);
      else run_field_cases<1, false>(field, args, csv, start, stop);
    }
    if (field.bit_width == 64) {
      if (limb_major) run_field_cases<2, true>(field, args, csv, start, stop);
      else run_field_cases<2, false>(field, args, csv, start, stop);
    }
    if (field.bit_width == 128) {
      if (limb_major) run_field_cases<4, true>(field, args, csv, start, stop);
      else run_field_cases<4, false>(field, args, csv, start, stop);
    }
    if (field.bit_width == 256) {
      if (limb_major) run_field_cases<8, true>(field, args, csv, start, stop);
      else run_field_cases<8, false>(field, args, csv, start, stop);
    }
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}
