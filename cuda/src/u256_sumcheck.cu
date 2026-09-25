#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
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

static constexpr int LIMBS = 8;
static constexpr uint32_t NPRIME = 0xffffffffu;

__device__ __constant__ uint32_t C_MOD[LIMBS] = {
    0x00000001u, 0xffffffffu, 0xfffe5bfeu, 0x53bda402u,
    0x09a1d805u, 0x3339d808u, 0x299d7d48u, 0x73eda753u,
};

__device__ __constant__ uint32_t C_MONT_POINTS[6][LIMBS] = {
    {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
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
};

struct U256 {
  uint32_t v[LIMBS];
};

__host__ __device__ inline uint32_t initial_limb0_value(int var, size_t off, int seed) {
  uint64_t x = static_cast<uint64_t>(static_cast<uint32_t>(seed));
  x += 7919ull * static_cast<uint64_t>(var + 1);
  x += 104729ull * static_cast<uint64_t>(off + 1);
  return static_cast<uint32_t>(x);
}

__device__ __forceinline__ U256 zero_u256() {
  U256 z{};
#pragma unroll
  for (int i = 0; i < LIMBS; ++i) z.v[i] = 0;
  return z;
}

__device__ __forceinline__ U256 load_u256(const uint32_t* ptr, size_t idx) {
  U256 x;
#pragma unroll
  for (int i = 0; i < LIMBS; ++i) x.v[i] = ptr[idx * LIMBS + i];
  return x;
}

__device__ __forceinline__ void store_u256(uint32_t* ptr, size_t idx, const U256& x) {
#pragma unroll
  for (int i = 0; i < LIMBS; ++i) ptr[idx * LIMBS + i] = x.v[i];
}

__device__ __forceinline__ U256 point_u256(int point_idx) {
  U256 x;
#pragma unroll
  for (int i = 0; i < LIMBS; ++i) x.v[i] = C_MONT_POINTS[point_idx][i];
  return x;
}

__device__ __forceinline__ int cmp_mod(const U256& a) {
#pragma unroll
  for (int i = LIMBS - 1; i >= 0; --i) {
    uint32_t ai = a.v[i];
    uint32_t mi = C_MOD[i];
    if (ai > mi) return 1;
    if (ai < mi) return -1;
  }
  return 0;
}

__device__ __forceinline__ U256 sub_raw(const U256& a, const U256& b, uint32_t* borrow_out) {
  U256 out;
  uint64_t borrow = 0;
#pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    uint64_t bi = static_cast<uint64_t>(b.v[i]) + borrow;
    uint64_t ai = static_cast<uint64_t>(a.v[i]);
    out.v[i] = static_cast<uint32_t>(ai - bi);
    borrow = ai < bi;
  }
  *borrow_out = static_cast<uint32_t>(borrow);
  return out;
}

__device__ __forceinline__ U256 add_raw(const U256& a, const U256& b, uint32_t* carry_out) {
  U256 out;
  uint64_t carry = 0;
#pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    uint64_t uv = static_cast<uint64_t>(a.v[i]) + b.v[i] + carry;
    out.v[i] = static_cast<uint32_t>(uv);
    carry = uv >> 32;
  }
  *carry_out = static_cast<uint32_t>(carry);
  return out;
}

__device__ __forceinline__ U256 add_mod(const U256& a, const U256& b) {
  uint32_t carry;
  U256 s = add_raw(a, b, &carry);
  if (carry || cmp_mod(s) >= 0) {
    U256 m;
#pragma unroll
    for (int i = 0; i < LIMBS; ++i) m.v[i] = C_MOD[i];
    uint32_t borrow;
    return sub_raw(s, m, &borrow);
  }
  return s;
}

__device__ __forceinline__ U256 sub_mod(const U256& a, const U256& b) {
  uint32_t borrow;
  U256 d = sub_raw(a, b, &borrow);
  if (!borrow) return d;
  U256 m;
#pragma unroll
  for (int i = 0; i < LIMBS; ++i) m.v[i] = C_MOD[i];
  uint32_t carry;
  return add_raw(d, m, &carry);
}

__device__ __forceinline__ U256 mont_mul(const U256& a, const U256& b) {
  uint32_t t[LIMBS + 1];
#pragma unroll
  for (int i = 0; i <= LIMBS; ++i) t[i] = 0;

#pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    uint64_t carry = 0;
#pragma unroll
    for (int j = 0; j < LIMBS; ++j) {
      uint64_t uv = static_cast<uint64_t>(t[j]) +
                    static_cast<uint64_t>(a.v[j]) * b.v[i] + carry;
      t[j] = static_cast<uint32_t>(uv);
      carry = uv >> 32;
    }
    uint64_t uvn = static_cast<uint64_t>(t[LIMBS]) + carry;
    t[LIMBS] = static_cast<uint32_t>(uvn);

    uint32_t q = t[0] * NPRIME;
    carry = 0;
#pragma unroll
    for (int j = 0; j < LIMBS; ++j) {
      uint64_t uv = static_cast<uint64_t>(t[j]) +
                    static_cast<uint64_t>(q) * C_MOD[j] + carry;
      uint32_t low = static_cast<uint32_t>(uv);
      carry = uv >> 32;
      if (j > 0) t[j - 1] = low;
    }
    uint64_t last = static_cast<uint64_t>(t[LIMBS]) + carry;
    t[LIMBS - 1] = static_cast<uint32_t>(last);
    t[LIMBS] = static_cast<uint32_t>(last >> 32);
  }

  U256 res;
#pragma unroll
  for (int i = 0; i < LIMBS; ++i) res.v[i] = t[i];
  if (t[LIMBS] || cmp_mod(res) >= 0) {
    U256 m;
#pragma unroll
    for (int i = 0; i < LIMBS; ++i) m.v[i] = C_MOD[i];
    uint32_t borrow;
    res = sub_raw(res, m, &borrow);
  }
  return res;
}

__device__ __forceinline__ U256 point_one_var(const U256& e, const U256& d, int point_idx) {
  U256 p = point_u256(point_idx);
  return add_mod(e, mont_mul(d, p));
}

__device__ __forceinline__ U256 eval_workload(int workload, const U256 p[6]) {
  if (workload == 1) return p[0];
  if (workload == 2) return mont_mul(p[0], p[1]);
  if (workload == 3) return add_mod(mont_mul(p[0], p[1]), p[2]);
  if (workload == 4) return mont_mul(mont_mul(p[0], p[1]), p[2]);
  if (workload == 5) {
    U256 aa = mont_mul(p[0], p[0]);
    U256 bb = mont_mul(p[1], p[1]);
    return mont_mul(mont_mul(aa, bb), p[2]);
  }
  if (workload == 6) {
    U256 abc = mont_mul(mont_mul(p[0], p[1]), p[2]);
    U256 de = mont_mul(p[3], p[4]);
    return add_mod(abc, de);
  }
  U256 abcg = mont_mul(mont_mul(mont_mul(p[0], p[1]), p[2]), p[3]);
  U256 deg = mont_mul(mont_mul(p[4], p[5]), p[3]);
  return add_mod(abcg, deg);
}

__global__ void fill_tables_kernel(uint32_t* tables, int num_vars, size_t n, int seed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = static_cast<size_t>(num_vars) * n;
  if (idx >= total) return;
  int var = static_cast<int>(idx / n);
  size_t off = idx - static_cast<size_t>(var) * n;
  uint32_t value = initial_limb0_value(var, off, seed);
  size_t base = idx * LIMBS;
  tables[base] = value;
#pragma unroll
  for (int i = 1; i < LIMBS; ++i) tables[base + i] = 0;
}

__global__ void eval_kernel(const uint32_t* tables, uint32_t* partials, size_t half_n,
                            int workload, int num_vars, int point_idx) {
  extern __shared__ uint32_t sh[];
  int tid = threadIdx.x;
  size_t off = blockIdx.x * blockDim.x + tid;
  size_t row_stride = half_n * 2;

  U256 p[6];
#pragma unroll
  for (int i = 0; i < 6; ++i) p[i] = zero_u256();

  if (off < half_n) {
    for (int var = 0; var < num_vars; ++var) {
      const uint32_t* base = tables + static_cast<size_t>(var) * row_stride * LIMBS;
      U256 e = load_u256(base, 2 * off);
      U256 o = load_u256(base, 2 * off + 1);
      U256 d = sub_mod(o, e);
      p[var] = point_one_var(e, d, point_idx);
    }
    p[0] = eval_workload(workload, p);
  } else {
    p[0] = zero_u256();
  }

#pragma unroll
  for (int limb = 0; limb < LIMBS; ++limb) sh[tid * LIMBS + limb] = p[0].v[limb];
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      U256 a, b;
#pragma unroll
      for (int limb = 0; limb < LIMBS; ++limb) {
        a.v[limb] = sh[tid * LIMBS + limb];
        b.v[limb] = sh[(tid + stride) * LIMBS + limb];
      }
      U256 s = add_mod(a, b);
#pragma unroll
      for (int limb = 0; limb < LIMBS; ++limb) sh[tid * LIMBS + limb] = s.v[limb];
    }
    __syncthreads();
  }

  if (tid == 0) {
#pragma unroll
    for (int limb = 0; limb < LIMBS; ++limb) partials[blockIdx.x * LIMBS + limb] = sh[limb];
  }
}

__global__ void reduce_pairs_kernel(uint32_t* partials, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t out_n = (n + 1) / 2;
  if (idx >= out_n) return;
  size_t lhs = 2 * idx;
  size_t rhs = lhs + 1;
  U256 a = load_u256(partials, lhs);
  U256 out = a;
  if (rhs < n) {
    U256 b = load_u256(partials, rhs);
    out = add_mod(a, b);
  }
  store_u256(partials, idx, out);
}

__global__ void fold_kernel(const uint32_t* tables, uint32_t* out, size_t half_n, int num_vars) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = static_cast<size_t>(num_vars) * half_n;
  if (idx >= total) return;
  int var = static_cast<int>(idx / half_n);
  size_t off = idx - static_cast<size_t>(var) * half_n;
  size_t row_stride = half_n * 2;
  const uint32_t* base = tables + static_cast<size_t>(var) * row_stride * LIMBS;
  U256 e = load_u256(base, 2 * off);
  U256 o = load_u256(base, 2 * off + 1);
  U256 d = sub_mod(o, e);
  U256 r = point_u256(1);
  U256 f = add_mod(e, mont_mul(d, r));
  store_u256(out + static_cast<size_t>(var) * half_n * LIMBS, off, f);
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
};


static bool contains(const std::vector<std::string>& xs, const char* x) {
  return std::find(xs.begin(), xs.end(), std::string(x)) != xs.end();
}

static size_t cdiv(size_t x, size_t y) { return (x + y - 1) / y; }

static void reduce_partials(uint32_t* partials, size_t n, int block) {
  while (n > 1) {
    size_t out_n = (n + 1) / 2;
    reduce_pairs_kernel<<<static_cast<unsigned>(cdiv(out_n, block)), block>>>(partials, n);
    CUDA_CHECK(cudaGetLastError());
    n = out_n;
  }
}

static uint64_t run_once(const Workload& w, uint32_t* initial, uint32_t* buf_a, uint32_t* buf_b,
                         uint32_t* partials, int rounds, int block) {
  const uint32_t* cur = initial;
  uint32_t* next_a = buf_a;
  uint32_t* next_b = buf_b;
  size_t cur_n = size_t{1} << rounds;
  uint64_t checksum = 0;

  for (int round = 0; round < rounds; ++round) {
    size_t half = cur_n / 2;
    size_t num_blocks = cdiv(half, block);
    for (int point = 0; point <= w.degree; ++point) {
      eval_kernel<<<static_cast<unsigned>(num_blocks), block, block * LIMBS * sizeof(uint32_t)>>>(
          cur, partials, half, w.id, w.vars, point);
      CUDA_CHECK(cudaGetLastError());
      reduce_partials(partials, num_blocks, block);
      uint32_t host_one[LIMBS];
      CUDA_CHECK(cudaMemcpy(host_one, partials, LIMBS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      checksum += host_one[0];
    }

    if (round + 1 < rounds) {
      size_t total = static_cast<size_t>(w.vars) * half;
      fold_kernel<<<static_cast<unsigned>(cdiv(total, block)), block>>>(cur, next_a, half, w.vars);
      CUDA_CHECK(cudaGetLastError());
      cur = next_a;
      std::swap(next_a, next_b);
      cur_n = half;
    }
  }
  return checksum;
}

struct Args {
  std::vector<int> rounds{8, 10, 12, 14, 16};
  std::vector<std::string> workloads;
  int warmups = 1;
  int repeats = 5;
  int seed = 1;
  int block = 128;
  std::string out = "results/cuda_poly_ladder.csv";
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
      while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        args.rounds.push_back(std::stoi(argv[++i]));
      }
    } else if (a == "--workloads") {
      args.workloads.clear();
      while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        args.workloads.emplace_back(argv[++i]);
      }
    } else if (a == "--warmups") {
      args.warmups = std::stoi(need("--warmups"));
    } else if (a == "--repeats") {
      args.repeats = std::stoi(need("--repeats"));
    } else if (a == "--seed") {
      args.seed = std::stoi(need("--seed"));
    } else if (a == "--block") {
      args.block = std::stoi(need("--block"));
    } else if (a == "--out") {
      args.out = need("--out");
    } else {
      std::cerr << "unknown argument: " << a << "\n";
      std::exit(2);
    }
  }
  return args;
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  if (args.block <= 0 || (args.block & (args.block - 1)) != 0) {
    std::cerr << "--block must be a positive power of two\n";
    return 2;
  }

  std::filesystem::path out_path(args.out);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }

  std::ofstream csv(args.out);
  if (!csv) {
    std::cerr << "failed to open output: " << args.out << "\n";
    return 1;
  }
  csv << "workload,polynomial,backend,bit_width,rounds,N,vars,degree,terms,warmups,repeats,"
         "median_ms,checksum,status\n";

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (const Workload& w : WORKLOADS) {
    if (!args.workloads.empty() && !contains(args.workloads, w.name)) continue;
    for (int rounds : args.rounds) {
      size_t n = size_t{1} << rounds;
      size_t elems = static_cast<size_t>(w.vars) * n;
      size_t words = elems * LIMBS;
      size_t max_partials = cdiv(n / 2, args.block) * LIMBS;

      uint32_t *initial = nullptr, *buf_a = nullptr, *buf_b = nullptr, *partials = nullptr;
      CUDA_CHECK(cudaMalloc(&initial, words * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&buf_a, words * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&buf_b, words * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&partials, max_partials * sizeof(uint32_t)));

      size_t fill_threads = static_cast<size_t>(w.vars) * n;
      fill_tables_kernel<<<static_cast<unsigned>(cdiv(fill_threads, args.block)), args.block>>>(
          initial, w.vars, n, args.seed + rounds);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      uint64_t checksum = 0;
      std::string status = "ok";

      for (int i = 0; i < args.warmups; ++i) {
        checksum += run_once(w, initial, buf_a, buf_b, partials, rounds, args.block);
      }
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<float> samples;
      samples.reserve(args.repeats);
      for (int rep = 0; rep < args.repeats; ++rep) {
        CUDA_CHECK(cudaEventRecord(start));
        checksum += run_once(w, initial, buf_a, buf_b, partials, rounds, args.block);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
      }
      std::sort(samples.begin(), samples.end());
      float median = samples[samples.size() / 2];

      csv << w.name << ',' << '"' << w.poly << '"' << ",cuda-u256-static-eval,256," << rounds
          << ',' << n << ',' << w.vars << ',' << w.degree << ',' << w.terms << ','
          << args.warmups << ',' << args.repeats << ',' << median << ',' << checksum << ','
          << status << "\n";
      csv.flush();

      std::cout << w.name << " rounds=" << rounds << " N=" << n << " median_ms=" << median
                << " checksum=" << checksum << std::endl;

      CUDA_CHECK(cudaFree(initial));
      CUDA_CHECK(cudaFree(buf_a));
      CUDA_CHECK(cudaFree(buf_b));
      CUDA_CHECK(cudaFree(partials));
    }
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}
