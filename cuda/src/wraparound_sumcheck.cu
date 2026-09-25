#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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

__host__ __device__ inline uint32_t initial_limb0_value(int var, size_t off, int seed) {
  uint64_t x = static_cast<uint64_t>(static_cast<uint32_t>(seed));
  x += 7919ull * static_cast<uint64_t>(var + 1);
  x += 104729ull * static_cast<uint64_t>(off + 1);
  return static_cast<uint32_t>(x);
}

template <int L>
struct UInt {
  uint32_t v[L];
};

template <int L>
__device__ __forceinline__ UInt<L> zero_uint() {
  UInt<L> z{};
#pragma unroll
  for (int i = 0; i < L; ++i) z.v[i] = 0;
  return z;
}

template <int L>
__device__ __forceinline__ UInt<L> load_uint(const uint32_t* ptr, size_t idx) {
  UInt<L> x;
#pragma unroll
  for (int i = 0; i < L; ++i) x.v[i] = ptr[idx * L + i];
  return x;
}

template <int L>
__device__ __forceinline__ void store_uint(uint32_t* ptr, size_t idx, const UInt<L>& x) {
#pragma unroll
  for (int i = 0; i < L; ++i) ptr[idx * L + i] = x.v[i];
}

template <int L>
__device__ __forceinline__ UInt<L> add_wrap(const UInt<L>& a, const UInt<L>& b) {
  UInt<L> out;
  uint64_t carry = 0;
#pragma unroll
  for (int i = 0; i < L; ++i) {
    uint64_t uv = static_cast<uint64_t>(a.v[i]) + b.v[i] + carry;
    out.v[i] = static_cast<uint32_t>(uv);
    carry = uv >> 32;
  }
  return out;
}

template <int L>
__device__ __forceinline__ UInt<L> sub_wrap(const UInt<L>& a, const UInt<L>& b) {
  UInt<L> out;
  uint64_t borrow = 0;
#pragma unroll
  for (int i = 0; i < L; ++i) {
    uint64_t bi = static_cast<uint64_t>(b.v[i]) + borrow;
    uint64_t ai = static_cast<uint64_t>(a.v[i]);
    out.v[i] = static_cast<uint32_t>(ai - bi);
    borrow = ai < bi;
  }
  return out;
}

template <int L>
__device__ __forceinline__ UInt<L> mul_low(const UInt<L>& a, const UInt<L>& b) {
  UInt<L> out;
  uint64_t carry = 0;
#pragma unroll
  for (int k = 0; k < L; ++k) {
    uint64_t acc = carry;
#pragma unroll
    for (int i = 0; i < L; ++i) {
      if (i <= k) acc += static_cast<uint64_t>(a.v[i]) * b.v[k - i];
    }
    out.v[k] = static_cast<uint32_t>(acc);
    carry = acc >> 32;
  }
  return out;
}

template <int L>
__device__ __forceinline__ UInt<L> mul_small(const UInt<L>& a, uint32_t scalar) {
  UInt<L> out;
  uint64_t carry = 0;
#pragma unroll
  for (int i = 0; i < L; ++i) {
    uint64_t uv = static_cast<uint64_t>(a.v[i]) * scalar + carry;
    out.v[i] = static_cast<uint32_t>(uv);
    carry = uv >> 32;
  }
  return out;
}

template <int L>
__device__ __forceinline__ UInt<L> point_one_var(const UInt<L>& e, const UInt<L>& d, uint32_t point) {
  return add_wrap(e, mul_small(d, point));
}

template <int L>
__device__ __forceinline__ UInt<L> eval_workload(int workload, const UInt<L> p[6]) {
  if (workload == 1) return p[0];
  if (workload == 2) return mul_low(p[0], p[1]);
  if (workload == 3) return add_wrap(mul_low(p[0], p[1]), p[2]);
  if (workload == 4) return mul_low(mul_low(p[0], p[1]), p[2]);
  if (workload == 5) {
    UInt<L> aa = mul_low(p[0], p[0]);
    UInt<L> bb = mul_low(p[1], p[1]);
    return mul_low(mul_low(aa, bb), p[2]);
  }
  if (workload == 6) {
    UInt<L> abc = mul_low(mul_low(p[0], p[1]), p[2]);
    UInt<L> de = mul_low(p[3], p[4]);
    return add_wrap(abc, de);
  }
  UInt<L> abcg = mul_low(mul_low(mul_low(p[0], p[1]), p[2]), p[3]);
  UInt<L> deg = mul_low(mul_low(p[4], p[5]), p[3]);
  return add_wrap(abcg, deg);
}

template <int L>
__global__ void fill_tables_kernel(uint32_t* tables, int num_vars, size_t n, int seed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = static_cast<size_t>(num_vars) * n;
  if (idx >= total) return;
  int var = static_cast<int>(idx / n);
  size_t off = idx - static_cast<size_t>(var) * n;
  size_t base = idx * L;
  tables[base] = initial_limb0_value(var, off, seed);
#pragma unroll
  for (int i = 1; i < L; ++i) tables[base + i] = 0;
}

template <int L>
__global__ void eval_kernel(const uint32_t* tables, uint32_t* partials, size_t half_n,
                            int workload, int num_vars, uint32_t point) {
  extern __shared__ uint32_t sh[];
  int tid = threadIdx.x;
  size_t off = blockIdx.x * blockDim.x + tid;
  size_t row_stride = half_n * 2;

  UInt<L> p[6];
#pragma unroll
  for (int i = 0; i < 6; ++i) p[i] = zero_uint<L>();

  if (off < half_n) {
    for (int var = 0; var < num_vars; ++var) {
      const uint32_t* base = tables + static_cast<size_t>(var) * row_stride * L;
      UInt<L> e = load_uint<L>(base, 2 * off);
      UInt<L> o = load_uint<L>(base, 2 * off + 1);
      UInt<L> d = sub_wrap(o, e);
      p[var] = point_one_var(e, d, point);
    }
    p[0] = eval_workload(workload, p);
  } else {
    p[0] = zero_uint<L>();
  }

#pragma unroll
  for (int limb = 0; limb < L; ++limb) sh[tid * L + limb] = p[0].v[limb];
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      UInt<L> a, b;
#pragma unroll
      for (int limb = 0; limb < L; ++limb) {
        a.v[limb] = sh[tid * L + limb];
        b.v[limb] = sh[(tid + stride) * L + limb];
      }
      UInt<L> s = add_wrap(a, b);
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
__global__ void reduce_pairs_kernel(uint32_t* partials, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t out_n = (n + 1) / 2;
  if (idx >= out_n) return;
  size_t lhs = 2 * idx;
  size_t rhs = lhs + 1;
  UInt<L> a = load_uint<L>(partials, lhs);
  UInt<L> out = a;
  if (rhs < n) {
    UInt<L> b = load_uint<L>(partials, rhs);
    out = add_wrap(a, b);
  }
  store_uint<L>(partials, idx, out);
}

template <int L>
__global__ void fold_kernel(const uint32_t* tables, uint32_t* out, size_t half_n, int num_vars,
                            uint32_t challenge) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = static_cast<size_t>(num_vars) * half_n;
  if (idx >= total) return;
  int var = static_cast<int>(idx / half_n);
  size_t off = idx - static_cast<size_t>(var) * half_n;
  size_t row_stride = half_n * 2;
  const uint32_t* base = tables + static_cast<size_t>(var) * row_stride * L;
  UInt<L> e = load_uint<L>(base, 2 * off);
  UInt<L> o = load_uint<L>(base, 2 * off + 1);
  UInt<L> d = sub_wrap(o, e);
  UInt<L> f = add_wrap(e, mul_small(d, challenge));
  store_uint<L>(out + static_cast<size_t>(var) * half_n * L, off, f);
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

template <int L>
static void reduce_partials(uint32_t* partials, size_t n, int block) {
  while (n > 1) {
    size_t out_n = (n + 1) / 2;
    reduce_pairs_kernel<L><<<static_cast<unsigned>(cdiv(out_n, block)), block>>>(partials, n);
    CUDA_CHECK(cudaGetLastError());
    n = out_n;
  }
}

template <int L>
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
      eval_kernel<L><<<static_cast<unsigned>(num_blocks), block, block * L * sizeof(uint32_t)>>>(
          cur, partials, half, w.id, w.vars, static_cast<uint32_t>(point));
      CUDA_CHECK(cudaGetLastError());
      reduce_partials<L>(partials, num_blocks, block);
      uint32_t host_one[L];
      CUDA_CHECK(cudaMemcpy(host_one, partials, L * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      for (int limb = 0; limb < L; ++limb) {
        checksum += static_cast<uint64_t>(limb + 1) * host_one[limb];
      }
    }

    if (round + 1 < rounds) {
      size_t total = static_cast<size_t>(w.vars) * half;
      fold_kernel<L><<<static_cast<unsigned>(cdiv(total, block)), block>>>(cur, next_a, half, w.vars, 1);
      CUDA_CHECK(cudaGetLastError());
      cur = next_a;
      std::swap(next_a, next_b);
      cur_n = half;
    }
  }
  return checksum;
}

struct Args {
  std::vector<int> bit_widths{32, 64, 128, 256};
  std::vector<int> rounds{8, 10, 12, 14, 16};
  std::vector<std::string> workloads;
  int warmups = 1;
  int repeats = 5;
  int seed = 1;
  int block = 128;
  std::string out = "results/cuda_wraparound_sweep.csv";
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
    if (a == "--bit-widths") {
      args.bit_widths.clear();
      while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        args.bit_widths.push_back(std::stoi(argv[++i]));
      }
    } else if (a == "--rounds") {
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

template <int L>
static void run_case(const Workload& w, int bit_width, int rounds, const Args& args, std::ofstream& csv) {
  size_t n = size_t{1} << rounds;
  size_t elems = static_cast<size_t>(w.vars) * n;
  size_t words = elems * L;
  size_t max_partials = cdiv(n / 2, args.block) * L;

  uint32_t *initial = nullptr, *buf_a = nullptr, *buf_b = nullptr, *partials = nullptr;
  CUDA_CHECK(cudaMalloc(&initial, words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&buf_a, words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&buf_b, words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&partials, max_partials * sizeof(uint32_t)));

  size_t fill_threads = static_cast<size_t>(w.vars) * n;
  fill_tables_kernel<L><<<static_cast<unsigned>(cdiv(fill_threads, args.block)), args.block>>>(
      initial, w.vars, n, args.seed + rounds);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  uint64_t checksum = 0;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  checksum += run_once<L>(w, initial, buf_a, buf_b, partials, rounds, args.block);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float first_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&first_ms, start, stop));

  for (int i = 0; i < args.warmups; ++i) {
    checksum += run_once<L>(w, initial, buf_a, buf_b, partials, rounds, args.block);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> samples;
  samples.reserve(args.repeats);
  for (int rep = 0; rep < args.repeats; ++rep) {
    CUDA_CHECK(cudaEventRecord(start));
    checksum += run_once<L>(w, initial, buf_a, buf_b, partials, rounds, args.block);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(ms);
  }
  std::sort(samples.begin(), samples.end());
  float median = samples[samples.size() / 2];
  double input_mib = static_cast<double>(w.vars) * static_cast<double>(n) * L * 4.0 / (1024.0 * 1024.0);

  csv << w.name << ',' << '"' << w.poly << '"' << ",cuda-sumcheck-bitwidth-wrap,wrap," << bit_width
      << ',' << L << ',' << rounds << ',' << n << ',' << w.vars << ',' << w.degree << ','
      << w.terms << ',' << args.warmups << ',' << args.repeats << ',' << first_ms << ','
      << median << ',' << input_mib << ',' << checksum << ",ok\n";
  csv.flush();

  std::cout << w.name << " bit_width=" << bit_width << " rounds=" << rounds << " N=" << n
            << " first_ms=" << first_ms << " median_ms=" << median << " checksum=" << checksum
            << std::endl;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(initial));
  CUDA_CHECK(cudaFree(buf_a));
  CUDA_CHECK(cudaFree(buf_b));
  CUDA_CHECK(cudaFree(partials));
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
  csv << "workload,polynomial,backend,arithmetic,bit_width,limbs,rounds,N,vars,degree,terms,"
         "warmups,repeats,first_ms,median_ms,input_mib,checksum,status\n";

  for (const Workload& w : WORKLOADS) {
    if (!args.workloads.empty() && !contains(args.workloads, w.name)) continue;
    for (int rounds : args.rounds) {
      for (int bit_width : args.bit_widths) {
        if (bit_width == 32) {
          run_case<1>(w, bit_width, rounds, args, csv);
        } else if (bit_width == 64) {
          run_case<2>(w, bit_width, rounds, args, csv);
        } else if (bit_width == 128) {
          run_case<4>(w, bit_width, rounds, args, csv);
        } else if (bit_width == 256) {
          run_case<8>(w, bit_width, rounds, args, csv);
        } else {
          std::cerr << "unsupported bit width: " << bit_width << "\n";
          return 2;
        }
      }
    }
  }
  return 0;
}
