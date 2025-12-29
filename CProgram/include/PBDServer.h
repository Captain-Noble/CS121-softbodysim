#pragma once
// Single shared header for a small multi-file project.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cmath>
#include <string>
#include <algorithm>
#include <functional>

// ---------------- platform sockets ----------------
#ifdef _WIN32
  #define NOMINMAX
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using socklen_t = int;
#else
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  #include <errno.h>
  using SOCKET = int;
  static constexpr int INVALID_SOCKET = -1;
  static constexpr int SOCKET_ERROR = -1;
#endif

// ---------------- net ----------------
void sockets_init();
void sockets_shutdown();
void sock_close(SOCKET s);

bool recv_exact(SOCKET s, void* dst, size_t n);
bool send_exact(SOCKET s, const void* src, size_t n);
SOCKET listen_and_accept(int port);

// ---------------- protocol ----------------
static constexpr uint32_t MAGIC_PBD1 = 0x31444250u; // 'PBD1' little-endian

enum MsgType : uint32_t {
  MSG_INIT      = 1,
  MSG_STEP      = 2,
  MSG_POSITIONS = 3,
  MSG_SHUTDOWN  = 4
};

#pragma pack(push, 1)
struct MsgHeader {
  uint32_t magic;
  uint32_t type;
  uint32_t size;
};
#pragma pack(pop)

bool read_header(SOCKET c, MsgHeader& h);
bool send_positions(SOCKET c, const std::vector<float>& pos);

// ---------------- perf (评测) ----------------
namespace perf {
  using clock = std::chrono::steady_clock;

  inline double ms_since(clock::time_point a, clock::time_point b) {
    return double(std::chrono::duration_cast<std::chrono::microseconds>(b - a).count()) / 1000.0;
  }

  struct StepStats {
    double predictMs = 0.0;
    double solveMs   = 0.0;
    double commitMs  = 0.0;
    double packMs    = 0.0;
    double totalMs   = 0.0;
  };

  struct Accum {
    uint64_t steps = 0;
    StepStats sum{};

    void add(const StepStats& s) {
      steps++;
      sum.predictMs += s.predictMs;
      sum.solveMs   += s.solveMs;
      sum.commitMs  += s.commitMs;
      sum.packMs    += s.packMs;
      sum.totalMs   += s.totalMs;
    }

    StepStats avg() const {
      StepStats a{};
      if (steps == 0) return a;
      double inv = 1.0 / double(steps);
      a.predictMs = sum.predictMs * inv;
      a.solveMs   = sum.solveMs   * inv;
      a.commitMs  = sum.commitMs  * inv;
      a.packMs    = sum.packMs    * inv;
      a.totalMs   = sum.totalMs   * inv;
      return a;
    }

    void reset() { steps = 0; sum = StepStats{}; }
  };

  struct ScopedAdd {
    clock::time_point t0;
    double* outMs;
    explicit ScopedAdd(double* dst) : t0(clock::now()), outMs(dst) {}
    ~ScopedAdd() {
      auto t1 = clock::now();
      *outMs += ms_since(t0, t1);
    }
  };
} // namespace perf

// ---------------- math + sim types ----------------
struct Vec3 {
  float x, y, z;
  Vec3() : x(0), y(0), z(0) {}
  Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
inline Vec3 operator*(const Vec3& a, float s)       { return Vec3(a.x*s, a.y*s, a.z*s); }
inline Vec3 operator*(float s, const Vec3& a)       { return Vec3(a.x*s, a.y*s, a.z*s); }

inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline float length(const Vec3& a) { return std::sqrt(dot(a,a)); }

inline float tet_volume(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3) {
  Vec3 a = p1 - p0;
  Vec3 b = p2 - p0;
  Vec3 c = p3 - p0;
  return dot(cross(a,b), c) / 6.0f;
}

struct SolverParams {
  uint32_t substeps = 2;
  uint32_t iterations = 6;
  float dtHint = 1.0f / 60.0f;
  float omega = 1.6f;

  float edgeCompliance = 5e-4f;
  float volumeCompliance = 0.0f;

  Vec3 gravity = Vec3(0.0f, -9.81f, 0.0f);

  uint32_t groundEnabled = 1;
  float groundY = 0.0f;
  float friction = 0.2f;
};

struct PBDState {
  uint32_t V=0, E=0, T=0;

  std::vector<Vec3> x;
  std::vector<Vec3> v;
  std::vector<Vec3> xStar;
  std::vector<float> w; // invMass

  std::vector<uint32_t> edgeI0, edgeI1;
  std::vector<float> edgeRest;
  std::vector<float> edgeLambda;

  std::vector<uint32_t> tetA, tetB, tetC, tetD;
  std::vector<float> tetRestVol;
  std::vector<float> tetLambda;

  SolverParams params;
};

void compute_inv_mass(PBDState& s, const std::vector<uint32_t>& pinned);
void build_rest(PBDState& s);

void solve_edges_xpbd_gs(PBDState& s, float dt);
void solve_tets_xpbd_gs(PBDState& s, float dt);

// ---------------- tiny thread pool (for parallel mode) ----------------
class ThreadPool {
public:
  explicit ThreadPool(unsigned workers);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  template <class Fn>
  void parallel_for(uint32_t begin, uint32_t end, uint32_t chunk, Fn&& fn) {
    if (end <= begin) return;
    if (workerCount_ == 1 || (end - begin) <= chunk) {
      fn(begin, end);
      return;
    }

    WorkCtx ctx;
    ctx.next.store(begin);
    ctx.end = end;
    ctx.chunk = std::max<uint32_t>(1, chunk);
    ctx.fn = [&](uint32_t a, uint32_t b) { fn(a, b); };
    ctx.remaining.store(int(workerCount_ + 1)); // workers + caller

    {
      std::lock_guard<std::mutex> lk(m_);
      current_ = &ctx;
      workId_++;
    }
    cv_.notify_all();

    run_ctx(ctx); // caller helps
    if (ctx.remaining.fetch_sub(1) == 1) {
      std::lock_guard<std::mutex> lk(ctx.doneM);
      ctx.doneCv.notify_one();
    }

    std::unique_lock<std::mutex> lk(ctx.doneM);
    ctx.doneCv.wait(lk, [&]{ return ctx.remaining.load() == 0; });

    {
      std::lock_guard<std::mutex> lk2(m_);
      current_ = nullptr;
    }
  }

private:
  struct WorkCtx {
    std::atomic<uint32_t> next{0};
    uint32_t end = 0;
    uint32_t chunk = 64;
    std::function<void(uint32_t,uint32_t)> fn;

    std::atomic<int> remaining{0};
    std::mutex doneM;
    std::condition_variable doneCv;
  };

  void run_ctx(WorkCtx& ctx);
  void worker_loop();

  std::mutex m_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};

  std::vector<std::thread> threads_;
  unsigned workerCount_ = 1;

  WorkCtx* current_ = nullptr;
  uint64_t workId_ = 0;
};

// ---------------- steppers (serial / parallel) ----------------
struct IStepper {
  virtual ~IStepper() = default;
  virtual const char* name() const = 0;
  virtual void step(PBDState& s, float dt, perf::StepStats& out) = 0;
  virtual void pack_positions(const PBDState& s, std::vector<float>& outPos, double& outPackMs) = 0;
};

struct SerialStepper final : IStepper {
  const char* name() const override { return "serial"; }
  void step(PBDState& s, float dt, perf::StepStats& out) override;
  void pack_positions(const PBDState& s, std::vector<float>& outPos, double& outPackMs) override;
};

struct ParallelStepper final : IStepper {
  ThreadPool pool;
  explicit ParallelStepper(unsigned threads);
  const char* name() const override { return "parallel"; }
  void step(PBDState& s, float dt, perf::StepStats& out) override;
  void pack_positions(const PBDState& s, std::vector<float>& outPos, double& outPackMs) override;
};

// ---------------- shared state + loops ----------------
struct Shared {
  std::mutex m;
  std::condition_variable cvStep;
  std::condition_variable cvOut;

  std::atomic<bool> running{true};

  bool haveInit = false;
  bool stepRequested = false;
  float stepDt = 1.0f/60.0f;

  bool outReady = false;
  std::vector<float> outPositions; // V*3 floats

  PBDState state;
  IStepper* stepper = nullptr;

  perf::Accum acc;
};

void sim_thread_fn(Shared* sh);
void comm_loop(SOCKET client, Shared* sh);
