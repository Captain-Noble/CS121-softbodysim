// PBDServer.cpp
// C++17, TCP server, comm thread + sim thread
// Build:
//   Linux/macOS: g++ -O3 -std=c++17 PBDServer.cpp -pthread -o PBDServer
//   Windows (MSVC): cl /O2 /std:c++17 PBDServer.cpp ws2_32.lib

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

#ifdef _WIN32
  #define NOMINMAX
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
  using socklen_t = int;
  static void sockets_init() {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2,2), &wsa) != 0) { std::fprintf(stderr, "WSAStartup failed\n"); std::exit(1); }
  }
  static void sockets_shutdown() { WSACleanup(); }
  static void sock_close(SOCKET s) { closesocket(s); }
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
  static void sockets_init() {}
  static void sockets_shutdown() {}
  static void sock_close(SOCKET s) { close(s); }
#endif

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

static bool recv_exact(SOCKET s, void* dst, size_t n) {
  uint8_t* p = reinterpret_cast<uint8_t*>(dst);
  size_t got = 0;
  while (got < n) {
#ifdef _WIN32
    int r = recv(s, reinterpret_cast<char*>(p + got), static_cast<int>(n - got), 0);
#else
    ssize_t r = recv(s, p + got, n - got, 0);
#endif
    if (r <= 0) return false;
    got += static_cast<size_t>(r);
  }
  return true;
}

static bool send_exact(SOCKET s, const void* src, size_t n) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
  size_t sent = 0;
  while (sent < n) {
#ifdef _WIN32
    int r = send(s, reinterpret_cast<const char*>(p + sent), static_cast<int>(n - sent), 0);
#else
    ssize_t r = send(s, p + sent, n - sent, 0);
#endif
    if (r <= 0) return false;
    sent += static_cast<size_t>(r);
  }
  return true;
}

struct Vec3 {
  float x, y, z;
  Vec3() : x(0), y(0), z(0) {}
  Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
};

static inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline Vec3 operator*(const Vec3& a, float s) { return Vec3(a.x*s, a.y*s, a.z*s); }
static inline Vec3 operator*(float s, const Vec3& a) { return Vec3(a.x*s, a.y*s, a.z*s); }

static inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static inline float length(const Vec3& a) { return std::sqrt(dot(a,a)); }

static float tet_volume(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3) {
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

  bool initialized = false;
};

static void compute_inv_mass(PBDState& s, const std::vector<uint32_t>& pinned) {
  s.w.assign(s.V, 0.0f);

  for (uint32_t ti = 0; ti < s.T; ++ti) {
    uint32_t a = s.tetA[ti], b = s.tetB[ti], c = s.tetC[ti], d = s.tetD[ti];
    float vol = tet_volume(s.x[a], s.x[b], s.x[c], s.x[d]);
    float mvol = std::fabs(vol);
    if (mvol > 1e-12f) {
      float inv = 4.0f / mvol;
      s.w[a] += inv; s.w[b] += inv; s.w[c] += inv; s.w[d] += inv;
    }
  }

  for (uint32_t idx : pinned) {
    if (idx < s.V) s.w[idx] = 0.0f;
  }
}

static void build_rest(PBDState& s) {
  s.edgeRest.resize(s.E);
  s.edgeLambda.assign(s.E, 0.0f);
  for (uint32_t ei = 0; ei < s.E; ++ei) {
    uint32_t i0 = s.edgeI0[ei], i1 = s.edgeI1[ei];
    s.edgeRest[ei] = length(s.x[i1] - s.x[i0]);
  }

  s.tetRestVol.resize(s.T);
  s.tetLambda.assign(s.T, 0.0f);
  for (uint32_t ti = 0; ti < s.T; ++ti) {
    uint32_t a = s.tetA[ti], b = s.tetB[ti], c = s.tetC[ti], d = s.tetD[ti];
    s.tetRestVol[ti] = tet_volume(s.x[a], s.x[b], s.x[c], s.x[d]);
  }
}

static void predict(PBDState& s, float dt) {
  Vec3 g = s.params.gravity;
  for (uint32_t i = 0; i < s.V; ++i) {
    if (s.w[i] == 0.0f) { s.xStar[i] = s.x[i]; continue; }
    s.v[i] = s.v[i] + g * dt;
    s.xStar[i] = s.x[i] + s.v[i] * dt;
  }
}

static void project_ground(PBDState& s) {
  if (!s.params.groundEnabled) return;
  float y0 = s.params.groundY;
  for (uint32_t i = 0; i < s.V; ++i) {
    if (s.w[i] == 0.0f) continue;
    Vec3 p = s.xStar[i];
    if (p.y < y0) { p.y = y0; s.xStar[i] = p; }
  }
}

static void solve_edges_xpbd_gs(PBDState& s, float dt) {
  float invDt2 = (dt > 1e-12f) ? (1.0f / (dt*dt)) : 0.0f;
  float comp = std::max(0.0f, s.params.edgeCompliance);

  for (uint32_t ei = 0; ei < s.E; ++ei) {
    uint32_t i0 = s.edgeI0[ei], i1 = s.edgeI1[ei];
    float w0 = s.w[i0], w1 = s.w[i1];
    float wSum = w0 + w1;
    if (wSum == 0.0f) continue;

    Vec3 p0 = s.xStar[i0];
    Vec3 p1 = s.xStar[i1];
    Vec3 d = p0 - p1;
    float len = length(d);
    if (len < 1e-12f) continue;

    float C = len - s.edgeRest[ei];
    float alpha = comp * invDt2;

    float lambda = s.edgeLambda[ei];
    float dlambda = (-C - alpha * lambda) / (wSum + alpha);
    lambda += dlambda;
    s.edgeLambda[ei] = lambda;

    Vec3 n = d * (1.0f / len);
    Vec3 corr = n * dlambda;

    s.xStar[i0] = s.xStar[i0] + corr * w0;
    s.xStar[i1] = s.xStar[i1] - corr * w1;
  }
}

static void solve_tets_xpbd_gs(PBDState& s, float dt) {
  float invDt2 = (dt > 1e-12f) ? (1.0f / (dt*dt)) : 0.0f;
  float comp = std::max(0.0f, s.params.volumeCompliance);

  for (uint32_t ti = 0; ti < s.T; ++ti) {
    uint32_t a = s.tetA[ti], b = s.tetB[ti], c = s.tetC[ti], d = s.tetD[ti];
    float wa = s.w[a], wb = s.w[b], wc = s.w[c], wd = s.w[d];
    if (wa + wb + wc + wd == 0.0f) continue;

    Vec3 pa = s.xStar[a];
    Vec3 pb = s.xStar[b];
    Vec3 pc = s.xStar[c];
    Vec3 pd = s.xStar[d];

    Vec3 ga = cross(pd - pb, pc - pb) * (1.0f / 6.0f);
    Vec3 gb = cross(pc - pa, pd - pa) * (1.0f / 6.0f);
    Vec3 gc = cross(pd - pa, pb - pa) * (1.0f / 6.0f);
    Vec3 gd = cross(pb - pa, pc - pa) * (1.0f / 6.0f);

    float wSum =
      wa * dot(ga, ga) +
      wb * dot(gb, gb) +
      wc * dot(gc, gc) +
      wd * dot(gd, gd);

    if (wSum < 1e-20f) continue;

    float vol = tet_volume(pa, pb, pc, pd);
    float C = vol - s.tetRestVol[ti];

    float alpha = comp * invDt2;
    float lambda = s.tetLambda[ti];
    float dlambda = (-C - alpha * lambda) / (wSum + alpha);
    lambda += dlambda;
    s.tetLambda[ti] = lambda;

    s.xStar[a] = s.xStar[a] + ga * (wa * dlambda);
    s.xStar[b] = s.xStar[b] + gb * (wb * dlambda);
    s.xStar[c] = s.xStar[c] + gc * (wc * dlambda);
    s.xStar[d] = s.xStar[d] + gd * (wd * dlambda);
  }
}

static void commit(PBDState& s, float dt) {
  float invDt = (dt > 1e-12f) ? (1.0f / dt) : 0.0f;
  float y0 = s.params.groundY;
  float fr = std::fmax(0.0f, std::fmin(1.0f, s.params.friction));

  for (uint32_t i = 0; i < s.V; ++i) {
    if (s.w[i] == 0.0f) {
      s.v[i] = Vec3(0,0,0);
      s.xStar[i] = s.x[i];
      continue;
    }
    Vec3 newX = s.xStar[i];
    Vec3 oldX = s.x[i];
    Vec3 vel = (newX - oldX) * invDt;

    if (s.params.groundEnabled && newX.y <= y0 + 1e-6f) {
      vel.x *= (1.0f - fr);
      vel.z *= (1.0f - fr);
      if (vel.y < 0.0f) vel.y = 0.0f;
    }

    s.v[i] = vel;
    s.x[i] = newX;
  }
}

static void step_sim(PBDState& s, float dt) {
  uint32_t ss = std::max(1u, s.params.substeps);
  float sdt = dt / float(ss);

  for (uint32_t k = 0; k < ss; ++k) {
    predict(s, sdt);

    for (uint32_t it = 0; it < s.params.iterations; ++it) {
      solve_edges_xpbd_gs(s, sdt);
      solve_tets_xpbd_gs(s, sdt);
      project_ground(s);
    }

    commit(s, sdt);
  }
}

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
};

static void sim_thread_fn(Shared* sh) {
  using clock = std::chrono::steady_clock;

  int frames = 0;
  auto last = clock::now();

  while (sh->running.load()) {
    float dt = 0.0f;

    {
      std::unique_lock<std::mutex> lk(sh->m);
      sh->cvStep.wait(lk, [&]{
        return !sh->running.load() || (sh->haveInit && sh->stepRequested);
      });
      if (!sh->running.load()) break;
      dt = sh->stepDt;
      sh->stepRequested = false;
    }

    step_sim(sh->state, dt);
    frames++;

    // Pack output positions
    if (sh->outPositions.size() != size_t(sh->state.V) * 3) {
      sh->outPositions.resize(size_t(sh->state.V) * 3);
    }
    for (uint32_t i = 0; i < sh->state.V; ++i) {
      sh->outPositions[i*3+0] = sh->state.x[i].x;
      sh->outPositions[i*3+1] = sh->state.x[i].y;
      sh->outPositions[i*3+2] = sh->state.x[i].z;
    }

    {
      std::lock_guard<std::mutex> lk(sh->m);
      sh->outReady = true;
    }
    sh->cvOut.notify_one();

    auto now = clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
    if (ms >= 1000) {
      double fps = double(frames) * 1000.0 / double(ms);
      std::printf("[PBDServer] FPS %.1f | V=%u E=%u T=%u\n", fps, sh->state.V, sh->state.E, sh->state.T);
      frames = 0;
      last = now;
    }
  }
}

static bool read_header(SOCKET c, MsgHeader& h) {
  if (!recv_exact(c, &h, sizeof(MsgHeader))) return false;
  if (h.magic != MAGIC_PBD1) return false;
  return true;
}

static bool send_positions(SOCKET c, const std::vector<float>& pos) {
  MsgHeader h;
  h.magic = MAGIC_PBD1;
  h.type = MSG_POSITIONS;
  h.size = (uint32_t)(pos.size() * sizeof(float));
  if (!send_exact(c, &h, sizeof(h))) return false;
  if (!pos.empty() && !send_exact(c, pos.data(), pos.size() * sizeof(float))) return false;
  return true;
}

static void comm_loop(SOCKET client, Shared* sh) {
  std::vector<uint8_t> payload;

  while (sh->running.load()) {
    MsgHeader h{};
    if (!read_header(client, h)) break;
    payload.resize(h.size);
    if (h.size > 0 && !recv_exact(client, payload.data(), h.size)) break;

    if (h.type == MSG_INIT) {
      const uint8_t* p = payload.data();
      auto rd_u32 = [&](uint32_t& out) {
        std::memcpy(&out, p, 4); p += 4;
      };
      auto rd_f32 = [&](float& out) {
        std::memcpy(&out, p, 4); p += 4;
      };

      uint32_t V,E,T;
      rd_u32(V); rd_u32(E); rd_u32(T);

      SolverParams params;
      rd_u32(params.substeps);
      rd_u32(params.iterations);
      rd_f32(params.dtHint);
      rd_f32(params.omega);
      rd_f32(params.edgeCompliance);
      rd_f32(params.volumeCompliance);
      rd_f32(params.gravity.x);
      rd_f32(params.gravity.y);
      rd_f32(params.gravity.z);
      rd_u32(params.groundEnabled);
      rd_f32(params.groundY);
      rd_f32(params.friction);

      uint32_t pinnedCount = 0;
      rd_u32(pinnedCount);
      std::vector<uint32_t> pinned(pinnedCount);
      if (pinnedCount > 0) {
        std::memcpy(pinned.data(), p, pinnedCount * sizeof(uint32_t));
        p += pinnedCount * sizeof(uint32_t);
      }

      std::vector<float> x0(size_t(V)*3);
      std::memcpy(x0.data(), p, x0.size() * sizeof(float));
      p += x0.size() * sizeof(float);

      std::vector<uint32_t> edgeIds(size_t(E)*2);
      std::memcpy(edgeIds.data(), p, edgeIds.size() * sizeof(uint32_t));
      p += edgeIds.size() * sizeof(uint32_t);

      std::vector<uint32_t> tetIds(size_t(T)*4);
      std::memcpy(tetIds.data(), p, tetIds.size() * sizeof(uint32_t));
      p += tetIds.size() * sizeof(uint32_t);

      PBDState st;
      st.V = V; st.E = E; st.T = T;
      st.params = params;
      st.x.resize(V);
      st.v.assign(V, Vec3(0,0,0));
      st.xStar.resize(V);

      for (uint32_t i=0;i<V;++i) {
        st.x[i] = Vec3(x0[i*3+0], x0[i*3+1], x0[i*3+2]);
        st.xStar[i] = st.x[i];
      }

      st.edgeI0.resize(E);
      st.edgeI1.resize(E);
      for (uint32_t ei=0; ei<E; ++ei) {
        st.edgeI0[ei] = edgeIds[ei*2+0];
        st.edgeI1[ei] = edgeIds[ei*2+1];
      }

      st.tetA.resize(T);
      st.tetB.resize(T);
      st.tetC.resize(T);
      st.tetD.resize(T);
      for (uint32_t ti=0; ti<T; ++ti) {
        st.tetA[ti] = tetIds[ti*4+0];
        st.tetB[ti] = tetIds[ti*4+1];
        st.tetC[ti] = tetIds[ti*4+2];
        st.tetD[ti] = tetIds[ti*4+3];
      }

      compute_inv_mass(st, pinned);
      build_rest(st);

      {
        std::lock_guard<std::mutex> lk(sh->m);
        sh->state = std::move(st);
        sh->haveInit = true;
      }
      sh->cvStep.notify_one();

      std::printf("[PBDServer] Init received. V=%u E=%u T=%u pinned=%u\n", V,E,T,pinnedCount);
    }
    else if (h.type == MSG_STEP) {
      if (h.size < sizeof(float)) break;
      float dt;
      std::memcpy(&dt, payload.data(), sizeof(float));

      {
        std::lock_guard<std::mutex> lk(sh->m);
        if (!sh->haveInit) continue;
        sh->stepDt = dt;
        sh->stepRequested = true;
      }
      sh->cvStep.notify_one();

      // Wait output and send
      std::unique_lock<std::mutex> lk(sh->m);
      sh->cvOut.wait(lk, [&]{ return !sh->running.load() || sh->outReady; });
      if (!sh->running.load()) break;
      sh->outReady = false;
      std::vector<float> toSend = sh->outPositions;
      lk.unlock();

      if (!send_positions(client, toSend)) break;
    }
    else if (h.type == MSG_SHUTDOWN) {
      break;
    }
    else {
      break;
    }
  }

  sh->running.store(false);
  sh->cvStep.notify_all();
  sh->cvOut.notify_all();
}

int main(int argc, char** argv) {
  sockets_init();

  int port = 7777;
  if (argc >= 2) port = std::atoi(argv[1]);

  SOCKET srv = socket(AF_INET, SOCK_STREAM, 0);
  if (srv == INVALID_SOCKET) { std::fprintf(stderr, "socket failed\n"); return 1; }

  int yes = 1;
#ifdef _WIN32
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));
#else
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons((uint16_t)port);

  if (bind(srv, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
    std::fprintf(stderr, "bind failed\n");
    sock_close(srv);
    sockets_shutdown();
    return 1;
  }

  if (listen(srv, 1) == SOCKET_ERROR) {
    std::fprintf(stderr, "listen failed\n");
    sock_close(srv);
    sockets_shutdown();
    return 1;
  }

  std::printf("[PBDServer] Listening on port %d...\n", port);

  sockaddr_in caddr{};
  socklen_t clen = sizeof(caddr);
  SOCKET client = accept(srv, (sockaddr*)&caddr, &clen);
  if (client == INVALID_SOCKET) {
    std::fprintf(stderr, "accept failed\n");
    sock_close(srv);
    sockets_shutdown();
    return 1;
  }
  std::printf("[PBDServer] Client connected.\n");

  Shared sh;
  std::thread sim(sim_thread_fn, &sh);
  comm_loop(client, &sh);

  if (sim.joinable()) sim.join();

  sock_close(client);
  sock_close(srv);
  sockets_shutdown();
  std::printf("[PBDServer] Shutdown.\n");
  return 0;
}
