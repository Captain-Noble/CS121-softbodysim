#include "PBDServer.h"

// ============================================================
// ThreadPool impl
// ============================================================
ThreadPool::ThreadPool(unsigned workers) {
  if (workers < 1) workers = 1;
  workerCount_ = workers;
  stop_.store(false);
  for (unsigned i = 0; i < workerCount_; ++i) {
    threads_.emplace_back([this] { this->worker_loop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(m_);
    stop_.store(true);
  }
  cv_.notify_all();
  for (auto& t : threads_) {
    if (t.joinable()) t.join();
  }
}

void ThreadPool::run_ctx(WorkCtx& ctx) {
  while (true) {
    uint32_t i = ctx.next.fetch_add(ctx.chunk);
    if (i >= ctx.end) break;
    uint32_t j = std::min<uint32_t>(ctx.end, i + ctx.chunk);
    ctx.fn(i, j);
  }
}

void ThreadPool::worker_loop() {
  uint64_t seen = 0;
  while (true) {
    WorkCtx* ctx = nullptr;

    {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [&]{
        return stop_.load() || (current_ != nullptr && workId_ != seen);
      });
      if (stop_.load()) return;
      ctx = current_;
      seen = workId_;
    }

    if (!ctx) continue;
    run_ctx(*ctx);

    if (ctx->remaining.fetch_sub(1) == 1) {
      std::lock_guard<std::mutex> lk(ctx->doneM);
      ctx->doneCv.notify_one();
    }
  }
}

// ============================================================
// PBD init helpers
// ============================================================
void compute_inv_mass(PBDState& s, const std::vector<uint32_t>& pinned) {
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

void build_rest(PBDState& s) {
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

// ============================================================
// Constraints (GS XPBD) - still serial
// ============================================================
void solve_edges_xpbd_gs(PBDState& s, float dt) {
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

void solve_tets_xpbd_gs(PBDState& s, float dt) {
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

// ============================================================
// Integrator helpers
// ============================================================
static void predict_serial(PBDState& s, float dt) {
  Vec3 g = s.params.gravity;
  for (uint32_t i = 0; i < s.V; ++i) {
    if (s.w[i] == 0.0f) { s.xStar[i] = s.x[i]; continue; }
    s.v[i] = s.v[i] + g * dt;
    s.xStar[i] = s.x[i] + s.v[i] * dt;
  }
}

static void project_ground_serial(PBDState& s) {
  if (!s.params.groundEnabled) return;
  float y0 = s.params.groundY;
  for (uint32_t i = 0; i < s.V; ++i) {
    if (s.w[i] == 0.0f) continue;
    Vec3 p = s.xStar[i];
    if (p.y < y0) { p.y = y0; s.xStar[i] = p; }
  }
}

static void commit_serial(PBDState& s, float dt) {
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

// parallelized versions (only safe stages)
static void predict_parallel(ThreadPool& pool, PBDState& s, float dt) {
  Vec3 g = s.params.gravity;
  pool.parallel_for(0, s.V, 256, [&](uint32_t a, uint32_t b){
    for (uint32_t i = a; i < b; ++i) {
      if (s.w[i] == 0.0f) { s.xStar[i] = s.x[i]; continue; }
      s.v[i] = s.v[i] + g * dt;
      s.xStar[i] = s.x[i] + s.v[i] * dt;
    }
  });
}

static void project_ground_parallel(ThreadPool& pool, PBDState& s) {
  if (!s.params.groundEnabled) return;
  float y0 = s.params.groundY;
  pool.parallel_for(0, s.V, 256, [&](uint32_t a, uint32_t b){
    for (uint32_t i = a; i < b; ++i) {
      if (s.w[i] == 0.0f) continue;
      Vec3 p = s.xStar[i];
      if (p.y < y0) { p.y = y0; s.xStar[i] = p; }
    }
  });
}

static void commit_parallel(ThreadPool& pool, PBDState& s, float dt) {
  float invDt = (dt > 1e-12f) ? (1.0f / dt) : 0.0f;
  float y0 = s.params.groundY;
  float fr = std::fmax(0.0f, std::fmin(1.0f, s.params.friction));

  pool.parallel_for(0, s.V, 256, [&](uint32_t a, uint32_t b){
    for (uint32_t i = a; i < b; ++i) {
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
  });
}

// ============================================================
// Steppers
// ============================================================
void SerialStepper::step(PBDState& s, float dt, perf::StepStats& out) {
  using perf::ScopedAdd;

  auto tAll0 = perf::clock::now();

  uint32_t ss = std::max(1u, s.params.substeps);
  float sdt = dt / float(ss);

  for (uint32_t k = 0; k < ss; ++k) {
    { ScopedAdd _(&out.predictMs); predict_serial(s, sdt); }

    {
      ScopedAdd _(&out.solveMs);
      for (uint32_t it = 0; it < s.params.iterations; ++it) {
        solve_edges_xpbd_gs(s, sdt);
        solve_tets_xpbd_gs(s, sdt);
        project_ground_serial(s);
      }
    }

    { ScopedAdd _(&out.commitMs); commit_serial(s, sdt); }
  }

  auto tAll1 = perf::clock::now();
  out.totalMs += perf::ms_since(tAll0, tAll1);
}

void SerialStepper::pack_positions(const PBDState& s, std::vector<float>& outPos, double& outPackMs) {
  perf::ScopedAdd _(&outPackMs);
  const size_t need = size_t(s.V) * 3;
  if (outPos.size() != need) outPos.resize(need);
  for (uint32_t i = 0; i < s.V; ++i) {
    outPos[i*3+0] = s.x[i].x;
    outPos[i*3+1] = s.x[i].y;
    outPos[i*3+2] = s.x[i].z;
  }
}

ParallelStepper::ParallelStepper(unsigned threads) : pool(std::max(1u, threads)) {}

void ParallelStepper::step(PBDState& s, float dt, perf::StepStats& out) {
  using perf::ScopedAdd;

  auto tAll0 = perf::clock::now();

  uint32_t ss = std::max(1u, s.params.substeps);
  float sdt = dt / float(ss);

  for (uint32_t k = 0; k < ss; ++k) {
    { ScopedAdd _(&out.predictMs); predict_parallel(pool, s, sdt); }

    {
      ScopedAdd _(&out.solveMs);

      // NOTE: 约束仍是 GS 串行（未来要“真正并行 solver”就是替换下面两行）
      for (uint32_t it = 0; it < s.params.iterations; ++it) {
        solve_edges_xpbd_gs(s, sdt);
        solve_tets_xpbd_gs(s, sdt);
        project_ground_parallel(pool, s);
      }
    }

    { ScopedAdd _(&out.commitMs); commit_parallel(pool, s, sdt); }
  }

  auto tAll1 = perf::clock::now();
  out.totalMs += perf::ms_since(tAll0, tAll1);
}

void ParallelStepper::pack_positions(const PBDState& s, std::vector<float>& outPos, double& outPackMs) {
  perf::ScopedAdd _(&outPackMs);
  const size_t need = size_t(s.V) * 3;
  if (outPos.size() != need) outPos.resize(need);

  pool.parallel_for(0, s.V, 512, [&](uint32_t a, uint32_t b){
    for (uint32_t i = a; i < b; ++i) {
      outPos[i*3+0] = s.x[i].x;
      outPos[i*3+1] = s.x[i].y;
      outPos[i*3+2] = s.x[i].z;
    }
  });
}

// ============================================================
// sim thread main loop
// ============================================================
void sim_thread_fn(Shared* sh) {
  using clock = perf::clock;

  int frames = 0;
  auto lastPrint = clock::now();

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

    perf::StepStats st{};
    if (sh->stepper) {
      sh->stepper->step(sh->state, dt, st);
      sh->stepper->pack_positions(sh->state, sh->outPositions, st.packMs);
    }
    frames++;

    {
      std::lock_guard<std::mutex> lk(sh->m);
      sh->outReady = true;
      sh->acc.add(st);
    }
    sh->cvOut.notify_one();

    auto now = clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPrint).count();
    if (ms >= 1000) {
      double fps = double(frames) * 1000.0 / double(ms);

      perf::StepStats avg{};
      {
        std::lock_guard<std::mutex> lk(sh->m);
        avg = sh->acc.avg();
        sh->acc.reset();
      }

      std::printf(
        "[PBDServer] Mode=%s FPS %.1f | V=%u E=%u T=%u | avg(ms): total=%.3f pred=%.3f solve=%.3f commit=%.3f pack=%.3f\n",
        sh->stepper ? sh->stepper->name() : "none",
        fps, sh->state.V, sh->state.E, sh->state.T,
        avg.totalMs, avg.predictMs, avg.solveMs, avg.commitMs, avg.packMs
      );

      frames = 0;
      lastPrint = now;
    }
  }
}
