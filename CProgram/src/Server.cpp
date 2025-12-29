#include "PBDServer.h"

// protocol helpers
bool read_header(SOCKET c, MsgHeader& h) {
  if (!recv_exact(c, &h, sizeof(MsgHeader))) return false;
  if (h.magic != MAGIC_PBD1) return false;
  return true;
}

bool send_positions(SOCKET c, const std::vector<float>& pos) {
  MsgHeader h;
  h.magic = MAGIC_PBD1;
  h.type  = MSG_POSITIONS;
  h.size  = (uint32_t)(pos.size() * sizeof(float));
  if (!send_exact(c, &h, sizeof(h))) return false;
  if (!pos.empty() && !send_exact(c, pos.data(), pos.size() * sizeof(float))) return false;
  return true;
}

void comm_loop(SOCKET client, Shared* sh) {
  std::vector<uint8_t> payload;

  while (sh->running.load()) {
    MsgHeader h{};
    if (!read_header(client, h)) break;

    payload.resize(h.size);
    if (h.size > 0 && !recv_exact(client, payload.data(), h.size)) break;

    if (h.type == MSG_INIT) {
      const uint8_t* p = payload.data();
      auto rd_u32 = [&](uint32_t& out) { std::memcpy(&out, p, 4); p += 4; };
      auto rd_f32 = [&](float& out)    { std::memcpy(&out, p, 4); p += 4; };

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
