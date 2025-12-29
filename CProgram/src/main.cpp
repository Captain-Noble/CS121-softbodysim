#include "PBDServer.h"

// simple args
enum class RunMode { Serial, Parallel };

static bool is_number(const char* s) {
  if (!s || !*s) return false;
  for (const char* p = s; *p; ++p) if (*p < '0' || *p > '9') return false;
  return true;
}

struct CmdArgs {
  int port = 7777;
  RunMode mode = RunMode::Serial;
  unsigned threads = std::max(1u, std::thread::hardware_concurrency());
};

static void print_usage(const char* exe) {
  std::printf(
    "Usage:\n"
    "  %s --port 7777 --mode serial|parallel [--threads N]\n"
    "  %s 7777 serial|parallel\n",
    exe, exe
  );
}

static CmdArgs parse_args(int argc, char** argv) {
  CmdArgs a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];

    if (s == "--help" || s == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }

    if (s == "--port" && i + 1 < argc) {
      a.port = std::atoi(argv[++i]);
      continue;
    }
    if (s == "--mode" && i + 1 < argc) {
      std::string m = argv[++i];
      if (m == "serial") a.mode = RunMode::Serial;
      else if (m == "parallel") a.mode = RunMode::Parallel;
      else {
        std::fprintf(stderr, "Unknown mode: %s\n", m.c_str());
        print_usage(argv[0]);
        std::exit(1);
      }
      continue;
    }
    if (s == "--threads" && i + 1 < argc) {
      a.threads = (unsigned)std::max(1, std::atoi(argv[++i]));
      continue;
    }

    // positional fallback
    if (is_number(argv[i])) { a.port = std::atoi(argv[i]); continue; }
    if (s == "serial")   { a.mode = RunMode::Serial; continue; }
    if (s == "parallel") { a.mode = RunMode::Parallel; continue; }

    std::fprintf(stderr, "Unknown arg: %s\n", s.c_str());
    print_usage(argv[0]);
    std::exit(1);
  }
  return a;
}

int main(int argc, char** argv) {
  sockets_init();

  CmdArgs args = parse_args(argc, argv);

  SerialStepper serial;
  ParallelStepper parallel(args.threads);

  Shared sh;
  sh.stepper = (args.mode == RunMode::Serial) ? (IStepper*)&serial : (IStepper*)&parallel;

  std::printf("[PBDServer] Start. mode=%s threads=%u port=%d\n",
              sh.stepper->name(), args.threads, args.port);

  SOCKET client = listen_and_accept(args.port);
  if (client == INVALID_SOCKET) {
    sockets_shutdown();
    return 1;
  }

  std::thread sim(sim_thread_fn, &sh);
  comm_loop(client, &sh);

  if (sim.joinable()) sim.join();

  sock_close(client);
  sockets_shutdown();
  std::printf("[PBDServer] Shutdown.\n");
  return 0;
}
