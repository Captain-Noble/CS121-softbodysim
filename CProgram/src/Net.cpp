#include "PBDServer.h"

void sockets_init() {
#ifdef _WIN32
  WSADATA wsa;
  if (WSAStartup(MAKEWORD(2,2), &wsa) != 0) {
    std::fprintf(stderr, "WSAStartup failed\n");
    std::exit(1);
  }
#endif
}

void sockets_shutdown() {
#ifdef _WIN32
  WSACleanup();
#endif
}

void sock_close(SOCKET s) {
#ifdef _WIN32
  closesocket(s);
#else
  close(s);
#endif
}

bool recv_exact(SOCKET s, void* dst, size_t n) {
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

bool send_exact(SOCKET s, const void* src, size_t n) {
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

SOCKET listen_and_accept(int port) {
  SOCKET srv = socket(AF_INET, SOCK_STREAM, 0);
  if (srv == INVALID_SOCKET) {
    std::fprintf(stderr, "socket failed\n");
    return INVALID_SOCKET;
  }

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
    return INVALID_SOCKET;
  }

  if (listen(srv, 1) == SOCKET_ERROR) {
    std::fprintf(stderr, "listen failed\n");
    sock_close(srv);
    return INVALID_SOCKET;
  }

  std::printf("[PBDServer] Listening on port %d...\n", port);

  sockaddr_in caddr{};
  socklen_t clen = sizeof(caddr);
  SOCKET client = accept(srv, (sockaddr*)&caddr, &clen);
  sock_close(srv);

  if (client == INVALID_SOCKET) {
    std::fprintf(stderr, "accept failed\n");
    return INVALID_SOCKET;
  }

  std::printf("[PBDServer] Client connected.\n");
  return client;
}
