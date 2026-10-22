#pragma once

#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace pixvqa {

// CPUID feature bit only. This does not execute AVX2 or AVX-512.
inline bool cpu_has_avx2() {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
  return __builtin_cpu_supports("avx2");
#else
  return false;
#endif
}

// Processors this process may run on. Affinity and processor-group limits
// win over the machine-wide count. The result is not an optimal thread count.
inline int logical_processors() {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    const int n = CPU_COUNT(&set);
    if (n > 0) return n;
  }
#elif defined(_WIN32)
  DWORD_PTR process_mask = 0;
  DWORD_PTR system_mask = 0;
  if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
    int n = 0;
    for (DWORD_PTR bit = process_mask; bit != 0; bit &= (bit - 1)) ++n;
    if (n > 0) return n;
  }
#endif
  const unsigned n = std::thread::hardware_concurrency();
  return n == 0 ? 1 : static_cast<int>(n);
}

}  // namespace pixvqa
