#pragma once

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace pixvqa {

class ThreadPool {
 public:
  explicit ThreadPool(int n);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  template <class F>
  auto submit(F&& f) -> std::future<decltype(f())> {
    using R = decltype(f());
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    auto fut = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mu_);
      jobs_.emplace_back([task]() { (*task)(); });
    }
    cv_.notify_one();
    return fut;
  }

  int size() const { return static_cast<int>(workers_.size()); }

 private:
  void worker();
  std::vector<std::thread> workers_;
  std::vector<std::function<void()>> jobs_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
};

// Persistent workers reused across frames, planes and scales.
// Concurrent parallel regions in one process are serialized (exclusive gate).
// Scratch that is thread_local on a worker survives between regions.
class PersistentFor {
 public:
  static PersistentFor& instance();
  void shutdown();

  template <class Fn>
  void run(int nthreads, int begin, int end, Fn&& fn) {
    const int n = end - begin;
    if (n <= 0) return;
    if (nthreads <= 1 || n == 1) {
      fn(begin, end);
      return;
    }
    nthreads = std::min(nthreads, n);
    const int chunk = (n + nthreads - 1) / nthreads;
    int nchunks = 0;
    for (int t = 0; t < nthreads; ++t) {
      const int lo = begin + t * chunk;
      if (lo >= end) break;
      ++nchunks;
    }
    using FnObj = std::remove_reference_t<Fn>;
    FnObj* ptr = &fn;
    auto tramp = [](void* ctx, int lo, int hi) { (*static_cast<FnObj*>(ctx))(lo, hi); };
    publish_and_wait(nthreads, begin, end, chunk, nchunks, tramp, ptr);
  }

  // One task per index in [0, n). Uses at most nthreads workers (chunk size 1).
  template <class Fn>
  void run_indexed(int nthreads, int n, Fn&& fn) {
    if (n <= 0) return;
    if (nthreads <= 1 || n == 1) {
      for (int i = 0; i < n; ++i) fn(i);
      return;
    }
    nthreads = std::min(nthreads, n);
    using FnObj = std::remove_reference_t<Fn>;
    FnObj* ptr = &fn;
    auto tramp = [](void* ctx, int lo, int hi) {
      auto* f = static_cast<FnObj*>(ctx);
      for (int i = lo; i < hi; ++i) (*f)(i);
    };
    publish_and_wait(nthreads, 0, n, /*chunk=*/1, n, tramp, ptr);
  }

 private:
  PersistentFor();
  ~PersistentFor();
  void ensure_workers(int helpers);
  void worker_main();
  // mu_ must be held. Counts the failed ticket and abandons tickets not yet taken
  // so waiters cannot deadlock. In-flight tickets still complete and increment done_.
  void fail_ticket_locked();
  void publish_and_wait(int nthreads, int begin, int end, int chunk, int nchunks,
                        void (*body)(void*, int, int), void* ctx);

  std::mutex gate_;
  std::mutex mu_;
  std::condition_variable cv_work_;
  std::condition_variable cv_done_;
  std::vector<std::thread> workers_;
  bool stop_ = false;
  uint64_t epoch_ = 0;
  int begin_ = 0;
  int end_ = 0;
  int chunk_ = 1;
  int nchunks_ = 0;
  int next_chunk_ = 0;
  int done_ = 0;
  void (*body_)(void*, int, int) = nullptr;
  void* ctx_ = nullptr;
  std::exception_ptr error_;
};

template <class Fn>
void parallel_for_ranges(int nthreads, int begin, int end, Fn&& fn) {
  PersistentFor::instance().run(nthreads, begin, end, std::forward<Fn>(fn));
}

}  // namespace pixvqa
