#include "pixvqa/thread_pool.hpp"

#include <cstdlib>

namespace pixvqa {

ThreadPool::ThreadPool(int n) {
  if (n < 1) n = 1;
  workers_.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    workers_.emplace_back([this]() { worker(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : workers_) t.join();
}

void ThreadPool::worker() {
  for (;;) {
    std::function<void()> job;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [this]() { return stop_ || !jobs_.empty(); });
      if (stop_ && jobs_.empty()) return;
      job = std::move(jobs_.back());
      jobs_.pop_back();
    }
    job();
  }
}

PersistentFor& PersistentFor::instance() {
  static PersistentFor inst;
  return inst;
}

PersistentFor::PersistentFor() {
  std::atexit([]() { PersistentFor::instance().shutdown(); });
}

PersistentFor::~PersistentFor() { shutdown(); }

void PersistentFor::shutdown() {
  std::lock_guard<std::mutex> gate(gate_);
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (stop_) return;
    stop_ = true;
  }
  cv_work_.notify_all();
  for (auto& t : workers_) {
    if (t.joinable()) t.join();
  }
  workers_.clear();
}

void PersistentFor::ensure_workers(int helpers) {
  if (helpers < 0) helpers = 0;
  while (static_cast<int>(workers_.size()) < helpers) {
    workers_.emplace_back([this]() { worker_main(); });
  }
}

void PersistentFor::fail_ticket_locked() {
  if (!error_) error_ = std::current_exception();
  ++done_;
  if (next_chunk_ < nchunks_) {
    done_ += (nchunks_ - next_chunk_);
    next_chunk_ = nchunks_;
  }
  if (done_ >= nchunks_) cv_done_.notify_all();
}

void PersistentFor::worker_main() {
  uint64_t seen = 0;
  for (;;) {
    void (*body)(void*, int, int) = nullptr;
    void* ctx = nullptr;
    int begin = 0, end = 0, chunk = 1, nchunks = 0;
    uint64_t epoch = 0;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_work_.wait(lock, [&]() { return stop_ || epoch_ != seen; });
      if (stop_ && epoch_ == seen) return;
      seen = epoch_;
      epoch = epoch_;
      body = body_;
      ctx = ctx_;
      begin = begin_;
      end = end_;
      chunk = chunk_;
      nchunks = nchunks_;
    }
    if (!body || nchunks <= 0) continue;
    try {
      for (;;) {
        int ticket = 0;
        {
          std::lock_guard<std::mutex> lock(mu_);
          if (epoch_ != epoch) break;
          if (next_chunk_ >= nchunks_) break;
          ticket = next_chunk_++;
        }
        const int lo = begin + ticket * chunk;
        const int hi = std::min(end, lo + chunk);
        if (lo < hi) body(ctx, lo, hi);
        std::lock_guard<std::mutex> lock(mu_);
        ++done_;
        if (done_ >= nchunks_) cv_done_.notify_all();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mu_);
      fail_ticket_locked();
    }
  }
}

void PersistentFor::publish_and_wait(int nthreads, int begin, int end, int chunk, int nchunks,
                                     void (*body)(void*, int, int), void* ctx) {
  std::lock_guard<std::mutex> gate(gate_);
  {
    std::lock_guard<std::mutex> lock(mu_);
    ensure_workers(nthreads - 1);
    begin_ = begin;
    end_ = end;
    chunk_ = chunk;
    nchunks_ = nchunks;
    next_chunk_ = 0;
    done_ = 0;
    body_ = body;
    ctx_ = ctx;
    error_ = nullptr;
    ++epoch_;
  }
  cv_work_.notify_all();

  // Caller participates so one core is not idle waiting.
  try {
    for (;;) {
      int ticket = -1;
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (next_chunk_ >= nchunks_) break;
        ticket = next_chunk_++;
      }
      const int lo = begin + ticket * chunk;
      const int hi = std::min(end, lo + chunk);
      if (lo < hi) body(ctx, lo, hi);
      std::lock_guard<std::mutex> lock(mu_);
      ++done_;
      if (done_ >= nchunks_) cv_done_.notify_all();
    }
  } catch (...) {
    std::lock_guard<std::mutex> lock(mu_);
    fail_ticket_locked();
  }

  {
    std::unique_lock<std::mutex> lock(mu_);
    cv_done_.wait(lock, [&]() { return done_ >= nchunks_; });
    body_ = nullptr;
    ctx_ = nullptr;
  }
  if (error_) std::rethrow_exception(error_);
}

}  // namespace pixvqa
