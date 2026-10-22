#pragma once

#include <cmath>

namespace pixvqa {

// Exact integral of the empirical step survival S(x) = P(Q >= x) over [lower, upper],
// normalized by the interval length and reported in percent:
//   100 * mean(clamp((Q - lower) / (upper - lower), 0, 1))
// Non-finite scores are ignored. This is not the published Python trapezoidal
// approximation of a linearly interpolated survival curve.
struct QcsAccum {
  double lower = 0;
  double upper = 1;
  double sum_clamped = 0;
  int finite = 0;

  void add(double q) {
    if (!std::isfinite(q)) return;
    const double width = upper - lower;
    double z = (q - lower) / width;
    if (z < 0.0) z = 0.0;
    if (z > 1.0) z = 1.0;
    sum_clamped += z;
    ++finite;
  }

  bool available() const { return finite > 0; }

  double percent() const { return 100.0 * sum_clamped / static_cast<double>(finite); }
};

}  // namespace pixvqa
