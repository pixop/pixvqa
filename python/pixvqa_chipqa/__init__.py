"""PixVQA adapter for upstream ChipQA (SDR, pooled video score)."""

PINNED_UPSTREAM_COMMIT = "53af39c571a46993adedfb89feb82da0809b71cc"
FEATURE_DIM = 221
GOLDEN_FEATURE_SHA256 = "268cedf84a7a11addb36e2ca4a690e2134fb0fdd9a4bd7a39eb8e9d4ada9e995"
GOLDEN_SCORE = 46.09665088

SCALER_NAME = "chipqa_livestream_minmaxscaler.z"
SVR_NAME = "chipqa_livestream_svr.z"
SCALER_SHA256 = "761c2b17892f95ec62db4f28a22a147c4405e71e5944a66a9ff0ca6ef47a3d14"
SVR_SHA256 = "f13b6f97b8824d585879f5f4fa519bcfd8dac89223002f1961a4222f22bbda44"
