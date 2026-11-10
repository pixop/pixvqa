#pragma once

// Product executable name. The repository remains pixop-vqa.
#ifndef PIXVQA_PRODUCT_NAME
#define PIXVQA_PRODUCT_NAME "pixvqa"
#endif

// Fallback for direct compilation without the PixVQA build scripts.
// Make and the legacy build scripts override this with the Git-derived version.
#ifndef PIXVQA_VERSION
#define PIXVQA_VERSION "0.3.0-dev"
#endif

// "cpu" or "cuda", set by the build script that compiles src/main.cpp.
#ifndef PIXVQA_BUILD_KIND
#define PIXVQA_BUILD_KIND "cpu"
#endif
