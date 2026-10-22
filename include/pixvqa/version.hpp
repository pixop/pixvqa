#pragma once

// Product executable name. The repository remains pixop-vqa.
#ifndef PIXVQA_PRODUCT_NAME
#define PIXVQA_PRODUCT_NAME "pixvqa"
#endif

// Development identifier. Not a public release number.
#ifndef PIXVQA_VERSION
#define PIXVQA_VERSION "0.3.0-dev"
#endif

// "cpu" or "cuda", set by the build script that compiles src/main.cpp.
#ifndef PIXVQA_BUILD_KIND
#define PIXVQA_BUILD_KIND "cpu"
#endif
