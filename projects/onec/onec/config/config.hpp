#pragma once

#include <cstdlib>
#include <cassert>
#include <iostream>

static_assert(sizeof(bool) == 1, "Bool must have a size of 1 byte");
static_assert(sizeof(short) == 2, "Short must have a size of 2 bytes");
static_assert(sizeof(int) == 4, "Int must have a size of 4 bytes");
static_assert(sizeof(long long int) == 8, "Long long int must have a size of 8 bytes");

#if defined(ONEC_DEBUG) && defined(ONEC_RELEASE)
    static_assert(false, "ONEC_DEBUG and ONEC_RELEASE must not be defined at the same time");
#endif

#if !defined(ONEC_DEBUG) && !defined(ONEC_RELEASE)
#   ifndef NDEBUG
#       define ONEC_DEBUG
#   else
#       define ONEC_RELEASE
#   endif
#endif

#ifdef ONEC_DEBUG
#   define ONEC_ERROR(message) std::cerr << "ONEC Error\n" \
                                         << "Description: " << message << "\n"\
                                         << "File: " << __FILE__ << "\n"\
                                         << "Line: " << __LINE__ << "\n";\
                               std::exit(EXIT_FAILURE)
#   define ONEC_ASSERT(condition, message) if (!(condition))\
                                           {\
                                               ONEC_ERROR(message);\
                                           }\
                                           static_cast<void>(0)
#   define ONEC_IF_DEBUG(code) code
#   define ONEC_IF_RELEASE(code)
#endif

#ifdef ONEC_RELEASE
#   define ONEC_ERROR(message) static_cast<void>(0)
#   define ONEC_ASSERT(condition, message) static_cast<void>(0)
#   define ONEC_IF_DEBUG(code)
#   define ONEC_IF_RELEASE(code) code
#endif
