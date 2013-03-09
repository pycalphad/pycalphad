// warning_disable.h -- disabling some excessive compiler warnings

#ifndef INCLUDED_WARNING_DISABLE
#define INCLUDED_WARNING_DISABLE
#include <boost/config/warning_disable.hpp>
#if defined(_MSC_VER)
#pragma warning (disable: 4503) //  name length exceeded warnings
#pragma warning (disable: 4244) // "conversion from '__int64' to 'const unsigned int', possible loss of data" in Boost utree lib
#endif
#endif