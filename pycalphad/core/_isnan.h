// Compatibility hack for Windows and non-Windows platforms
#ifdef _WIN32
#include <float.h>
#define isnan  _isnan
#else
#include <math.h>
#endif