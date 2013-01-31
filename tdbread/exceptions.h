// exceptions.h -- definitions for exception handling

#ifndef INCLUDED_EXCEPTIONS
#define INCLUDED_EXCEPTIONS

#include <boost/exception/all.hpp>

struct exception_base: virtual std::exception, virtual boost::exception { };

struct parse_error: virtual exception_base { };
struct syntax_error: virtual parse_error { };
typedef boost::error_info<struct spec_err,std::string> specific_errinfo; // info for specific token that caused exception
typedef boost::error_info<struct str_err,std::string> str_errinfo; // user friendly error message

struct io_error: virtual exception_base { };
struct file_error: virtual io_error { };
struct read_error: virtual io_error { };
struct file_read_error: virtual file_error, virtual read_error { };

#endif