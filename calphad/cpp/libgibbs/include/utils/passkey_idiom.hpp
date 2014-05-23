/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis
	Copyright (c) 2010 "GManNickG" on StackOverflow
	From http://stackoverflow.com/questions/3324898/can-we-increase-the-re-usability-of-this-key-oriented-access-protection-pattern

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

/*
 * The idea of the passkey idiom is to allow classes to implement
 * functions that can only be accessed by certain other classes.
 * Rather than allowing a friend class full access to private data members,
 * this idiom gives that class finer-grained control over access.
 */

#ifndef PASSKEY_IDIOM_HPP_
#define PASSKEY_IDIOM_HPP_


// each class has its own unique key only it can create
// (it will try to get friendship by "showing" its passkey)
template <typename T>
class passkey
{
private:
    friend T; // C++0x, MSVC allows as extension
    passkey() {}

    // noncopyable
    passkey(const passkey&) = delete;
    passkey& operator=(const passkey&) = delete;
};

// functions still require a macro. this
// is because a friend function requires
// the entire declaration, which is not
// just a type, but a name as well. we do
// this by creating a tag and specializing
// the passkey for it, friending the function
#define PASSKEY_EXPAND(pX) pX

// we use variadic macro parameters to allow
// functions with commas, it all gets pasted
// back together again when we friend it
#define PASSKEY_FUNCTION(pTag, pFunc, ...)               \
        struct PASSKEY_EXPAND(pTag);                             \
                                                         \
        template <>                                      \
        class passkey<PASSKEY_EXPAND(pTag)>                      \
        {                                                \
        private:                                         \
            friend pFunc __VA_ARGS__;                    \
            passkey() {}                                 \
                                                         \
            passkey(const passkey&) = delete;            \
            passkey& operator=(const passkey&) = delete; \
        }

// meta function determines if a type
// is contained in a parameter pack
template<typename T, typename... List>
struct is_contained : std::false_type {};

template<typename T, typename... List>
struct is_contained<T, T, List...> : std::true_type {};

template<typename T, typename Head, typename... List>
struct is_contained<T, Head, List...> : is_contained<T, List...> {};

// this class can only be created with allowed passkeys
template <typename... Keys>
class allow
{
public:
    // check if passkey is allowed
    template <typename Key>
    allow(const passkey<Key>&)
    {
        static_assert(is_contained<Key, Keys>::value,
                        "Passkey is not allowed.");
    }

private:
    // noncopyable
    allow(const allow&) = delete;
    allow& operator=(const allow&) = delete;
};

#endif /* PASSKEY_IDIOM_HPP_ */
