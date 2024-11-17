module;

#include <format>
#include <stdexcept>

export module cpp_matrix:log;

namespace cpp_matrix {

export void LogAndThrow(std::string msg)
{
    printf("%s\n", msg.c_str());
    throw std::runtime_error { std::move(msg) };
}

}