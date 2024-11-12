module;

#include <format>
#include <stdexcept>

export module gpu:log;

namespace gpu {

export void LogAndThrow(std::string msg)
{
    printf("%s\n", msg.c_str());
    throw std::runtime_error { std::move(msg) };
}

}