module;

#include <functional>
#include <map>
#include <string>
#include <variant>

export module keras:model;
import :layer;
import :losses;

namespace keras {

export class Model {
    using HeadLabel = std::string;
    using LossFuncNameOrLossFunc = std::variant<std::string, losses::Loss>;

public:
    struct CompileArgs {
        std::string optimizer { "rmsprop" };
        std::map<HeadLabel, LossFuncNameOrLossFunc> loss {};
    };

    Model(Layer input, std::initializer_list<Layer> outputs) { }

    void compile(CompileArgs args) { }
};

}