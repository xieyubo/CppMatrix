#include <map>
#include <utility>
#include <string>

import keras;

using namespace keras;

int main()
{
    auto inp = Input({ 21 });

    auto l1 = Dense(128, /*activation=*/"relu")(inp);
    auto l2 = Dense(128, /*activation=*/"relu")(l1);
    auto l3 = Dense(128, /*activation=*/"relu")(l2);
    auto l4 = Dense(128, /*activation=*/"relu")(l3);
    auto l5 = Dense(128, /*activation=*/"relu")(l4);

    auto policyOut = Dense(28, /*name=*/"policyHead", /*activation=*/"softmax")(l5);
    auto valueOut = Dense(1, /*name=*/"valueHead", /*activation=*/"tanh")(l5);

    auto bce = keras::losses::CategoricalCrossentropy(/*from_logits=*/false);
    auto model = Model(inp, { policyOut, valueOut });
    model.compile({.optimizer="SGD", .loss={{"valueHead", "mean_squared_error"}, {"policyHead", std::move(bce)}}});
    return 0;
}