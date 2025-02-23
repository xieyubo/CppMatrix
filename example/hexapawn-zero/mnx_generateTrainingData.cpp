#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

import hexapawn_zero;

std::pair<std::pair<int, int>, long> getBestMoveRes(Board& board)
{
    auto bestMove = std::pair<int, int> { board.None, board.None };
    auto bestVal = 1000000000;
    for (auto m : board.generateMoves()) {
        auto tmp = board;
        tmp.applyMove(m);
        auto mVal = minimax(tmp, 30, tmp.turn == board.WHITE);
        if (board.turn == board.WHITE && mVal > bestVal) {
            bestVal = mVal;
            bestMove = m;
        }
        if (board.turn == board.BLACK && mVal < bestVal) {
            bestVal = mVal;
            bestMove = m;
        }
    }
    return { bestMove, bestVal };
}

std::vector<std::vector<int>> positions;
std::vector<std::vector<int>> moveProbs;
std::vector<int> outcomes;

std::vector<int> terminals;

void visitNodes(Board& board)
{
    auto [term, _] = board.isTerminal();
    if (term) {
        terminals.push_back(1);
        return;
    } else {
        auto [bestMove, bestVal] = getBestMoveRes(board);
        positions.push_back(board.toNetworkInput());
        auto moveProb = std::vector<int>(28);
        auto idx = board.getNetworkOutputIndex(bestMove);
        moveProb[idx] = 1;
        moveProbs.push_back(std::move(moveProb));
        if (bestVal > 0) {
            outcomes.push_back(1);
        }
        if (bestVal == 0) {
            outcomes.push_back(0);
        }
        if (bestVal < 0) {
            outcomes.push_back(-1);
        }
        for (auto m : board.generateMoves()) {
            auto next = board;
            next.applyMove(m);
            visitNodes(next);
        }
    }
}

void save(std::string name, nlohmann::json value)
{
    if (!name.ends_with(".json")) {
        name.append(".json");
    }
    std::ofstream { name } << value.dump();
}

int main()
{
    auto board = Board();
    board.setStartingPosition();
    visitNodes(board);

    save("positions", positions);
    save("moveprobs", moveProbs);
    save("outcomes", "outcomes");
    return 0;
}