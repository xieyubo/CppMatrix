module;

#include <algorithm>

export module hexapawn_zero:mnx_minimax;
import :game;

export long minimax(Board& board, int depth, bool maximize) {
    auto [isTerminal, winner] = board.isTerminal();
    if (isTerminal) {
        if (winner == board.WHITE) {
            return 1000;
        }
        if (winner == board.BLACK) {
            return -1000;
        }
        if (winner == board.DRAW) {
            return 0;
        }
    }
    auto moves = board.generateMoves();
    if (maximize) {
        auto bestVal = -999999999999;
        for (const auto& move: moves) {
            auto next = board;
            next.applyMove(move);
            bestVal = std::max(bestVal, minimax(next, depth - 1, !maximize));
        }
        return bestVal;
    } else {
        auto bestVal = 999999999999;
        for (const auto& move: moves) {
            auto next = board;
            next.applyMove(move);
            bestVal = std::min(bestVal, minimax(next, depth - 1, !maximize));
        }
        return bestVal;
    }
}