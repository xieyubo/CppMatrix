module;

#include <array>
#include <map>
#include <vector>
#include <string>

export module hexapawn_zero:game;

export class Board {
public:
    const int None = -1;
    const int DRAW = -1;
    const int EMPTY = 0;
    const int WHITE = 1;
    const int BLACK = 2;

    Board()
    {
        self.board = {self.EMPTY, self.EMPTY, self.EMPTY,
                      self.EMPTY, self.EMPTY, self.EMPTY,
                      self.EMPTY, self.EMPTY, self.EMPTY};

        self.turn = self.WHITE;

        self.outputIndex = {};
        self.legal_moves = {};

        // white forward moves
        self.outputIndex[{6, 3}] = 0;
        self.outputIndex[{7, 4}] = 1;
        self.outputIndex[{8, 5}] = 2;
        self.outputIndex[{3, 0}] = 3;
        self.outputIndex[{4, 1}] = 4;
        self.outputIndex[{5, 2}] = 5;

        // black forward moves
        self.outputIndex[{0, 3}] = 6;
        self.outputIndex[{1, 4}] = 7;
        self.outputIndex[{2, 5}] = 8;
        self.outputIndex[{3, 6}] = 9;
        self.outputIndex[{4, 7}] = 10;
        self.outputIndex[{5, 8}] = 11;

        // white capture moves
        self.outputIndex[{6, 4}] = 12;
        self.outputIndex[{7, 3}] = 13;
        self.outputIndex[{7, 5}] = 14;
        self.outputIndex[{8, 4}] = 15;
        self.outputIndex[{3, 1}] = 16;
        self.outputIndex[{4, 0}] = 17;
        self.outputIndex[{4, 2}] = 18;
        self.outputIndex[{5, 1}] = 19;

        // black pawn moves
        self.outputIndex[{0, 4}] = 20;
        self.outputIndex[{1, 3}] = 21;
        self.outputIndex[{1, 5}] = 22;
        self.outputIndex[{2, 4}] = 23;
        self.outputIndex[{3, 7}] = 24;
        self.outputIndex[{4, 6}] = 25;
        self.outputIndex[{4, 8}] = 26;
        self.outputIndex[{5, 7}] = 27;

        // enumerate fields such that
        // WHITE_PAWN_CAPTURES[FROM IDX] lists all possible
        // target capture squares
        self.WHITE_PAWN_CAPTURES = {
            {},
            {},
            {},
            {1},
            {0,2},
            {1},
            {4},
            {3,5},
            {4}
        };

        self.BLACK_PAWN_CAPTURES = {
            {4},
            {3,5},
            {4},
            {7},
            {6,8},
            {7},
            {},
            {},
            {}
        };
    }

    std::pair<bool, int> isTerminal()
    {
        auto winner = None;
        // black wins if she has placed a pawn on the 0th row
        if (self.board[6] == self.BLACK ||
                self.board[7] == self.BLACK ||
                self.board[8] == self.BLACK) {
            winner = self.BLACK;
        }
        // white wins if he placed a pawn on the 3rd row
        if (self.board[0] == self.WHITE ||
                self.board[1] == self.WHITE ||
                self.board[2] == self.WHITE) {
            winner = self.WHITE;
        }
        if (winner != None) {
            return {true, winner};
        } else {
            // stalemate if there is no winner
            // and current player can't move
            if (self.generateMoves().size() == 0) {
                if (self.turn == self.WHITE) {
                    return {true, self.BLACK};
                } else {
                    return {true, self.WHITE};
                }
            } else {
                return {false, None};
            }
        }
    }

    std::string toString()
    {
        if (self.turn == self.WHITE) {
            std::string str = "w:";
            for (const auto& x: self.board) {
                str += std::to_string(x);
            }
            return str;
        } else {
            std::string str = "b:";
            for (const auto& x: self.board) {
                str += std::to_string(x);
            }
            return str;
        }
    }

    std::string toDisplayString()
    {
        std::string s = "";
        for (auto i = 0; i < 3; ++i) {
            if (self.board[i] == self.WHITE) {
                s += "W";
            }
            if (self.board[i] == self.BLACK) {
                s += "B";
            }
            if (self.board[i] == self.EMPTY) {
                s += "_";
            }
        }
        s += "\n";
        for (auto i = 3; i < 6; ++i) {
            if (self.board[i] == self.WHITE) {
                s += "W";
            }
            if (self.board[i] == self.BLACK) {
                s += "B";
            }
            if (self.board[i] == self.EMPTY) {
                s += "_";
            }
        }
        s += "\n";
        for (auto i = 6; i < 9; ++i) {
            if (self.board[i] == self.WHITE) {
                s += "W";
            }
            if (self.board[i] == self.BLACK) {
                s += "B";
            }
            if (self.board[i] == self.EMPTY) {
                s += "_";
            }
        }
        s += "\n";
        return s;
    }

    // turn the position + turn into
    // input for the network
    std::vector<int> toNetworkInput()
    {
        std::vector<int> posVec = {};
        // white pawns
        for (auto i = 0; i < 9; ++i) {
            if (self.board[i] == self.WHITE) {
                posVec.push_back(1);
            } else {
                posVec.push_back(0);
            }
        }
        // black pawns
        for (auto i = 0; i < 9; ++i) {
            if (self.board[i] == self.BLACK) {
                posVec.push_back(1);
            } else {
                posVec.push_back(0);
            }
        }
        for (auto i = 0; i < 3; ++i) {
            if (self.board[i] == self.WHITE) {
                posVec.push_back(1);
            } else {
                posVec.push_back(0);
            }
        }
        return posVec;
    }

    // given a move, get the index of the correspnding move output
    // of the network
    int getNetworkOutputIndex(std::pair<int, int> move)
    {
        return self.outputIndex[move];
    }

    void setStartingPosition()
    {
        // BLACK BLACK BLACK
        // ##### ##### #####
        // WHITE WHITE WHITE
        self.board = {self.BLACK, self.BLACK, self.BLACK,
                      self.EMPTY, self.EMPTY, self.EMPTY,
                      self.WHITE, self.WHITE, self.WHITE};
    }

    void applyMove(std::pair<int, int> move)
    {
        auto fromSquare = move.first;
        auto toSquare = move.second;
        self.board[toSquare] = self.board[fromSquare];
        self.board[fromSquare] = self.EMPTY;
        if (self.turn == self.WHITE) {
            self.turn = self.BLACK;
        } else {
            self.turn = self.WHITE;
        }
        self.legal_moves = {};
    }

    std::vector<std::pair<int, int>> generateMoves()
    {
        if (self.legal_moves.empty()) {
            std::vector<std::pair<int, int>> moves = {};
            for (auto i = 0; i < 9; ++i) {
                if (self.board[i] == self.turn) {
                    if (self.turn == self.WHITE) {
                        // check if we can move one square up
                        auto toSqure = i - 3;
                        if (toSqure >= 0) {
                            if (self.board[toSqure] == self.EMPTY) {
                                moves.push_back({i, toSqure});
                            }
                        }
                        // check if we can capture to the left or right
                        const auto& potCaptureSquares = self.WHITE_PAWN_CAPTURES[i];
                        for (auto toSquare: potCaptureSquares) {
                            if (self.board[toSquare] == self.BLACK) {
                                moves.push_back({i, toSquare});
                            }
                        }
                    }
                    if (self.turn == self.BLACK) {
                        // check if we can move one square down
                        auto toSquare = i + 3;
                        if (toSquare < 9) {
                            if (self.board[toSquare] == self.EMPTY) {
                                moves.push_back({i, toSquare});
                            }
                        }
                        // check if we can capture to the left or right
                        const auto& potCaptureSquares = self.BLACK_PAWN_CAPTURES[i];
                        for (auto toSquare: potCaptureSquares) {
                            if (self.board[toSquare] == self.WHITE) {
                                moves.push_back({i, toSquare});
                            }
                        }
                    }
                }
            }
            self.legal_moves = std::move(moves);
        }
        return self.legal_moves;
    }

    int turn;

    Board(const Board& board)
        : turn { board.turn }
        , board { board.board }
        , outputIndex { board.outputIndex }
        , legal_moves { board.legal_moves }
        , WHITE_PAWN_CAPTURES { board.WHITE_PAWN_CAPTURES }
        , BLACK_PAWN_CAPTURES { board.BLACK_PAWN_CAPTURES }
    {
    }

private:
    Board& self {*this};
    std::vector<int> board;
    std::map<std::pair<int, int>, int> outputIndex;
    std::vector<std::pair<int, int>> legal_moves;
    std::vector<std::vector<int>> WHITE_PAWN_CAPTURES;
    std::vector<std::vector<int>> BLACK_PAWN_CAPTURES;
};
