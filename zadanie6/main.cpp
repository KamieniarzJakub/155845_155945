#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <list>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <memory>
#include <chrono>
#include <climits>
#include <set>

using namespace std;

mt19937 rng(random_device{}());

enum class NeighType { VERTEX_SWAP, EDGE_SWAP };
enum class LSMode    { STEEPEST, GREEDY };


class Instance {
public:
    int n;
    vector<int> profit;
    vector<vector<int>> dist;

    static Instance loadFromCSV(const string& filename, bool precomputed=false) {
        ifstream in(filename);
        if(!in) { cerr << "Cannot open " << filename << endl; exit(1); }
        Instance inst;
        string line;

        if(precomputed) {
            while(getline(in, line)) {
                if(line.empty()) continue;
                replace(line.begin(), line.end(), ';', ' ');
                stringstream ss(line);
                int val; vector<int> row;
                ss >> val; inst.profit.push_back(val);
                while(ss >> val) row.push_back(val);
                inst.dist.push_back(row);
            }
            inst.n = inst.profit.size();
        } else {
            vector<double> X,Y;
            while(getline(in,line)) {
                if(line.empty()) continue;
                replace(line.begin(), line.end(), ';', ' ');
                stringstream ss(line);
                double x,y; int p;
                ss >> x >> y >> p;
                X.push_back(x); Y.push_back(y); inst.profit.push_back(p);
            }
            inst.n = X.size();
            inst.dist.assign(inst.n, vector<int>(inst.n));
            for(int i=0;i<inst.n;i++)
                for(int j=0;j<inst.n;j++)
                    inst.dist[i][j] = round(sqrt((X[i]-X[j])*(X[i]-X[j]) + (Y[i]-Y[j])*(Y[i]-Y[j])));
        }
        return inst;
    }

    void saveDistanceCSV(const string& filename) const {
        ofstream f(filename);
        for(int i=0;i<n;i++) {
            f << profit[i];
            for(int j=0;j<n;j++) f << ";" << dist[i][j];
            f << "\n";
        }
    }

    int cycleLength(const vector<int>& c) const {
        int s=0;
        for(size_t i=0;i<c.size();i++) s += dist[c[i]][c[(i+1)%c.size()]];
        return s;
    }

    int cycleProfit(const vector<int>& c) const {
        int s=0; for(int v:c) s+=profit[v]; return s;
    }

    int deltaInsert(int i,int j,int v) const { return dist[i][v]+dist[v][j]-dist[i][j]; }
    int deltaRemove(int prev,int k,int next) const { return dist[prev][k]+dist[k][next]-dist[prev][next]-profit[k]; }
};

class Solution {
public:
    vector<int> cycle;
    int length=0, profitSum=0, lengthPhase1=0;

    int objective() const { return profitSum - length; }

    void computeStats(const Instance& inst) {
        if(cycle.empty()) return;
        length = inst.cycleLength(cycle);
        profitSum = inst.cycleProfit(cycle);
    }

    void phaseIIRemove(const Instance& inst, size_t min_size=3) {
        list<int> cyc(cycle.begin(), cycle.end());
        bool changed=true;
        while(changed && cyc.size() > min_size) {
            changed = false; 
            int bestGain = 0; 
            auto bestIt = cyc.begin();

            for(auto it = cyc.begin(); it != cyc.end(); ++it) {
                auto prev_it = (it == cyc.begin() ? prev(cyc.end()) : prev(it));
                auto next_it = next(it); 
                if(next_it == cyc.end()) next_it = cyc.begin();
                
                int delta = inst.deltaRemove(*prev_it, *it, *next_it);
                if(delta > bestGain) { 
                    bestGain = delta; 
                    bestIt = it; 
                }
            }
            if(bestGain > 0) { 
                cyc.erase(bestIt); 
                changed = true; 
            }
        }
        cycle.assign(cyc.begin(), cyc.end());
        computeStats(inst);
    }
};

class Heuristic {
public:
    virtual Solution solve(const Instance& inst)=0;
    virtual ~Heuristic(){}
};

// --- Pomocnicze klasy (LS, Regret) ---

class RandomSolution: public Heuristic {
public:
    Solution solve(const Instance& inst) override {
        Solution sol;
        vector<int> perm(inst.n); iota(perm.begin(),perm.end(),0);
        shuffle(perm.begin(),perm.end(), rng);
        int k = max(2, inst.n / 2);
        sol.cycle.assign(perm.begin(), perm.begin()+k);
        sol.computeStats(inst);
        sol.lengthPhase1 = sol.length;
        return sol;
    }
};

class Regret2: public Heuristic {
    bool weighted; double w;
public:
    Regret2(bool wg=false,double ww=1.0):weighted(wg),w(ww){}

    Solution solve(const Instance& inst) override {
        int n = inst.n;
        vector<bool> used(n, false);
        int start = uniform_int_distribution<int>(0, n-1)(rng);
        used[start] = true;

        int nxt = -1, bestd = INT_MAX;
        for (int v = 0; v < n; v++)
            if (!used[v] && inst.dist[start][v] < bestd) { bestd = inst.dist[start][v]; nxt = v; }
        used[nxt] = true;

        vector<int> cyc = {start, nxt};
        vector<int> unused; unused.reserve(n - 2);
        for (int v = 0; v < n; v++) if (!used[v]) unused.push_back(v);

        return runRegretLoop(inst, cyc, unused);
    }

    void repair(const Instance& inst, Solution& partialSol) {
        int n = inst.n;
        vector<bool> used(n, false);
        vector<int> cyc = partialSol.cycle;
        for (int v : cyc) used[v] = true;

        if (cyc.empty()) {
            int start = uniform_int_distribution<int>(0, n - 1)(rng);
            cyc.push_back(start);
            used[start] = true;
        }

        vector<int> unused; unused.reserve(n - (int)cyc.size());
        for (int v = 0; v < n; v++) if (!used[v]) unused.push_back(v);

        partialSol = runRegretLoop(inst, cyc, unused);
    }

private:
    Solution runRegretLoop(const Instance& inst, vector<int>& cyc, vector<int>& unused) {
        int target_k = max(2, inst.n / 2);
        int cyc_size = (int)cyc.size();

        vector<double> incs;
        incs.reserve(cyc_size + target_k);

        while (cyc_size < target_k) {
            double bestRG = -1e18;
            int bestV = -1, bestVIdx = -1, bestPos = -1;

            for (int ui = 0; ui < (int)unused.size(); ui++) {
                int v = unused[ui];
                incs.clear();
                for (int i = 0; i < cyc_size; i++) {
                    int a = cyc[i], b = cyc[(i + 1) % cyc_size];
                    incs.push_back(inst.deltaInsert(a, b, v));
                }
                double min1 = 1e18, min2 = 1e18;
                for (double x : incs) {
                    if (x < min1) { min2 = min1; min1 = x; }
                    else if (x < min2) { min2 = x; }
                }
                if (min2 >= 1e17) continue; 
                double rg = min2 - min1;
                if (weighted) rg += w * (-min1);
                if (rg > bestRG) { bestRG = rg; bestV = v; bestVIdx = ui; }
            }


            double bestInc = 1e18;
            for (int i = 0; i < cyc_size; i++) {
                int a = cyc[i], b = cyc[(i + 1) % cyc_size];
                double inc = inst.deltaInsert(a, b, bestV);
                if (inc < bestInc) { bestInc = inc; bestPos = i + 1; }
            }


            cyc.insert(cyc.begin() + bestPos, bestV);
            cyc_size++;

            unused[bestVIdx] = unused.back();
            unused.pop_back();
        }

        Solution sol;
        sol.cycle = cyc;
        sol.computeStats(inst);
        return sol;
    }
};

class LocalSearchWithMoveList : public Heuristic {
    RandomSolution rndSol;
    enum class MoveType { ADD, REMOVE, EDGE_SWAP };
    struct Move { int delta; MoveType type; int a, b, c, d; };
    enum class EdgeSwapApplicability { NOT_APPLICABLE, FORWARD, REVERSED };
    vector<Move> moveList;

public:
    Solution solve(const Instance& inst) override {
        return solveFrom(inst, rndSol.solve(inst));
    }

    Solution solveFrom(const Instance& inst, Solution sol) {
        if (sol.cycle.empty()) {
            sol.computeStats(inst);
            return sol;
        }

        vector<int> next(inst.n, -1);
        vector<int> prev(inst.n, -1);
        vector<char> inCycle(inst.n, 0);

        buildLinkedCycle(sol.cycle, next, prev, inCycle);
        int startVertex = sol.cycle[0];
        int cycleSize = (int)sol.cycle.size();
        buildImprovingMoveList(inst, next, prev, inCycle, startVertex, cycleSize, moveList);

        while (true) {
            bool applied = false;
            while (!moveList.empty()) {
                Move m = moveList.back();
                moveList.pop_back();

                if (!isApplicable(m, next, prev, inCycle)) continue;

                applyMove(m, inst, next, prev, inCycle, startVertex, cycleSize);
                applied = true;
                break;
            }

            if (!applied) {
                buildImprovingMoveList(inst, next, prev, inCycle, startVertex, cycleSize, moveList);
                if (moveList.empty()) break;
            }
        }

        sol.cycle = materializeCycle(startVertex, next, cycleSize);
        sol.computeStats(inst);
        return sol;
    }

private:
    void buildLinkedCycle(const vector<int>& cyc, vector<int>& next, vector<int>& prev, vector<char>& inCycle) {
        fill(next.begin(), next.end(), -1);
        fill(prev.begin(), prev.end(), -1);
        fill(inCycle.begin(), inCycle.end(), 0);
        int n = (int)cyc.size();
        for (int i = 0; i < n; i++) {
            int v = cyc[i];
            int vn = cyc[(i + 1) % n];
            int vp = cyc[(i - 1 + n) % n];
            next[v] = vn; prev[v] = vp; inCycle[v] = 1;
        }
    }

    vector<int> materializeCycle(int startVertex, const vector<int>& next, int cycleSize) {
        vector<int> cyc; cyc.reserve(cycleSize);
        int v = startVertex;
        for (int step = 0; step < cycleSize; step++) {
            cyc.push_back(v); v = next[v];
        }
        return cyc;
    }

    void rebuildLinkedCycleFromLinearOrder(const vector<int>& cyc, vector<int>& next, vector<int>& prev) {
        int n = (int)cyc.size();
        for (int i = 0; i < n; i++) {
            int v = cyc[i];
            next[v] = cyc[(i + 1) % n];
            prev[v] = cyc[(i - 1 + n) % n];
        }
    }

    void insertBetween(int left, int right, int v, vector<int>& next, vector<int>& prev, vector<char>& inCycle) {
        next[left] = v; prev[v] = left;
        next[v] = right; prev[right] = v;
        inCycle[v] = 1;
    }

    void removeVertex(int v, vector<int>& next, vector<int>& prev, vector<char>& inCycle) {
        int left = prev[v]; int right = next[v];
        next[left] = right; prev[right] = left;
        next[v] = -1; prev[v] = -1; inCycle[v] = 0;
    }

    void reverseSubpath(vector<int>& cyc, int start, int end) {
        int n = (int)cyc.size();
        if (start <= end) {
            reverse(cyc.begin() + start, cyc.begin() + end + 1);
        } else {
            vector<int> temp; temp.reserve(n - start + end + 1);
            for (int k = start; k < n; k++) temp.push_back(cyc[k]);
            for (int k = 0; k <= end; k++) temp.push_back(cyc[k]);
            reverse(temp.begin(), temp.end());
            int idx = 0;
            for (int k = start; k < n; k++) cyc[k] = temp[idx++];
            for (int k = 0; k <= end; k++) cyc[k] = temp[idx++];
        }
    }

    void buildImprovingMoveList(const Instance& inst, const vector<int>& next, const vector<int>& prev, const vector<char>& inCycle, int startVertex, int cycleSize, vector<Move>& lm) {
        lm.clear();
        vector<int> cyc = materializeCycle(startVertex, next, cycleSize);
        int n = (int)cyc.size();

        vector<int> outside; outside.reserve(inst.n - n);
        for (int v = 0; v < inst.n; v++) if (!inCycle[v]) outside.push_back(v);

        for (int i = 0; i < n; i++) {
            int left = cyc[i], right = cyc[(i + 1) % n];
            for (int v : outside) {
                int delta = inst.profit[v] - inst.deltaInsert(left, right, v);
                if (delta > 0) lm.push_back({delta, MoveType::ADD, left, right, v, -1});
            }
        }

        if (n > 2) {
            for (int idx = 0; idx < n; idx++) {
                int left = cyc[(idx - 1 + n) % n], v = cyc[idx], right = cyc[(idx + 1) % n];
                int delta = inst.deltaRemove(left, v, right);
                if (delta > 0) lm.push_back({delta, MoveType::REMOVE, left, v, right, -1});
            }
        }

        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 2; j < n; j++) {
                if (i == 0 && j == n - 1) continue;
                int a = cyc[i], b = cyc[(i + 1) % n], c = cyc[j], d = cyc[(j + 1) % n];
                int delta = inst.dist[a][b] + inst.dist[c][d] - inst.dist[a][c] - inst.dist[b][d];
                if (delta > 0) lm.push_back({delta, MoveType::EDGE_SWAP, a, b, c, d});
            }
        }

        sort(lm.begin(), lm.end(), [](const Move& lhs, const Move& rhs) {
            if (lhs.delta != rhs.delta) return lhs.delta < rhs.delta;
            return (int)lhs.type < (int)rhs.type;
        });
    }

    bool isApplicable(const Move& m, const vector<int>& next, const vector<int>& prev, const vector<char>& inCycle) const {
        if (m.type == MoveType::ADD) return (!inCycle[m.c] && next[m.a] == m.b);
        if (m.type == MoveType::REMOVE) return (inCycle[m.b] && prev[m.b] == m.a && next[m.b] == m.c);
        return getEdgeSwapApplicability(m, next) != EdgeSwapApplicability::NOT_APPLICABLE;
    }

    EdgeSwapApplicability getEdgeSwapApplicability(const Move& m, const vector<int>& next) const {
        if (next[m.a] == m.b && next[m.c] == m.d) return EdgeSwapApplicability::FORWARD;
        if (next[m.b] == m.a && next[m.d] == m.c) return EdgeSwapApplicability::REVERSED;
        return EdgeSwapApplicability::NOT_APPLICABLE;
    }

    void applyMove(const Move& m, const Instance& inst, vector<int>& next, vector<int>& prev, vector<char>& inCycle, int& startVertex, int& cycleSize) {
        if (m.type == MoveType::ADD) {
            insertBetween(m.a, m.b, m.c, next, prev, inCycle);
            cycleSize++; return;
        }
        if (m.type == MoveType::REMOVE) {
            removeVertex(m.b, next, prev, inCycle);
            cycleSize--;
            if (m.b == startVertex) startVertex = m.c;
            return;
        }
        applyEdgeSwapMove(m, next, prev, startVertex, cycleSize);
    }

    void applyEdgeSwapMove(const Move& m, vector<int>& next, vector<int>& prev, int& startVertex, int cycleSize) {
        vector<int> cyc = materializeCycle(startVertex, next, cycleSize);
        int n = (int)cyc.size();
        vector<int> position(next.size(), -1);
        for (int i = 0; i < n; i++) position[cyc[i]] = i;

        EdgeSwapApplicability state = getEdgeSwapApplicability(m, next);
        if (state == EdgeSwapApplicability::NOT_APPLICABLE) return;

        int start_rev, end_rev;
        if (state == EdgeSwapApplicability::FORWARD) {
            start_rev = (position[m.a] + 1) % n;
            end_rev = position[m.c];
        } else {
            start_rev = (position[m.b] + 1) % n;
            end_rev = position[m.d];
        }

        reverseSubpath(cyc, start_rev, end_rev);
        rebuildLinkedCycleFromLinearOrder(cyc, next, prev);
        startVertex = cyc[0];
    }
};



class IteratedRegret : public Heuristic {
    double time_limit_ms;
public:
    int iterationsCount = 0;
    IteratedRegret(double limit) : time_limit_ms(limit) {}

    Solution solve(const Instance& inst) override {
        iterationsCount = 0;
        auto t0 = chrono::steady_clock::now();
        Solution best;
        best.profitSum = -1e9; // Start z bardzo niskim wynikiem

        while (chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count() < time_limit_ms) {
            int n = inst.n;
            vector<bool> used(n, false);
            int start = uniform_int_distribution<int>(0, n - 1)(rng);
            list<int> cyc = {start};
            used[start] = true;

            // Szukanie drugiego wierzchołka
            int nxt = -1, bestd = INT_MAX;
            for(int v = 0; v < n; v++) 
                if(!used[v] && inst.dist[start][v] < bestd) { bestd = inst.dist[start][v]; nxt = v; }
            
            cyc.push_back(nxt);
            used[nxt] = true;

            // Budowanie pełnego cyklu Hamiltona
            while ((int)cyc.size() < n) {
                double bestRG = -1e18;
                int bestV = -1;

                for(int v = 0; v < n; v++) {
                    if(!used[v]) {
                        vector<double> incs;
                        for(auto it = cyc.begin(); it != cyc.end(); ++it) {
                            auto next_it = next(it);
                            if(next_it == cyc.end()) next_it = cyc.begin();
                            incs.push_back(inst.deltaInsert(*it, *next_it, v));
                        }
                        sort(incs.begin(), incs.end());
                        double rg = incs[1] - incs[0];
                        if(rg > bestRG) { bestRG = rg; bestV = v; }
                    }
                }

                double bestInc = 1e18;
                auto bestPos = cyc.begin();
                for(auto it = cyc.begin(); it != cyc.end(); ++it) {
                    auto next_it = next(it);
                    if(next_it == cyc.end()) next_it = cyc.begin();
                    double inc = inst.deltaInsert(*it, *next_it, bestV);
                    if(inc < bestInc) { bestInc = inc; bestPos = next_it; }
                }
                used[bestV] = true;
                cyc.insert(bestPos, bestV);
            }

            Solution sol;
            sol.cycle.assign(cyc.begin(), cyc.end());
            sol.phaseIIRemove(inst); // Redukcja do najbardziej opłacalnego podzbioru

            if (iterationsCount == 0 || sol.objective() > best.objective()) {
                best = sol;
            }
            iterationsCount++;
        }
        return best;
    }
};

// Wrapper dla pojedynczego LS, aby działał przez określony czas
class IteratedLS : public Heuristic {
    LocalSearchWithMoveList ls;
    double time_limit_ms;
public:
    int iterationsCount = 0;
    IteratedLS(double limit) : time_limit_ms(limit) {}

    Solution solve(const Instance& inst) override {
        iterationsCount = 0;
        auto t0 = chrono::steady_clock::now();
        Solution best;
        best.profitSum = -1e9;

        while (chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count() < time_limit_ms) {
            Solution s = ls.solve(inst); // solve() wewnątrz ma RandomSolution + LS
            if (iterationsCount == 0 || s.objective() > best.objective()) best = s;
            iterationsCount++;
        }
        return best;
    }
};

// --- Klasa HAE ---

class HAE : public Heuristic {
    int opType; bool useLS; double timeLimitMs;
    Regret2 repairAlgo; LocalSearchWithMoveList ls;

public:
    int iterationsCount = 0;
    HAE(int op, bool ls_f, double lim) : opType(op), useLS(ls_f), timeLimitMs(lim) {}

    Solution solve(const Instance& inst) override {
        this->iterationsCount = 0;
        vector<Solution> pop;
        auto t0 = chrono::steady_clock::now();

        // 1. Inicjalizacja populacji (Minima lokalne)
        while (pop.size() < 20) {
            Solution s = ls.solve(inst);
            if (isUnique(pop, s)) pop.push_back(s);
        }

        // 2. Pętla ewolucyjna (Steady-state)
        while (chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count() < timeLimitMs) {
            int i = rand() % 20, j;
            do { j = rand() % 20; } while (i == j);

            Solution child = combine(pop[i], pop[j], inst);
            destroyMacro(inst, child);
            // Naprawa metodą Regret (jak w LNS)
            repairAlgo.repair(inst, child);
            
            // Opcjonalne Lokalne Przeszukiwanie
            if (useLS) child = ls.solveFrom(inst, child);
            else child.computeStats(inst);

            this->iterationsCount++;

            // Zastępowanie najgorszego
            int worstIdx = 0;
            for (int k = 1; k < 20; k++) {
                if (pop[k].objective() < pop[worstIdx].objective()) worstIdx = k;
            }

            if (child.objective() > pop[worstIdx].objective() && isUnique(pop, child)) {
                pop[worstIdx] = child;
            }
        }

        Solution best = pop[0];
        for (auto& s : pop) if (s.objective() > best.objective()) best = s;
        return best;
    }

private:
    bool isUnique(const vector<Solution>& pop, const Solution& s) {
        for (const auto& p : pop) {
            if (p.objective() == s.objective()) return false;
        }
        return true;
    }

    // Metoda pomocnicza: łączy rozłączne fragmenty w jeden cykl (losowo)
    void connectSubpaths(vector<vector<int>>& paths, Solution& child) {
        if (paths.empty()) return;
        
        // Losowa kolejność i orientacja podścieżek
        shuffle(paths.begin(), paths.end(), rng);
        for (auto& p : paths) {
            if (rand() % 2) reverse(p.begin(), p.end());
            for (int v : p) child.cycle.push_back(v);
        }
    }

    Solution combine(const Solution& p1, const Solution& p2, const Instance& inst) {
        Solution child;
        
        if (opType == 1) {
            // OPERATOR 1: Wspólne wierzchołki i krawędzie (podścieżki)
            set<pair<int, int>> edges2;
            for (size_t i = 0; i < p2.cycle.size(); i++) {
                int u = p2.cycle[i], v = p2.cycle[(i + 1) % p2.cycle.size()];
                edges2.insert({min(u, v), max(u, v)});
            }

            vector<vector<int>> subpaths;
            vector<int> currentPath;
            for (size_t i = 0; i < p1.cycle.size(); i++) {
                int u = p1.cycle[i], v = p1.cycle[(i + 1) % p1.cycle.size()];
                currentPath.push_back(u);
                if (edges2.find({min(u, v), max(u, v)}) == edges2.end()) {
                    subpaths.push_back(currentPath);
                    currentPath.clear();
                }
            }
            if (!currentPath.empty()) subpaths.push_back(currentPath);
            connectSubpaths(subpaths, child);

        } else if (opType == 2) {
            // OPERATOR 2: Usuwanie krawędzi i wierzchołków nieobecnych w p2
            set<int> v2(p2.cycle.begin(), p2.cycle.end());
            set<pair<int, int>> e2;
            for (size_t i = 0; i < p2.cycle.size(); i++) {
                e2.insert({min(p2.cycle[i], p2.cycle[(i+1)%p2.cycle.size()]), 
                           max(p2.cycle[i], p2.cycle[(i+1)%p2.cycle.size()])});
            }

            vector<vector<int>> subpaths;
            vector<int> currentPath;
            for (size_t i = 0; i < p1.cycle.size(); i++) {
                int u = p1.cycle[i], v = p1.cycle[(i+1)%p1.cycle.size()];
                if (v2.count(u)) {
                    currentPath.push_back(u);
                    // Jeśli krawędź u-v nie istnieje w p2, kończymy podścieżkę
                    if (e2.find({min(u, v), max(u, v)}) == e2.end()) {
                        subpaths.push_back(currentPath);
                        currentPath.clear();
                    }
                }
            }
            connectSubpaths(subpaths, child);

        } else if (opType == 3) {
            // OPERATOR 3: Usuwanie tylko wierzchołków nieobecnych w p2 (zachowanie krawędzi p1)
            set<int> v2(p2.cycle.begin(), p2.cycle.end());
            for (int v : p1.cycle) {
                if (v2.count(v)) child.cycle.push_back(v);
            }
        }

        // Zabezpieczenie przed pustym rozwiązaniem
        if (child.cycle.empty()) {
            child.cycle.push_back(p1.cycle[rand() % p1.cycle.size()]);
        }
        return child;
    }
    void destroyMacro(const Instance& inst, Solution& sol) {
        int n_cyc = (int)sol.cycle.size();
        int to_remove = max(2, (int)(n_cyc * uniform_real_distribution<double>(0.15, 0.45)(rng)));

        int num_segments = 3;
        int per_segment = to_remove / num_segments;

        vector<bool> keep(inst.n, true);
        vector<bool> removed_pos(n_cyc, false);
        int total_removed = 0;

        for (int s = 0; s < num_segments && total_removed < to_remove; s++) {
            vector<int> candidates;
            for (int i = 0; i < n_cyc; i++)
                if (!removed_pos[i]) candidates.push_back(i);
            if (candidates.empty()) break;

            int start_pos = candidates[uniform_int_distribution<int>(0, (int)candidates.size()-1)(rng)];
            int start_v = sol.cycle[start_pos];

            // Shaw: posortuj pozostałe nieusunięte wierzchołki po podobieństwie do start_v
            vector<pair<double, int>> similarity;
            similarity.reserve(n_cyc - total_removed - 1);
            for (int i = 0; i < n_cyc; i++) {
                int v = sol.cycle[i];
                if (!keep[v]) continue; // już usunięty
                if (v == start_v) continue;
                double sim = inst.dist[start_v][v] - abs(inst.profit[start_v] - inst.profit[v]);
                similarity.push_back({sim, v});
            }
            sort(similarity.begin(), similarity.end());

            // usuń start_v + per_segment-1 najbardziej podobnych
            keep[start_v] = false;
            removed_pos[start_pos] = true;
            total_removed++;

            for (int i = 0; i < per_segment - 1 && i < (int)similarity.size() && total_removed < to_remove; i++) {
                int v = similarity[i].second;
                keep[v] = false;
                // oznacz pozycję w cyklu
                for (int j = 0; j < n_cyc; j++)
                    if (sol.cycle[j] == v) { removed_pos[j] = true; break; }
                total_removed++;
            }
        }

        vector<int> new_cyc;
        new_cyc.reserve(n_cyc - total_removed);
        for (int v : sol.cycle) if (keep[v]) new_cyc.push_back(v);
        sol.cycle = new_cyc;
        sol.computeStats(inst);
    }
};

// --- Inne metody i Benchmark ---

class MSLS : public Heuristic {
    LocalSearchWithMoveList ls;
    RandomSolution rndSol;
public:
    Solution solve(const Instance& inst) override {
        Solution bestSol;
        bestSol.profitSum = -1e9;

        for (int i = 0; i < 200; i++) {
            Solution current = ls.solveFrom(inst, rndSol.solve(inst));
            if (i == 0 || current.objective() > bestSol.objective()) bestSol = current;
        }
        return bestSol;
    }
};

class ILS : public Heuristic {
    LocalSearchWithMoveList ls;
    RandomSolution rndSol;
    double time_limit_ms;
public:
    int iterationsCount = 0;
    ILS(double limit) : time_limit_ms(limit) {}

    Solution solve(const Instance& inst) override {
        iterationsCount = 0;
        auto t0 = chrono::steady_clock::now();

        Solution x = ls.solveFrom(inst, rndSol.solve(inst));
        Solution best = x;

        while (true) {
            auto t1 = chrono::steady_clock::now();
            if (chrono::duration<double, milli>(t1 - t0).count() >= time_limit_ms) break;

            Solution y = x;
            perturbMicro(inst, y);
            y = ls.solveFrom(inst, y);
            iterationsCount++;

            if (y.objective() >= x.objective()) {
                x = y;
                if (x.objective() > best.objective()) best = x;
            }
        }
        return best;
    }

private:
    void perturbMicro(const Instance& inst, Solution& sol) {
        vector<int>& cyc = sol.cycle;
        int n = (int)cyc.size();
        if (n < 8) return;

        // double-bridge: wybierz 4 losowe pozycje i przestaw segmenty A+C+B+D
        vector<int> pos(4);
        do {
            for (int& p : pos) p = uniform_int_distribution<int>(0, n - 1)(rng);
            sort(pos.begin(), pos.end());
        } while (pos[0] == pos[1] || pos[1] == pos[2] || pos[2] == pos[3]);

        int i = pos[0], j = pos[1], k = pos[2];
        vector<int> newcyc;
        newcyc.reserve(n);

        // A: [0..i], C: [j+1..k], B: [i+1..j], D: [k+1..n-1]
        for (int x = 0;   x <= i; x++) newcyc.push_back(cyc[x]);
        for (int x = j+1; x <= k; x++) newcyc.push_back(cyc[x]);
        for (int x = i+1; x <= j; x++) newcyc.push_back(cyc[x]);
        for (int x = k+1; x <  n; x++) newcyc.push_back(cyc[x]);

        cyc = newcyc;
        sol.computeStats(inst);
    }
};

class LNS : public Heuristic {
    LocalSearchWithMoveList ls;
    Regret2 repairAlgo;
    RandomSolution rndSol;
    double time_limit_ms;
    bool use_ls;
public:
    int iterationsCount = 0;
    LNS(double limit, bool ls_flag) : time_limit_ms(limit), use_ls(ls_flag) {}

    Solution solve(const Instance& inst) override {
        iterationsCount = 0;
        auto t0 = chrono::steady_clock::now();

        Solution x = ls.solveFrom(inst, rndSol.solve(inst));
        Solution best = x;

        while (true) {
            auto t1 = chrono::steady_clock::now();
            if (chrono::duration<double, milli>(t1 - t0).count() >= time_limit_ms) break;

            Solution y = x;
            destroyMacro(inst, y);
            repairAlgo.repair(inst, y);
            if (use_ls) y = ls.solveFrom(inst, y);
            iterationsCount++;

        if (y.objective() >= x.objective()) { 
            x = y;
            if (x.objective() > best.objective()) best = x;
        }
        }
        return best;
    }

private:
    void destroyMacro(const Instance& inst, Solution& sol) {
        int n_cyc = (int)sol.cycle.size();
        int to_remove = max(2, (int)(n_cyc * uniform_real_distribution<double>(0.15, 0.45)(rng)));

        int num_segments = 3;
        int per_segment = to_remove / num_segments;

        vector<bool> keep(inst.n, true);
        vector<bool> removed_pos(n_cyc, false);
        int total_removed = 0;

        for (int s = 0; s < num_segments && total_removed < to_remove; s++) {
            vector<int> candidates;
            for (int i = 0; i < n_cyc; i++)
                if (!removed_pos[i]) candidates.push_back(i);
            if (candidates.empty()) break;

            int start_pos = candidates[uniform_int_distribution<int>(0, (int)candidates.size()-1)(rng)];
            int start_v = sol.cycle[start_pos];

            // Shaw: posortuj pozostałe nieusunięte wierzchołki po podobieństwie do start_v
            vector<pair<double, int>> similarity;
            similarity.reserve(n_cyc - total_removed - 1);
            for (int i = 0; i < n_cyc; i++) {
                int v = sol.cycle[i];
                if (!keep[v]) continue; // już usunięty
                if (v == start_v) continue;
                double sim = inst.dist[start_v][v] - abs(inst.profit[start_v] - inst.profit[v]);
                similarity.push_back({sim, v});
            }
            sort(similarity.begin(), similarity.end());

            // usuń start_v + per_segment-1 najbardziej podobnych
            keep[start_v] = false;
            removed_pos[start_pos] = true;
            total_removed++;

            for (int i = 0; i < per_segment - 1 && i < (int)similarity.size() && total_removed < to_remove; i++) {
                int v = similarity[i].second;
                keep[v] = false;
                // oznacz pozycję w cyklu
                for (int j = 0; j < n_cyc; j++)
                    if (sol.cycle[j] == v) { removed_pos[j] = true; break; }
                total_removed++;
            }
        }

        vector<int> new_cyc;
        new_cyc.reserve(n_cyc - total_removed);
        for (int v : sol.cycle) if (keep[v]) new_cyc.push_back(v);
        sol.cycle = new_cyc;
        sol.computeStats(inst);
    }
};


// --- System zapisywania i Main ---

void makeDir(const string& path) {
#ifdef _WIN32
    system(("mkdir \"" + path + "\" 2>nul").c_str());
#else
    system(("mkdir -p \"" + path + "\"").c_str());
#endif
}

void saveBest(const string& path, const Solution& sol) {
    ofstream f(path);
    f << "objective=" << sol.objective() << "\nprofit=" << sol.profitSum << "\nlength=" << sol.length << "\ncount=" << sol.cycle.size() << "\ncycle:\n";
    for (int v : sol.cycle) f << v << "\n";
}

void saveAllCSV(const string& path, const vector<Solution>& sols, const vector<double>& times) {
    ofstream f(path);
    f << "rep,objective,profit,length,time_ms,cycle\n";
    for (size_t i = 0; i < sols.size(); i++) {
        f << i << "," << sols[i].objective() << "," << sols[i].profitSum << "," << sols[i].length << "," << (int)(times[i] * 1000) << ",\"";
        for (size_t j = 0; j < sols[i].cycle.size(); j++) { f << sols[i].cycle[j]; if (j + 1 < sols[i].cycle.size()) f << " "; }
        f << "\"\n";
    }
}

struct BenchmarkResult {
    double avgObj = 0.0;
    int minObj = 0, maxObj = 0;
    double avgTimeMs = 0.0, minTimeMs = 0.0, maxTimeMs = 0.0;
    double avgIter = 0.0;
    int minIter = 0, maxIter = 0;
    Solution best;
    vector<Solution> allSols;
    vector<double> timesSeconds;
};

BenchmarkResult runBenchmark(Heuristic& solver, const Instance& inst, int repetitions = 20) {
    BenchmarkResult result;
    vector<int> objs; objs.reserve(repetitions);
    vector<int> iters;
    result.timesSeconds.reserve(repetitions);
    result.allSols.reserve(repetitions);
    bool hasBest = false;

    for (int rep = 0; rep < repetitions; rep++) {
        auto t0 = chrono::steady_clock::now();
        Solution s = solver.solve(inst);
        double dt = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

        objs.push_back(s.objective());
        result.timesSeconds.push_back(dt);
        result.allSols.push_back(s);

        if (auto* h = dynamic_cast<HAE*>(&solver))
            iters.push_back(h->iterationsCount);
        else if (auto* ils = dynamic_cast<ILS*>(&solver))
            iters.push_back(ils->iterationsCount);
        else if (auto* lns = dynamic_cast<LNS*>(&solver))
            iters.push_back(lns->iterationsCount);
        else if (auto* ir = dynamic_cast<IteratedRegret*>(&solver))
    iters.push_back(ir->iterationsCount);
        else if (auto* ils_wrap = dynamic_cast<IteratedLS*>(&solver))
            iters.push_back(ils_wrap->iterationsCount);

        if (!hasBest || s.objective() > result.best.objective()) {
            result.best = s; hasBest = true;
        }
    }

    result.avgObj = accumulate(objs.begin(), objs.end(), 0.0) / objs.size();
    result.minObj = *min_element(objs.begin(), objs.end());
    result.maxObj = *max_element(objs.begin(), objs.end());

    result.avgTimeMs = 1000.0 * accumulate(result.timesSeconds.begin(), result.timesSeconds.end(), 0.0) / result.timesSeconds.size();
    result.minTimeMs = 1000.0 * (*min_element(result.timesSeconds.begin(), result.timesSeconds.end()));
    result.maxTimeMs = 1000.0 * (*max_element(result.timesSeconds.begin(), result.timesSeconds.end()));

    if (!iters.empty()) {
        result.avgIter = accumulate(iters.begin(), iters.end(), 0.0) / iters.size();
        result.minIter = *min_element(iters.begin(), iters.end());
        result.maxIter = *max_element(iters.begin(), iters.end());
    }

    return result;
}


int main() {
    vector<Instance> insts = { Instance::loadFromCSV("TSPA.csv"), Instance::loadFromCSV("TSPB.csv") };
    vector<string> tags = {"A", "B"};
    makeDir("output");

    for (size_t ii = 0; ii < insts.size(); ii++) {
        const auto& inst = insts[ii]; string tag = tags[ii]; string base = "output/" + tag;
        makeDir(base); makeDir(base + "/solutions"); makeDir(base + "/solutions_all");
        ofstream stats(base + "/stats.csv");
        stats << "method,avg_obj,min_obj,max_obj,avg_time_ms,avg_iter,min_iter,max_iter\n";

        auto run = [&](Heuristic& s, string name) {
            BenchmarkResult r = runBenchmark(s, inst);
            stats << name << "," << r.avgObj << "," << r.minObj << "," << r.maxObj << "," << r.avgTimeMs << "," << r.avgIter << "," << r.minIter << "," << r.maxIter << "\n";
            saveBest(base + "/solutions/" + name + ".txt", r.best);
            saveAllCSV(base + "/solutions_all/" + name + ".csv", r.allSols, r.timesSeconds);
            cout << "[" << tag << "] " << name << " done.\n";
            return r.avgTimeMs;
        };
        
        
        MSLS msls; double limit = run(msls, "MSLS");
        ILS ils(limit); run(ils, "ILS");
        LNS lns(limit, true); run(lns, "LNS");
        LNS lnsa(limit, false); run(lnsa, "LNSa");

        HAE h1(1, true, limit); run(h1, "HAE_Op1_LS");
        HAE h2(2, true, limit); run(h2, "HAE_Op2_LS");
        HAE h2n(2, false, limit); run(h2n, "HAE_Op2_noLS");
        HAE h3(3, true, limit); run(h3, "HAE_Op3_LS");
        HAE h3n(3, false, limit); run(h3n, "HAE_Op3_noLS");
        IteratedRegret ir(limit);
        run(ir, "Regret2");

        IteratedLS ils_wrap(limit);
        run(ils_wrap, "LS_MOVE_LIST");
    }
    return 0;
}