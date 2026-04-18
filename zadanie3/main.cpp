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
#include <queue>
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
        length = inst.cycleLength(cycle);
        profitSum = inst.cycleProfit(cycle);
    }

    void phaseIIRemove(const Instance& inst, size_t min_size=2) {
        list<int> cyc(cycle.begin(), cycle.end());
        bool changed=true;
        while(changed && cyc.size()>min_size) {
            changed=false; int bestGain=0; auto bestIt=cyc.begin();
            for(auto it=cyc.begin(); it!=cyc.end(); ++it) {
                auto prev_it = (it==cyc.begin()? prev(cyc.end()):prev(it));
                auto next_it = next(it); if(next_it==cyc.end()) next_it=cyc.begin();
                int delta = inst.deltaRemove(*prev_it,*it,*next_it);
                if(delta>bestGain){ bestGain=delta; bestIt=it; }
            }
            if(bestGain>0){ cyc.erase(bestIt); changed=true; }
        }
        cycle.assign(cyc.begin(),cyc.end());
        computeStats(inst);
    }
};

class Heuristic { public: virtual Solution solve(const Instance& inst)=0; virtual ~Heuristic(){} };

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
        int n=inst.n; vector<bool> used(n,false);
        int start = uniform_int_distribution<int>(0, n-1)(rng);
        list<int> cyc={start}; used[start]=true;
        int nxt=-1;
        int bestd=INT_MAX;
        for(int v=0;v<n;v++) if(!used[v] && inst.dist[start][v]<bestd){ bestd=inst.dist[start][v]; nxt=v; }
        cyc.push_back(nxt); used[nxt]=true;
        while((int)cyc.size()<n){
            double bestRG=-1e18; int bestV=-1; auto bestPos=cyc.begin();
            for(int v=0;v<n;v++) if(!used[v]){
                vector<double> incs;
                for(auto it=cyc.begin();it!=cyc.end();++it){
                    auto next_it=next(it); if(next_it==cyc.end()) next_it=cyc.begin();
                    incs.push_back(inst.deltaInsert(*it,*next_it,v));
                }
                sort(incs.begin(),incs.end());
                if(incs.size()<2) continue;
                double rg=incs[1]-incs[0]; if(weighted) rg+=w*(-incs[0]);
                if(rg>bestRG){ bestRG=rg; bestV=v; }
            }
            double bestInc=1e18;
            for(auto it=cyc.begin();it!=cyc.end();++it){
                auto next_it=next(it); if(next_it==cyc.end()) next_it=cyc.begin();
                double inc=inst.deltaInsert(*it,*next_it,bestV);
                if(inc<bestInc){ bestInc=inc; bestPos=next_it; }
            }
            used[bestV]=true; cyc.insert(bestPos,bestV);
        }
        Solution sol; sol.cycle.assign(cyc.begin(),cyc.end());
        sol.computeStats(inst); sol.lengthPhase1=sol.length;
        sol.phaseIIRemove(inst);
        return sol;
    }
};



class LocalSearch : public Heuristic {
    NeighType neighType;
    LSMode    mode;
    bool      startRandom;
    RandomSolution rndSol;
    Regret2 reg2Sol;

public:
    LocalSearch(NeighType nt, LSMode lm, bool sr)
        : neighType(nt), mode(lm), startRandom(sr) {}

    Solution solve(const Instance& inst) override {
        Solution sol = startRandom ? rndSol.solve(inst) : reg2Sol.solve(inst);
        localSearch(inst, sol);
        return sol;
    }

private:

    void applyAdd(Solution& sol, const Instance& inst,
                vector<int>& cyc, int v, int pos) {
        int n = cyc.size();
        int prev_v = cyc[(pos - 1 + n) % n];
        int next_v = cyc[pos % n];
        sol.length    += inst.deltaInsert(prev_v, next_v, v);
        sol.profitSum += inst.profit[v];
        cyc.insert(cyc.begin() + pos, v);
    }

    int deltaRem(const Instance& inst, const vector<int>& cyc, int idx) const {
        int n = cyc.size();
        int prev_v = cyc[(idx-1+n)%n];
        int next_v = cyc[(idx+1)%n];
        int k = cyc[idx];
        return inst.deltaRemove(prev_v, k, next_v);
    }

    void applyRemove(Solution& sol, const Instance& inst,
                    vector<int>& cyc, int idx) {
        int n = cyc.size();
        int prev_v = cyc[(idx - 1 + n) % n];
        int next_v = cyc[(idx + 1) % n];
        int k    = cyc[idx];
        sol.length    -= (inst.dist[prev_v][k] + inst.dist[k][next_v] - inst.dist[prev_v][next_v]);
        sol.profitSum -= inst.profit[k];
        cyc.erase(cyc.begin() + idx);
    }

  int deltaVertexSwap(const Instance& inst, const vector<int>& cyc, int i, int j) const {
    int n = cyc.size();
    if(n < 3 || i == j) return 0;

    int im1 = (i - 1 + n) % n;
    int ip1 = (i + 1) % n;
    int jm1 = (j - 1 + n) % n;
    int jp1 = (j + 1) % n;

    int vi = cyc[i];
    int vj = cyc[j];

    // sąsiedzi (i przed j)
    if(ip1 == j) {
        int oldCost = inst.dist[cyc[im1]][vi] + inst.dist[vi][vj] + inst.dist[vj][cyc[jp1]];
        int newCost = inst.dist[cyc[im1]][vj] + inst.dist[vj][vi] + inst.dist[vi][cyc[jp1]];
        return oldCost - newCost;
    }

    // sąsiedzi cykliczni (j przed i)
    if(jp1 == i) {
        int oldCost = inst.dist[cyc[jm1]][vj] + inst.dist[vj][vi] + inst.dist[vi][cyc[ip1]];
        int newCost = inst.dist[cyc[jm1]][vi] + inst.dist[vi][vj] + inst.dist[vj][cyc[ip1]];
        return oldCost - newCost;
    }

    // ogólny przypadek
    int oldCost = inst.dist[cyc[im1]][vi] + inst.dist[vi][cyc[ip1]]
                + inst.dist[cyc[jm1]][vj] + inst.dist[vj][cyc[jp1]];

    int newCost = inst.dist[cyc[im1]][vj] + inst.dist[vj][cyc[ip1]]
                + inst.dist[cyc[jm1]][vi] + inst.dist[vi][cyc[jp1]];

    return oldCost - newCost;
}
    void applyVertexSwap(Solution& sol, const Instance& inst,
                        vector<int>& cyc, int i, int j) {
        sol.length -= deltaVertexSwap(inst, cyc, i, j);
        swap(cyc[i], cyc[j]);
    }

    int deltaEdgeSwap(const Instance& inst, const vector<int>& cyc, int i, int j) const {
        int n = cyc.size();
        int a = cyc[i], b = cyc[(i+1)%n];
        int c = cyc[j], d = cyc[(j+1)%n];
        return inst.dist[a][b] + inst.dist[c][d]
             - inst.dist[a][c] - inst.dist[b][d];
    }

    void applyEdgeSwap(Solution& sol, const Instance& inst,
                    vector<int>& cyc, int i, int j) {
        sol.length -= deltaEdgeSwap(inst, cyc, i, j);
        int n = cyc.size();
        int l = (i+1)%n, r = j;
        while(l < r) { swap(cyc[l], cyc[r]); l++; r--; }
    }

    void localSearch(const Instance& inst, Solution& sol) {
        if(mode == LSMode::STEEPEST)
            steepest(inst, sol);
        else
            greedy(inst, sol);
        sol.computeStats(inst);
    }

    void steepest(const Instance& inst, Solution& sol) {
        bool improved = true;
        while(improved) {
            
            vector<int>& cyc = sol.cycle;
            improved = false;
            int bestDelta = 0;
            int bestType = -1;
            int bestI = -1, bestJ = -1, bestV = -1, bestPos = -1;


            int n = cyc.size();
            vector<bool> inCycle(inst.n, false);
            for(int v : cyc) inCycle[v] = true;

            // ADD: przegladamy wszystkie wierzcholki spoza cyklu i wszystkie pozycje wstawienia
            for(int v = 0; v < inst.n; v++) {
                if(inCycle[v]) continue;
                for(int i = 0; i < n; i++) {
                    int j = (i+1)%n;
                    int d = inst.profit[v] - inst.deltaInsert(cyc[i], cyc[j], v);
                    if(d > bestDelta) {
                        bestDelta = d; bestType = 0;
                        bestV = v; bestPos = j;
                    }
                }
            }

            // REMOVE: przegladamy wszystkie wierzcholki w cyklu
            if(n > 2) {
                for(int idx = 0; idx < n; idx++) {
                    int d = deltaRem(inst, cyc, idx);
                    if(d > bestDelta) {
                        bestDelta = d; bestType = 1; bestI = idx;
                    }
                }
            }

            // INTRA: przegladamy wszystkie pary (wg wybranego typu sasiedztwa)
            if(neighType == NeighType::VERTEX_SWAP) {
                for(int i = 0; i < n-1; i++)
                    for(int j = i+1; j < n; j++) {
                        int d = deltaVertexSwap(inst, cyc, i, j);
                        if(d > bestDelta) {
                            bestDelta = d; bestType = 2; bestI = i; bestJ = j;
                        }
                        
                    }
            } else {
                for(int i = 0; i < n-1; i++)
                    for(int j = i+2; j < n; j++) {
                        if(i==0 && j==n-1) continue;
                        int d = deltaEdgeSwap(inst, cyc, i, j);
                        if(d > bestDelta) {
                            bestDelta = d; bestType = 3; bestI = i; bestJ = j;
                        }
                    }
            }

            if(bestDelta > 0) {
                improved = true;
                if(bestType == 0)      applyAdd(sol, inst, cyc, bestV, bestPos);
                else if(bestType == 1) applyRemove(sol, inst, cyc, bestI);
                else if(bestType == 2) applyVertexSwap(sol, inst, cyc, bestI, bestJ);
                else if(bestType == 3) applyEdgeSwap(sol, inst, cyc, bestI, bestJ);
            }
        }
    }
void greedy(const Instance& inst, Solution& sol) {
    bool improved = true;

    while (improved) {
        improved = false;

        vector<int>& cyc = sol.cycle;
        int n = cyc.size();

        vector<bool> inCycle(inst.n, false);
        for (int v : cyc) inCycle[v] = true;

        vector<int> outside;
        for (int v = 0; v < inst.n; v++) if (!inCycle[v]) outside.push_back(v);
        shuffle(outside.begin(), outside.end(), rng);


        vector<int> cycleIndices(n);
        iota(cycleIndices.begin(), cycleIndices.end(), 0);
        shuffle(cycleIndices.begin(), cycleIndices.end(), rng);


        vector<int> moves = {0, 1, 2};
        shuffle(moves.begin(), moves.end(), rng);

        for (int move : moves) {
            if (improved) break;

            if (move == 0) { // ADD
                for (int v : outside) {
                    if (improved) break;
                    for (int i : cycleIndices) {
                        int j = (i + 1) % n;
                        int d = inst.profit[v] - inst.deltaInsert(cyc[i], cyc[j], v);
                        if (d > 0) { 
                            applyAdd(sol, inst, cyc, v, j); 
                            improved = true; 
                            break; 
                        }
                    }
                }
            }
            else if (move == 1 && n > 2) { // REMOVE
                for (int idx : cycleIndices) {
                    int d = deltaRem(inst, cyc, idx);
                    if (d > 0) { 
                        applyRemove(sol, inst, cyc, idx); 
                        improved = true; 
                        break; 
                    }
                }
            }
            else if (move == 2) { // INTRA
                for (int i : cycleIndices) {
                    if (improved) break;
                    // Ponieważ i oraz j biorą się z przetasowanej listy, 
                    // sprawdzamy pary w całkowicie losowej kolejności
                    for (int j : cycleIndices) {
                        if (i >= j) continue; // Zapewnia brak duplikatów (a < b) i omija i == j

                        if (neighType == NeighType::VERTEX_SWAP) {
                            int d = deltaVertexSwap(inst, cyc, i, j);
                            if (d > 0) { 
                                applyVertexSwap(sol, inst, cyc, i, j); 
                                improved = true; 
                                break; 
                            }
                        } else {
                            if (i == 0 && j == n - 1) continue;
                            if (j < i + 2) continue; // Ignorujemy sąsiadujące krawędzie
                            int d = deltaEdgeSwap(inst, cyc, i, j);
                            if (d > 0) { 
                                applyEdgeSwap(sol, inst, cyc, i, j); 
                                improved = true; 
                                break; 
                            }
                        }
                    }
                }
            }
        }
    }
}
};






//MOVE LIST
class LocalSearchWithMoveList : public Heuristic {
    RandomSolution rndSol;

    enum class MoveType {
        ADD,        // dodanie wierzchołka spoza cyklu
        REMOVE,     // usunięcie wierzchołka z cyklu
        EDGE_SWAP   // ruch wewnątrztrasowy typu 2-opt (wymiana dwóch krawędzi)
    };

    // - ADD:       a = left,  b = right, c = v
    // - REMOVE:    a = left,  b = v,     c = right
    // - EDGE_SWAP: a = a,     b = b,     c = c, d = d
    struct Move {
        int delta;
        MoveType type;
        int a, b, c, d;
    };

    enum class EdgeSwapApplicability {
        NOT_APPLICABLE, // przynajmniej jedna usuwana krawędź już nie występuje
        FORWARD,        // obie usuwane krawędzie występują w zapamiętanym kierunku
        REVERSED        // obie usuwane krawędzie występują jednocześnie odwrócone
    };

    // Lista ruchów poprawiających, posortowana od najgorszego do najlepszego.
    // Dzięki temu najlepszy ruch jest na końcu i można go pobierać przez pop_back().
    vector<Move> moveList;

public:
    LocalSearchWithMoveList() {}

    Solution solve(const Instance& inst) override {
        Solution sol = rndSol.solve(inst);
        if (sol.cycle.empty()) {
            sol.computeStats(inst);
            return sol;
        }

        // Reprezentacja cyklu:
        // next[v]    - następnik wierzchołka v w cyklu
        // prev[v]    - poprzednik wierzchołka v w cyklu
        // inCycle[v] - czy wierzchołek v należy do cyklu
        vector<int> next(inst.n, -1);
        vector<int> prev(inst.n, -1);
        vector<char> inCycle(inst.n, 0);

        buildLinkedCycle(sol.cycle, next, prev, inCycle);
        int startVertex = sol.cycle[0];
        int cycleSize = (int)sol.cycle.size();
        buildImprovingMoveList(inst, next, prev, inCycle, startVertex, cycleSize, moveList);

        // Alternatywna wersja z wykładu:
        // 1. Dopóki istnieje aplikowalny ruch na liście LM, próbujemy go znaleźć.
        // 2. Jeżeli lista się wyczerpie, budujemy ją ponownie od zera.
        // 3. Kończymy, gdy po pełnym przeglądzie nowa LM jest pusta.
        while (true) {
            bool applied = false;

            while (!moveList.empty()) {
                Move m = moveList.back();
                moveList.pop_back();

                // Ruch nie jest już aplikowalny do bieżącego rozwiązania
                if (!isApplicable(m, next, prev, inCycle)) {
                    continue;
                }

                // Znaleziono ruch aplikowalny
                applyMove(m, inst, next, prev, inCycle, startVertex, cycleSize);
                applied = true;
                break;
            }

            // Jeżeli nie znaleziono żadnego ruchu aplikowalnego,
            // budujemy nową listę ruchów dla bieżącego rozwiązania
            if (!applied) {
                buildImprovingMoveList(inst, next, prev, inCycle, startVertex, cycleSize, moveList);

                // Brak ruchów poprawiających => osiągnięto lokalne optimum
                if (moveList.empty()) {
                    break;
                }
            }
        }

        sol.cycle = materializeCycle(startVertex, next, cycleSize);
        sol.computeStats(inst);

        return sol;
    }

private:
    // Buduje reprezentację next / prev / inCycle na podstawie cyklu
    // zapisanego jako wektor.
    void buildLinkedCycle(const vector<int>& cyc,
                          vector<int>& next,
                          vector<int>& prev,
                          vector<char>& inCycle) {
        fill(next.begin(), next.end(), -1);
        fill(prev.begin(), prev.end(), -1);
        fill(inCycle.begin(), inCycle.end(), 0);

        int n = (int)cyc.size();
        for (int i = 0; i < n; i++) {
            int v = cyc[i];
            int vn = cyc[(i + 1) % n];
            int vp = cyc[(i - 1 + n) % n];

            next[v] = vn;
            prev[v] = vp;
            inCycle[v] = 1;
        }
    }

    vector<int> materializeCycle(int startVertex,
                                 const vector<int>& next,
                                 int cycleSize) {
        vector<int> cyc;
        cyc.reserve(cycleSize);

        int v = startVertex;
        for (int step = 0; step < cycleSize; step++) {
            cyc.push_back(v);
            v = next[v];
        }

        return cyc;
    }

    void rebuildLinkedCycleFromLinearOrder(const vector<int>& cyc,
                                           vector<int>& next,
                                           vector<int>& prev) {
        int n = (int)cyc.size();
        for (int i = 0; i < n; i++) {
            int v = cyc[i];
            next[v] = cyc[(i + 1) % n];
            prev[v] = cyc[(i - 1 + n) % n];
        }
    }

    // Wstawienie v pomiędzy left i right.
    // Przed operacją:
    // left -> right
    // Po operacji:
    // left -> v -> right
    void insertBetween(int left, int right, int v,
                       vector<int>& next,
                       vector<int>& prev,
                       vector<char>& inCycle) {
        next[left] = v;
        prev[v] = left;

        next[v] = right;
        prev[right] = v;

        inCycle[v] = 1;
    }

    // Usunięcie v z cyklu.
    // Przed operacją:
    // left -> v -> right
    // Po operacji:
    // left -> right
    void removeVertex(int v,
                      vector<int>& next,
                      vector<int>& prev,
                      vector<char>& inCycle) {
        int left = prev[v];
        int right = next[v];

        next[left] = right;
        prev[right] = left;

        next[v] = -1;
        prev[v] = -1;
        inCycle[v] = 0;
    }

    // Odwrócenie fragmentu cyklu w linearyzacji.
    // Obsługuje również przypadek zawijania przez koniec wektora.
    void reverseSubpath(vector<int>& cyc, int start, int end) {
        int n = (int)cyc.size();

        if (start <= end) {
            reverse(cyc.begin() + start, cyc.begin() + end + 1);
        } else {
            vector<int> temp;
            temp.reserve(n - start + end + 1);

            for (int k = start; k < n; k++) temp.push_back(cyc[k]);
            for (int k = 0; k <= end; k++) temp.push_back(cyc[k]);

            reverse(temp.begin(), temp.end());

            int idx = 0;
            for (int k = start; k < n; k++) cyc[k] = temp[idx++];
            for (int k = 0; k <= end; k++) cyc[k] = temp[idx++];
        }
    }

    // Buduje listę wszystkich ruchów poprawiających aplikowalnych do bieżącego rozwiązania.
    // Lista jest sortowana rosnąco po delta, aby najlepszy ruch znajdował się na końcu
    void buildImprovingMoveList(const Instance& inst,
                                const vector<int>& next,
                                const vector<int>& prev,
                                const vector<char>& inCycle,
                                int startVertex,
                                int cycleSize,
                                vector<Move>& lm) {
        lm.clear();

        vector<int> cyc = materializeCycle(startVertex, next, cycleSize);
        int n = (int)cyc.size();

        vector<int> position(inst.n, -1);
        for (int i = 0; i < n; i++) {
            position[cyc[i]] = i;
        }

        // RUCHY MIĘDZYTRASOWE: ADD
        // Dla każdego wierzchołka spoza cyklu sprawdzamy wszystkie możliwe pozycje wstawienia.
        vector<int> outside;
        outside.reserve(inst.n - n);

        for (int v = 0; v < inst.n; v++) {
            if (!inCycle[v]) outside.push_back(v);
        }

        for (int i = 0; i < n; i++) {
            int left = cyc[i];
            int right = cyc[(i + 1) % n];

            for (int v : outside) {
                int delta = inst.profit[v] - inst.deltaInsert(left, right, v);
                if (delta > 0) {
                    lm.push_back({delta, MoveType::ADD, left, right, v, -1});
                }
            }
        }

        // RUCHY MIĘDZYTRASOWE: REMOVE
        if (n > 2) {
            for (int idx = 0; idx < n; idx++) {
                int left = cyc[(idx - 1 + n) % n];
                int v = cyc[idx];
                int right = cyc[(idx + 1) % n];

                int delta = inst.deltaRemove(left, v, right);
                if (delta > 0) {
                    lm.push_back({delta, MoveType::REMOVE, left, v, right, -1});
                }
            }
        }

        // RUCHY WEWNĄTRZTRASOWE: EDGE_SWAP (2-opt)
        // Używamy sąsiedztwa wymiany dwóch krawędzi.
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 2; j < n; j++) {
                // Pomijamy krawędzie sąsiednie cyklicznie
                if (i == 0 && j == n - 1) continue;

                int a = cyc[i];
                int b = cyc[(i + 1) % n];
                int c = cyc[j];
                int d = cyc[(j + 1) % n];

                int delta =
                    inst.dist[a][b] + inst.dist[c][d] -
                    inst.dist[a][c] - inst.dist[b][d];

                if (delta > 0) {
                    lm.push_back({delta, MoveType::EDGE_SWAP, a, b, c, d});
                }
            }
        }

        // Najlepszy ruch będzie na końcu
        sort(lm.begin(), lm.end(),
             [](const Move& lhs, const Move& rhs) {
                 if (lhs.delta != rhs.delta) return lhs.delta < rhs.delta;
                 return (int)lhs.type < (int)rhs.type;
             });
    }

    // Sprawdzenie aplikowalności ruchu.
    //
    // Dla EDGE_SWAP dopuszczamy trzy sytuacje:
    // - NOT_APPLICABLE:
    //   przynajmniej jednej z usuwanych krawędzi nie ma już w rozwiązaniu
    // - FORWARD:
    //   obie usuwane krawędzie występują dokładnie w zapisanym kierunku
    // - REVERSED:
    //   obie usuwane krawędzie występują jednocześnie w kierunku odwróconym
    bool isApplicable(const Move& m,
                      const vector<int>& next,
                      const vector<int>& prev,
                      const vector<char>& inCycle) const {
        if (m.type == MoveType::ADD) {
            int left = m.a;
            int right = m.b;
            int v = m.c;

            return (!inCycle[v] && next[left] == right);
        }

        if (m.type == MoveType::REMOVE) {
            int left = m.a;
            int v = m.b;
            int right = m.c;

            return (inCycle[v] && prev[v] == left && next[v] == right);
        }

        return getEdgeSwapApplicability(m, next) != EdgeSwapApplicability::NOT_APPLICABLE;
    }

    // Określa, czy EDGE_SWAP jest aplikowalny:
    // - FORWARD  : krawędzie (a,b) i (c,d) istnieją w tym kierunku
    // - REVERSED : krawędzie istnieją obie odwrócone: (b,a) i (d,c)
    // - NOT_APPLICABLE : w przeciwnym razie
    EdgeSwapApplicability getEdgeSwapApplicability(const Move& m,
                                                   const vector<int>& next) const {
        int a = m.a, b = m.b, c = m.c, d = m.d;

        if (next[a] == b && next[c] == d) {
            return EdgeSwapApplicability::FORWARD;
        }

        if (next[b] == a && next[d] == c) {
            return EdgeSwapApplicability::REVERSED;
        }

        return EdgeSwapApplicability::NOT_APPLICABLE;
    }

    void applyMove(const Move& m,
                   const Instance& inst,
                   vector<int>& next,
                   vector<int>& prev,
                   vector<char>& inCycle,
                   int& startVertex,
                   int& cycleSize) {
        if (m.type == MoveType::ADD) {
            int left = m.a;
            int right = m.b;
            int v = m.c;

            insertBetween(left, right, v, next, prev, inCycle);
            cycleSize++;
            return;
        }

        if (m.type == MoveType::REMOVE) {
            int v = m.b;
            int right = m.c;

            removeVertex(v, next, prev, inCycle);
            cycleSize--;

            // Jeśli usunięto wierzchołek startowy, przechodzimy na jego następnika
            if (v == startVertex) {
                startVertex = right;
            }
            return;
        }

        applyEdgeSwapMove(m, next, prev, startVertex, cycleSize);
    }

    // Aplikacja ruchu EDGE_SWAP.
    //
    // Operujemy na chwilowej linearyzacji cyklu, wykonujemy odwrócenie
    // odpowiedniego fragmentu, a następnie odbudowujemy next / prev.
    void applyEdgeSwapMove(const Move& m,
                           vector<int>& next,
                           vector<int>& prev,
                           int& startVertex,
                           int cycleSize) {
        vector<int> cyc = materializeCycle(startVertex, next, cycleSize);
        int n = (int)cyc.size();

        vector<int> position(next.size(), -1);
        for (int i = 0; i < n; i++) {
            position[cyc[i]] = i;
        }

        EdgeSwapApplicability state = getEdgeSwapApplicability(m, next);
        if (state == EdgeSwapApplicability::NOT_APPLICABLE) {
            return;
        }

        int start_rev, end_rev;

        if (state == EdgeSwapApplicability::FORWARD) {
            // Usuwamy krawędzie (a,b) i (c,d),
            // odwracamy fragment od b do c.
            int i = position[m.a];
            int j = position[m.c];

            start_rev = (i + 1) % n;
            end_rev = j;
        } else {
            // Usuwane krawędzie występują jako (b,a) i (d,c),
            // odwracamy fragment od a do d.
            int i = position[m.b];
            int j = position[m.d];

            start_rev = (i + 1) % n;
            end_rev = j;
        }

        reverseSubpath(cyc, start_rev, end_rev);
        rebuildLinkedCycleFromLinearOrder(cyc, next, prev);
        startVertex = cyc[0];
    }
};





// RUCHY KANDYDACKIE:
class LocalSearchWithCandidateMoves : public Heuristic {
    RandomSolution rndSol;
    vector<vector<int>> candidateNearestNeighbors;
    int numCandidates;
    int cachedNumCandidates;

    enum class MoveType { NONE, INTRA_1, INTRA_2, ADD_NEXT, ADD_PREV, REMOVE };

public:
    explicit LocalSearchWithCandidateMoves(int numCandidates_ = 10)
        : numCandidates(max(1, numCandidates_)), cachedNumCandidates(-1) {}

    void precomputeCandidateMoves(const Instance& inst, int numCandidates) {
        candidateNearestNeighbors.assign(inst.n, vector<int>());

        for (int n1 = 0; n1 < inst.n; n1++) {
            vector<pair<int, int>> distances;
            distances.reserve(inst.n - 1);

            for (int n2 = 0; n2 < inst.n; n2++) {
                if (n1 != n2) {
                    distances.push_back({inst.dist[n1][n2], n2});
                }
            }

            sort(distances.begin(), distances.end());

            int lim = min(numCandidates, (int)distances.size());
            candidateNearestNeighbors[n1].reserve(lim);
            for (int i = 0; i < lim; i++) {
                candidateNearestNeighbors[n1].push_back(distances[i].second);
            }
        }
    }

    Solution solve(const Instance& inst) override {
        // 1. Prekalkulacja sąsiedztwa kandydackiego
        if (candidateNearestNeighbors.empty() ||
            (int)candidateNearestNeighbors.size() != inst.n ||
            cachedNumCandidates != numCandidates) {
            precomputeCandidateMoves(inst, numCandidates);
            cachedNumCandidates = numCandidates;
        }

        Solution sol = rndSol.solve(inst);
        if (sol.cycle.empty()) {
            sol.computeStats(inst);
            return sol;
        }

        // Cykl przechowujemy jako listę dwukierunkową w tablicach next/prev.
        //
        // Dzięki temu:
        // - ADD wykonujemy w O(1)
        // - REMOVE wykonujemy w O(1)
        //
        // Vector<int> cyc będzie tylko chwilową linearyzacją cyklu, budowaną raz na iterację lokalnego przeszukiwania.
        vector<int> next(inst.n, -1);
        vector<int> prev(inst.n, -1);
        vector<char> inCycle(inst.n, 0);

        buildLinkedCycle(sol.cycle, next, prev, inCycle);

        int startVertex = sol.cycle[0];
        int cycleSize = (int)sol.cycle.size();
        vector<int> position(inst.n, -1);

        bool improved = true;

        while (improved) {
            improved = false;

            int best_delta = 0;
            MoveType best_type = MoveType::NONE;
            int best_i = -1;
            int best_j_or_v = -1;

            vector<int> cyc = materializeCycle(startVertex, next, cycleSize);
            int n = (int)cyc.size();

            for (int i = 0; i < n; i++) {
                position[cyc[i]] = i;
            }

            // ZBIERANIE RUCHÓW KANDYDACKICH 
            for (int i = 0; i < n; i++) {
                int n1 = cyc[i];
                int n1_next = cyc[(i + 1) % n];
                int n1_prev = cyc[(i - 1 + n) % n];

                for (int n2 : candidateNearestNeighbors[n1]) {
                    int j = inCycle[n2] ? position[n2] : -1;

                    if (j != -1) {
                        // RUCHY WEWNĄTRZTRASOWE
                        if (i == j || i == (j + 1) % n || j == (i + 1) % n) continue;

                        int n2_next = cyc[(j + 1) % n];
                        int n2_prev = cyc[(j - 1 + n) % n];

                        // RUCH 1:
                        // Odwracamy fragment od n1_next do n2_prev.
                        int delta1 =
                            (inst.dist[n1][n1_next] + inst.dist[n2_prev][n2]) -
                            (inst.dist[n1][n2_prev] + inst.dist[n1_next][n2]);

                        if (delta1 > best_delta) {
                            best_delta = delta1;
                            best_type = MoveType::INTRA_1;
                            best_i = i;
                            best_j_or_v = j;
                        }

                        // RUCH 2:
                        // Odwracamy fragment od n1 do n2.
                        int delta2 =
                            (inst.dist[n1_prev][n1] + inst.dist[n2][n2_next]) -
                            (inst.dist[n1_prev][n2] + inst.dist[n1][n2_next]);

                        if (delta2 > best_delta) {
                            best_delta = delta2;
                            best_type = MoveType::INTRA_2;
                            best_i = i;
                            best_j_or_v = j;
                        }

                    } else {
                        // RUCHY MIĘDZYTRASOWE (ADD)

                        // Opcja 1: wstaw n2 pomiędzy n1 a n1_next
                        int d_add_next = inst.profit[n2] - inst.deltaInsert(n1, n1_next, n2);
                        if (d_add_next > best_delta) {
                            best_delta = d_add_next;
                            best_type = MoveType::ADD_NEXT;
                            best_i = i;
                            best_j_or_v = n2;
                        }

                        // Opcja 2: wstaw n2 pomiędzy n1_prev a n1
                        int d_add_prev = inst.profit[n2] - inst.deltaInsert(n1_prev, n1, n2);
                        if (d_add_prev > best_delta) {
                            best_delta = d_add_prev;
                            best_type = MoveType::ADD_PREV;
                            best_i = i;
                            best_j_or_v = n2;
                        }
                    }
                }
            }

            // ZBIERANIE RUCHÓW USUWANIA (REMOVE)
            if (n > 2) {
                for (int idx = 0; idx < n; idx++) {
                    int prev_v = cyc[(idx - 1 + n) % n];
                    int next_v = cyc[(idx + 1) % n];
                    int v = cyc[idx];

                    int d_rem = inst.deltaRemove(prev_v, v, next_v);
                    if (d_rem > best_delta) {
                        best_delta = d_rem;
                        best_type = MoveType::REMOVE;
                        best_i = idx;
                    }
                }
            }

            // APLIKACJA NAJLEPSZEGO RUCHU
            if (best_delta > 0) {
                improved = true;

                if (best_type == MoveType::INTRA_1) {
                    int start_rev = (best_i + 1) % n;
                    int end_rev = (best_j_or_v - 1 + n) % n;

                    reverseSubpath(cyc, start_rev, end_rev);
                    rebuildLinkedCycleFromLinearOrder(cyc, next, prev);
                    startVertex = cyc[0];

                } else if (best_type == MoveType::INTRA_2) {
                    reverseSubpath(cyc, best_i, best_j_or_v);
                    rebuildLinkedCycleFromLinearOrder(cyc, next, prev);
                    startVertex = cyc[0];

                } else if (best_type == MoveType::ADD_NEXT) {
                    int n1 = cyc[best_i];
                    int old_next = next[n1];
                    int v = best_j_or_v;

                    insertBetween(n1, old_next, v, next, prev, inCycle);
                    cycleSize++;

                } else if (best_type == MoveType::ADD_PREV) {
                    int n1 = cyc[best_i];
                    int old_prev = prev[n1];
                    int v = best_j_or_v;

                    insertBetween(old_prev, n1, v, next, prev, inCycle);
                    cycleSize++;

                    if (n1 == startVertex) {
                        startVertex = v;
                    }

                } else if (best_type == MoveType::REMOVE) {
                    int v = cyc[best_i];
                    int successor = next[v];

                    removeVertex(v, next, prev, inCycle);
                    cycleSize--;

                    if (v == startVertex) {
                        startVertex = successor;
                    }
                }
            }
        }

        sol.cycle = materializeCycle(startVertex, next, cycleSize);
        sol.computeStats(inst);
        return sol;
    }

private:
    void buildLinkedCycle(const vector<int>& cyc,
                          vector<int>& next,
                          vector<int>& prev,
                          vector<char>& inCycle) {
        fill(next.begin(), next.end(), -1);
        fill(prev.begin(), prev.end(), -1);
        fill(inCycle.begin(), inCycle.end(), 0);

        int n = (int)cyc.size();
        for (int i = 0; i < n; i++) {
            int v = cyc[i];
            int vn = cyc[(i + 1) % n];
            int vp = cyc[(i - 1 + n) % n];

            next[v] = vn;
            prev[v] = vp;
            inCycle[v] = 1;
        }
    }

    void rebuildLinkedCycleFromLinearOrder(const vector<int>& cyc,
                                           vector<int>& next,
                                           vector<int>& prev) {
        int n = (int)cyc.size();
        for (int i = 0; i < n; i++) {
            int v = cyc[i];
            next[v] = cyc[(i + 1) % n];
            prev[v] = cyc[(i - 1 + n) % n];
        }
    }

    vector<int> materializeCycle(int startVertex,
                                 const vector<int>& next,
                                 int cycleSize) {
        vector<int> cyc;
        cyc.reserve(cycleSize);

        int v = startVertex;
        for (int step = 0; step < cycleSize; step++) {
            cyc.push_back(v);
            v = next[v];
        }

        return cyc;
    }

    void insertBetween(int left, int right, int v,
                       vector<int>& next,
                       vector<int>& prev,
                       vector<char>& inCycle) {
        next[left] = v;
        prev[v] = left;

        next[v] = right;
        prev[right] = v;

        inCycle[v] = 1;
    }

    void removeVertex(int v,
                      vector<int>& next,
                      vector<int>& prev,
                      vector<char>& inCycle) {
        int left = prev[v];
        int right = next[v];

        next[left] = right;
        prev[right] = left;

        next[v] = -1;
        prev[v] = -1;
        inCycle[v] = 0;
    }

    void reverseSubpath(vector<int>& cyc, int start, int end) {
        int n = (int)cyc.size();

        if (start <= end) {
            reverse(cyc.begin() + start, cyc.begin() + end + 1);
        } else {
            vector<int> temp;
            temp.reserve(n - start + end + 1);

            for (int k = start; k < n; k++) temp.push_back(cyc[k]);
            for (int k = 0; k <= end; k++) temp.push_back(cyc[k]);

            reverse(temp.begin(), temp.end());

            int idx = 0;
            for (int k = start; k < n; k++) cyc[k] = temp[idx++];
            for (int k = 0; k <= end; k++) cyc[k] = temp[idx++];
        }
    }
};

void makeDir(const string& path){
#ifdef _WIN32
    system(("mkdir \"" + path + "\" 2>nul").c_str());
#else
    system(("mkdir -p \"" + path + "\"").c_str());
#endif
}

void saveBest(const string& path, const Solution& sol){
    ofstream f(path);
    f<<"objective="<<sol.objective()<<"\nprofit="<<sol.profitSum<<"\nlength="<<sol.length<<"\ncount="<<sol.cycle.size()<<"\ncycle:\n";
    for(int v:sol.cycle) f<<v<<"\n";
}

void saveAllCSV(const string& path, const vector<Solution>& sols, const vector<double>& times) {
    ofstream f(path);
    f<<"rep,objective,profit,length,time_ms,cycle\n";
    for(size_t i=0;i<sols.size();i++){
        f<<i<<","<<sols[i].objective()<<","<<sols[i].profitSum<<","<<sols[i].length<<","<<(int)(times[i]*1000)<<",\"";
        for(size_t j=0;j<sols[i].cycle.size();j++){ f<<sols[i].cycle[j]; if(j+1<sols[i].cycle.size()) f<<" "; }
        f<<"\"\n";
    }
}

enum class SolverKind {
    CLASSIC_LS,
    MOVE_LIST_LS,
    CANDIDATE_LS
};

struct MethodDef {
    string name;
    SolverKind kind;
    NeighType nt;
    LSMode mode;
    bool startRandom;
    int candidateCount;
};

struct BenchmarkResult {
    double avgObj = 0.0;
    int minObj = 0;
    int maxObj = 0;

    double avgTimeMs = 0.0;
    double minTimeMs = 0.0;
    double maxTimeMs = 0.0;

    Solution best;
    vector<Solution> allSols;
    vector<double> timesSeconds;
};

unique_ptr<Heuristic> buildSolver(const MethodDef& def) {
    if (def.kind == SolverKind::MOVE_LIST_LS) {
        return make_unique<LocalSearchWithMoveList>();
    }   
    if (def.kind == SolverKind::CANDIDATE_LS) {
        return make_unique<LocalSearchWithCandidateMoves>(def.candidateCount);
    }
    

    return make_unique<LocalSearch>(def.nt, def.mode, def.startRandom);
}

BenchmarkResult runBenchmark(Heuristic& solver, const Instance& inst, int repetitions = 100) {
    BenchmarkResult result;

    vector<int> objs;
    objs.reserve(repetitions);
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

        if (!hasBest || s.objective() > result.best.objective()) {
            result.best = s;
            hasBest = true;
        }
    }

    result.avgObj = accumulate(objs.begin(), objs.end(), 0.0) / objs.size();
    result.minObj = *min_element(objs.begin(), objs.end());
    result.maxObj = *max_element(objs.begin(), objs.end());

    result.avgTimeMs = 1000.0 * accumulate(result.timesSeconds.begin(), result.timesSeconds.end(), 0.0) / result.timesSeconds.size();
    result.minTimeMs = 1000.0 * (*min_element(result.timesSeconds.begin(), result.timesSeconds.end()));
    result.maxTimeMs = 1000.0 * (*max_element(result.timesSeconds.begin(), result.timesSeconds.end()));

    return result;
}

int main() {
    bool usePrecomputed = false;
    const int repetitions = 100;

    vector<Instance> insts = {
        Instance::loadFromCSV("TSPA.csv", usePrecomputed),
        Instance::loadFromCSV("TSPB.csv", usePrecomputed)
    };

    if (!usePrecomputed) {
        insts[0].saveDistanceCSV("TSPA_matrix.csv");
        insts[1].saveDistanceCSV("TSPB_matrix.csv");
    }

    vector<string> tags = {"A", "B"};

    vector<MethodDef> methods = {
        {"LS_STEEPEST_ESWAP_RAND",      SolverKind::CLASSIC_LS,   NeighType::EDGE_SWAP, LSMode::STEEPEST, true,  0},
        {"LS_MOVE_LIST_RAND", SolverKind::MOVE_LIST_LS, NeighType::EDGE_SWAP, LSMode::STEEPEST, true, 0},
        {"LS_CANDIDATE_MOVES_K10_RAND", SolverKind::CANDIDATE_LS, NeighType::EDGE_SWAP, LSMode::STEEPEST, true, 10},
        {"LS_CANDIDATE_MOVES_K30_RAND", SolverKind::CANDIDATE_LS, NeighType::EDGE_SWAP, LSMode::STEEPEST, true, 30},
        {"LS_CANDIDATE_MOVES_K50_RAND", SolverKind::CANDIDATE_LS, NeighType::EDGE_SWAP, LSMode::STEEPEST, true, 50},
        {"LS_CANDIDATE_MOVES_K100_RAND", SolverKind::CANDIDATE_LS, NeighType::EDGE_SWAP, LSMode::STEEPEST, true, 100},
        
    };

    makeDir("output");

    for (size_t ii = 0; ii < insts.size(); ii++) {
        const auto& inst = insts[ii];
        string tag = tags[ii];
        string base = "output/" + tag;

        makeDir(base);
        makeDir(base + "/solutions");
        makeDir(base + "/solutions_all");

        ofstream stats(base + "/stats.csv");
        stats << "method,avg_obj,min_obj,max_obj,avg_time_ms,min_time_ms,max_time_ms\n";

        {
            Regret2 reg;
            BenchmarkResult res = runBenchmark(reg, inst, repetitions);

            stats << "REGRET2,"
                  << res.avgObj << ","
                  << res.minObj << ","
                  << res.maxObj << ","
                  << res.avgTimeMs << ","
                  << res.minTimeMs << ","
                  << res.maxTimeMs << "\n";

            saveBest(base + "/solutions/REGRET2.txt", res.best);
            saveAllCSV(base + "/solutions_all/REGRET2.csv", res.allSols, res.timesSeconds);

            cerr << "[" << tag << "] REGRET2 done\n";
        }

        for (const auto& def : methods) {
            unique_ptr<Heuristic> solver = buildSolver(def);
            BenchmarkResult res = runBenchmark(*solver, inst, repetitions);

            stats << def.name << ","
                  << res.avgObj << ","
                  << res.minObj << ","
                  << res.maxObj << ","
                  << res.avgTimeMs << ","
                  << res.minTimeMs << ","
                  << res.maxTimeMs << "\n";

            saveBest(base + "/solutions/" + def.name + ".txt", res.best);
            saveAllCSV(base + "/solutions_all/" + def.name + ".csv", res.allSols, res.timesSeconds);

            cerr << "[" << tag << "] " << def.name
                 << " avg_obj=" << res.avgObj
                 << " avg_time=" << res.avgTimeMs << "ms\n";
        }
    }

    return 0;
}