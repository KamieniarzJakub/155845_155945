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


class RandomWalk : public Heuristic {
    double timeLimitSeconds;
    RandomSolution rndSol;

public:
    RandomWalk(double tl=1.0) : timeLimitSeconds(tl) {}

    void setTimeLimit(double tl) { timeLimitSeconds = tl; }

    Solution solve(const Instance& inst) override {
        Solution sol = rndSol.solve(inst);
        Solution best = sol;

        auto start = chrono::steady_clock::now();
        while(true) {
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now-start).count();
            if(elapsed >= timeLimitSeconds) break;

            vector<int>& cyc = sol.cycle;
            int n = cyc.size();
            vector<bool> inCycle(inst.n, false);
            for(int v : cyc) inCycle[v] = true;

            int mtype = uniform_int_distribution<int>(0,2)(rng);
            if(mtype==0) {
                vector<int> outside;
                for(int v=0;v<inst.n;v++) if(!inCycle[v]) outside.push_back(v);
                if(outside.empty()) { mtype=1; }
                else {
                    int v = outside[uniform_int_distribution<int>(0,(int)outside.size()-1)(rng)];
                    int pos = uniform_int_distribution<int>(0,n-1)(rng);
                    cyc.insert(cyc.begin()+pos, v);
                    sol.computeStats(inst);
                }
            }
            if(mtype==1) {
                if(n<=2) continue;
                int idx = uniform_int_distribution<int>(0,n-1)(rng);
                cyc.erase(cyc.begin()+idx);
                sol.computeStats(inst);
            } else if(mtype==2) {
                if(n<4) continue;
                int i = uniform_int_distribution<int>(0,n-1)(rng);
                int j = uniform_int_distribution<int>(0,n-1)(rng);
                if(i==j) continue;
                if(i>j) swap(i,j);
                if(uniform_int_distribution<int>(0,1)(rng)==0) swap(cyc[i],cyc[j]);
                else {
                    if(i==0 && j==n-1) continue;
                    int l=i+1,r=j;
                    while(l<r){swap(cyc[l],cyc[r]);l++;r--;}
                }
                sol.computeStats(inst);
            }

            if(sol.objective() > best.objective()) best = sol;
        }
        return best;
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

class LocalSearchWithMoveList : public Heuristic {
    NeighType neighType;
    LSMode mode;
    bool startRandom;
    RandomSolution rndSol;
    vector<pair<int, vector<int>>> moveList; // lista ruchów (ocenionych wcześniej)

public:
    LocalSearchWithMoveList(NeighType nt, LSMode lm, bool sr)
        : neighType(nt), mode(lm), startRandom(sr) {}

    Solution solve(const Instance& inst) override {
        Solution sol = startRandom ? rndSol.solve(inst) : solveWithMoveList(inst);
        localSearch(inst, sol);
        return sol;
    }

private:
    void localSearch(const Instance& inst, Solution& sol) {
        if(mode == LSMode::STEEPEST)
            steepest(inst, sol);
        else
            greedy(inst, sol);
        sol.computeStats(inst);
    }

    Solution solveWithMoveList(const Instance& inst) {
        Solution sol = rndSol.solve(inst);
        vector<int>& cyc = sol.cycle;
        moveList.clear();  // Inicjalizujemy pustą listę ruchów

        // Zbieramy wszystkie ruchy aplikowalne do cyklu i dodajemy je do listy
        for(int i = 0; i < inst.n; i++) {
            for(int j = 0; j < cyc.size(); j++) {
                int d = inst.deltaInsert(cyc[i], cyc[j], i);
                if(d > 0) { // dodajemy tylko ruchy, które przynoszą poprawę
                    moveList.push_back({d, {i, j}});
                }
            }
        }

        // Sortujemy listę ruchów według oceny (od najlepszych do najgorszych)
        sort(moveList.begin(), moveList.end(), greater<>());

        return sol;
    }

    void steepest(const Instance& inst, Solution& sol) {
        bool improved = true;
        while(improved) {
            vector<int>& cyc = sol.cycle;
            improved = false;
            int bestDelta = 0;
            int bestType = -1;
            int bestI = -1, bestJ = -1, bestV = -1, bestPos = -1;

            // Przeglądanie listy ruchów od najlepszych do najgorszych
            for (auto& move : moveList) {
                int delta = move.first;
                if (delta > bestDelta) {
                    bestDelta = delta;
                    bestV = move.second[0]; // Wartość ruchu
                    bestPos = move.second[1];
                    improved = true;
                }
            }

            if(bestDelta > 0) {
                // Wykonanie ruchu
                cyc.insert(cyc.begin() + bestPos, bestV);
                sol.computeStats(inst);
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

    // Funkcje pomocnicze (applyAdd, applyRemove, deltaRem, deltaVertexSwap, applyVertexSwap, deltaEdgeSwap, applyEdgeSwap)
    void applyAdd(Solution& sol, const Instance& inst, vector<int>& cyc, int v, int pos) {
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

    void applyRemove(Solution& sol, const Instance& inst, vector<int>& cyc, int idx) {
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

    void applyVertexSwap(Solution& sol, const Instance& inst, vector<int>& cyc, int i, int j) {
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

    void applyEdgeSwap(Solution& sol, const Instance& inst, vector<int>& cyc, int i, int j) {
        sol.length -= deltaEdgeSwap(inst, cyc, i, j);
        int n = cyc.size();
        int l = (i+1)%n, r = j;
        while(l < r) { swap(cyc[l], cyc[r]); l++; r--; }
    }
};



// RUCHY KANDYDACKIE:
class LocalSearchWithCandidateMoves : public Heuristic {
    RandomSolution rndSol;
    vector<pair<int, pair<int, int>>> candidateMoves; // Lista kandydatów (delta, wierzchołki)

public:
    LocalSearchWithCandidateMoves() {}

    Solution solve(const Instance& inst) override {
        Solution sol = rndSol.solve(inst);
        vector<int>& cyc = sol.cycle;
        candidateMoves.clear();  // Inicjalizujemy pustą listę ruchów kandydackich

        // Zbieramy ruchy kandydackie
        for (int n1 = 0; n1 < inst.n; n1++) {
            // Generujemy 10 najbliższych sąsiadów wierzchołka n1
            vector<pair<int, int>> closestNeighbors;
            for (int n2 = 0; n2 < inst.n; n2++) {
                if (n1 != n2) {
                    closestNeighbors.push_back({inst.dist[n1][n2], n2});
                }
            }

            // Sortowanie sąsiadów po odległości
            sort(closestNeighbors.begin(), closestNeighbors.end());

            // Wybieramy 10 najbliższych
            for (int i = 0; i < 10 && i < closestNeighbors.size(); i++) {
                int n2 = closestNeighbors[i].second;
                int delta = inst.deltaInsert(n1, n2, n2);
                if (delta > 0) { // dodajemy tylko te ruchy, które poprawiają rozwiązanie
                    candidateMoves.push_back({delta, {n1, n2}});
                }
            }
        }

        // Sortowanie ruchów kandydackich według oceny (najpierw najlepsze)
        sort(candidateMoves.begin(), candidateMoves.end(), greater<>());

        // Wykonaj najlepsze dostępne ruchy kandydackie
        for (auto& move : candidateMoves) {
            int delta = move.first;
            int n1 = move.second.first;
            int n2 = move.second.second;

            // Sprawdzamy, czy ten ruch jest możliwy i aplikujemy go
            // Wstawiamy krawędź między n1 i n2 do cyklu
            cyc.push_back(n2); // Prosty przykład dodania nowego wierzchołka
            sol.computeStats(inst);
        }

        return sol;
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

int main(){
    bool usePrecomputed=false;

    vector<Instance> insts={Instance::loadFromCSV("TSPA.csv",usePrecomputed),
                            Instance::loadFromCSV("TSPB.csv",usePrecomputed)};
    if(!usePrecomputed){ insts[0].saveDistanceCSV("TSPA_matrix.csv"); insts[1].saveDistanceCSV("TSPB_matrix.csv"); }

    vector<string> tags={"A","B"};

    struct MethodDef {
        string name;
        NeighType nt;
        LSMode mode;
        bool startRandom;
    };
    vector<MethodDef> lsDefs = {
        {"LS_STEEPEST_ESWAP_RAND",   NeighType::EDGE_SWAP,   LSMode::STEEPEST, true},
        {"LS_MOVE_LIST_RAND",   NeighType::EDGE_SWAP,   LSMode::STEEPEST, true},
        {"LS_CANDIDATE_MOVES_RAND",   NeighType::EDGE_SWAP,   LSMode::STEEPEST, true},

    };

    makeDir("output");

    for(size_t ii=0;ii<insts.size();ii++) {
        const auto& inst = insts[ii];
        string tag = tags[ii];
        string base = "output/"+tag;
        makeDir(base);
        makeDir(base+"/solutions");
        makeDir(base+"/solutions_all");

        double maxAvgTime = 0.0;

        ofstream stats(base+"/stats.csv");
        stats<<"method,avg_obj,min_obj,max_obj,avg_time_ms,min_time_ms,max_time_ms\n";

        // REGRET2
        {
            Regret2 reg;
            vector<int> objs; vector<double> times; vector<Solution> allSols;
            Solution best; bool hasBest=false;
            for(int rep=0;rep<100;rep++){
                auto t0=chrono::steady_clock::now();
                Solution s=reg.solve(inst);
                double dt=chrono::duration<double>(chrono::steady_clock::now()-t0).count();
                objs.push_back(s.objective()); times.push_back(dt);
                allSols.push_back(s);
                if(!hasBest||s.objective()>best.objective()){best=s;hasBest=true;}
            }
            double avgT=accumulate(times.begin(),times.end(),0.0)/times.size();
            stats<<"REGRET2,"<<accumulate(objs.begin(),objs.end(),0.0)/objs.size()<<","
                 <<*min_element(objs.begin(),objs.end())<<","<<*max_element(objs.begin(),objs.end())<<","
                 <<avgT*1000<<","<<*min_element(times.begin(),times.end())*1000<<","<<*max_element(times.begin(),times.end())*1000<<"\n";
            saveBest(base+"/solutions/REGRET2.txt",best);
            saveAllCSV(base+"/solutions_all/REGRET2.csv",allSols,times);
            cerr << "[" << tag << "] REGRET2 done\n";
        }

        // Local search
        for(auto& def : lsDefs) {
            LocalSearch ls(def.nt, def.mode, def.startRandom);
            vector<int> objs; vector<double> times; vector<Solution> allSols;
            Solution best; bool hasBest=false;
            for(int rep=0;rep<100;rep++){   
                auto t0=chrono::steady_clock::now();
                Solution s=ls.solve(inst);
                double dt=chrono::duration<double>(chrono::steady_clock::now()-t0).count();
                objs.push_back(s.objective()); times.push_back(dt);
                allSols.push_back(s);
                if(!hasBest||s.objective()>best.objective()){best=s;hasBest=true;}
            }
            double avgT=accumulate(times.begin(),times.end(),0.0)/times.size();
            if(avgT > maxAvgTime) maxAvgTime = avgT;
            stats<<def.name<<","<<accumulate(objs.begin(),objs.end(),0.0)/objs.size()<<","
                 <<*min_element(objs.begin(),objs.end())<<","<<*max_element(objs.begin(),objs.end())<<","
                 <<avgT*1000<<","<<*min_element(times.begin(),times.end())*1000<<","<<*max_element(times.begin(),times.end())*1000<<"\n";
            saveBest(base+"/solutions/"+def.name+".txt",best);
            saveAllCSV(base+"/solutions_all/"+def.name+".csv",allSols,times);

            cerr << "[" << tag << "] " << def.name << " avg_obj=" << accumulate(objs.begin(),objs.end(),0.0)/objs.size()
                 << " avg_time=" << avgT*1000 << "ms\n";
        }
    }

    return 0;
}