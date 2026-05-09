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
#include <chrono>
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

    static Instance loadFromCSV(const string& filename) {
        ifstream in(filename);
        if(!in) { cerr << "Nie mozna otworzyc " << filename << endl; exit(1); }
        Instance inst;
        string line;
        vector<double> X, Y;
        while(getline(in, line)) {
            if(line.empty()) continue;
            replace(line.begin(), line.end(), ';', ' ');
            stringstream ss(line);
            double x, y; int p;
            ss >> x >> y >> p;
            X.push_back(x); Y.push_back(y); inst.profit.push_back(p);
        }
        inst.n = X.size();
        inst.dist.assign(inst.n, vector<int>(inst.n));
        for(int i=0; i<inst.n; i++)
            for(int j=0; j<inst.n; j++)
                inst.dist[i][j] = round(sqrt(pow(X[i]-X[j], 2) + pow(Y[i]-Y[j], 2)));
        return inst;
    }
};

class Solution {
public:
    vector<int> cycle;
    int length = 0, profitSum = 0;

    int objective() const { return profitSum - length; }

    void computeStats(const Instance& inst) {
        if(cycle.empty()) { length = 0; profitSum = 0; return; }
        length = 0; profitSum = 0;
        for(size_t i=0; i<cycle.size(); i++) {
            length += inst.dist[cycle[i]][cycle[(i+1)%cycle.size()]];
            profitSum += inst.profit[cycle[i]];
        }
    }

    set<int> getNodes() const { return set<int>(cycle.begin(), cycle.end()); }
    
    set<pair<int, int>> getEdges() const {
        set<pair<int, int>> edges;
        for(size_t i=0; i<cycle.size(); i++) {
            int u = cycle[i], v = cycle[(i+1)%cycle.size()];
            edges.insert({min(u, v), max(u, v)});
        }
        return edges;
    }
};

// --- Funkcja wczytująca najlepsze rozwiązania ---
Solution loadBestSolution(const string& filename, const Instance& inst) {
    ifstream in(filename);
    Solution sol; string line;
    bool inCycle = false;
    while(getline(in, line)) {
        if(line.find("cycle:") != string::npos) { inCycle = true; continue; }
        if(inCycle && !line.empty()) sol.cycle.push_back(stoi(line));
    }
    sol.computeStats(inst);
    return sol;
}

class LocalSearch {
    NeighType nt;
public:
    LocalSearch(NeighType type) : nt(type) {}

    void solveGreedy(const Instance& inst, Solution& sol) {
        bool improved = true;
        while (improved) {
            improved = false;
            vector<int>& cyc = sol.cycle;
            int n = cyc.size();
            
            vector<bool> inCycle(inst.n, false);
            for (int v : cyc) inCycle[v] = true;

            // Kolejność ruchów: ADD, REMOVE, INTRA 
            // 1. ADD
            for (int v = 0; v < inst.n && !improved; v++) {
                if (inCycle[v]) continue;
                for (int i = 0; i < n; i++) {
                    int j = (i + 1) % n;
                    if (inst.profit[v] - (inst.dist[cyc[i]][v] + inst.dist[v][cyc[j]] - inst.dist[cyc[i]][cyc[j]]) > 0) {
                        cyc.insert(cyc.begin() + j, v);
                        sol.computeStats(inst);
                        improved = true; break;
                    }
                }
            }
            // 2. REMOVE
            if (!improved && n > 2) {
                for (int i = 0; i < n; i++) {
                    int p = cyc[(i-1+n)%n], c = cyc[i], nx = cyc[(i+1)%n];
                    if (inst.dist[p][c] + inst.dist[c][nx] - inst.dist[p][nx] - inst.profit[c] > 0) {
                        cyc.erase(cyc.begin() + i);
                        sol.computeStats(inst);
                        improved = true; break;
                    }
                }
            }
            // 3. INTRA
            if (!improved) {
                for (int i = 0; i < n && !improved; i++) {
                    for (int j = i + 1; j < n; j++) {
                        if (nt == NeighType::VERTEX_SWAP) {
                            int oldD = inst.dist[cyc[(i-1+n)%n]][cyc[i]] + inst.dist[cyc[i]][cyc[(i+1)%n]] +
                                       inst.dist[cyc[(j-1+n)%n]][cyc[j]] + inst.dist[cyc[j]][cyc[(j+1)%n]];
                            swap(cyc[i], cyc[j]);
                            int newD = inst.dist[cyc[(i-1+n)%n]][cyc[i]] + inst.dist[cyc[i]][cyc[(i+1)%n]] +
                                       inst.dist[cyc[(j-1+n)%n]][cyc[j]] + inst.dist[cyc[j]][cyc[(j+1)%n]];
                            if (oldD - newD > 0) { sol.computeStats(inst); improved = true; break; }
                            else swap(cyc[i], cyc[j]);
                        } else {
                            if (i == 0 && j == n - 1) continue;
                            if (j < i + 2) continue;
                            int gain = (inst.dist[cyc[i]][cyc[i+1]] + inst.dist[cyc[j]][cyc[(j+1)%n]]) -
                                       (inst.dist[cyc[i]][cyc[j]] + inst.dist[cyc[i+1]][cyc[(j+1)%n]]);
                            if (gain > 0) {
                                reverse(cyc.begin() + i + 1, cyc.begin() + j + 1);
                                sol.computeStats(inst);
                                improved = true; break;
                            }
                        }
                    }
                }
            }
        }
    }
};

// --- Funkcje podobieństwa ---
int countCommonNodes(const set<int>& a, const set<int>& b) {
    int common = 0;
    for(int v : a) if(b.count(v)) common++;
    return common;
}

int countCommonEdges(const set<pair<int, int>>& a, const set<pair<int, int>>& b) {
    int common = 0;
    for(auto const& e : a) if(b.count(e)) common++;
    return common;
}

// --- Eksperyment ---
void runExperiment(string label, string instFile, string bestFile, NeighType nt) {
    Instance inst = Instance::loadFromCSV(instFile);
    Solution bestKnown = loadBestSolution(bestFile, inst);
    auto bestNodes = bestKnown.getNodes();
    auto bestEdges = bestKnown.getEdges();

    LocalSearch ls(nt);
    vector<Solution> optima;
    cout << "[" << label << "] Generowanie 1000 optimow..." << endl;

    for(int i=0; i<1000; i++) {
        Solution sol;
        vector<int> p(inst.n); iota(p.begin(), p.end(), 0);
        shuffle(p.begin(), p.end(), rng);
        sol.cycle.assign(p.begin(), p.begin() + (inst.n / 2));
        sol.computeStats(inst);
        ls.solveGreedy(inst, sol);
        optima.push_back(sol);
        if(i % 100 == 0) cout << "." << flush;
    }
    cout << " Done." << endl;

    string outName = "convexity_" + label + "_" + (nt == NeighType::EDGE_SWAP ? "ESWAP" : "VSWAP") + ".csv";
    ofstream out(outName);
    out << "objective,nodes_to_best,edges_to_best,nodes_avg,edges_avg\n";

    for(int i=0; i<1000; i++) {
        auto iNodes = optima[i].getNodes();
        auto iEdges = optima[i].getEdges();

        double simNBest = countCommonNodes(iNodes, bestNodes);
        double simEBest = countCommonEdges(iEdges, bestEdges);

        double sumNAvg = 0, sumEAvg = 0;
        for(int j=0; j<1000; j++) {
            if(i == j) continue;
            sumNAvg += countCommonNodes(iNodes, optima[j].getNodes());
            sumEAvg += countCommonEdges(iEdges, optima[j].getEdges());
        }

        out << optima[i].objective() << "," << simNBest << "," << simEBest << "," 
            << sumNAvg/999.0 << "," << sumEAvg/999.0 << "\n";
    }
}

int main() {
    // Generujemy 4 zestawy danych (2 instancje * 2 typy sąsiedztwa)
    runExperiment("A", "TSPA.csv", "best_A.txt", NeighType::EDGE_SWAP);
    runExperiment("A", "TSPA.csv", "best_A.txt", NeighType::VERTEX_SWAP);
    runExperiment("B", "TSPB.csv", "best_B.txt", NeighType::EDGE_SWAP);
    runExperiment("B", "TSPB.csv", "best_B.txt", NeighType::VERTEX_SWAP);
    return 0;
}