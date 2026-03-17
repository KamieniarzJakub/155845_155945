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
using namespace std;

// --------------------------------------------------------------
// KLASA INSTANCJI
// --------------------------------------------------------------
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

// --------------------------------------------------------------
// KLASA ROZWIĄZANIA
// --------------------------------------------------------------
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

// --------------------------------------------------------------
// KLASA HEURYSTYKI (INTERFEJS)
// --------------------------------------------------------------
class Heuristic { public: virtual Solution solve(const Instance& inst)=0; virtual ~Heuristic(){} };

// --------------------------------------------------------------
// RANDOM HEURYSTYKA (TYLKO LOSOWANIE, BEZ FAZY II)
// --------------------------------------------------------------
class RandomSolution: public Heuristic {
public:
    Solution solve(const Instance& inst) override {
        Solution sol;
        vector<int> perm(inst.n); iota(perm.begin(),perm.end(),0);
        shuffle(perm.begin(),perm.end(), mt19937(random_device{}()));
        int k = 2 + rand()%(inst.n-1);
        sol.cycle.assign(perm.begin(), perm.begin()+k);
        sol.computeStats(inst);
        sol.lengthPhase1 = sol.length; // zachowujemy długość fazy I
        return sol;
    }
};

// --------------------------------------------------------------
// NEAREST NEIGHBOR HEURYSTYKA
// --------------------------------------------------------------
class NearestNeighbor: public Heuristic {
    bool ignore_profit;
public:
    NearestNeighbor(bool ig=false):ignore_profit(ig){}
    Solution solve(const Instance& inst) override {
        int n=inst.n; vector<bool> used(n,false);
        int start=rand()%n; list<int> cyc={start}; used[start]=true;
        for(int iter=1;iter<n;iter++){
            int last=cyc.back(), best=-1; double bestScore=1e18;
            for(int v=0;v<n;v++) if(!used[v]){
                double s = inst.dist[last][v]; if(!ignore_profit) s-=inst.profit[v];
                if(s<bestScore){ bestScore=s; best=v; }
            }
            if(best==-1) break; used[best]=true; cyc.push_back(best);
        }
        Solution sol; sol.cycle.assign(cyc.begin(),cyc.end());
        sol.computeStats(inst); sol.lengthPhase1=sol.length;
        sol.phaseIIRemove(inst);
        return sol;
    }
};

// --------------------------------------------------------------
// GREEDY CYCLE HEURYSTYKA
// --------------------------------------------------------------
class GreedyCycle: public Heuristic {
    bool ignore_profit;
public:
    GreedyCycle(bool ig=false):ignore_profit(ig){}
    Solution solve(const Instance& inst) override {
        int n=inst.n; vector<bool> used(n,false);
        int start=rand()%n; list<int> cyc={start}; used[start]=true;
        int sec=-1; double best=1e18;
        for(int v=0;v<n;v++) if(!used[v]){
            double s=inst.dist[start][v]; if(!ignore_profit) s-=inst.profit[v];
            if(s<best){ best=s; sec=v; }
        }
        cyc.push_back(sec); used[sec]=true;
        while(cyc.size()<n){
            double bestInc=1e18; int bestV=-1; auto bestPos=cyc.begin();
            for(int v=0;v<n;v++) if(!used[v]){
                for(auto it=cyc.begin();it!=cyc.end();++it){
                    auto next_it=next(it); if(next_it==cyc.end()) next_it=cyc.begin();
                    double inc=inst.deltaInsert(*it,*next_it,v);
                    if(!ignore_profit) inc-=inst.profit[v];
                    if(inc<bestInc){ bestInc=inc; bestV=v; bestPos=next_it; }
                }
            }
            used[bestV]=true; cyc.insert(bestPos,bestV);
        }
        Solution sol; sol.cycle.assign(cyc.begin(),cyc.end());
        sol.computeStats(inst); sol.lengthPhase1=sol.length;
        sol.phaseIIRemove(inst);
        return sol;
    }
};

// --------------------------------------------------------------
// REGRET2 HEURYSTYKA
// --------------------------------------------------------------
class Regret2: public Heuristic {
    bool weighted; double w;
public:
    Regret2(bool wg=false,double ww=1.0):weighted(wg),w(ww){}
    Solution solve(const Instance& inst) override {
        int n=inst.n; vector<bool> used(n,false);
        int start=rand()%n; list<int> cyc={start}; used[start]=true;
        int nxt=-1, bestd=1e9;
        for(int v=0;v<n;v++) if(!used[v] && inst.dist[start][v]<bestd){ bestd=inst.dist[start][v]; nxt=v; }
        cyc.push_back(nxt); used[nxt]=true;
        while(cyc.size()<n){
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

// --------------------------------------------------------------
// ZAPIS STATYSTYK
// --------------------------------------------------------------
void makeDir(const string& path){
#ifdef _WIN32
    system(("_mkdir \"" + path + "\" 2>nul").c_str());
#else
    system(("mkdir -p \"" + path + "\"").c_str());
#endif
}

void saveBest(const string& path, const Solution& sol){
    ofstream f(path);
    f<<"objective="<<sol.objective()<<"\nprofit="<<sol.profitSum<<"\nlength="<<sol.length<<"\ncount="<<sol.cycle.size()<<"\ncycle:\n";
    for(int v:sol.cycle) f<<v<<"\n";
}

void saveAllCSV(const string& path, const vector<Solution>& sols){
    ofstream f(path); f<<"rep,objective,profit,length,length_phase1,cycle\n";
    for(size_t i=0;i<sols.size();i++){
        f<<i<<","<<sols[i].objective()<<","<<sols[i].profitSum<<","<<sols[i].length<<","<<sols[i].lengthPhase1<<",\"";
        for(size_t j=0;j<sols[i].cycle.size();j++){ f<<sols[i].cycle[j]; if(j+1<sols[i].cycle.size()) f<<" "; }
        f<<"\"\n";
    }
}

// --------------------------------------------------------------
// MAIN
// --------------------------------------------------------------
int main(){
    srand(time(NULL));
    bool usePrecomputed=false;

    vector<Instance> insts={Instance::loadFromCSV("TSPA.csv",usePrecomputed),
                            Instance::loadFromCSV("TSPB.csv",usePrecomputed)};
    if(!usePrecomputed){ insts[0].saveDistanceCSV("TSPA_matrix.csv"); insts[1].saveDistanceCSV("TSPB_matrix.csv"); }

    vector<string> tags={"A","B"};

    vector<pair<string, unique_ptr<Heuristic>>> methods;
    methods.emplace_back("RANDOM", make_unique<RandomSolution>());
    methods.emplace_back("NN", make_unique<NearestNeighbor>(false));
    methods.emplace_back("NNa", make_unique<NearestNeighbor>(true));
    methods.emplace_back("GC", make_unique<GreedyCycle>(false));
    methods.emplace_back("GCa", make_unique<GreedyCycle>(true));
    methods.emplace_back("REGRET2", make_unique<Regret2>(false));
    methods.emplace_back("WREGRET2_w1.0", make_unique<Regret2>(true,1.0));
    methods.emplace_back("WREGRET2_w0.5", make_unique<Regret2>(true,0.5));

    makeDir("output");

    for(size_t i=0;i<insts.size();i++){
        const auto& inst=insts[i]; string tag=tags[i]; string base="output/"+tag;
        makeDir(base); makeDir(base+"/solutions"); makeDir(base+"/solutions_all");

        ofstream stats(base+"/stats.csv");
        stats<<"method,avg_obj,min_obj,max_obj,avg_len1,min_len1,max_len1,avg_len2,min_len2,max_len2\n";

        for(auto& m:methods){
            vector<int> objs,len1,len2; vector<Solution> allSols;
            Solution best; bool hasBest=false;
            for(int rep=0;rep<200;rep++){
                Solution s=m.second->solve(inst);
                objs.push_back(s.objective()); len1.push_back(s.lengthPhase1); len2.push_back(s.length);
                allSols.push_back(s);
                if(!hasBest || s.objective()>best.objective()){ best=s; hasBest=true; }
            }
            auto avg=[](const vector<int>&v){ return accumulate(v.begin(),v.end(),0.0)/v.size(); };
            stats<<m.first<<","<<avg(objs)<<","<<*min_element(objs.begin(),objs.end())<<","<<*max_element(objs.begin(),objs.end())<<","
                 <<avg(len1)<<","<<*min_element(len1.begin(),len1.end())<<","<<*max_element(len1.begin(),len1.end())<<","
                 <<avg(len2)<<","<<*min_element(len2.begin(),len2.end())<<","<<*max_element(len2.begin(),len2.end())<<"\n";

            saveBest(base+"/solutions/"+m.first+".txt",best);
            saveAllCSV(base+"/solutions_all/"+m.first+".csv",allSols);
        }
    }

    return 0;
}