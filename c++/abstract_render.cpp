#include <iostream>
#include <fstream>
#include <cassert>
#include <bits/stdc++.h>
#include <limits>
#include "CImg.h"
#include "boost/program_options.hpp"
#include "boost/foreach.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_USE_DOUBLE
#include "tiny_obj_loader.h"

#define PI 3.14159265359
#define EPS 1e-6
#define MAX_ZONO_SIZE 1000

using namespace std;


class Double {
public:
    double x;
    Double():x(0){};
    Double(double _x):x(_x){};
    Double(double _x, double _y):x(_x){};  // Perturbation
    void operator+=(const Double& rhs) {x+=rhs.x;}
    void operator-=(const Double& rhs) {x-=rhs.x;}
    void operator*=(const Double& rhs) {x*=rhs.x;}
    void operator/=(const Double& rhs) {x/=rhs.x;}
    Double operator-() const {return Double(-x);}
    Double operator+(const Double& rhs) const {return Double(x+rhs.x);}
    Double operator-(const Double& rhs) const {return Double(x-rhs.x);}
    Double operator*(const Double& rhs) const {return Double(x*rhs.x);}
    Double operator/(const Double& rhs) const {return Double(x/rhs.x);}
    bool operator!=(const Double& rhs) {return x!=rhs.x;}
    bool operator==(const Double& rhs) {return x==rhs.x;}
    bool operator>=(const Double& rhs) {return x>=rhs.x;}
    bool operator<=(const Double& rhs) {return x<=rhs.x;}
    bool operator>(const Double& rhs) {return x>rhs.x;}
    bool operator<(const Double& rhs) {return x<rhs.x;}
    Double _cos() {return cos(x);}
    Double _abs() {return abs(x);}
    Double _sin() {return sin(x);}
    Double _sqrt() {return sqrt(x);}
    Double _square() {return x*x;}
    double _min() {return x;}
    double _max() {return x;}
    Double _union(const Double& rhs) const {return Double(x);}
    Double _intersect(const Double& rhs) const {return Double(x);}
    bool _is_empty() {return false;}
    bool _is_finite() const {return isfinite(x);}
};
std::ostream &operator<<(std::ostream &os, Double const &m) {
    return os << m.x;
}



class BBB {
public:
    double a,b;
    BBB():a(std::numeric_limits<double>::infinity()),
            b(-std::numeric_limits<double>::infinity()){};
    BBB(double _a):a(_a),b(_a){};
    BBB(double _a, double _b):a(_a),b(_b) {make_sane();};
    void make_sane() {
            if(a>b) {
                make_empty();
            }
        }
    void make_empty() {
            a = std::numeric_limits<double>::infinity();
            b = -std::numeric_limits<double>::infinity();
        }
    void operator+=(const BBB& rhs) {a+=rhs.a; b+=rhs.b; make_sane();}
    void operator-=(const BBB& rhs) {a-=rhs.b; b-=rhs.a; make_sane();}
    void operator*=(const BBB& rhs) {
            if(rhs._is_empty() or _is_empty()) {
                make_empty();
            }
            else {
                vector<double> vv{a*rhs.a, a*rhs.b, b*rhs.a, b*rhs.b};
                a = *min_element(vv.begin(), vv.end());
                b = *max_element(vv.begin(), vv.end());
            }
        }
    void operator/=(const BBB& rhs) {
            if(rhs._is_empty() or _is_empty()) {
                make_empty();
            }
            else if (rhs.a*rhs.b <= 0) {
                a = -std::numeric_limits<double>::infinity();
                b = std::numeric_limits<double>::infinity();
            }
            else {
                vector<double> vv{a/rhs.a, a/rhs.b, b/rhs.a, b/rhs.b};
                a = *min_element(vv.begin(), vv.end());
                b = *max_element(vv.begin(), vv.end());
            }
        }
    BBB operator-() const {return BBB(-b,-a);}
    BBB operator+(const BBB& rhs) const {
            if(rhs._is_empty() or _is_empty()) {
                return BBB();
            }
            return BBB(a+rhs.a, b+rhs.b);
        }
    BBB operator-(const BBB& rhs) const {
            if(rhs._is_empty() or _is_empty()) {
                return BBB();
            }
            return BBB(a-rhs.b, b-rhs.a);
        }
    BBB operator*(const BBB& rhs) const {
            if(rhs._is_empty() or _is_empty()) {
                return BBB();
            }
            else {
                vector<double> vv{a*rhs.a, a*rhs.b, b*rhs.a, b*rhs.b};
                double _a = *min_element(vv.begin(), vv.end());
                double _b = *max_element(vv.begin(), vv.end());
                return BBB(_a,_b);
            }

        }
    BBB operator/(const BBB& rhs) const {
            double _a,_b;
            if(rhs._is_empty() or _is_empty()) {
                return BBB();
            }
            else if (rhs.a*rhs.b <= 0) {
                _a = -std::numeric_limits<double>::infinity();
                _b = std::numeric_limits<double>::infinity();
            }
            else {
                vector<double> vv{a/rhs.a, a/rhs.b, b/rhs.a, b/rhs.b};
                _a = *min_element(vv.begin(), vv.end());
                _b = *max_element(vv.begin(), vv.end());
            }
            return BBB(_a,_b);
        }
    bool operator!=(const BBB& rhs) {return !((a==b) && (rhs.a==rhs.b) && (a==rhs.a));}  // Returning true if 'Any'
    bool operator==(const BBB& rhs) {return !((a>rhs.b) || (b<rhs.a));}  // Returning true if 'Any'
    bool operator>=(const BBB& rhs) {return !(b<rhs.a);}  // Returning true if 'Any'
    bool operator<=(const BBB& rhs) {return !(a>rhs.b);}  // Returning true if 'Any'
    bool operator>(const BBB& rhs) {return !(b<=rhs.a);}    // Returning true if 'Any'
    bool operator<(const BBB& rhs) {return !(a>=rhs.b);}    // Returning true if 'Any'
    BBB _cos() {
            if(_is_empty()) {
                return BBB();
            }
            // Find values of n s.t. a <= n*pi <=b
            int n_min = ceil(a/PI);
            int n_max = floor(b/PI);
            vector<double> vv{cos(a), cos(b)};
            if(n_min <= n_max) {
                if(n_min%2==0)
                    vv.push_back(1);
                if(n_min%2==1)
                    vv.push_back(-1);
                if(n_max%2==0)
                    vv.push_back(1);
                if(n_max%2==1)
                    vv.push_back(-1);
            }
            double _a = *min_element(vv.begin(), vv.end());
            double _b = *max_element(vv.begin(), vv.end());
            return BBB(_a,_b);
        }
    BBB _sin() {
            if(_is_empty()) {
                return BBB();
            }
            // Find values of n s.t. a <= (n+0.5)*pi <=b
            int n_min = ceil(a/PI-0.5);
            int n_max = floor(b/PI-0.5);
            vector<double> vv{sin(a), sin(b)};
            if(n_min <= n_max) {
                if(n_min%2==0)
                    vv.push_back(1);
                if(n_min%2==1)
                    vv.push_back(-1);
                if(n_max%2==0)
                    vv.push_back(1);
                if(n_max%2==1)
                    vv.push_back(-1);
            }
            double _a = *min_element(vv.begin(), vv.end());
            double _b = *max_element(vv.begin(), vv.end());
            return BBB(_a,_b);
        }
    BBB _sqrt() {
            if(_is_empty()) {
                return BBB();
            }
            assert(a>=0);
            return BBB(sqrt(a),sqrt(b));
        }
    BBB _square() {
            if(_is_empty()) {
                return BBB();
            }
            vector<double> vv{a*a, b*b};
            if (a*b<=0) {
                vv.push_back(0);
            }
            double _a = *min_element(vv.begin(), vv.end());
            double _b = *max_element(vv.begin(), vv.end());
            return BBB(_a,_b);
        }
    BBB _abs() {
            if(_is_empty()) {
                return BBB();
            }
            vector<double> vv{abs(a), abs(b)};
            if (a*b<=0) {
                vv.push_back(0);
            }
            double _a = *min_element(vv.begin(), vv.end());
            double _b = *max_element(vv.begin(), vv.end());
            return BBB(_a,_b);
        }
    double _min() {return a;}
    double _max() {return b;}
    BBB _union(const BBB& rhs) const {
            double _a = min(a,rhs.a);
            double _b = max(b,rhs.b);
            return BBB(_a, _b);
        }
    BBB _intersect(const BBB& rhs) const {
            double _a = max(a,rhs.a);
            double _b = min(b,rhs.b);
            return BBB(_a, _b);
        }
    bool _is_empty() const {return a>b;}
    bool _is_finite() const {return isfinite(a) && isfinite(b);}
};
std::ostream &operator<<(std::ostream &os, BBB const &m) {
    return os << "[" << m.a << "," << m.b << "]";
}


struct {
    vector<BBB> boxes;
    // TODO: Hashmap for boxes_operators
    map<tuple<int,int>, int> hashmap_mult;
    map<int, int> hashmap_cos;
    map<int, int> hashmap_sin;
    int mult(int i, int j) {
        auto tt = make_tuple(i,j);
        auto tt1 = make_tuple(j,i);
        auto it = hashmap_mult.find(tt);
        if(it != hashmap_mult.end())
            return it->second;

        if(num_boxes()>1000){
            return -1;
        }

        BBB mbox = boxes[i]*boxes[j];
        boxes.push_back(mbox);
        int nid = boxes.size()-1;
        hashmap_mult[tt] = nid;
        hashmap_mult[tt1] = nid;
        return nid;
    }
    int box(double i, double j) {
        assert(i<j);
        boxes.push_back(BBB(i,j));
        int nid = boxes.size()-1;
        return nid;
    }
    int num_boxes() {
        return boxes.size();
    }
    int cos(int i) {
        auto it = hashmap_cos.find(i);
        if(it != hashmap_cos.end())
            return it->second;
        BBB mbox = boxes[i]._cos();
        boxes.push_back(mbox);
        int nid = boxes.size()-1;
        hashmap_cos[i] = nid;
        return nid;
    }
    int sin(int i) {
        auto it = hashmap_sin.find(i);
        if(it != hashmap_sin.end())
            return it->second;
        BBB mbox = boxes[i]._sin();
        boxes.push_back(mbox);
        int nid = boxes.size()-1;
        hashmap_sin[i] = nid;
        return nid;
    }
    void reset() {
        boxes.clear();
        hashmap_cos.clear();
        hashmap_sin.clear();
        hashmap_mult.clear();
    }
} ZonoManager;

double get_with_default(const map<int,double>&m, int key) {
    auto it = m.find( key );
    if ( it == m.end() ) {
        return 0;
    }
    else {
        return it->second;
    }
}

class Zono {
public:
    // value is v0 + \sum_i coeffs[i]*BOXES[box_ids[i]]
    double v0;
    map<int, double> coeffs_map;
    double ub, lb; // Upper and lower bounds
    bool empty, full;

    void make_empty() {
            lb = std::numeric_limits<double>::infinity();
            ub = -std::numeric_limits<double>::infinity();
            empty = true;
            coeffs_map.clear();
        }
    void make_full() {
            lb = -std::numeric_limits<double>::infinity();
            ub = std::numeric_limits<double>::infinity();
            full = true;
            coeffs_map.clear();
        }
    Zono():v0(0),ub(0),lb(0),empty(true),full(false){make_empty();};
    Zono(double x):v0(x),ub(x),lb(x),empty(false),full(false){};
    Zono(double x, double y):v0(0),ub(y),lb(x),empty(false),full(false){
            if(y<x) {
                make_empty();
            }
            else if (y!=x) {
                int bid = ZonoManager.box(x, y);
                coeffs_map[bid] = 1;
            }
        };
    Zono(double _v0, const map<int, double>& _coeffs_map):v0(_v0),coeffs_map(_coeffs_map),empty(false),full(false){
        compute_lb_ub();
    }
    void compute_lb_ub() {
        if (empty) {
            make_empty();
        }
        else if(full) {
            make_full();
        }
        else {
            double _lb = v0;
            double _ub = v0;
            for(auto i:support()) {
                BBB bb = ZonoManager.boxes[i] * get_with_default(coeffs_map, i);
                _lb += bb.a;
                _ub += bb.b;
            }
            lb = _lb;
            ub = _ub;
        }
    }
    Zono operator-() const {
        if (empty) {
            return Zono();
        }
        if (full) {
            return *this;
        }
        double _v0 = -v0;
        map<int, double> _coeffs_map;
        for(auto i:support()) {
            auto tt = coeffs_map.find(i);
            double _c = get_with_default(this->coeffs_map, i);
            if (_c!=0) {
                _coeffs_map[i] = -_c;
            }
        }
        return Zono(_v0, _coeffs_map);
    }
    Zono operator+(const Zono& rhs) const {
        if (empty || rhs.empty) {
            return Zono();
        }
        if (full) {
            return *this;
        }
        if (rhs.full) {
            return rhs;
        }
        double _v0 = v0 + rhs.v0;
        map<int, double> _coeffs_map;

        auto s1 = support();
        auto s2 = rhs.support();
        vector<int> sm(s1.size() + s2.size());
        auto sm_it = set_union (s1.begin(),s1.end(),s2.begin(),s2.end(),sm.begin());
        sm.resize(sm_it-sm.begin());

        for(auto i:sm) {
            double c1 = get_with_default(coeffs_map, i);
            double c2 = get_with_default(rhs.coeffs_map, i);
            double _c = c1+c2;
            if (_c!=0) {
                _coeffs_map[i] = _c;
            }
        }
        return Zono(_v0, _coeffs_map);
    }
    void operator+=(const Zono& rhs) {
        if (empty || rhs.empty) {
            return;
        }
        if (full) {
            return;
        }
        if (rhs.full) {
            make_full();
            return;
        }
        double _v0 = v0 + rhs.v0;
        map<int, double> _coeffs_map;

        auto s1 = support();
        auto s2 = rhs.support();
        vector<int> sm(s1.size() + s2.size());
        auto sm_it = set_union (s1.begin(),s1.end(),s2.begin(),s2.end(),sm.begin());
        sm.resize(sm_it-sm.begin());

        for(auto i:sm) {
            double c1 = get_with_default(coeffs_map, i);
            double c2 = get_with_default(rhs.coeffs_map, i);
            double _c = c1+c2;
            if (_c!=0) {
                _coeffs_map[i] = _c;
            }
        }
        v0 = _v0;
        coeffs_map = _coeffs_map;
        compute_lb_ub();
    }
    Zono operator-(const Zono& rhs) const {
        if (empty || rhs.empty) {
            return Zono();
        }
        if (full) {
            return *this;
        }
        if (rhs.full) {
            return rhs;
        }
        double _v0 = v0 - rhs.v0;
        map<int, double> _coeffs_map;

        auto s1 = support();
        auto s2 = rhs.support();
        vector<int> sm(s1.size() + s2.size());
        auto sm_it = set_union (s1.begin(),s1.end(),s2.begin(),s2.end(),sm.begin());
        sm.resize(sm_it-sm.begin());

        for(auto i:sm) {
            double c1 = get_with_default(coeffs_map, i);
            double c2 = get_with_default(rhs.coeffs_map, i);
            double _c = c1-c2;
            if (_c!=0) {
                _coeffs_map[i] = _c;
            }
        }
        return Zono(_v0, _coeffs_map);
    }
    Zono operator*(const Zono& rhs) const {
        if (empty || rhs.empty) {
            return Zono();
        }
        if ((rhs.is_single_val() && rhs.lb==0)
            || (is_single_val() && lb==0)) {
            return Zono(0);
        }
        if (full) {
            return *this;
        }
        if (rhs.full) {
            return rhs;
        }
        double _v0 = v0 * rhs.v0;
        map<int, double> _coeffs_map;
        BBB out_of_memory_stuff(0);

        vector<int> s1 = support();
        vector<int> s2 = rhs.support();
        vector<int> sm(s1.size() + s2.size());
        vector<int> si(s1.size() + s2.size());
        auto sm_it = set_union (s1.begin(),s1.end(),s2.begin(),s2.end(),sm.begin());
        sm.resize(sm_it-sm.begin());
        // auto si_it = set_intersection (s1.begin(),s1.end(),s2.begin(),s2.end(),si.begin());
        // si.resize(si_it-si.begin());

        for(auto i:sm) {
            double c1 = get_with_default(coeffs_map, i);
            double c2 = get_with_default(rhs.coeffs_map, i);
            double _c = c1*rhs.v0 + c2*v0;
            if (_c!=0) {
                _coeffs_map[i] = _c;
            }
        }
        if (ZonoManager.num_boxes() < MAX_ZONO_SIZE) {
            for(auto i:s1) {
                for(auto j:s2) {
                    double c1 = get_with_default(coeffs_map, i);
                    double c2 = get_with_default(rhs.coeffs_map, j);
                    double _c = c1*c2;
                    if (_c!=0) {
                        int bid = ZonoManager.mult(i,j);
                        if (bid > 0) {
                            _coeffs_map[bid] = get_with_default(_coeffs_map, bid) + _c;
                        }
                        else {
                            assert(bid == -1); // Max size reached
                            out_of_memory_stuff += ZonoManager.boxes[i]*ZonoManager.boxes[j]*_c;
                        }
                    }
                }
            }
        }
        else {
            BBB bb1(0);
            BBB bb2(0);
            for(auto i:s1) {
                bb1 += ZonoManager.boxes[i]*get_with_default(coeffs_map, i);
            }
            for(auto i:s2) {
                bb2 += ZonoManager.boxes[i]*get_with_default(rhs.coeffs_map, i);
            }
            out_of_memory_stuff += bb1*bb2;
        }
        return Zono(_v0, _coeffs_map) + Zono(out_of_memory_stuff.a, out_of_memory_stuff.b);
    }
    Zono operator/(const Zono& rhs) const {
        if (empty || rhs.empty) {
            return Zono();
        }
        if (full || (rhs.lb <=0 && rhs.ub>=0)) {
            Zono res(0);
            res.make_full();
            return res;
        }
        if (rhs.is_single_val()) {
            double _v0 = v0/rhs.lb;
            map<int, double> _coeffs_map;
            for(auto i:support()) {
                double c1 = get_with_default(coeffs_map, i);
                double _c = c1/rhs.lb;
                if (_c!=0) {
                    _coeffs_map[i] = _c;
                }
            }
            return Zono(_v0, _coeffs_map);
        }
        BBB c = BBB(lb, ub)/BBB(rhs.lb, rhs.ub);
        return Zono(c.a, c.b);                  // TODO: FIX
    }
    bool is_single_val() const {return (!empty && (lb==ub));}
    bool operator!=(const Zono& rhs) {return !((lb==ub) && (rhs.lb==rhs.ub) && (lb==rhs.lb));}  // Returning true if 'Any'
    bool operator==(const Zono& rhs) {return !((lb>rhs.ub) || (ub<rhs.lb));}  // Returning true if 'Any'
    bool operator>=(const Zono& rhs) {return !(ub<rhs.lb);}  // Returning true if 'Any'
    bool operator<=(const Zono& rhs) {return !(lb>rhs.ub);}  // Returning true if 'Any'
    bool operator>(const Zono& rhs) {return !(ub<=rhs.lb);}    // Returning true if 'Any'
    bool operator<(const Zono& rhs) {return !(lb>=rhs.ub);}    // Returning true if 'Any'

    vector<int> nonzero_coeffs() {
        vector<int> res;
        for(int i=0; i<ZonoManager.num_boxes(); i++) {
            double c1 = get_with_default(coeffs_map, i);
            if(c1!=0)
                res.push_back(i);
        }
        return res;
    }

    Zono _cos() {
            auto nz = nonzero_coeffs();
            if(v0==0 && nz.size()==1 && coeffs_map[nz[0]]==1) {
                int bid = ZonoManager.cos(nz[0]);
                Zono res(0);
                res.coeffs_map[bid] = 1;
                res.compute_lb_ub();
                return res;
            }
            else {
                BBB c = BBB(lb, ub)._cos();
                return Zono(c.a, c.b);
            }
        }
    Zono _sin() {
            auto nz = nonzero_coeffs();
            if(v0==0 && nz.size()==1 && coeffs_map[nz[0]]==1) {
                int bid = ZonoManager.sin(nz[0]);
                Zono res(0);
                res.coeffs_map[bid] = 1;
                res.compute_lb_ub();
                return res;
            }
            else {
                BBB c = BBB(lb, ub)._sin();
                return Zono(c.a, c.b);
            }
        }
    Zono _sqrt() {
        // TODO: Make Better
            BBB c = BBB(lb, ub)._sqrt();
            return Zono(c.a, c.b);
        }
    Zono _square() {
        // if(!empty and !full) {
        //     cout <<"oho"<<endl;
        // }
            return (*this)*(*this);
        }
    Zono _abs() {
        // TODO: Make Better
            BBB c = BBB(lb, ub)._abs();
            return Zono(c.a, c.b);
        }
    double _min() {return lb;}
    double _max() {return ub;}
    Zono _union(const Zono& rhs) const {
        // TODO: Make Better
            BBB c = BBB(lb, ub)._union(BBB(rhs.lb, rhs.ub));
            return Zono(c.a, c.b);
        }
    Zono _intersect(const Zono& rhs) const {
        // TODO: Make Better
            BBB c = BBB(lb, ub)._intersect(BBB(rhs.lb, rhs.ub));
            return Zono(c.a, c.b);
        }
    bool _is_empty() const {return lb>ub;}
    bool _is_finite() const {return isfinite(lb) && isfinite(ub);}
    vector<int> support() const {
        pair<int,double> me; // what a map<int, int> is made of
        vector<int> v;
        BOOST_FOREACH(me, coeffs_map) {
            v.push_back(me.first);
        }
        sort(v.begin(), v.end());
        return v;
    }
};
std::ostream &operator<<(std::ostream &os, Zono const &m) {
    return os << "Z[" << m.lb << "," << m.ub << "]";
}

template<typename T>
T cos(T x) {return x._cos();};
template<typename T>
T sin(T x) {return x._sin();};
template<typename T>
T sqrt(T x) {return x._sqrt();};
template<typename T>
T square(T x) {return x._square();};
template<>
double square(double x) {return x*x;};
template<typename T>
T abs(T x) {return x._abs();};
template<>
double abs(double x) {return abs(x);};
template<typename T>
T _union(T x, T y) {return x._union(y);};
template<>
double _union<double>(double x, double y) { return x;}
template<typename T>
T _intersect(T x, T y) {return x._intersect(y);};
template<>
double _intersect<double>(double x, double y) { return x;}
template<typename T>
bool _is_empty(T x) {return x._is_empty();};
template<>
bool _is_empty<double>(double x) { return false;}

template <typename T>
class Point3d {
public:
    T x, y, z;
    Point3d():x(0),y(0),z(0){};
    Point3d(T _x, T _y, T _z):x(_x),y(_y),z(_z){};
    void print(ostream &s, string end = "") const {
        s << "(" << x << "," << y << "," << z << ")" << end;
    }
    void operator+=(const Point3d<T>& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
    }
    void operator-=(const Point3d<T>& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
    }
    void operator+=(const T& rhs) {
        x += rhs;
        y += rhs;
        z += rhs;
    }
    void operator-=(const T& rhs) {
        x -= rhs;
        y -= rhs;
        z -= rhs;
    }
    void operator*=(const T& rhs) {
        x *= rhs;
        y *= rhs;
        z *= rhs;
    }
    Point3d<T> operator+(const Point3d<T>& rhs) const {
        T _x = x+rhs.x;
        T _y = y+rhs.y;
        T _z = z+rhs.z;
        return Point3d<T>(_x,_y,_z);
    }
    Point3d<T> operator-(const Point3d<T>& rhs) const {
        T _x = x-rhs.x;
        T _y = y-rhs.y;
        T _z = z-rhs.z;
        return Point3d<T>(_x,_y,_z);
    }
    Point3d<T> operator+(const T& rhs) const {
        T _x = x+rhs;
        T _y = y+rhs;
        T _z = z+rhs;
        return Point3d<T>(_x,_y,_z);
    }
    Point3d<T> operator-(const T& rhs) const {
        T _x = x-rhs;
        T _y = y-rhs;
        T _z = z-rhs;
        return Point3d<T>(_x,_y,_z);
    }
    Point3d<T> operator-() const {
        T _x = -x;
        T _y = -y;
        T _z = -z;
        return Point3d<T>(_x,_y,_z);
    }
    Point3d<T> operator*(const T& rhs) const {
        T _x = x*rhs;
        T _y = y*rhs;
        T _z = z*rhs;
        return Point3d<T>(_x,_y,_z);
    }
    Point3d<T> operator/(const T& rhs) const {
        T _x = x/rhs;
        T _y = y/rhs;
        T _z = z/rhs;
        return Point3d<T>(_x,_y,_z);
    }
    Point3d<T> operator/(const Point3d<T>& rhs) const {
        T _x = x/rhs.x;
        T _y = y/rhs.y;
        T _z = z/rhs.z;
        return Point3d<T>(_x,_y,_z);
    }
    Point3d<T> cross(const Point3d<T>& rhs) const {
        T _x = y*rhs.z - z*rhs.y;
        T _y = z*rhs.x - x*rhs.z;
        T _z = x*rhs.y - y*rhs.x;
        return Point3d<T>(_x,_y,_z);
    }
    T dot(const Point3d<T>& rhs) const {
        T _x = x * rhs.x;
        T _y = y * rhs.y;
        T _z = z * rhs.z;
        return _x + _y + _z;
    }
    T norm() const {
        return sqrt(square(x) + square(y) + square(z));
    }
};

class Face {
public:
    int v0, v1, v2;
    Face():v0(-1),v1(-1),v2(-1){};
    Face(int _v0, int _v1, int _v2):v0(_v0),v1(_v1),v2(_v2){};
    void print(ostream &s, string end = "") {
        s << "(" << v0 << "," << v1 << "," << v2 << ")" << end;
    }
};

template <typename T>
class Matrix33 {
public:
    Point3d<T> r0;
    Point3d<T> r1;
    Point3d<T> r2;
    Matrix33() {};
    Matrix33(Point3d<T> _r0, Point3d<T> _r1, Point3d<T> _r2):
        r0(_r0),r1(_r1),r2(_r2) {};
    Point3d<T> col0() const {
        return Point3d<T>(r0.x, r1.x, r2.x);
    }
    Point3d<T> col1() const {
        return Point3d<T>(r0.y, r1.y, r2.y);
    }
    Point3d<T> col2() const {
        return Point3d<T>(r0.z, r1.z, r2.z);
    }
    Matrix33<T> matmul(const Matrix33<T>& rhs) const {
        auto c0 = rhs.col0();
        auto c1 = rhs.col1();
        auto c2 = rhs.col2();
        Point3d<T> _r0(r0.dot(c0), r0.dot(c1), r0.dot(c2));
        Point3d<T> _r1(r1.dot(c0), r1.dot(c1), r1.dot(c2));
        Point3d<T> _r2(r2.dot(c0), r2.dot(c1), r2.dot(c2));
        return Matrix33<T>(_r0, _r1, _r2);
    }
    Point3d<T> dot(const Point3d<T>& rhs) const {
        return Point3d<T>(r0.dot(rhs), r1.dot(rhs), r2.dot(rhs));
    }
    Matrix33<T> t() { // Transpose
        Point3d<T> _rx(r0.x, r1.x, r2.x);
        Point3d<T> _ry(r0.y, r1.y, r2.y);
        Point3d<T> _rz(r0.z, r1.z, r2.z);
        return Matrix33<T>(_rx, _ry, _rz);
    }
    Matrix33<T> inverse() {
        T s00 = r1.y*r2.z - r2.y*r1.z;
        T s01 = r0.z*r2.y - r2.z*r0.y;
        T s02 = r0.y*r1.z - r1.y*r0.z;

        T s10 = r1.z*r2.x - r2.z*r1.x;
        T s11 = r0.x*r2.z - r2.x*r0.z;
        T s12 = r0.z*r1.x - r1.z*r0.x;

        T s20 = r1.x*r2.y - r2.x*r1.y;
        T s21 = r0.y*r2.x - r2.y*r0.x;
        T s22 = r0.x*r1.y - r1.x*r0.y;

        T d = r0.x*s00 + r0.y*s01 + r0.z*s02;

        auto _r0 = Point3d<T>(s00, s01, s02)/d;
        auto _r1 = Point3d<T>(s10, s11, s12)/d;
        auto _r2 = Point3d<T>(s20, s21, s22)/d;

        return Matrix33<T>(_r0, _r1, _r2);
    }
    void print(ostream&s) {
        r0.print(s, ";\n");
        r1.print(s, ";\n");
        r2.print(s, ";\n");
    }
};


struct {
    string mesh;
    string out_id;
    double rescale;
    double theta_max;
    double theta_eps;
    double delta_t_max;
    double delta_t_eps;
    double t_z;
    int img_h;
    int img_w;
    int img_upsample;
    double light_ambient;
    double light_color;
    Point3d<double> light_src;
    int debug_i;
    int debug_j;
} CONFIG;

void imshow(vector<vector<double>>& img) {
    int img_h = img.size();
    int img_w = img[0].size();
    cimg_library::CImg<unsigned char> image(img_h,img_w,1,1,0);
    cimg_forXY(image,x,y) { image(x,y) = (int)(min(max(img[y][x],0.),1.)*255); }
    image.display();
}
void imsave(vector<vector<double>>& img, string fname) {
    int img_h = img.size();
    int img_w = img[0].size();
    cimg_library::CImg<unsigned char> image(img_h,img_w,1,1,0);
    cimg_forXY(image,x,y) { image(x,y) = (int)(min(max(img[y][x],0.),1.)*255); }
    image.save(fname.c_str());
}


template <typename T>
vector<vector<T>> downsample(vector<vector<T>> image, int factor) {
    int img_h = image.size();
    int img_w = image[0].size();
    assert(img_h%factor==0);
    assert(img_w%factor==0);
    vector<vector<T>> result(img_h/factor, vector<T>(img_w/factor));
    for (int i = 0; i < img_h/factor; i++) {
        for (int j = 0; j < img_w/factor; j++) {
            T sum = T(0);
            T coeffs = T(0);
            for (int ii = 0; ii < factor; ii++) {
                for (int jj = 0; jj < factor; jj++) {
                    double coeff = (ii-factor/2.)*(ii-factor/2.) + (jj-factor/2.)*(jj-factor/2.);
                    coeff = exp(-coeff/(2 * factor/6. * factor/6.));
                    coeffs += coeff;
                    sum += T(coeff)*image[factor*i+ii][factor*j+jj];
                }
            }
            result[i][j] = sum/(coeffs);
        }
    }
    return result;
}

void parse_arguments(int argc, char const *argv[]) {
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Help screen")
      ("mesh", po::value<string>()->required(), "Mesh obj file")
      ("out_id", po::value<string>()->required(), "Output ID")
      ("rescale", po::value<float>()->default_value(0), "Mesh rescaling size")
      ("theta_max", po::value<float>()->default_value(90), "Rotation variation (in degrees)")
      ("theta_eps", po::value<float>()->default_value(10), "Rotation robustness perturbation (in degrees)")
      ("delta_t_max", po::value<float>()->default_value(0.5), "Translation variation")
      ("delta_t_eps", po::value<float>()->default_value(0.1), "Translation robustness perturbation")
      ("t_z", po::value<float>()->default_value(3), "Mean z translation")
      ("img_h", po::value<int>()->default_value(64), "Image height")
      ("img_w", po::value<int>()->default_value(64), "Image width")
      ("img_upsample", po::value<int>()->default_value(8), "Image upsampling factor")
      ("light_src_x", po::value<float>()->default_value(-1), "Light source intensity")
      ("light_src_y", po::value<float>()->default_value(-1), "Light source intensity")
      ("light_src_z", po::value<float>()->default_value(2.5), "Light source intensity")
      ("light_color", po::value<float>()->default_value(1), "Light source intensity")
      ("light_ambient", po::value<float>()->default_value(0.2), "Ambient Light")
      ("debug_i", po::value<int>()->default_value(20), "Image width")
      ("debug_j", po::value<int>()->default_value(20), "Image width")
      ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        exit(0);
    }

    CONFIG.mesh             = vm["mesh"].as<string>();
    CONFIG.out_id           = vm["out_id"].as<string>();
    CONFIG.rescale          = vm["rescale"].as<float>();
    CONFIG.theta_max        = vm["theta_max"].as<float>();
    CONFIG.theta_eps        = vm["theta_eps"].as<float>();
    CONFIG.delta_t_max      = vm["delta_t_max"].as<float>();
    CONFIG.delta_t_eps      = vm["delta_t_eps"].as<float>();
    CONFIG.t_z              = vm["t_z"].as<float>();
    CONFIG.img_h            = vm["img_h"].as<int>();
    CONFIG.img_w            = vm["img_w"].as<int>();
    CONFIG.img_upsample     = vm["img_upsample"].as<int>();
    CONFIG.light_ambient    = vm["light_ambient"].as<float>();
    CONFIG.light_color      = vm["light_color"].as<float>();
    double light_src_x      = vm["light_src_x"].as<float>();
    double light_src_y      = vm["light_src_y"].as<float>();
    double light_src_z      = vm["light_src_z"].as<float>();
    CONFIG.light_src        = Point3d<double>(light_src_x, light_src_y, light_src_z);
    CONFIG.debug_i          = vm["debug_i"].as<int>();
    CONFIG.debug_j          = vm["debug_j"].as<int>();

    cout << "CONFIG.mesh             : " << CONFIG.mesh             << endl;
    cout << "CONFIG.out_id           : " << CONFIG.out_id           << endl;
    cout << "CONFIG.rescale          : " << CONFIG.rescale          << endl;
    cout << "CONFIG.theta_max        : " << CONFIG.theta_max        << endl;
    cout << "CONFIG.theta_eps        : " << CONFIG.theta_eps        << endl;
    cout << "CONFIG.delta_t_max      : " << CONFIG.delta_t_max      << endl;
    cout << "CONFIG.delta_t_eps      : " << CONFIG.delta_t_eps      << endl;
    cout << "CONFIG.t_z              : " << CONFIG.t_z              << endl;
    cout << "CONFIG.img_h            : " << CONFIG.img_h            << endl;
    cout << "CONFIG.img_w            : " << CONFIG.img_w            << endl;
    cout << "CONFIG.img_upsample     : " << CONFIG.img_upsample     << endl;
    cout << "CONFIG.light_ambient    : " << CONFIG.light_ambient    << endl;
    cout << "CONFIG.light_color      : " << CONFIG.light_color      << endl;
    cout << "CONFIG.light_src        : "; CONFIG.light_src.print(cout, "\n");
    cout << "CONFIG.debug_i          : " << CONFIG.debug_i          << endl;
    cout << "CONFIG.debug_j          : " << CONFIG.debug_j          << endl;
}

template<typename T> T area(T x1, T y1, T x2, T y2, T x3, T y3) {
    return (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
}

template<typename T> bool lies_in_triangle(T x1, T y1, T x2, T y2, T x3, T y3, T xp, T yp) {
    T area123 = area(x1, y1, x2, y2, x3, y3);
    T area1 = area(xp, yp, x2, y2, x3, y3);
    T area2 = area(x1, y1, xp, yp, x3, y3);
    T area3 = area(x1, y1, x2, y2, xp, yp);
    T area_diff = area1 + area2 + area3 - area123;
    return area_diff==0;
}

template <typename T>
class Mesh {
public:
    vector<Point3d<T>> vertices;
    vector<Face> faces;
    vector<Point3d<T>> normals;
    Mesh(){};
    Mesh(vector<Point3d<T>> _v, vector<Face> _f):
        vertices(_v), faces(_f) {
            // Compute Normals
            for (auto f:faces) {
                auto edge_01 = vertices[f.v1]-vertices[f.v0];
                auto edge_02 = vertices[f.v2]-vertices[f.v0];
                auto normal = edge_01.cross(edge_02);
                normals.push_back(normal/normal.norm());
            }
        };
    Mesh(vector<Point3d<T>> _v, vector<Face> _f, vector<Point3d<T>> _n):
        vertices(_v), faces(_f), normals(_n) {};
    void print(ostream& s) const {
        s << "Vertices:\n";
        for(auto v:vertices) {
            v.print(s, "\n");
        }
        s << "\n\n";

        s << "Faces:\n";
        for(auto f:faces) {
            f.print(s, "\n");
        }
        s << "\n\n";

        s << "Normals:\n";
        for(auto v:normals) {
            v.print(s, "\n");
        }
        s << "\n\n";
    }
    void rotate(Matrix33<T> R) {
        for(auto&v : vertices) {
            v = R.dot(v);
        }
        for(auto&n : normals) {
            n = R.dot(n);
        }
    }
    void translate(Point3d<T> t) {
        for(auto&v : vertices) {
            v += t;
        }
    }
    void rescale(double scale) {
        if(scale==0)
            return;
        double max_v = 0;
        for(auto v:vertices) {
            max_v = max(max_v, abs(v.x));
            max_v = max(max_v, abs(v.y));
            max_v = max(max_v, abs(v.z));
        }
        cout << "Rescaling " << max_v << " -> " << scale << endl;
        for(auto&v:vertices) {
            v *= scale/max_v;
        }
    }
    vector<vector<T>> render(int img_h, int img_w,
                            double light_ambient = 0.2,
                            double light_color = 1,
                            Point3d<T> light_src = Point3d<T>(-1,-1,2),
                            int img_upsample = 4) {

        cout << "Rendering Image(" << img_h << "," << img_w << ") upsampled by "
            << img_upsample << "... " << endl;

        img_h *= img_upsample;
        img_w *= img_upsample;

        vector<vector<T>> image(img_h, vector<T>(img_w, 0));

        double img_wf = (float)img_w;
        double img_hf = (float)img_h;
        Matrix33<T> K(Point3d<T>(img_wf,  0,      img_wf/2),
                    Point3d<T>(0,      img_hf, img_hf/2),
                    Point3d<T>(0,      0,      1));
        auto K_inv = K.inverse();

        vector<Matrix33<T>> v012;
        vector<Matrix33<T>> v012_inv;
        for(auto&f:faces) {
            Matrix33<T> mm(vertices[f.v0], vertices[f.v1], vertices[f.v2]);
            v012.push_back(mm.t());
        }
        cout << "V012:" << endl;
        for(auto m:v012) {
            m.print(cout); cout << endl;
        }

        // cout << "Precomputing inverses... ";
        // for (int i = 0, done=-1; i < v012.size(); i++) {
        //     int doing =  (int)(i/((float)v012.size())*10);
        //     if(doing > done) {
        //         done = doing;
        //         cout << done*10 << " ";
        //         cout.flush();
        //     }
        //     v012_inv.push_back(v012[i].inverse());
        // }
        // cout << endl;

        // cout << "Inverses:" << endl;
        // for(auto m:v012_inv) {
        //     m.print(cout); cout << endl;
        // }

        cout << "ZonoManager Boxes: (" << ZonoManager.num_boxes() << ")" << endl;
        for(auto b:ZonoManager.boxes) {
            cout << b << endl;
        }

        cout << "Coloring... ";
        int done = -1;
        for (int i = 0, done=-1; i < img_h; i++) {
            int doing =  (int)(i/img_hf*10);
            if(doing > done) {
                done = doing;
                cout << done*10 << " ";
                cout.flush();
            }
            for (int j = 0; j < img_w; j++) {
                // cout << "j" << j << " "; cout.flush();
                // cout << "ZonoManager Boxes: (" << ZonoManager.num_boxes() << ")" << endl;
                // Project each face onto image
                vector<Point3d<T>> proj_j;
                vector<Point3d<T>> proj_i;
                for(auto m:v012) {
                    auto proj_homo = K.matmul(m);
                    auto p_j = proj_homo.r0/proj_homo.r2;
                    auto p_i = proj_homo.r1/proj_homo.r2;
                    proj_j.push_back(p_j);
                    proj_i.push_back(p_i);
                }

                // Check if (i,j) lies in (proj_i, proj_j)
                vector<int> over_possible_f;
                vector<int> under_possible_f;
                for (int f = 0; f < faces.size(); f++) {
                    T Ax = proj_i[f].x;
                    T Bx = proj_i[f].y;
                    T Cx = proj_i[f].z;
                    T Ay = proj_j[f].x;
                    T By = proj_j[f].y;
                    T Cy = proj_j[f].z;

                    // Check sector opp. A
                    T E0x = Bx-Ax;
                    T E0y = By-Ay;
                    T E1x = Cx-Ax;
                    T E1y = Cy-Ay;
                    T Epx = -Ax+i;
                    T Epy = -Ay+j;
                    T E0_cross_Ep = E0x*Epy - E0y*Epx;
                    T Ep_cross_E1 = Epx*E1y - Epy*E1x;
                    T inside_secA = E0_cross_Ep*Ep_cross_E1; // if > 0

                    // Check sector opp. B, using E0, E2
                    T E2x = Bx-Cx;
                    T E2y = By-Cy;
                    Epx = -Bx+i;
                    Epy = -By+j;
                    E0_cross_Ep = E0x*Epy - E0y*Epx;
                    T Ep_cross_E2 = Epx*E2y - Epy*E2x;
                    T inside_secB = E0_cross_Ep*Ep_cross_E2; // if > 0

                    // Check sector opp. C, using E1, E2
                    Epx = -Cx+i;
                    Epy = -Cy+j;
                    Ep_cross_E1 = Epx*E1y - Epy*E1x;
                    Ep_cross_E2 = Epx*E2y - Epy*E2x;
                    T inside_secC = Ep_cross_E1*Ep_cross_E2; // if > 0

                    // Compare sum of areas
                    T E1_cross_E2 = E1x*E2y - E1y*E2x;
                    T areaC = abs(E0_cross_Ep);
                    T areaB = abs(Ep_cross_E1);
                    T areaA = abs(Ep_cross_E2);
                    T areaABC = abs(E1_cross_E2);
                    T area_diff = abs(areaA+areaB+areaC-areaABC);

                    if(i==CONFIG.debug_i && j==CONFIG.debug_j) {
                        cout << "\nA"<<f<<" " <<areaA << " " << areaB << " " << areaC << endl;
                        cout << areaABC << " " << area_diff << endl;
                    }

                    // if ((inside_secA>0) && (inside_secB>0) && (inside_secC>0) && ((area_diff-EPS)<0)) {
                    if (((area_diff-EPS)<0)) {
                    // if (lies_in_triangle<T>(Ax,Ay,Bx,By,Cx,Cy,T((double)i),T((double)j))) {
                        over_possible_f.push_back(f);
                        under_possible_f.push_back(f);
                    }

                    // if ((inside_secA>0) && (inside_secB>0) && (inside_secC>0) && ((area_diff-EPS)<0)) {
                    // if (!((area_diff-EPS)>0)) {
                    // if (lies_in_triangle<T>(Ax,Ay,Bx,By,Cx,Cy,T((double)i),T((double)j))) {
                    // }
                }


                // Find point of interesection with each face
                //      ray     = K @ (v0;v1;v2) @ (a,b,c)
                // =>   (a,b,c) = (v0;v1;v2)_inv @ K_inv @ ray
                Point3d<T> ray(j,i,1);
                vector<Point3d<T>> coeffs;
                if(v012_inv.size() == v012.size()) {
                    for(auto&mm_inv:v012_inv) {
                        auto abc = mm_inv.dot(K_inv.dot(ray));
                        coeffs.push_back(abc);
                    }
                }
                else {
                    for(auto&mm:v012) {
                        auto mm_inv = mm.inverse();
                        auto abc = mm_inv.dot(K_inv.dot(ray));
                        coeffs.push_back(abc);
                        v012_inv.push_back(mm_inv);
                    }
                }


                // Distances and colors of all intersecting triangles
                vector<T> over_distances;
                vector<T> under_distances;
                vector<T> over_colors;
                vector<T> under_colors;
                vector<int> over_fff;
                vector<int> under_fff;

                for (auto f:over_possible_f) {
                    auto abc = coeffs[f];
                    T sum = abc.x + abc.y + abc.z;
                    if(sum != 0) {
                        abc = abc/sum;
                        if(abc.x>=0 && abc.y>=0 && abc.z>=0) {
                            // Ray intersects triangle
                            auto P = v012[f].dot(abc);
                            T dist = square(P.x) + square(P.y) + square(P.z);

                            // Use distances of vertices also
                            Point3d v0 = vertices[faces[f].v0];
                            Point3d v1 = vertices[faces[f].v1];
                            Point3d v2 = vertices[faces[f].v2];
                            T d0 = square(v0.x) + square(v0.y) + square(v0.z);
                            T d1 = square(v1.x) + square(v1.y) + square(v1.z);
                            T d2 = square(v2.x) + square(v2.y) + square(v2.z);
                            T v_dist = _union(d0, _union(d1, d2));
                            T final_distance = _intersect(dist, v_dist);
                            if (_is_empty(final_distance)) {
                                continue;
                            }
                            over_distances.push_back(final_distance);
                            if(i==CONFIG.debug_i && j==CONFIG.debug_j) {
                                cout << "abc "; abc.print(cout, "\n");
                                cout << "sum " << sum  << endl;
                                cout << "Distances " << f << endl;
                                cout << d0 << endl;
                                cout << d1 << endl;
                                cout << d2 << endl;
                                cout << v_dist << endl;
                                cout << dist << endl;
                                P.print(cout, "\n");
                            }

                            auto light_ray = light_src;
                            light_ray = light_ray/light_ray.norm();
                            T ii = normals[f].dot(light_ray);
                            // ii = max(ii, 0.);
                            ii = (ii+1)/2.;
                            T color = ii*light_color + (-ii+1)*light_ambient;
                            over_colors.push_back(color);
                            over_fff.push_back(f);
                        }
                    }
                }

                for (auto f:under_possible_f) {
                    auto abc = coeffs[f];
                    T sum = abc.x + abc.y + abc.z;
                    if(!(sum == 0)) {
                        abc = abc/sum;
                        if(!(abc.x<0) && !(abc.y<0) && !(abc.z<0)) {
                            // Ray intersects triangle
                            auto P = v012[f].dot(abc);
                            T dist = square(P.x) + square(P.y) + square(P.z);

                            // Use distances of vertices also
                            Point3d v0 = vertices[faces[f].v0];
                            Point3d v1 = vertices[faces[f].v1];
                            Point3d v2 = vertices[faces[f].v2];
                            T d0 = square(v0.x) + square(v0.y) + square(v0.z);
                            T d1 = square(v1.x) + square(v1.y) + square(v1.z);
                            T d2 = square(v2.x) + square(v2.y) + square(v2.z);
                            T v_dist = _union(d0, _union(d1, d2));
                            if(i==CONFIG.debug_i && j==CONFIG.debug_j) {
                                cout << "abc "; abc.print(cout, "\n");
                                cout << "sum " << sum  << endl;
                                cout << "Distances " << f << endl;
                                cout << d0 << endl;
                                cout << d1 << endl;
                                cout << d2 << endl;
                                cout << v_dist << endl;
                                cout << dist << endl;
                                P.print(cout, "\n");
                            }
                            T final_distance = dist;//_intersect(dist, v_dist);
                            under_distances.push_back(final_distance);
                            if (_is_empty(final_distance)) {
                                continue;
                            }

                            auto light_ray = light_src;
                            light_ray = light_ray/light_ray.norm();
                            T ii = normals[f].dot(light_ray);
                            // ii = max(ii, 0.);
                            ii = (ii+1)/2.;
                            T color = ii*light_color + (-ii+1)*light_ambient;
                            under_colors.push_back(color);
                            under_fff.push_back(f);
                        }
                    }
                }

                if(i==CONFIG.debug_i && j==CONFIG.debug_j) {
                    cout << "\n\n" << i << " " << j;
                    cout << "\ninverses:" << endl;
                    for(auto m:v012_inv) {m.print(cout); cout << endl;}
                    cout << "\ncoeffs" << endl;
                    for(auto v:coeffs) v.print(cout,"\n");
                    cout << "\nover_possible_f" << endl;
                    for(auto v:over_possible_f) cout << v << " ";
                    cout << "\nover_Distances" << endl;
                    for(auto v:over_distances) cout << v << endl;
                    cout << "\nover_Colors" << endl;
                    for(auto v:over_colors) cout << v << endl;
                    cout << "\nover_Faces" << endl;
                    for(auto v:over_fff) cout << v << endl;
                    cout << "\nunder_possible_f" << endl;
                    for(auto v:under_possible_f) cout << v << " ";
                    cout << "\nunder_Distances" << endl;
                    for(auto v:under_distances) cout << v << endl;
                    cout << "\nunder_Colors" << endl;
                    for(auto v:under_colors) cout << v << endl;
                    cout << "\nunder_Faces" << endl;
                    for(auto v:under_fff) cout << v << endl;
                }

                // Choose color corresponding to minimum distance
                vector<T> over_possible_min_d;
                vector<T> over_possible_min_c;
                vector<T> under_possible_min_d;
                vector<T> under_possible_min_c;
                for (int fi = 0; fi < over_distances.size(); fi++) {
                    bool flag = true;
                    for (int fii = 0; fii < over_distances.size(); fii++) {
                        if((fi!=fii) && !(over_distances[fi] < over_distances[fii])) {
                            flag=false;
                            break;
                        }
                    }
                    if(flag) {
                        over_possible_min_c.push_back(over_colors[fi]);
                        over_possible_min_d.push_back(over_distances[fi]);
                    }
                }
                for (int fi = 0; fi < under_distances.size(); fi++) {
                    bool flag = true;
                    for (int fii = 0; fii < under_distances.size(); fii++) {
                        if((fi!=fii) && (under_distances[fi] >= under_distances[fii])) {
                            flag=false;
                            break;
                        }
                    }
                    if(flag) {
                        under_possible_min_c.push_back(under_colors[fi]);
                        under_possible_min_d.push_back(under_distances[fi]);
                    }
                }

                if(i==CONFIG.debug_i && j==CONFIG.debug_j) {
                    cout << "over_possible_min_d" << endl;
                    for(auto v:over_possible_min_d) cout << v << endl;
                    cout << "over_possible_min_c" << endl;
                    for(auto v:over_possible_min_c) cout << v << endl;
                    cout << "under_possible_min_d" << endl;
                    for(auto v:under_possible_min_d) cout << v << endl;
                    cout << "under_possible_min_c" << endl;
                    for(auto v:under_possible_min_c) cout << v << endl;
                }

                T over_min_c;
                T under_min_c;
                if(over_possible_min_c.size()>0) {
                    over_min_c = over_possible_min_c[0];
                    for(auto mc:over_possible_min_c) {
                        over_min_c = _union(over_min_c, mc);
                    }
                }
                else {
                    over_min_c = T(0);
                }
                if(under_possible_min_c.size()>0) {
                    under_min_c = under_possible_min_c[0];
                    for(auto mc:under_possible_min_c) {
                        under_min_c = _union(under_min_c, mc);
                    }
                }
                else {
                    under_min_c = T(0);
                }
                if(i==CONFIG.debug_i && j==CONFIG.debug_j) {
                    cout << "over_min_c: " << over_min_c << endl;
                    cout << "under_min_c: " << under_min_c << endl;
                }
                // ////////////////////////////////////////////////////
                // /////////////// END: OverApproximation /////////////
                // ////////////////////////////////////////////////////

                image[i][j] = _union(over_min_c, under_min_c);
            }
        }
        cout << endl;

        image = downsample<T>(image, img_upsample);

        return image;
    }
};

template <typename T>
Matrix33<T> R_from_thetaX_abstract(T theta) {
    T c = cos(theta);
    T s = sin(theta);
    auto r0 = Point3d<T>(1,0,0);
    auto r1 = Point3d<T>(0,c,-s);
    auto r2 = Point3d<T>(0,s,c);
    return Matrix33<T>(r0,r1,r2);
}
template <typename T>
Matrix33<T> R_from_thetaY_abstract(T theta) {
    T c = cos(theta);
    T s = sin(theta);
    auto r0 = Point3d<T>(c,0,s);
    auto r1 = Point3d<T>(0,1,0);
    auto r2 = Point3d<T>(-s,0,c);
    return Matrix33<T>(r0,r1,r2);
}
template <typename T>
Matrix33<T> R_from_thetaZ_abstract(T theta) {
    T c = cos(theta);
    T s = sin(theta);
    auto r0 = Point3d<T>(c,-s,0);
    auto r1 = Point3d<T>(s,c,0);
    auto r2 = Point3d<T>(0,0,1);
    return Matrix33<T>(r0,r1,r2);
}
template <typename T>
Matrix33<T> R_from_thetas_abstract(Point3d<T> thetas) {
    auto Rx = R_from_thetaX_abstract(thetas.x);
    auto Ry = R_from_thetaY_abstract(thetas.y);
    auto Rz = R_from_thetaZ_abstract(thetas.z);
    auto Ryx = Ry.matmul(Rx);
    auto Rzyx = Rz.matmul(Ryx);
    return Rzyx;
}

Matrix33<double> R_from_thetaX(double theta) {
    double c = cos(theta);
    double s = sin(theta);
    auto r0 = Point3d<double>(1,0,0);
    auto r1 = Point3d<double>(0,c,-s);
    auto r2 = Point3d<double>(0,s,c);
    return Matrix33(r0,r1,r2);
}
Matrix33<double> R_from_thetaY(double theta) {
    double c = cos(theta);
    double s = sin(theta);
    auto r0 = Point3d<double>(c,0,s);
    auto r1 = Point3d<double>(0,1,0);
    auto r2 = Point3d<double>(-s,0,c);
    return Matrix33(r0,r1,r2);
}
Matrix33<double> R_from_thetaZ(double theta) {
    double c = cos(theta);
    double s = sin(theta);
    auto r0 = Point3d<double>(c,-s,0);
    auto r1 = Point3d<double>(s,c,0);
    auto r2 = Point3d<double>(0,0,1);
    return Matrix33(r0,r1,r2);
}
Matrix33<double> R_from_thetas(Point3d<double> thetas) {
    auto Rx = R_from_thetaX(thetas.x);
    auto Ry = R_from_thetaY(thetas.y);
    auto Rz = R_from_thetaZ(thetas.z);
    return Rz.matmul(Ry.matmul(Rx));
}

Mesh<double> load_mesh(string inputfile) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str(), NULL, true);

    if (!warn.empty()) {
    std::cout << warn << std::endl;
    }

    if (!err.empty()) {
    std::cerr << err << std::endl;
    }

    if (!ret) {
    exit(1);
    }

    auto shape = shapes[0];
    vector<Point3d<double>> vertices;
    vector<Face> faces;

    assert(attrib.vertices.size()%3==0);
    int N = attrib.vertices.size()/3;
    int M = shape.mesh.num_face_vertices.size();

    // Loop over vertices
    for (size_t i = 0; i < N; i++) {
        double x = attrib.vertices[3*i+0];
        double y = attrib.vertices[3*i+1];
        double z = attrib.vertices[3*i+2];
        vertices.push_back(Point3d<double>(x,y,z));
    }

    // Loop over faces
    for (size_t f = 0; f < M; f++) {
        assert(shape.mesh.num_face_vertices[f] == 3);
        int v0 = shape.mesh.indices[3*f + 0].vertex_index;
        int v1 = shape.mesh.indices[3*f + 1].vertex_index;
        int v2 = shape.mesh.indices[3*f + 2].vertex_index;
        faces.push_back(Face(v0,v1,v2));
    }

    return Mesh(vertices, faces);
}

template<typename T>
Point3d<T> make_point_abstract(Point3d<double> p) {
    return Point3d<T>(T(p.x), T(p.y), T(p.z));
}
template<typename T>
Matrix33<T> make_matrix_abstract(Matrix33<double> m) {
    return Matrix33<T>(make_point_abstract<T>(m.r0),
                        make_point_abstract<T>(m.r1),
                        make_point_abstract<T>(m.r2));
}
template<typename T>
Mesh<T> make_mesh_abstract(Mesh<double> mesh) {
    vector<Point3d<T>> vertices;
    vector<Point3d<T>> normals;
    for(auto v:mesh.vertices) {
        vertices.push_back(make_point_abstract<T>(v));
    }
    for(auto v:mesh.normals) {
        normals.push_back(make_point_abstract<T>(v));
    }
    return Mesh<T>(vertices, mesh.faces, normals);
}

// Choose Abstraction
#ifdef ABSTRACT_ZONO
typedef Zono AbstractClass;
#else
typedef BBB AbstractClass;
#endif

int main(int argc, char const *argv[])
{
    parse_arguments(argc, argv);
    // string mesh_file = "3dmodels/box_stack.obj";
    // string mesh_file = "../3dmodels/cuboid.obj";
    // string mesh_file = "../3dmodels/tetrahedron.obj";
    // string mesh_file = "../3dmodels/guitar.obj";

    auto mesh = load_mesh(CONFIG.mesh);
    mesh.rescale(CONFIG.rescale);
    cout << "Num Vertices: " << mesh.vertices.size() << endl;
    cout << "Num Faces:    " << mesh.faces.size() << endl;
    mesh.print(cout);

    default_random_engine generator;
    uniform_real_distribution<double> distribution(-1.0,1.0);

    for(int run_id=0; run_id<10; run_id++) {
        // Reset Managers
        ZonoManager.reset();

        auto mesh_copy = mesh;
        // R, t
        Point3d thetas(distribution(generator)*(CONFIG.theta_max)/180*PI,
                        distribution(generator)*(CONFIG.theta_max)/180*PI,
                        distribution(generator)*(CONFIG.theta_max)/180*PI);
        Point3d t(0.,0.,CONFIG.t_z);
        Point3d delta_t(distribution(generator)*CONFIG.delta_t_max,
                        distribution(generator)*CONFIG.delta_t_max,
                        distribution(generator)*CONFIG.delta_t_max);
        t += delta_t;

        cout << "thetas: "; thetas.print(cout, "\n");
        cout << "t:      "; t.print(cout, "\n");

        Matrix33<double> R = R_from_thetas(thetas);

        // Apply R,t
        mesh_copy.rotate(R);
        mesh_copy.translate(t);

        // Perturb R,t
        double theta_max = CONFIG.theta_eps/180*PI;
        Point3d<AbstractClass> thetas_eps(AbstractClass(-theta_max, theta_max),
                                        AbstractClass(-theta_max, theta_max),
                                        AbstractClass(-theta_max, theta_max));
        Point3d<AbstractClass> t_eps(AbstractClass(-CONFIG.delta_t_eps, CONFIG.delta_t_eps),
                                        AbstractClass(-CONFIG.delta_t_eps, CONFIG.delta_t_eps),
                                        AbstractClass(-CONFIG.delta_t_eps, CONFIG.delta_t_eps));
        Matrix33<AbstractClass> R_eps = R_from_thetas_abstract(thetas_eps);
        Mesh<AbstractClass> mesh_abstract = make_mesh_abstract<AbstractClass>(mesh);

        mesh_abstract.print(cout);

        auto R_abs = make_matrix_abstract<AbstractClass>(R);
        auto t_abs = make_point_abstract<AbstractClass>(t);
        auto t_sum = t_abs+t_eps;
        mesh_abstract.rotate(R_abs);
        mesh_abstract.rotate(R_eps);
        mesh_abstract.translate(t_sum);

        cout << "thetas_eps:"; thetas_eps.print(cout, "\n");
        cout << "R_abs:"; R_abs.print(cout);
        cout << "R_eps:"; R_eps.print(cout);
        cout << "t_abs:"; t_abs.print(cout, "\n");
        cout << "t_eps:"; t_eps.print(cout, "\n");
        cout << "t_sum:"; t_sum.print(cout, "\n");

        mesh_abstract.print(cout);

        // Render
        auto img_concrete = mesh_copy.render(CONFIG.img_h, CONFIG.img_w,
                                    CONFIG.light_ambient,
                                    CONFIG.light_color,
                                    CONFIG.light_src,
                                    CONFIG.img_upsample);

        // Render
        auto img_abstract = mesh_abstract.render(CONFIG.img_h, CONFIG.img_w,
                                    CONFIG.light_ambient,
                                    CONFIG.light_color,
                                    make_point_abstract<AbstractClass>(CONFIG.light_src),
                                    CONFIG.img_upsample);

        vector<vector<double>> img_abs_min(CONFIG.img_h, vector<double>(CONFIG.img_w));
        vector<vector<double>> img_abs_max(CONFIG.img_h, vector<double>(CONFIG.img_w));
        for (int i = 0; i < CONFIG.img_h; i++) {
            for (int j = 0; j < CONFIG.img_w; j++) {
                img_abs_min[i][j] = img_abstract[i][j]._min();
                img_abs_max[i][j] = img_abstract[i][j]._max();
            }
        }

        imshow(img_concrete);
        imshow(img_abs_min);
        imshow(img_abs_max);
        string prefix = "images/"+CONFIG.out_id+"_" + to_string(run_id) + "_dtheta" + to_string(CONFIG.theta_eps) + "_dt" + to_string(CONFIG.delta_t_eps); 
        imsave(img_concrete, prefix + "_concrete.png");
        imsave(img_abs_min, prefix + "_abs_min.png");
        imsave(img_abs_max, prefix + "_abs_max.png");
    }

    return 0;
}
