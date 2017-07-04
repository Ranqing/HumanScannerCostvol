#ifndef AGGR_METHOD_H
#define AGGR_METHOD_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cmath>
using namespace std;

typedef unsigned char uchar;

#define DEFAULT_BF_WND 9
#define DEFAULT_BOX_WND 9

class AggrMethod {
public:
    AggrMethod(const int w, const int h, const int range, const int wnd_sz = 0): m_range(range), m_w(w), m_h(h), m_wnd_sz(wnd_sz) {}

    virtual ~AggrMethod() {}

    virtual void aggr_cost_vol(const vector<float>& img, const vector<uchar>& msk, vector<vector<float> >& cost_vol) = 0;

public:
    int m_w, m_h, m_range, m_wnd_sz;
};

#endif // AGGR_METHOD_H
