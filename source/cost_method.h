#ifndef COST_METHOD_H
#define COST_METHOD_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cmath>
using namespace std;

#include "../../Qing/qing_disp.h"

#define DEFAULT_CEN_WND 9
#define DEFAULT_NCC_WND 9

typedef unsigned char uchar;

class CostMethod
{
public:
    CostMethod(const int w, const int h, const int range, const float max_d, const float min_d, const int wnd_sz = 0):
        m_range(range), m_w(w), m_h(h), m_max_d(max_d), m_min_d(min_d), m_wnd_sz(wnd_sz) {}

    virtual ~CostMethod() {}

    virtual void build_cost_vol_l(const vector<float> &img_l, const vector<float> &img_r,
                                  const vector<float> &avg_l, const vector<float> &avg_r,
                                  const vector<uchar> &msk_l, const vector<uchar> &msk_r,
                                  vector<vector<float> > &cost_vol) = 0;

    virtual void build_cost_vol_r(const vector<float> &img_l, const vector<float> &img_r,
                                  const vector<float> &avg_l, const vector<float> &avg_r,
                                  const vector<uchar> &msk_l, const vector<uchar> &msk_r,
                                  vector<vector<float> > &cost_vol) = 0;

//    virtual void calc_cluster_costs_l(const vector<float> &img_l, const vector<float> &img_r,
//                                      const vector<float> &avg_l, const vector<float> &avg_r,
//                                      const vector<uchar> &msk_l, const vector<uchar> &msk_r,
//                                      const int lx, const int ly, const int stk, const int edk,
//                                      vector<float>& mcosts) = 0;

//    virtual void calc_cluster_costs_r(const vector<float> &img_l, const vector<float> &img_r,
//                                      const vector<float> &avg_l, const vector<float> &avg_r,
//                                      const vector<uchar> &msk_l, const vector<uchar> &msk_r,
//                                      const int rx, const int ry, const int stk, const int edk,
//                                      vector<float>& mcosts) = 0;

public:
    int m_w, m_h, m_range, m_wnd_sz;
    float m_max_d, m_min_d;
};



#endif // COST_METHOD_H
