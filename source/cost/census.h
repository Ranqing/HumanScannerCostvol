#ifndef CENSUS_H
#define CENSUS_H

#include "../cost_method.h"

#define CENSUS_WND 9
#define CENSUS_BIT (CENSUS_WND * CENSUS_WND) - 1

typedef bitset<CENSUS_BIT> census_code;
typedef bitset<CENSUS_BIT> * census_code_ptr;

class Census : public CostMethod
{
public:
    Census(const int w, const int h, const int range, const float max_d, const float min_d, const int wnd_sz = 0 ) ;
    ~Census();

    void build_cost_vol_l(const vector<float> &img_l, const vector<float> &img_r,
                          const vector<float> &avg_l, const vector<float> &avg_r,
                          const vector<uchar> &msk_l, const vector<uchar> &msk_r,
                          vector<vector<float> > &cost_vol);

    void build_cost_vol_r(const vector<float> &img_l, const vector<float> &img_r,
                          const vector<float> &avg_l, const vector<float> &avg_r,
                          const vector<uchar> &msk_l, const vector<uchar> &msk_r,
                          vector<vector<float> > &cost_vol);

    void census_transform(const vector<float>& img_l, const vector<float>& img_r);

private:
    census_code_ptr m_code_l, m_code_r;
};

inline Census::Census(const int w, const int h, const int range, const float max_d, const float min_d, const int wnd_sz): CostMethod(w, h, range, max_d, min_d, wnd_sz) {
    printf("\tcensus for cost computation...%d x %d, %f ~ %f, wnd = %d\n",w, h, min_d, max_d, wnd_sz);

    m_code_l = new census_code[m_h * m_w];
    m_code_r = new census_code[m_h * m_w];
}

inline Census::~Census() {
    if(NULL != m_code_l) { delete[] m_code_l; m_code_l = NULL; }
    if(NULL != m_code_r) { delete[] m_code_r; m_code_r = NULL; }
}

#endif // CENSUS_H
