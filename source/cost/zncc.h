#ifndef ZNCC_H
#define ZNCC_H

#include "../cost_method.h"

class ZNCC: public CostMethod
{
public:
    ZNCC(const int w, const int h, const int range, const float max_d, const float min_d, const int wnd_sz);
    ~ZNCC();

    void build_cost_vol_l(const vector<float> &img_l, const vector<float> &img_r,
                          const vector<float> &avg_l, const vector<float> &avg_r,
                          const vector<uchar> &msk_l, const vector<uchar> &msk_r,
                          vector<vector<float> > &cost_vol);

    void build_cost_vol_r(const vector<float> &img_l, const vector<float> &img_r,
                          const vector<float> &avg_l, const vector<float> &avg_r,
                          const vector<uchar> &msk_l, const vector<uchar> &msk_r,
                          vector<vector<float> > &cost_vol);

//    void calc_cluster_costs_l(const vector<float> &img_l, const vector<float> &img_r,
//                              const vector<float> &avg_l, const vector<float> &avg_r,
//                              const vector<uchar> &msk_l, const vector<uchar> &msk_r,
//                              const int lx, const int ly, const int stk, const int edk,
//                              vector<float> &mcosts);

//    void calc_cluster_costs_r(const vector<float> &img_l, const vector<float> &img_r,
//                              const vector<float> &avg_l, const vector<float> &avg_r,
//                              const vector<uchar> &msk_l, const vector<uchar> &msk_r,
//                              const int rx, const int ry, const int stk, const int edk,
//                              vector<float> &mcosts);
};

inline ZNCC::ZNCC(const int w, const int h, const int range, const float max_d, const float min_d, const int wnd_sz): CostMethod(w, h, range, max_d, min_d, wnd_sz) {
    printf("\tzncc for cost computation...%d x %d, %f ~ %f, wnd = %d\n", w, h, min_d, max_d, wnd_sz);

}

inline ZNCC::~ZNCC() {

}

#endif // ZNCC_H
