#ifndef TAD_H
#define TAD_H

#include "../cost_method.h"

class TAD: public CostMethod
{
public:
    TAD(const int w, const int h, const int range, const float max_d, const float min_d, const int wnd_sz = 0 ) ;
    ~TAD();

    void build_cost_vol_l(const vector<float> &img_l, const vector<float> &img_r,
                          const vector<float> &avg_l, const vector<float> &avg_r,
                          const vector<uchar> &msk_l, const vector<uchar> &msk_r,
                          vector<vector<float> > &cost_vol);
    void build_cost_vol_r(const vector<float> &img_l, const vector<float> &img_r,
                          const vector<float> &avg_l, const vector<float> &avg_r,
                          const vector<uchar> &msk_l, const vector<uchar> &msk_r,
                          vector<vector<float> > &cost_vol);
};

inline TAD::TAD(const int w, const int h, const int range, const float max_d, const float min_d, const int wnd_sz): CostMethod(w, h, range, max_d, min_d, wnd_sz) {
    printf("\ttrunated-ad for cost computation...%d x %d, %f ~ %f, wnd = %d\n",w, h, min_d, max_d, wnd_sz);
}

inline TAD::~TAD() {

}

#endif // ADCENSUS_H
