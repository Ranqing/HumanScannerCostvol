#ifndef BOX_AGGR_H
#define BOX_AGGR_H

#include "../aggr_method.h"

class BoxCA: public AggrMethod
{
public:
    BoxCA(const int w, const int h, const int range, const int wnd_sz);
    ~BoxCA();

    void aggr_cost_vol(const vector<float> &img, const vector<uchar> &msk, vector<vector<float> > &cost_vol);
};

inline BoxCA::BoxCA(const int w, const int h, const int range, const int wnd_sz): AggrMethod(w,h,range,wnd_sz) {
    printf("\tbox filter for cost computation...%d x %d, %d, %d\n", w, h, range, wnd_sz);
}

inline BoxCA::~BoxCA() {

}

#endif // BOX_AGGR_H
