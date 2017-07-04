#ifndef GF_AGGR_H
#define GF_AGGR_H

#include "../aggr_method.h"

class GFCA: public AggrMethod
{
public:
    GFCA(const int w, const int h, const int range, const int wnd_sz);
    ~GFCA();

    void aggr_cost_vol(const vector<float> &img, const vector<uchar> &msk, vector<vector<float> > &cost_vol);

};

inline GFCA::GFCA(const int w, const int h, const int range, const int wnd_sz): AggrMethod(w,h,range,wnd_sz) {
    printf("\tguided filter for cost computation...%d x %d, %d, %d\n", w, h, range, wnd_sz);
}

inline GFCA::~GFCA() {

}

#endif // GF_AGGR_H
