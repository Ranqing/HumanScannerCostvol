#ifndef BF_H
#define BF_H

#include "../aggr_method.h"

class BFCA: public AggrMethod
{
public:
    BFCA(const int w, const int h, const int range, const int wnd_sz);
    ~BFCA();

    void aggr_cost_vol(const vector<float> &img, const vector<uchar> &msk, vector<vector<float> > &cost_vol);
};

inline BFCA::BFCA(const int w, const int h, const int range, const int wnd_sz): AggrMethod(w, h, range, wnd_sz) {
    printf("\tbilateral filter for cost computation... %d x %d, %d, %d\n", w, h, range, wnd_sz);
}

inline BFCA::~BFCA() {
}





#endif // BF_H
