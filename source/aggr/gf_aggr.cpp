#include "gf_aggr.h"
#include "../../../Qing/qing_guided_filter.h"

void GFCA::aggr_cost_vol(const vector<float> &img, const vector<uchar> &msk, vector<vector<float> > &cost_vol) {
    for(int k = 0; k <= m_range; ++k) {
        qing_guided_filter_1ch(img, m_h, m_w, cost_vol[k]);
    }
}
