#include "box_aggr.h"

#include "../../../Qing/qing_box_filter.h"

void BoxCA::aggr_cost_vol(const vector<float> &img, const vector<uchar> &msk, vector<vector<float> > &cost_vol) {
    for(int k = 0; k <= m_range; ++k) {
        qing_box_filter(cost_vol[k], m_h, m_w);
    }
}
