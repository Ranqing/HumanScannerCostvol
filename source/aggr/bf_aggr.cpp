#include "bf_aggr.h"

#include "../../../Qing/qing_bilateral_filter.h"
#include "../../../Qing/qing_timer.h"

void BFCA::aggr_cost_vol(const vector<float> &img, const vector<uchar> &msk, vector<vector<float> > &cost_vol) {

    QingTimer timer;
    for(int k = 0; k <= m_range; ++k) {
        qing_bilateral_filter_1ch(img, msk, m_w, m_h, cost_vol[k], m_wnd_sz);
    }
    cout << "\tbilateral aggregation done..." << timer.duration() << "s\n";
}
