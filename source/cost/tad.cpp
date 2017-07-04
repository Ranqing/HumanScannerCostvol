#include "tad.h"

//TAD

#define BORDER_THRES 0.011764

#define TAU_1 0.02745
#define TAU_2 0.00784
#define ALPHA 0.11

void TAD::build_cost_vol_l(const vector<float> &img_l, const vector<float> &img_r,
                           const vector<float> &avg_l, const vector<float> &avg_r,
                           const vector<uchar> &msk_l, const vector<uchar> &msk_r,
                           vector<vector<float> > &cost_vol) {
    for(int k = 0; k <= m_range; ++k)
    {
        float d = qing_k_2_disp(m_max_d, m_min_d, k);
        vector<float>& cost_of_k = cost_vol[k];

        for(int y = 0; y < m_h; ++y)
        {
            for(int x = 0; x < m_w ; ++x)
            {
                int cur_idx_l = y * m_w + x;
                int cur_idx_r = cur_idx_l - d;

                if( 0 == msk_l[cur_idx_l] ) continue;

                if( x-d >= 0)
                    cost_of_k[cur_idx_l] = 0.33333 * ( fabs(img_l[cur_idx_l * 3 + 0] - img_r[cur_idx_r * 3 + 0]) +
                                                       fabs(img_l[cur_idx_l * 3 + 1] - img_r[cur_idx_r * 3 + 1]) +
                                                       fabs(img_l[cur_idx_l * 3 + 2] - img_r[cur_idx_r * 3 + 2]) );

                else
                    cost_of_k[cur_idx_l] = 0.33333 * ( fabs(img_l[cur_idx_l * 3 + 0]) +
                                                       fabs(img_l[cur_idx_l * 3 + 1]) +
                                                       fabs(img_l[cur_idx_l * 3 + 2]) );
            }
        }
    }

    cout << "\ttruncate-ad left cost volume calculation done...\n" ;
}


void TAD::build_cost_vol_r(const vector<float> &img_l, const vector<float> &img_r,
                           const vector<float> &avg_l, const vector<float> &avg_r,
                           const vector<uchar> &msk_l, const vector<uchar> &msk_r,
                           vector<vector<float> > &cost_vol) {
    for(int k = 0; k <= m_range; ++k)
    {
        float d = qing_k_2_disp(m_max_d, m_min_d, k);
        vector<float>& cost_of_k = cost_vol[k];

        for(int y = 0; y < m_h; ++y)
        {
            for(int x = 0; x < m_w ; ++x)
            {
                int cur_idx_r = y * m_w + x;
                int cur_idx_l = cur_idx_r + d;

                if( 0 == msk_r[cur_idx_r] ) continue;

                if( x+d < m_w)
                    cost_of_k[cur_idx_r] = 0.33333 * ( fabs(img_l[cur_idx_l * 3 + 0] - img_r[cur_idx_r * 3 + 0]) +
                                                       fabs(img_l[cur_idx_l * 3 + 1] - img_r[cur_idx_r * 3 + 1]) +
                                                       fabs(img_l[cur_idx_l * 3 + 2] - img_r[cur_idx_r * 3 + 2]) );

                else
                    cost_of_k[cur_idx_r] = 0.33333 * ( fabs(img_r[cur_idx_r * 3 + 0]) +
                                                       fabs(img_r[cur_idx_r * 3 + 1]) +
                                                       fabs(img_r[cur_idx_r * 3 + 2]) );
            }
        }
    }

    cout << "\ttruncate-ad right cost volume calculation done...\n" ;
}
