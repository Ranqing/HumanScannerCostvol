#include "census.h"

void Census::census_transform(const vector<float> &img_l, const vector<float> &img_r) {
    int offset = CENSUS_WND * 0.5;
    census_code_ptr ptr_code_l = m_code_l;
    census_code_ptr ptr_code_r = m_code_r;

    for(int y = 0; y < m_h; ++y)
    {
        for(int x = 0; x < m_w; ++x)
        {
            int bit_cnt = 0;
            int cen_idx = y * m_w + x;

            for(int dy = -offset; dy <= offset; ++dy)
            {
                int cury = (y + dy + m_h) % m_h;
                for(int dx = -offset; dx <= offset; ++dx)
                {
                    if(0 == dx && 0 == dy) continue;      //不考虑中心像素

                    int curx = (x + dx + m_w) % m_w;
                    int cur_idx = cury * m_w + curx;

                    (*ptr_code_l)[ bit_cnt ] = ( img_l[cen_idx] > img_l[cur_idx] );
                    (*ptr_code_r)[ bit_cnt ] = ( img_r[cen_idx] > img_r[cur_idx] );
                    bit_cnt ++;
                }
            }

            ptr_code_l++;
            ptr_code_r++;
        }
    }
}


void Census::build_cost_vol_l(const vector<float> &img_l, const vector<float> &img_r,
                              const vector<float> &avg_l, const vector<float> &avg_r,
                              const vector<uchar> &msk_l, const vector<uchar> &msk_r,
                              vector<vector<float> > &cost_vol) {
    census_code value_l, value_r;
    census_code_ptr ptr_code_l, ptr_code_r;

    ptr_code_l = m_code_l;
    ptr_code_r = m_code_r;

    for(int k = 0; k <= m_range; ++k)
    {
        float d = qing_k_2_disp(m_max_d, m_min_d, k);
        vector<float>& cost_of_k = cost_vol[k];

        for(int y = 0; y < m_h; ++y)
        {
            for(int x = 0; x < m_w && x-d >= 0; ++x)
            {
                int idx_l = y * m_w + x;
                int idx_r = idx_l - d;

                if( 0 == msk_l[idx_l] || 0 == msk_r[idx_r] ) continue;

                value_l = *(ptr_code_l+idx_l);
                value_r = *(ptr_code_r+idx_r);

                cost_of_k[idx_l] = (value_l ^ value_r).count();
            }
        }
    }
    cout << "\tcensus left cost volume calculation done...\n" ;
}

void Census::build_cost_vol_r(const vector<float> &img_l, const vector<float> &img_r,
                              const vector<float> &avg_l, const vector<float> &avg_r,
                              const vector<uchar> &msk_l, const vector<uchar> &msk_r,
                              vector<vector<float> > &cost_vol) {
    census_code value_l, value_r;
    census_code_ptr ptr_code_l, ptr_code_r;

    ptr_code_l = m_code_l;
    ptr_code_r = m_code_r;

    for(int k = 0; k <= m_range; ++k)
    {
        float d = qing_k_2_disp(m_max_d, m_min_d, k);
        vector<float>& cost_of_k = cost_vol[k];

        for(int y = 0; y < m_h; ++y)
        {
            for(int x = 0; x < m_w && x+d < m_w; ++x)
            {
                int idx_r = y * m_w + x;
                int idx_l = idx_l + d;

                if( 0 == msk_l[idx_l] || 0 == msk_r[idx_r] ) continue;

                value_l = *(ptr_code_l+idx_l);
                value_r = *(ptr_code_r+idx_r);

                cost_of_k[idx_r] = (value_l ^ value_r).count();
            }
        }
    }
    cout << "\tcensus right cost volume calculation done...\n" ;
}
