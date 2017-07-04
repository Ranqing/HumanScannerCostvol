#include "zncc.h"
#include "../../../Qing/qing_image.h"
#include "../../../Qing/qing_timer.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;

void ZNCC::build_cost_vol_l(const vector<float> &img_l, const vector<float> &img_r,
                            const vector<float> &avg_l, const vector<float> &avg_r,
                            const vector<uchar> &msk_l, const vector<uchar> &msk_r,
                            vector<vector<float> > &cost_vol) {
# if 0
    Mat img(m_h, m_w, CV_32FC1);
    Mat uchar_img(m_h, m_w, CV_8UC1);
    qing_vec_2_img<float>(img_l, img);
    img.convertTo(uchar_img, CV_8UC1, 255);
    imwrite("gray_l_3.jpg", uchar_img);

    qing_vec_2_img<float>(img_r, img);
    img.convertTo(uchar_img, CV_8UC1, 255);
    imwrite("gray_r_3.jpg", uchar_img);

    qing_vec_2_img<float>(avg_l, img);
    img.convertTo(uchar_img, CV_8UC1, 255);
    imwrite("mean_l_3.jpg", uchar_img);

    qing_vec_2_img<float>(avg_r, img);
    img.convertTo(uchar_img, CV_8UC1, 255);
    imwrite("mean_r_3.jpg", uchar_img);

    fstream fout("gray_l.txt", ios::out);
    for(int i = 0, sz = img_l.size(); i < sz; i++) {
        fout << img_l[i] << ' ';
        int y = i / m_w;
        int x = i % m_w;
        if(x==m_w-1) fout << endl;
    }
    fout.close();
    fout.open("gray_r.txt", ios::out);
    for(int i = 0, sz = img_r.size(); i < sz; i++) {
        fout << img_r[i] << ' ';
        int y = i / m_w;
        int x = i % m_w;
        if(x==m_w -1) fout << endl;
    }
    fout.close();

    fout.open("mean_gray_l.txt", ios::out);
    for(int i = 0, sz = avg_l.size(); i < sz; i++) {
        fout << avg_l[i] << ' ';
        int y = i / m_w;
        int x = i % m_w;
        if(x==m_w-1) fout << endl;
    }
    fout.close();
    fout.open("mean_gray_r.txt", ios::out);
    for(int i = 0, sz = avg_r.size(); i < sz; i++) {
        fout << avg_r[i] << ' ';
        int y = i / m_w;
        int x = i % m_w;
        if(x==m_w-1) fout << endl;
    }
    fout.close();
    cout << "DEBUG in ZNCC done..." << endl;

    exit(1);
# endif

    QingTimer timer ;

    int offset = m_wnd_sz * 0.5, wndsz2 = m_wnd_sz * m_wnd_sz;
    for(int k = 0; k <= m_range; ++k)
    {
        vector<float>& cost_of_k = cost_vol[k];
        float d = qing_k_2_disp(m_max_d, m_min_d, k);

        for(int y = 0; y < m_h; ++y)
        {
            for(int x = 0; x < m_w ; ++x)
            {
                if( (x - d) < 0) continue;
                int cen_idx_l = y * m_w + x;
                int cen_idx_r = cen_idx_l - d;

                if( 0 == msk_l[cen_idx_l] || 0 == msk_r[cen_idx_r] ) continue;

                double fenzi = 0., fenmu = 0., fenmu1 = 0., fenmu2 = 0.;

                vector<double> delta_lefts(wndsz2,0.), delta_rights(wndsz2, 0.);
                for(int dy = -offset; dy <= offset; ++dy)
                {
                    int cury = y + dy;

                    if(0 > cury || m_h <= cury) continue;

                    for(int dx = -offset; dx <= offset; ++dx)
                    {
                        int curx_l = x + dx;
                        int curx_r = curx_l - d;

                        if(0 > curx_l || m_w <= curx_l || 0 > curx_r || m_w <= curx_r) continue;

                        int cur_idx_l = cury * m_w + curx_l;
                        int cur_idx_r = cur_idx_l - d;

                        if(0 == msk_l[cur_idx_l] || 0 == msk_r[cur_idx_r]) continue;

                        double delta_l = img_l[cur_idx_l] - avg_l[cen_idx_l];
                        double delta_r = img_r[cur_idx_r] - avg_r[cen_idx_r];
                        fenzi  += delta_l * delta_r;
                        fenmu1 += delta_l * delta_l;
                        fenmu2 += delta_r * delta_r;

# if 0
                        if(y==6&&k==14) {
                            int tidx = (dy+offset) * m_wnd_sz + (dx+offset);
                            delta_lefts[tidx] = delta_l;
                            delta_rights[tidx] = delta_r;
                        }
# endif
                    }
                }

                fenmu = sqrt(fenmu1) * sqrt(fenmu2) ;
                if(fenmu != 0.0)
                    cost_of_k[cen_idx_l] = fenzi / fenmu ;
                else
                    cost_of_k[cen_idx_l] = 1.f;  //max-value

# if 0
                if(y==6&&k==14&&x>120&&x<140) {
                    cout << x << ": " << endl;
                    for(int m = 0; m < wndsz2; ++m) {
                        cout << delta_lefts[m] << ' ';
                    }
                    cout << endl;
                    for(int m = 0; m < wndsz2; ++m) {
                        cout << delta_rights[m] << ' ';
                    }
                    cout << endl;
                    cout << cost_of_k[cen_idx_l] << endl << endl;

                }
# endif
            }
        }
    }
    cout << "\tzncc left cost volume calculation done..." << timer.duration() << "s\n" ;
}

void ZNCC::build_cost_vol_r(const vector<float> &img_l, const vector<float> &img_r,
                            const vector<float> &avg_l, const vector<float> &avg_r,
                            const vector<uchar> &msk_l, const vector<uchar> &msk_r,
                            vector<vector<float> > &cost_vol) {
    QingTimer timer;
    int offset = m_wnd_sz * 0.5;
    for(int k = 0; k <= m_range; ++k)
    {
        vector<float>& cost_of_k = cost_vol[k];
        float d = qing_k_2_disp(m_max_d, m_min_d, k);

        for(int y = 0; y < m_h; ++y)
        {
            for(int x = 0; x < m_w ; ++x)
            {
                if(x+d >= m_w) continue;

                int cen_idx_r = y * m_w + x;
                int cen_idx_l = cen_idx_r + d;

                if( 0 == msk_l[cen_idx_l] || 0 == msk_r[cen_idx_r] ) continue;

                double fenzi = 0., fenmu = 0., fenmu1 = 0., fenmu2 = 0.;

                for(int dy = -offset; dy <= offset; ++dy)
                {
                    int cury = y + dy;

                    if(0 > cury || m_h <= cury) continue;

                    for(int dx = -offset; dx <= offset; ++dx)
                    {
                        int curx_r = x + dx;
                        int curx_l = curx_r + d;

                        if(0 > curx_l || m_w <= curx_l || 0 > curx_r || m_w <= curx_r) continue;

                        int cur_idx_r = cury * m_w + curx_r;
                        int cur_idx_l = cur_idx_r + d;

                        if(0 == msk_l[cur_idx_l] || 0 == msk_r[cur_idx_r]) continue;

                        double delta_l = img_l[cur_idx_l] - avg_l[cen_idx_l];
                        double delta_r = img_r[cur_idx_r] - avg_r[cen_idx_r];
                        fenzi  += delta_l * delta_r;
                        fenmu1 += delta_l * delta_l;
                        fenmu2 += delta_r * delta_r;
                    }
                }

                fenmu = sqrt(fenmu1) * sqrt(fenmu2) ;
                if(fenmu != 0.0)
                    cost_of_k[cen_idx_r]  = fenzi / fenmu ;
                else
                    cost_of_k[cen_idx_r] = 1.f;
            }
        }
    }
    cout << "\tzncc right cost volume calculation done..." << timer.duration() << "s\n" ;
}

//void ZNCC::calc_cluster_costs_l(const vector<float> &img_l, const vector<float> &img_r,
//                                const vector<float> &avg_l, const vector<float> &avg_r,
//                                const vector<uchar> &msk_l, const vector<uchar> &msk_r,
//                                const int lx, const int ly, const int stk, const int edk,
//                                vector<float> &mcosts) {

//    int range = edk - stk + 1;
//    mcosts.resize(range); memeset(&mcosts.front(), 0, sizeof(float) * range);

//}


//void ZNCC::calc_cluster_costs_r(const vector<float> &img_l, const vector<float> &img_r,
//                                const vector<float> &avg_l, const vector<float> &avg_r,
//                                const vector<uchar> &msk_l, const vector<uchar> &msk_r,
//                                const int rx, const int ry, const int stk, const int edk,
//                                vector<float> &mcosts) {
//    int range = edk - stk + 1;
//    mcosts.resize(range); memeset(&mcosts.front(), 0, sizeof(float) * range);

//}
