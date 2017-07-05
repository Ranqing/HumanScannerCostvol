#include "stereo_flow.h"
#include "qx_upsampling/qx_zdepth_upsampling_using_rbf.h"

#include "../../Qing/qing_timer.h"
#include "../../Qing/qing_memory.h"
#include "../../Qing/qing_disp.h"
#include "../../Qing/qing_median_filter.h"
#include "../../Qing/qing_matching_cost.h"

void StereoFlow::calc_mean_images() {

    //unsigned char [0,255]
    cv::blur(m_mat_gray_l, m_mat_mean_l, Size(m_wnd_size, m_wnd_size));
    cv::blur(m_mat_gray_r, m_mat_mean_r, Size(m_wnd_size, m_wnd_size));

    //float [0,255]
    Mat meanL, meanR;
    m_mat_mean_l.convertTo(meanL, CV_32FC1); qing_img_2_vec<float>(meanL, m_mean_l);             //imshow("mean_l", m_mat_mean_l); waitKey(0); destroyWindow("mean_l");
    m_mat_mean_r.convertTo(meanR, CV_32FC1); qing_img_2_vec<float>(meanR, m_mean_r);             //imshow("mean_r", m_mat_mean_r); waitKey(0); destroyWindow("mean_r");

# if 0
 //   m_ncc_mean_l.resize(m_total); copy(m_mean_l.begin(), m_mean_l.end(), m_ncc_mean_l.begin());
 //   m_ncc_mean_r.resize(m_total); copy(m_mean_r.begin(), m_mean_r.end(), m_ncc_mean_r.begin());
 //   cout << "\tmean image with default_ncc_wnd done...." << endl;
# endif
}

void StereoFlow::calc_support_region() {
    m_support_region->init(m_h,m_w,m_wnd_size);
    m_support_region->calc_patch_borders(m_view_l, m_view_r, m_mask_l, m_mask_r);
    QingTimer timer;
    m_support_region->aggr_for_normalize(0);
    cout << "\taggregate cross shaped region for normalization left..";
    m_support_region->aggr_for_normalize(1);
    cout << "right..done..\t" << timer.duration() << "s" << endl;
}

//2017.05.31

void StereoFlow::median_filter() {
    int idx;

    fill_n(m_best_k_l.begin(), m_total, 0);
    fill_n(m_best_k_r.begin(), m_total, 0);
    for(idx = 0; idx < m_total; ++idx) {
        if(0!=m_mask_l[idx]) m_best_k_l[idx] = m_disp_l[idx];
        if(0!=m_mask_r[idx]) m_best_k_r[idx] = m_disp_r[idx];
    }

    vector<int> new_best_k_l(m_total, 0), new_best_k_r(m_total, 0);
    cout << "\tapplying median filter..";
    QingTimer timer;
    qing_median_filter(new_best_k_l, m_best_k_l, m_mask_l, m_h, m_w, m_wnd_size, m_disp_ranges+1);
    cout << "\tleft.." ;
    qing_median_filter(new_best_k_r, m_best_k_r, m_mask_r, m_h, m_w, m_wnd_size, m_disp_ranges+1);
    cout << "\tright..\t" << timer.duration() << "s." << endl;

    copy(new_best_k_l.begin(), new_best_k_l.end(), m_best_k_l.begin());
    copy(new_best_k_r.begin(), new_best_k_r.end(), m_best_k_r.begin());
    for(idx = 0; idx < m_total; ++idx) {
        if(0!=m_mask_l[idx]) m_disp_l[idx] = m_best_k_l[idx];
        if(0!=m_mask_r[idx]) m_disp_r[idx] = m_best_k_r[idx];
    }
}


void StereoFlow::subpixel_approximation() {
    cout << "applying subpixel approximation ..." << endl;
}


void StereoFlow::matching_cost() {
    m_hwd_costvol_l.clear();
    m_hwd_costvol_r.clear();

    qing_allocf_3(m_hwd_costvol_l, m_h, m_w, m_disp_ranges+1);
    matching_cost_from_zncc();
# if 0
    string mcost_folder = "./matching-cost-ncc/";
    qing_create_dir(mcost_folder);
    string filename ;
    float * temp_mcost = new float[m_total];
    for(int d = 0; d < m_disp_ranges+1; ++d) {
        memset(temp_mcost, 0, sizeof(float)*m_total);
        for(int y = 0, idx = 0; y < m_h; ++y) {
            for(int x = 0; x < m_w; ++x) {
                temp_mcost[idx++] = m_hwd_costvol_l[y][x][d];
            }
        }
        filename = mcost_folder + "zncc_" + qing_int_2_string(d) + ".txt";
        qing_save_mcost_txt(filename, temp_mcost, m_total, m_w);
    }
# endif

    qing_allocf_3(m_hwd_costvol_r, m_h, m_w, m_disp_ranges+1);
    qing_stereo_flip_cost_vol(m_hwd_costvol_r, m_hwd_costvol_l, m_h, m_w, m_disp_ranges+1);

# if 0
    Mat uimg(m_h, m_w, CV_8UC1, Scalar(0));

    qing_depth_max_cost(m_disp_l, m_hwd_costvol_l, m_mask_l, m_h, m_w, m_disp_ranges + 1);
    qing_float_vec_2_uchar_img(m_disp_l, m_scale, uimg);
    imwrite("mcost_disp_l.jpg", uimg);

    qing_depth_max_cost(m_disp_r, m_hwd_costvol_r, m_mask_r, m_h, m_w, m_disp_ranges + 1);
    qing_float_vec_2_uchar_img(m_disp_r, m_scale, uimg);
    imwrite("mcost_disp_r.jpg", uimg);
# endif
}

void StereoFlow::matching_cost_from_zncc() {

    int wndsz2 = m_wnd_size * m_wnd_size, idx;
    float * ncc_vec_l = new float[wndsz2];
    float * ncc_vec_r = new float[wndsz2];

    for(int i = 0; i <= m_disp_ranges; ++i) {
        for(int y = 0; y < m_h; ++y) {
            qing_get_ncc_vec(ncc_vec_r, y, 0, m_gray_r, m_mean_r, m_h, m_w, 1, m_wnd_size);
            for(int x = 0; x < i; ++x) {
                idx = y * m_w + x;
                if(0==m_mask_l[idx]) continue;
                qing_get_ncc_vec(ncc_vec_l, y, x, m_gray_l, m_mean_l, m_h, m_w, 1, m_wnd_size);
                m_hwd_costvol_l[y][x][i] = qing_ncc_value(ncc_vec_l, ncc_vec_r, wndsz2);
            }
            for(int x = i; x < m_w; ++x) {
                idx = y * m_w + x;
                if(0==m_mask_l[idx]) continue;
                qing_get_ncc_vec(ncc_vec_r, y, x-i, m_gray_r, m_mean_r, m_h, m_w, 1, m_wnd_size);
                qing_get_ncc_vec(ncc_vec_l, y, x,   m_gray_l, m_mean_l, m_h, m_w, 1, m_wnd_size);
                m_hwd_costvol_l[y][x][i] = qing_ncc_value(ncc_vec_l, ncc_vec_r, wndsz2);

# if 0
                if(y==480&&i==120&&x>820&&x<830) {
                    cout << x << ": " << endl;
                    for(int m = 0; m < wndsz2 ; ++m) {
                        cout << ncc_vec_l[m] << ' ' ;
                    }
                    cout << endl;
                    for(int m = 0; m < wndsz2 ; ++m) {
                        cout << ncc_vec_r[m] << ' ' ;
                    }
                    cout << endl;
                    cout << m_hwd_costvol_l[y][x][i] << endl << endl;
                }
# endif
            }
        }
    }
}

void StereoFlow::matching_cost_aggregation(const int direction, vector<vector<vector<float> > >& hwd_cost_vol) {
    QingTimer timer;
    timer.restart();

    cout << "\tmatching cost aggregation start.." ;
    cout << "\t" << (direction ? "right.." : "left..") ;
    vector<int>& borders_u = direction ? m_support_region->get_u_borders_r() : m_support_region->get_u_borders_l();
    vector<int>& borders_d = direction ? m_support_region->get_d_borders_r() : m_support_region->get_d_borders_l();
    vector<int>& borders_l = direction ? m_support_region->get_l_borders_r() : m_support_region->get_l_borders_l();
    vector<int>& borders_r = direction ? m_support_region->get_r_borders_r() : m_support_region->get_r_borders_l();
    vector<int>& aggr_borders = direction ? m_support_region->get_aggr_borders_r() : m_support_region->get_aggr_borders_l();
    vector<unsigned char> & mask = direction ? m_mask_r : m_mask_l;

    int idx, limit_l, limit_r, limit_d, limit_u, i, y, x, dx, dy, didx;

    vector<vector<vector<float> > > tmp_hwd_cost_vol;
    tmp_hwd_cost_vol.clear();
    qing_allocf_3(tmp_hwd_cost_vol, m_h, m_w, m_disp_ranges+1);

    //horizontal
    for(i = 0; i <= m_disp_ranges; ++i) {
        for(idx = -1, y = 0; y < m_h; ++y) {
            for(x = 0; x < m_w; ++x) {
                if(0==mask[++idx]) continue;
                limit_l = -borders_l[idx];
                limit_r =  borders_r[idx];

                if(limit_l == limit_r) tmp_hwd_cost_vol[y][x][i] = hwd_cost_vol[y][x][i];
                else {
                    for(dx = limit_l ; dx <= limit_r; ++dx) {
                        if (0 > x + dx || m_w <= x + dx) continue;
                        didx = idx + dx;
                        if (0 == mask[didx]) continue;

                        tmp_hwd_cost_vol[y][x][i] += hwd_cost_vol[y][x + dx][i];
                    }
                }
            }
        }
    }

    //vertical
    for(i = 0; i <= m_disp_ranges; ++i) {
        for(idx = -1, y = 0; y < m_h; ++y) {
            for (x = 0; x < m_w; ++x) {
                hwd_cost_vol[y][x][i] = 0.f;

                if (0 == mask[++idx]) continue;
                limit_u = -borders_u[idx];
                limit_d =  borders_d[idx];

                if(limit_u == limit_d) hwd_cost_vol[y][x][i] = tmp_hwd_cost_vol[y][x][i];
                else {
                    for (dy = limit_u; dy <= limit_d; ++dy) {
                        if (0 > y + dy || m_h <= y + dy) continue;
                        didx = idx + dy * m_w;
                        if (0 == mask[didx]) continue;

                        hwd_cost_vol[y][x][i] += tmp_hwd_cost_vol[y + dy][x][i];
                    }
                }
            }
        }
    }

    //normalization
    for(i = 0; i <= m_disp_ranges; ++i) {
        for(idx = -1, y = 0; y < m_h; ++y) {
            for(x = 0; x < m_w; ++x) {
                if(0==mask[++idx]) continue;
                if(0==aggr_borders[idx]) continue;
                hwd_cost_vol[y][x][i] /= aggr_borders[idx]*1.f;
            }
        }
    }
    cout << "\tnormalization..\tdone..\t" << timer.duration()  << "s."<< endl;

}

void StereoFlow::calc_init_disparity_from_cost() {
    matching_cost_aggregation(0, m_hwd_costvol_l);
    matching_cost_aggregation(1, m_hwd_costvol_r);

    fill_n(m_disp_l.begin(), m_total, 0.f);
    fill_n(m_disp_r.begin(), m_total, 0.f);
    qing_depth_max_cost(m_disp_l, m_hwd_costvol_l, m_mask_l, m_h, m_w, m_disp_ranges + 1);
    qing_depth_max_cost(m_disp_r, m_hwd_costvol_r, m_mask_r, m_h, m_w, m_disp_ranges + 1);

    fill_n(m_best_k_l.begin(), m_total, 0);
    fill_n(m_best_k_r.begin(), m_total, 0);
    for(int i = 0; i < m_total; ++i) {
        m_best_k_l[i] = m_disp_l[i];
        m_best_k_r[i] = m_disp_r[i];
    }
    cout << "\tcopy disparity to discrete level done.." << endl;
}


void StereoFlow::calc_init_disparity_from_cost(const vector<int> &pre_bestk_l, const vector<int> &pre_bestk_r, const int pre_w, const int pre_h) {

    matching_cost_aggregation(0, m_hwd_costvol_l);
    matching_cost_aggregation(1, m_hwd_costvol_r);
    fill_n(m_best_k_l.begin(), m_total, 0); //cout << "debug:\t" << m_best_k_l.size() << endl;
    fill_n(m_best_k_r.begin(), m_total, 0); //cout << "debug:\t" << m_best_k_r.size() << endl;

    int y, x, idx, pre_x, pre_y, pre_idx, pre_k, cur_stk , cur_edk;
    int bestk;

    QingTimer timer;
    for(idx = -1, y = 0; y < m_h; ++y) {
        for(x = 0; x < m_w; ++x) {
            if(0==m_mask_l[++idx]) continue;
            pre_x = x / 2;
            pre_y = y / 2;
            pre_idx = pre_y * pre_w + pre_x;
            pre_k = pre_bestk_l[pre_idx];
            cur_stk = max(0, 2 * pre_k - 2);
            cur_edk = min(2 * pre_k + 2, m_disp_ranges);

        //    cout << "idx = " << idx << ", pre_idx = " << pre_idx << ", pre_k = " << pre_k << ", cur_stk = " << cur_stk << ", cur_edk = " << cur_edk << endl;

            qing_vec_max_pos(bestk, m_hwd_costvol_l[y][x], cur_stk, cur_edk);
            m_best_k_l[idx] = bestk;
        }
    }
    cout << "\tcalculate disparity with previous infos..\tleft..";

    for(idx = -1, y = 0; y < m_h; ++y) {
        for(x = 0; x < m_w; ++x) {
            if(0==m_mask_r[++idx]) continue;
            pre_x = x / 2;
            pre_y = y / 2;
            pre_idx = pre_y * pre_w + pre_x;
            pre_k = pre_bestk_r[pre_idx];

            cur_stk = max(0, 2 * pre_k - 2);
            cur_edk = min(2 * pre_k + 2, m_disp_ranges);

            qing_vec_max_pos(bestk, m_hwd_costvol_r[y][x], cur_stk, cur_edk);
            m_best_k_r[idx] = bestk;
        }
    }
    cout << "\tright..\tdone..\t" << timer.duration() << "s." << endl;

    fill_n(m_disp_l.begin(),m_total, 0.f);
    fill_n(m_disp_r.begin(),m_total, 0.f);
    for(int i = 0; i < m_total; ++i) {
        m_disp_l[i] = m_best_k_l[i];
        m_disp_r[i] = m_best_k_r[i];
    }
}

//disparities of a pixel and its neighbors should be satisfied with local smoothness
//half of the neighbors in a 3x3 neighbors diff by a disparity less than one pixel
void StereoFlow::local_smoothness_check() {
    int y, x, idx, wnd_sz = 3, sum_diff, sum, d;
    int dx, dy, didx, offset = (int) (wnd_sz * 0.5);

    cout << "\tlocal smoothness check for pixels..\t";
  //  cout << m_outliers_l.size() << '\t' << m_outliers_r.size() ;

    QingTimer timer;
    int cnt = 0;
    for (idx = -1, y = 0; y < m_h; ++y) {
        for (x = 0; x < m_w; ++x) {
            if (0 == m_mask_l[++idx]) continue;
            if (1 == m_outliers_l[idx] || 0.f == m_disp_l[idx]) continue;

            d = (int) m_disp_l[idx];
            sum_diff = 0;
            sum = 0;
            for (dy = -offset; dy <= offset; ++dy) {
                if (0 > (y + dy) || m_h <= (y + dy)) continue;
                for (int dx = -offset; dx <= offset; ++dx) {
                    if (0 > (x + dx) || m_w <= (x + dx)) continue;
                    didx = (y + dy) * m_w + (x + dx);
                    if (0 == m_mask_l[didx]) continue;
                    sum++;
                    sum_diff += (abs((int) m_disp_l[didx] - d) <= 1);
                }
            }
            if (sum_diff * 2 <= sum)  {
                m_outliers_l[idx] = 1;
                cnt ++;
            }
        }
    }
    cout << "\tleft.." << cnt << " outliers..";

    cnt = 0;
    for (idx = -1, y = 0; y < m_h; ++y) {
        for (x = 0; x < m_w; ++x) {
            if (0 == m_mask_r[++idx]) continue;
            if (1 == m_outliers_r[idx] || 0.f == m_disp_r[idx]) continue;

            d = (int) m_disp_r[idx];
            sum_diff = 0;
            sum = 0;
            for (dy = -offset; dy <= offset; ++dy) {
                if (0 > (y + dy) || m_h <= (y + dy)) continue;
                for (int dx = -offset; dx <= offset; ++dx) {
                    if (0 > (x + dx) || m_w <= (x + dx)) continue;
                    didx = (y + dy) * m_w + (x + dx);
                    if (0 == m_mask_r[didx]) continue;
                    sum++;
                    sum_diff += (abs((int) m_disp_r[didx] - d) <= 1);
                }
            }
            if (sum_diff * 2 <= sum) {
                m_outliers_r[idx] = 1;
                cnt ++;
            }
        }
    }
    cout << "\tright.." << cnt << " outliers..\tdone..\t" << timer.duration() << "s." << endl;

}

void StereoFlow::ordering_check() {
    cout << "\tordering-check then find large disparity intervals..";

    QingTimer timer;

    int idx, y, x, cnt = 0;
    int rx, rx_add, lx, lx_add;
    for(idx = -1, y = 0; y < m_h; ++y) {
        for(x = 0; x < m_w - 1; ++x) {
            if(0==m_mask_l[++idx] || 1==m_outliers_l[idx]) continue;
            if(0==m_disp_l[idx] || 0==m_disp_l[idx+1]) continue;

            rx = x - m_disp_l[idx];
            rx_add = x + 1 - m_disp_l[idx+1];
            if( rx >= rx_add || m_disp_l[idx] > m_disp_l[idx+1] + 1) {
                m_outliers_l[idx] = 1;
                cnt ++;
            }
        }
    }
    cout << "\tleft.." << cnt << " outliers..";

    cnt = 0;
    for (idx = -1, y = 0; y < m_h ; ++y) {
        for(x = 0; x < m_w - 1; ++x) {
            if(0==m_mask_r[++idx] || 1==m_outliers_r[idx]) continue;
            if(0==m_disp_r[idx] || 0==m_disp_r[idx+1]) continue;

            lx = x + m_disp_r[idx];
            lx_add = x + 1 + m_disp_r[idx+1];
            if(lx >= lx_add || m_disp_r[idx] > m_disp_r[idx+1] + 1) {
                m_outliers_r[idx] = 1;
                cnt ++;
            }
        }
    }
    cout << "\tright.." << cnt << " outliers..\tdone..\t" << timer.duration() << "s."  << endl;
}

void StereoFlow::cross_check() {
    cout << "\tcross-check then find mis-match and occlusion pixels.." ;

    QingTimer timer;

    int idx, y, x, rx, ridx, lx, lidx;
    for(idx = -1, y = 0; y < m_h; ++y) {
        for(x = 0; x < m_w; ++x) {
            if(0==m_mask_l[++idx]) continue;
            if(0==m_disp_l[idx]) { m_outliers_l[idx] = 1; }
            if(1==m_outliers_l[idx]) continue;

            rx = ((int)(x - m_disp_l[idx] + m_w )) % m_w;
            ridx = y * m_w + rx;
            if(m_disp_l[idx] != m_disp_r[ridx]) {
                m_outliers_l[idx] = 1;
                m_outliers_r[ridx] = 1;
            }
        }
    }
    cout << "\tleft.." ;

    for(idx = -1, y = 0; y < m_h; ++y) {
        for(x = 0; x < m_w; ++x) {
            if(0==m_mask_r[++idx]) continue;
            if(0==m_disp_r[idx]) { m_outliers_r[idx] = 1; }
            if(1==m_outliers_r[idx]) continue;

            lx = ((int)(x + m_disp_r[idx] + m_w)) % m_w;
            lidx = y * m_w + lx;
            if(m_disp_l[lidx] != m_disp_r[idx]) {
                m_outliers_r[idx] = 1;
            }
        }
    }
    cout << "\tright..\tdone..\t" << timer.duration() << "s." << endl;
}

void StereoFlow::rematch_using_rbf(const Mat &rgb_view, const vector<uchar> &mask, const vector<uchar> &outliers,
                                   vector<float> &disp) {
    float sigma_spatial = 0.005f;
    float sigma_range = 0.1f;

    vector<uchar> guidance(m_total * 3);
    vector<uchar> gradient_x(m_total), gradient_y(m_total);                                                                                     //color image gradients
    vector<float> out(m_total), in(m_total), temp(m_total), temp_2w(2*m_w), ones(m_total), ones_temp(m_total), ones_temp_2w(2*m_w);             //disparity

    //rgb image as guidance
    memcpy(&guidance.front(),rgb_view.data, sizeof(uchar)*m_total*3);
    memset(&in.front(), 0, sizeof(float)*m_total);
    copy(disp.begin(), disp.end(), in.begin());
    for(int idx = 0; idx < m_total; ++idx) {
        if(0==mask[idx] || 1 == outliers[idx])
            in[idx] =0.f;
    }

    qx_zdepth_upsampling_using_rbf(&out[0], &in[0], &guidance[0], &temp[0], &temp_2w[0], &gradient_x[0], &gradient_y[0],
                                   &ones[0], &ones_temp[0], &ones_temp_2w[0], m_h, m_w, sigma_spatial, sigma_range);


    fill_n(disp.begin(), m_total, 0.f);
    for(int idx = 0; idx < m_total; idx++) {
        if(0==mask[idx]) continue;
        if(out[idx] >= 0 && out[idx] <= m_disp_ranges) disp[idx] = out[idx];
    }
}

void StereoFlow::rematch_using_rbf() {
    cout << "\trematch invalid pixels using rbf.." ;
    QingTimer timer;

    rematch_using_rbf(m_mat_view_l, m_mask_l, m_outliers_l, m_disp_l);
    cout << "\tleft..";
    rematch_using_rbf(m_mat_view_r, m_mask_r, m_outliers_r, m_disp_r);
    cout << "\tright..\tdone...\t" << timer.duration() << "s" << endl;

    fill_n(m_best_k_l.begin(), m_total, 0);
    fill_n(m_best_k_r.begin(), m_total, 0);
    for(int i = 0; i < m_total; ++i) {
        m_best_k_l[i] = m_disp_l[i];
        m_best_k_r[i] = m_disp_r[i];
    }
    cout << "\tcopy disparity to discrete level done.." << endl;
}

//quadratic formula
//d_min = d - (f(d+1)-f(d-1))/(2*(f(d+1)+f(d-1)-f(d)))
void StereoFlow::subpixel_enhancement() {
    cout << endl << "applying subpixel enhancement.." ;
    QingTimer timer;

    int x, y, idx, d;
    float mcost, mcost_add, mcost_sub, dmax;   //f(d), f(d+1), f(d-1)

    for(idx = -1, y = 0; y < m_h; ++y) {
        for(x = 0; x < m_w; ++x) {
            if(0 == m_mask_l[++idx]) continue;
            if(0.f == m_disp_l[idx]) continue;

            d = (int)(m_disp_l[idx]);
            mcost = m_hwd_costvol_l[y][x][d];
            mcost_add = m_hwd_costvol_l[y][x][d+1];
            mcost_sub = m_hwd_costvol_l[y][x][d-1];

            if( mcost > mcost_add && mcost > mcost_sub) {
                dmax = d - ((mcost_add - mcost_sub)/(2*(mcost_add + mcost_sub - 2 * mcost)));
                m_disp_l[idx] = dmax;
            }
//          else {
//              cerr << "error" << endl;
//          }
        }
    }
    cout << "\t" << timer.duration() << " s."<< endl;
}

void StereoFlow::trans_costvol_for_beeler() {
    //convert cost vol from zncc to (1-zncc)/2
    int y, x, d;
    float epsilon;
    for(y = 0; y < m_h; ++y) {
        for(x = 0; x < m_w; ++x) {
            for(d = 0; d <= m_disp_ranges ; ++d) {
                epsilon = m_hwd_costvol_l[y][x][d];
                m_hwd_costvol_l[y][x][d]=(1-epsilon)*0.5f;
            }
        }
    }
}

void StereoFlow::beeler_subpixel_refinement(const float& dstep, const double& thresh) {
    int y, x, d, idx;
//    float tstep = 0.5f, refine_disp;
    float refine_disp;
    double ddata, wdata, dsmooth, wsmooth;

    vector<float> disp_cp(m_total, 0.f);
    copy(m_disp_l.begin(), m_disp_l.end(), disp_cp.begin());

    QingTimer timer;
    int cnt = 0;
    for (y = 1; y < m_h - 1; y++) {
        for (x = 1; x < m_w - 1; x++) {
            idx = y * m_w + x;
            if (0 == m_mask_l[idx]) continue;
            if (0 == m_mask_l[idx - 1] && 0 == m_mask_l[idx + 1] && 0 == m_mask_l[idx + m_w] && 0 == m_mask_l[idx - m_w])   continue;

            compute_data_item(ddata, wdata, y, x, dstep);
            compute_smooth_item(dsmooth, wsmooth, y, x);
            refine_disp = (ddata * wdata + dsmooth * wsmooth) / (wdata + wsmooth);
        //    cout << "y = " << y << "\tx = " << x << "\tddata = " << ddata << "\twdata = " << wdata << "\tdsmooth = " << dsmooth << "\twsmooth = " << wsmooth << "\trefine =" << refine_disp << endl;

            if(abs(refine_disp - m_disp_l[idx]) <= thresh) {
                disp_cp[idx] = refine_disp;
                cnt ++;
            }
        }
    }
    copy(disp_cp.begin(), disp_cp.end(), m_disp_l.begin());
    cout << cnt << " points refined..\t"<< timer.duration() << " s.." << endl;
}

//matching cost after sub-pixel
void StereoFlow::compute_data_item(double &ddata, double &wdata, const int &y, const int &x, const float &dstep) {
    int idx = y * m_w + x;
    int d = (int)m_disp_l[idx];
    int sub_d = max(0,d-1);
    int add_d = min(d+1,m_disp_ranges);

    double sub_epsilon = m_hwd_costvol_l[y][x][d-1];
    double src_epsilon = m_hwd_costvol_l[y][x][d];
    double add_epsilon = m_hwd_costvol_l[y][x][d+1];
    double temp;

    assert(sub_epsilon > 0 && sub_epsilon < 1);
    assert(src_epsilon > 0 && src_epsilon < 1);
    assert(add_epsilon > 0 && add_epsilon < 1);

    if(sub_epsilon < src_epsilon && sub_epsilon < add_epsilon) {
        wdata = sub_epsilon - src_epsilon;
        ddata = d - dstep;
    }
    else if(src_epsilon < sub_epsilon && src_epsilon < add_epsilon) {
        temp = sub_epsilon + add_epsilon - 2 * src_epsilon;
        wdata = dstep * temp;
        ddata = d + dstep * (sub_epsilon - add_epsilon) / temp;
    }
    else if(add_epsilon < src_epsilon && add_epsilon < sub_epsilon) {
        wdata = add_epsilon - src_epsilon;
        ddata = d + dstep;
    }
    else { //add_epsilon == src_epsilon && sub_epsilon == src_epsilon
            wdata = 1;
            ddata = d;
    }
}


void StereoFlow::compute_smooth_item(double &dsmooth, double &wsmooth, const int &y, const int &x) {

    wsmooth = 0.005;

    double alpha = 1.5, beta = 2.0;
    int idx = y * m_w + x;
    int d = m_disp_l[idx];
    int u_d = (int)m_disp_l[idx - m_w];  //d(x,y-1)
    int d_d = (int)m_disp_l[idx + m_w];  //d(x,y+1)
    int l_d = (int)m_disp_l[idx - 1];    //d(x-1,y)
    int r_d = (int)m_disp_l[idx + 1];    //d(x+1,y)

    double wx, wy;

    wx = min(max(abs(l_d - d) - abs(r_d - d),-10), 10);
    wy = min(max(abs(u_d - d) - abs(d_d - d),-10), 10);
    // cout << "hah \twx = " << wx << "\twy = " << wy ;

    wx = exp(- (wx * wx) );
    wy = exp(- (wy * wy) );
    //  cout << "\twx = " << wx << "\twy = " << wy << "\tu_d = " << u_d << "\td_d = " << u_d << "\tl_d = " << l_d << "\tr_d = " << r_d << "\td = " << d << endl;


    dsmooth = wx * (l_d + r_d) + wy * (u_d + d_d);
    dsmooth = dsmooth / ( 2 * (wx + wy) );
    //sweight = wsmooth * (1 + alpha * epsilon + beta * epsilon * epsilon);
    //return true;
}

#if 0
void StereoFlow::scanline_optimize(const int direction) {

}
#endif


