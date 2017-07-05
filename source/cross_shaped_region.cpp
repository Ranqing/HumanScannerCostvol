#include "cross_shaped_region.h"

void CrossShapedRegion::init(const int h, const int w, const int sz) {
    set_image_sz(w, h);
    set_patch_params(sz);
}

void CrossShapedRegion::set_patch_params(const int sz) {
    m_dis2 = sz;
    m_dis1 = 2 * m_dis2;
    m_tau1 = TAU1;
    m_tau2 = TAU2;

# if 0
    cout << "\tcross shaped region: dis1 = " << m_dis1 << ", dis2 = " << m_dis2 << ", tau1 = " << m_tau1 << ", tau2 = " << m_tau2 ;
# endif
}

void CrossShapedRegion::set_image_sz(const int w, const int h) {
    m_w = w;
    m_h = h;
}

void CrossShapedRegion::calc_patch_borders(const vector<float> &imgL, const vector<float> &imgR, const vector<uchar> &mskL, const vector<uchar> &mskR) {

    cout << "\tcalculate cross shaped region.." << "\t dis1 = " << m_dis1 << ", dis2 = " << m_dis2 << ", tau1 = " << m_tau1 << ", tau2 = " << m_tau2 << ".\t" ;
    double duration = (double)getTickCount();
    cout << "up.." ;       calc_patch_borders(imgL, mskL,  0, -1, m_u_borders_l); calc_patch_borders(imgR, mskR, 0, -1, m_u_borders_r);
    cout << "down..";      calc_patch_borders(imgL, mskL,  0,  1, m_d_borders_l); calc_patch_borders(imgR, mskR, 0,  1, m_d_borders_r);
    cout << "left..";      calc_patch_borders(imgL, mskL, -1,  0, m_l_borders_l); calc_patch_borders(imgR, mskR, -1, 0, m_l_borders_r);
    cout << "right..";     calc_patch_borders(imgL, mskL,  1,  0, m_r_borders_l); calc_patch_borders(imgR, mskR,  1, 0, m_r_borders_r);
    duration = ((double)getTickCount() - duration) / getTickFrequency();
    printf("\t%.2lf s\n", duration);

# if 0
    Mat test_border(m_h, m_w, CV_32FC3);
    qing_vec_2_img<float>(imgL, test_border);
    cvtColor(test_border, test_border, CV_RGB2BGR);

    int px = 188, py = 388;
    int index = py * m_w + px;
    int uborder = m_u_borders_l[index];
    int dborder = m_d_borders_l[index];
    int lborder = m_l_borders_l[index];
    int rborder = m_r_borders_l[index];
    cout << "......test..... u_border = " << uborder << ", d_border = " << dborder << ", l_border = " << lborder << ", r_border = " << rborder << endl;

    for(int i = 0; i < uborder; i++) {
        int cx = px;
        int cy = max(py - i, 0);
        test_border.at<Vec3f>(cy, cx) = Vec3f(1.f, 0.f, 0.f);         //blue up
    }
    for(int i = 0; i < dborder; i++) {
        int cx = px;
        int cy = max(py + i, 0);
        test_border.at<Vec3f>(cy, cx) = Vec3f(0.f, 1.f, 0.f);         //green down
    }
    for(int i = 0; i < lborder; i++) {
        int cx = px - i;
        int cy = py;
        test_border.at<Vec3f>(cy, cx) = Vec3f(0.f, 0.f, 0.0f);        //black left
    }
    for(int i = 0; i < rborder; i++) {
        int cx = px + i;
        int cy = py;
        test_border.at<Vec3f>(cy, cx) = Vec3f(1.f, 0.f, 1.f);         //purple right
    }
    circle(test_border, Point2f(px, py), 2, Scalar(0.f, 0.f, 1.f));   //red center

    imshow("test_border", test_border);
    waitKey(0);
    destroyWindow("test_border");
# endif
}

void CrossShapedRegion::calc_patch_borders(const vector<float> &img, const vector<uchar> &msk, const int dx, const int dy, vector<int> &border) {
    border.resize(m_w*m_h);
    fill_n(border.begin(), m_w*m_h, 0);

//#pragma omp parallel for
    for(int y = 0; y < m_h; ++y) {
        for(int x = 0; x < m_w; ++x) {
            int index = y * m_w + x;
            if(255 == msk[index]) {
                border[index] = calc_border(img, msk, x, y, dx, dy);
//                cout << y << ", " << x <<  ": " << border[index] << endl;
            }
        }
    }
}

int CrossShapedRegion::calc_border(const vector<float> &img, const vector<uchar> &msk, const int x, const int y, const int dx, const int dy) {
    int d = 1;
    int index = y * m_w + x;

    //center color
    float cen_r = img[index * 3 + 0];
    float cen_g = img[index * 3 + 1];
    float cen_b = img[index * 3 + 2];
    //predecerssor color
    float pre_r = cen_r, pre_g = cen_g, pre_b = cen_b;

    int xpos = x + d * dx;
    int ypos = y + d * dy;

    while(0 <= xpos && xpos < m_w && 0 <= ypos && ypos < m_h) {
        index = ypos * m_w + xpos;

        float cur_r = img[index * 3 + 0];
        float cur_g = img[index * 3 + 1];
        float cur_b = img[index * 3 + 2];

        float delta_r_1 = cur_r - cen_r; float delta_g_1 = cur_g - cen_g; float delta_b_1 = cur_b - cen_b;
        float delta_r_2 = cur_r - pre_r; float delta_g_2 = cur_g - pre_g; float delta_b_2 = cur_b - pre_b;

        //color differene between cur_pixel and cen_pixel
        float color_diff_1 = sqrt(delta_r_1 * delta_r_1 + delta_g_1 * delta_g_1 + delta_b_1 * delta_b_1) * 0.3;
        //color difference between cur_pixel and pre_pixel
        float color_diff_2 = sqrt(delta_r_2 * delta_r_2 + delta_g_2 * delta_g_2 + delta_b_2 * delta_b_2) * 0.3;

        bool condition1 = msk[index];
        bool condition2 = d < m_dis1;
        bool condition3 = ( color_diff_1 < m_tau1 ) && ( color_diff_2 < m_tau1 );      //difference with center pixel and previous pixel small than m_tau1
        bool condition4 = d < m_dis2 || ( d > m_dis2 && color_diff_1 < m_tau2 );

        if( !(condition1 && condition2 && condition3 && condition4)) break;

        ++d;
        pre_r = cur_r, pre_g = cur_g, pre_b = cur_b;
        xpos += dx;
        ypos += dy;
    }
    return d - 1;
}


void CrossShapedRegion::aggr_for_normalize(const int direction /* = 0*/) {
    vector<int>& borders_u = direction ? get_u_borders_r() : get_u_borders_l();
    vector<int>& borders_d = direction ? get_d_borders_r() : get_d_borders_l();
    vector<int>& borders_l = direction ? get_l_borders_r() : get_l_borders_l();
    vector<int>& borders_r = direction ? get_r_borders_r() : get_r_borders_l();
    vector<int>& aggr_borders = direction ? m_aggr_borders_r : m_aggr_borders_l;
    aggr_borders.clear();
    aggr_borders.resize(m_h*m_w, 0);

    int y, x, idx = -1, limit_l, limit_r, limit_d, limit_u, dx, dy, didx;

    vector<int> tmp_borders(m_h*m_w, 0);

    //horizontal
    for(y = 0; y < m_h; ++y) {
        for(x = 0; x < m_w; ++x) {
            ++idx;
            tmp_borders[idx] = borders_r[idx] + borders_l[idx] + 1;
        }
    }

    //vertical
    for(idx = -1, y = 0; y < m_h; ++y) {
        for (x = 0; x < m_w; ++x) {
            aggr_borders[++idx] = 0;
            limit_u = -borders_u[idx];
            limit_d = borders_d[idx];
            if (limit_u == limit_d) aggr_borders[idx] = tmp_borders[idx];
            else {
                for (dy = limit_u; dy <= limit_d; ++dy) {
                    if (0 > y + dy || m_h <= y + dy) continue;
                    didx = idx + dy * m_w;

                    aggr_borders[idx] += tmp_borders[didx];
                }
            }

        }
    }

}