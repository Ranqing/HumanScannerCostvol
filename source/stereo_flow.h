#ifndef STEREOFLOW_H
#define STEREOFLOW_H

#include "common.h"
#include "cross_shaped_region.h"
#include "region_voting.h"


class StereoFlow
{
public:
    StereoFlow(const Mat& imgL, const Mat& imgR, const Mat& mskL, const Mat& mskR, const float& max_disp, const float& min_disp, const float scale );
    StereoFlow();

    ~StereoFlow();

    void calc_mean_images();    //for zncc calculation in stereo_flow class
    void calc_support_region();

    //2017.05.31
    void matching_cost();
    void matching_cost_aggregation(const int direction, vector<vector<vector<float> > >& hwd_cost_vol);
    void calc_init_disparity_from_cost();
    void calc_init_disparity_from_cost(const vector<int>& pre_bestk_l, const vector<int>& pre_bestk_r, const int pre_w, const int pre_h);
    void cross_check();
    void local_smoothness_check();
    void ordering_check();
    void rematch_using_rbf();
    void rematch_using_rbf(const Mat& rgb_view, const vector<uchar>& mask, const vector<uchar>& outliers, vector<float>& disp);

    void subpixel_enhancement();
    void subpixel_approximation();
    void trans_costvol_for_beeler();
    void beeler_subpixel_refinement(const float& dstep, const double& thresh);
    void median_filter();

private:
    vector<vector<vector<float> > > m_hwd_costvol_l, m_hwd_costvol_r;

    void matching_cost_from_zncc();
    void compute_data_item(double& ddata, double& wdata, const int& y, const int& x, const float& tstep);
    void compute_smooth_item(double& dsmooth, double& wsmooth, const int& y, const int& x);

private:
    vector<float> m_view_l, m_gray_l, m_mean_l, m_view_r, m_gray_r, m_mean_r;            //image datas
    vector<uchar> m_mask_l, m_mask_r;                                                    //mask datas
    vector<float> m_disp_l, m_disp_r, m_disp;                               //disparity datas
    vector<float> m_best_mcost, m_best_mcost_l, m_best_mcost_r;                          //best matching cost datas
    vector<float> m_best_prior, m_best_prior_l, m_best_prior_r;                          //best prior datas
    vector<int> m_best_k, m_best_k_l, m_best_k_r;                                        //best disp discrete level
    vector<uchar> m_outliers_l, m_outliers_r;

    Mat m_mat_view_l, m_mat_view_r;             //rgb unsgined char [0,255]
    Mat m_mat_gray_l, m_mat_gray_r;             //grayscale unsigned char [0,255]
    Mat m_mat_mean_l, m_mat_mean_r;             //grayscale unsigned char [0,255]
    Mat m_mat_mask_l, m_mat_mask_r;             //grayscale unsigned char [0,255]

    CrossShapedRegion * m_support_region;
    RegionVoter * m_region_voter;

    //ScanlineOptimizer * m_scanline_optimizer;
    //scanline optimization
    //float m_pi1, m_pi2;
    //int m_tau_so;                   //params for scanline optimization
    float m_max_disp, m_min_disp, m_scale;
    int m_w, m_h, m_total, m_wnd_size, m_disp_ranges;

public:
    void set_wnd_size(const int wnd_sz) { m_wnd_size = wnd_sz; }
    int  get_wnd_size() { return m_wnd_size; }

    vector<float>& get_disp_l()     { return m_disp_l; }
    vector<float>& get_disp_r()     { return m_disp_r; }
    vector<float>& get_disp()       { return m_disp; }

    int get_w() { return m_w; }
    int get_h() { return m_h; }
    int get_disp_range()  { return m_disp_ranges; }
    float get_scale() { return m_scale; }

    Mat get_mat_mask_l()  { return m_mat_mask_l; }
    Mat get_mat_mask_r()  { return m_mat_mask_r; }
    Mat get_mat_mean_l()  { return m_mat_mean_l; }
    Mat get_mat_mean_r()  { return m_mat_mean_r; }
    Mat get_mat_view_l()  { return m_mat_view_l; }
    Mat get_mat_view_r()  { return m_mat_view_r; }
    Mat get_mat_gray_l()  { return m_mat_gray_l; }
    Mat get_mat_gray_r()  { return m_mat_gray_r; }

    vector<int>& get_bestk_l() { return m_best_k_l; }
    vector<int>& get_bestk_r() { return m_best_k_r; }
    vector<float>& get_best_mcost_l() { return m_best_mcost_l; }
    vector<float>& get_best_mcost_r() { return m_best_mcost_r; }
    vector<float>& get_best_prior_l() { return m_best_prior_l; }
    vector<float>& get_best_prior_r() { return m_best_prior_r; }

    vector<uchar>& get_outliers_l() { return m_outliers_l;}
    vector<uchar>& get_outliers_r() { return m_outliers_r;}
    vector<uchar>& get_mask_l() { return m_mask_l;}
    vector<uchar>& get_mask_r() { return m_mask_r;}
};

inline StereoFlow::StereoFlow(const Mat &imgL, const Mat &imgR, const Mat &mskL, const Mat &mskR, const float &max_disp, const float &min_disp, const float scale):
    m_max_disp(max_disp), m_min_disp(min_disp), m_scale(scale) {

    //bgr view : uchar [0,255]
    cvtColor(imgL, m_mat_view_l, CV_BGR2RGB);
    cvtColor(imgR, m_mat_view_r, CV_BGR2RGB);

    //gray view: uchar [0,255]
    cvtColor(m_mat_view_l, m_mat_gray_l, CV_RGB2GRAY);
    cvtColor(m_mat_view_r, m_mat_gray_r, CV_RGB2GRAY);

    //mask: uchar [0,255]
    m_mat_mask_l = mskL.clone();
    m_mat_mask_r = mskR.clone();

    m_w = imgL.size().width;
    m_h = imgL.size().height;
    m_total = m_w * m_h;
    m_disp_ranges = (m_max_disp - m_min_disp) / DISP_STEP;

    Mat viewL, viewR, grayL, grayR;
    m_mat_view_l.convertTo(viewL, CV_32FC3, 1.0f);
    m_mat_view_r.convertTo(viewR, CV_32FC3, 1.0f);
    cvtColor(viewL, grayL, CV_RGB2GRAY);
    cvtColor(viewR, grayR, CV_RGB2GRAY);

    //rgb float [0,255]
    m_view_l.resize(m_total * 3); memcpy(&m_view_l.front(), viewL.data, sizeof(float)*m_total*3);
    m_view_r.resize(m_total * 3); memcpy(&m_view_r.front(), viewR.data, sizeof(float)*m_total*3);
    //gray float [0,255]
    m_gray_l.resize(m_total); memcpy(&m_gray_l.front(), grayL.data, sizeof(float)*m_total);
    m_gray_r.resize(m_total); memcpy(&m_gray_r.front(), grayR.data, sizeof(float)*m_total);
    //mask uchar [0,255]
    m_mask_l.resize(m_total); memcpy(&m_mask_l.front(), mskL.data, sizeof(uchar)*m_total);
    m_mask_r.resize(m_total); memcpy(&m_mask_r.front(), mskR.data, sizeof(uchar)*m_total);

    //disparity
    m_disp_l.resize(m_total); m_disp_r.resize(m_total); m_disp.resize(m_total);
    //matching cost
    m_best_mcost_l.resize(m_total); m_best_mcost_r.resize(m_total); m_best_mcost.resize(m_total);
    //priority
    m_best_prior_l.resize(m_total); m_best_prior_r.resize(m_total); m_best_prior.resize(m_total);
    //discrete disparity
    m_best_k_l.resize(m_total); m_best_k_r.resize(m_total); m_best_k.resize(m_total);

    m_support_region = new CrossShapedRegion();
    m_region_voter = NULL;

    m_outliers_l.clear(); m_outliers_l.resize(m_total, 0);
    m_outliers_r.clear(); m_outliers_r.resize(m_total, 0);

#if 0
    cout << m_w << '\t' << m_h << endl;
    Mat test_view = Mat::zeros(m_h, m_w, CV_32FC3);
    Mat test_gray = Mat::zeros(m_h, m_w, CV_32FC1);
    Mat test_mask = Mat::zeros(m_h, m_w, CV_8UC1);
    Mat uchar_test_view,uchar_test_gray;

    qing_vec_2_img<float>(m_view_l, test_view);
    qing_vec_2_img<float>(m_gray_l, test_gray);
    qing_vec_2_img<uchar>(m_mask_l, test_mask);

    cvtColor(test_view, test_view, CV_RGB2BGR);
    test_view.convertTo(uchar_test_view, CV_8UC3, 1);
    test_gray.convertTo(uchar_test_gray, CV_8UC1, 1);

    imshow("test_view_l", uchar_test_view);
    imshow("test_gray_l", uchar_test_gray);
    imshow("test_mask_l", test_mask);

    qing_vec_2_img<float>(m_view_r, test_view);
    qing_vec_2_img<float>(m_gray_r, test_gray);
    qing_vec_2_img<uchar>(m_mask_r, test_mask);

    cvtColor(test_view, test_view, CV_RGB2BGR);
    test_view.convertTo(uchar_test_view, CV_8UC3, 1);
    test_gray.convertTo(uchar_test_gray, CV_8UC1, 1);

    imshow("test_view_r", uchar_test_view);
    imshow("test_gray_r", uchar_test_gray);
    imshow("test_mask_r", test_mask);

    waitKey(0);
    destroyAllWindows();
#endif

}

inline StereoFlow::StereoFlow() {
}

inline StereoFlow::~StereoFlow() {
    if(NULL != m_support_region) { delete m_support_region;   m_support_region = NULL; }
    if(NULL != m_region_voter) { delete m_region_voter;  m_region_voter = NULL; }
}


#endif // STEREOFLOW_H
