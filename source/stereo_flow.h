#ifndef STEREOFLOW_H
#define STEREOFLOW_H

#include "common.h"
#include "cross_shaped_region.h"
#include "cost_method.h"
#include "aggr_method.h"
#include "match.h"
#include "match_value.h"
#include "region_voting.h"

class MatchHash;

const float c_thresh_zncc  = 0.85;	//0.85,1.15
const float c_thresh_prior = 1.20;
const float c_thresh_e_zncc  = 0.5;
const float c_thresh_e_prior = 1.02;

class StereoFlow
{
public:
    StereoFlow(const Mat& imgL, const Mat& imgR, const Mat& mskL, const Mat& mskR, const float& max_disp, const float& min_disp, const float scale );
    StereoFlow();

    ~StereoFlow();

    void calc_mean_images();    //for zncc calculation in stereo_flow class
    void calc_support_region();

    void calc_cost_vol();
    void aggr_cost_vol();
    void set_disp_by_wta();
    void max_takes_all(const vector<uchar>& mask, const vector<vector<float> >& cost_vol,
                       vector<float>& disp, vector<int>& bestk, vector<float>& mcost, vector<float>& prior);
    void min_takes_all(const vector<uchar>& mask, const vector<vector<float> >& cost_vol,
                       vector<float>& disp, vector<int>& bestk, vector<float>& mcost, vector<float>& prior);

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

   // void removal_isolated_propagation();
   // void check_ordering();

private:
    vector<vector<vector<float> > > m_hwd_costvol_l, m_hwd_costvol_r;
    vector<vector<vector<float> > > m_ncc_vecs_l, m_ncc_vecs_r;                                   //each pixel's ncc vector
    vector<float> m_seed_prior;                                                                   //priority of seed matches

  //  int removal_isolated_seeds(const int& rsize, const int& thresh);
  //  int removal_boundary_seeds(const int& rsize);
  //  int removal_isolated_disparities(vector<float>& disp, vector<int>& bestk, vector<float>& mcost, vector<float>& prior, const vector<uchar>& mask, const int& rsize, const int& thresh);

    void matching_cost_from_zncc();
    void compute_data_item(double& ddata, double& wdata, const int& y, const int& x, const float& tstep);
    void compute_smooth_item(double& dsmooth, double& wsmooth, const int& y, const int& x);

    //end of 2017.05.31

private:
    vector<float> m_view_l, m_gray_l, m_mean_l, m_view_r, m_gray_r, m_mean_r;            //image datas
    vector<uchar> m_mask_l, m_mask_r;                                                    //mask datas
    vector<float> m_disp_l, m_disp_r, m_seed_disp, m_disp;                               //disparity datas
    vector<float> m_best_mcost, m_best_mcost_l, m_best_mcost_r;                          //best matching cost datas
    vector<float> m_best_prior, m_best_prior_l, m_best_prior_r;                          //best prior datas
    vector<int> m_best_k, m_best_k_l, m_best_k_r;                                        //best disp discrete level
    vector<uchar> m_outliers_l, m_outliers_r;

    Mat m_mat_view_l, m_mat_view_r;          //rgb unsgined char [0,255]
    Mat m_mat_gray_l, m_mat_gray_r;          //grayscale unsigned char [0,255]
    Mat m_mat_mean_l, m_mat_mean_r;          //grayscale unsigned char [0,255]
    Mat m_mat_mask_l, m_mat_mask_r;          //grayscale unsigned char [0,255]

#if 1
    //blur size == DEFAULT_NCC_WND
    vector<float> m_ncc_mean_l, m_ncc_mean_r;
#endif

    vector<Match> m_matches_l, m_matches_r;                                    //seed matches

    CostMethod * m_cost_mtd;
    AggrMethod * m_aggr_mtd;
    CrossShapedRegion * m_support_region;
    RegionVoter * m_region_voter;

    //ScanlineOptimizer * m_scanline_optimizer;
    //scanline optimization
    //    float m_pi1, m_pi2;
    //    int m_tau_so;                   //params for scanline optimization

    float m_max_disp, m_min_disp, m_scale;
    int m_w, m_h, m_total, m_wnd_size, m_disp_ranges;
    int m_valid_pixels_l, m_valid_pixels_r;

    vector<vector<float> > m_cost_vol_l, m_cost_vol_r;

public:
    void set_patch_params(const int sz);
    CostMethod * set_cost_type(const CCType name);
    AggrMethod * set_aggr_type(const CAType name);

    void calc_cluster_costs_l(const int st_k, const int ed_k, const int lx, const int ly, vector<float>& mcosts);                   //calculate a cluster of matching cost - zncc
    void calc_cluster_costs_r(const int st_k, const int ed_k, const int rx, const int ry, vector<float>& mcosts);                   //calculate a cluster of matching cost - zncc

    void calc_init_disparity();                                                                         //calculate initial disparity
    void calc_init_disparity(const vector<int>& pre_bestk_l, const vector<int>& pre_bestk_r);           //calculate initial disparity
    void calc_seed_disparity();                                                                         //calculate seed disparity

    void disp_2_matches();                                                                              //disparity to matches to be propagated
    void matches_2_disp();                                                                              //matches to disparity after propagation

    void propagate(const int direction, priority_queue<Match>& queue, MatchHash *& hashmap);
    void seed_propagate(const int direction);                                                           //propagate

    void copy_disp_2_disp_l();
    void copy_disp_2_disp_r();
    void calc_rematch_borders(const vector<float>& disp, const vector<uchar>& mask, const int scanline, vector<int>& border0, vector<int>& border1);
    void re_match_l();                                                                                  //fill in unexpanded pixels
    void re_match_r();

    //fill in unexpanded pixels using guidance filter
    void upsampling_using_rbf();
    void upsampling_using_rbf(const Mat& rgb_view, const vector<uchar>& mask,
                              vector<float>& disp, vector<int>& best_k, vector<float>& best_mcost, vector<float>& best_prior);

#if 0
    void scanline_optimize(const int direction);
#endif

    void check_outliers_l();
    void check_outliers_r();
    void init_region_voter();
    void region_voting();
    void region_voting(const int direction, const vector<uchar>& mask, vector<float>& disp,
                       vector<int>& best_k, vector<float>& best_mcost, vector<float>& best_prior,
                       vector<uchar>& outliers);

    void median_filter();
//    void subpixel_enhancement();
    void cross_validation();

public:
    void set_wnd_size(const int wnd_sz) { m_wnd_size = wnd_sz; }
    int  get_wnd_size() { return m_wnd_size; }

    vector<float>& get_mean_l()     { return m_mean_l; }
    vector<float>& get_mean_r()     { return m_mean_r; }

    vector<float>& get_disp_l()     { return m_disp_l; }
    vector<float>& get_disp_r()     { return m_disp_r; }
    vector<float>& get_disp()       { return m_disp; }
    vector<float>& get_disp_seed()  { return m_seed_disp; }

    float get_scale() { return m_scale; }
    int get_w() { return m_w; }
    int get_h() { return m_h; }
    int get_disp_range()  { return m_disp_ranges; }

    Mat get_mat_mask_l()  { return m_mat_mask_l; }
    Mat get_mat_mask_r()  { return m_mat_mask_r; }
    Mat get_mat_mean_l()  { return m_mat_mean_l; }
    Mat get_mat_mean_r()  { return m_mat_mean_r; }
    Mat get_mat_view_l()  { return m_mat_view_l; }
    Mat get_mat_view_r()  { return m_mat_view_r; }
    Mat get_mat_gray_l() { return m_mat_gray_l; }
    Mat get_mat_gray_r() { return m_mat_gray_r; }

    vector<int>& get_bestk_l() { return m_best_k_l; }
    vector<int>& get_bestk_r() { return m_best_k_r; }
    vector<float>& get_best_mcost_l() { return m_best_mcost_l; }
    vector<float>& get_best_mcost_r() { return m_best_mcost_r; }
    vector<float>& get_best_prior_l() { return m_best_prior_l; }
    vector<float>& get_best_prior_r() { return m_best_prior_r; }
    vector<float>& get_mcost_l(const int k) { return m_cost_vol_l[k];}
    vector<float>& get_mcost_r(const int k) { return m_cost_vol_r[k];}

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

    m_valid_pixels_l = countNonZero(mskL);
    m_valid_pixels_r = countNonZero(mskR);

    //disparity
    m_disp_l.resize(m_total); m_disp_r.resize(m_total); m_disp.resize(m_total); m_seed_disp.resize(m_total);
    //matching cost
    m_best_mcost_l.resize(m_total); m_best_mcost_r.resize(m_total); m_best_mcost.resize(m_total);
    //priority
    m_best_prior_l.resize(m_total); m_best_prior_r.resize(m_total); m_best_prior.resize(m_total);
    //discrete disparity
    m_best_k_l.resize(m_total);
    m_best_k_r.resize(m_total);
    m_best_k.resize(m_total);

    m_support_region = new CrossShapedRegion();
    m_region_voter = NULL;
    m_cost_mtd = NULL;
    m_aggr_mtd = NULL;

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

inline float calc_min_prior(const float min_mcost, const float sec_mcost) {

    if(numeric_limits<float>::max() == sec_mcost || (min_mcost * sec_mcost) < 0) {   //min_mcost <<<<<<<<<<<<<<<<<< sec_mcost
        return c_thresh_prior;
    }

    return fabs(sec_mcost)/fabs(min_mcost);     //bigger the better
}

inline float calc_max_prior(const float max_mcost, const float sec_mcost) {
    if(-1 == sec_mcost || (max_mcost * sec_mcost) < 0 ) {     //max_mcost >>>>>>>>> sec_mcost
        return c_thresh_prior;
    }
    return fabs(max_mcost)/fabs(sec_mcost);                   //bigger the better
}

inline void cluster_max_takes_all(const vector<float>& mcosts, int& maxk, float& max_mcost, float& sec_mcost) {

    if(mcosts.size() == 1) { /*cout << "here" << endl;*/ maxk = 0; max_mcost = mcosts[0];  sec_mcost = max_mcost/c_thresh_prior;  return; }

    maxk = 0;
    max_mcost = mcosts[0], sec_mcost = -1;
    for(int k = 1, sz = mcosts.size(); k < sz; ++k ) {
        if(mcosts[k] > max_mcost) {
            sec_mcost = max_mcost;
            max_mcost = mcosts[k];
            maxk = k;
        }
        else if(mcosts[k] > sec_mcost) {
            sec_mcost = mcosts[k];
        }
    }
}


#endif // STEREOFLOW_H
