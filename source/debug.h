#ifndef DEBUGGER_H
#define DEBUGGER_H

#include "../../Qing/qing_common.h"

class StereoFlow;

class Debugger
{
public:
    Debugger(const string& dir);
    ~Debugger();

    void save_init_infos(const int level);
    void save_seed_infos(const int level);
    void save_prop_infos(const int level);
    void save_clean_prop_infos(const int level);
    void save_order_check_infos(const int level, const string& savename);
    void save_rematch_infos(const int level);
    void save_upsamling_infos(const int level);
    void save_so_infos(const int level);
    void save_rv_infos(const int level);
    void save_median_infos(const int level);
    void save_subpixel_infos(const int level);
    void save_final_infos(const int level);

    void set_data_source(StereoFlow * stereo_fl);
    void set_triangulate_info(const float& scale, const Point2i& crop_l, const Point2i& crop_r, const Mat& qmatrix);

    void get_disp_datas(Mat& disp_l, Mat& disp_r, Mat& disp);
    void save_disp_data(const string& savefn, const Mat& disp);
    void save_float_vec_data(const string& savefn, const int w, const int h, const vector<float>& vec);
    void save_int_vec_data(const string& savefn, const int w, const int h, const vector<int>& vec);

    void fast_check_sgbm(const string sgbmname);
    void fast_check_disp_by_depth(const string filename, float * mdisp);
    void fast_check_by_histogram(const string histname, int disp_range = 0, int y_min = 0, int y_max = 0);
    void fast_check_by_diff(const string diffname, const int diff_thresh = 1);

    void save_disparity(const string& prefix, const string& lvl, const string& suffix);
    void save_disparity_with_outliers(const string& prefix, const string& lvl, const string& suffix, const char& c);

    void compare_init_final_disp(const int level) ;

private:
    string m_save_dir;                  //save dir
    float m_scale;                      //disparity scale
    StereoFlow * m_stereo_fl;           //data sources

    int m_w, m_h, m_total;
    Point2i m_crop_l, m_crop_r;
    float m_crop_d;                    //crop disparity
    Mat m_qmatrix;

};


#endif // DEBUGGER_H
