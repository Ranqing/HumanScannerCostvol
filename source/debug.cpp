#include "debug.h"
#include "stereo_flow.h"

#include "../../Qing/qing_image.h"
#include "../../Qing/qing_ply.h"
#include "../../Qing/qing_disp.h"

Debugger::Debugger(const string& dir):m_save_dir(dir)  {
}

Debugger::~Debugger() {

}

void Debugger::set_data_source(StereoFlow *stereo_fl) {
    m_stereo_fl = stereo_fl;
    m_scale = m_stereo_fl->get_scale();
    m_w = m_stereo_fl->get_w();
    m_h = m_stereo_fl->get_h();
    m_total = m_w * m_h;
}

void Debugger::set_triangulate_info(const float &scale, const Point2i &crop_l, const Point2i &crop_r, const Mat &qmatrix) {
    m_crop_l = crop_l * scale;
    m_crop_r = crop_r * scale;
    m_qmatrix = qmatrix.clone();

    double * qmtx = (double *)m_qmatrix.ptr<double>(0);
    qmtx[0*4+3] *= scale;
    qmtx[1*4+3] *= scale;
    qmtx[2*4+3] *= scale;

    m_crop_d = (crop_l.x - crop_r.x)*scale;

# if 0
    cout << scale << endl;
    cout << m_crop_l << endl;
    cout << m_crop_r << endl;
    cout << m_crop_d << endl;
    cout << m_qmatrix << endl;
# endif
}

void Debugger::get_disp_datas(Mat& disp_l, Mat& disp_r, Mat& disp) {
    vector<float>& disp_vec_l = m_stereo_fl->get_disp_l();
    vector<float>& disp_vec_r = m_stereo_fl->get_disp_r();
    vector<float>& disp_vec = m_stereo_fl->get_disp();

    disp_l = Mat::zeros(m_h, m_w, CV_32FC1);
    disp_r = Mat::zeros(m_h, m_w, CV_32FC1);
    disp = Mat::zeros(m_h, m_w, CV_32FC1);

    qing_vec_2_img<float>(disp_vec_l, disp_l);
    qing_vec_2_img<float>(disp_vec_r, disp_r);
    qing_vec_2_img<float>(disp_vec, disp);
}

void Debugger::save_disp_data(const string &savefn, const Mat &disp) {
    Mat disp_img;
    disp.convertTo(disp_img, CV_8UC1, m_scale);
    imwrite(savefn, disp_img);
}

void Debugger::save_float_vec_data(const string& savefn, const int w, const int h, const vector<float>& vec){
    ofstream fout(savefn, ios::out);
    if(fout.is_open() == true) {
        for(int x = 0, cnt = 0; x < vec.size(); ++x) {
            fout << setprecision(4) << vec[x] << ' ';
            if(cnt == w)  {
                fout << endl;
                cnt = 0;
            }
            else cnt++;
        }
    }
    fout.close();

}

void Debugger::save_int_vec_data(const string &savefn, const int w, const int h, const vector<int> &vec) {
    ofstream fout(savefn, ios::out);
    if(fout.is_open() == true) {
        for(int x = 0, cnt = 0; x < vec.size(); ++x) {
            fout <<  vec[x] << ' ';
            if(cnt == w)  {
                fout << endl;
                cnt = 0;
            }
            else cnt++;
        }
    }
    fout.close();
}

void Debugger::save_init_infos(const int level) {
    Mat disp_l, disp_r, disp;
    get_disp_datas(disp_l, disp_r, disp);

    string str = int2string(level);
    string save_fn_l = m_save_dir + "/init_disp_l_" + str + ".jpg";
    string save_fn_r = m_save_dir + "/init_disp_r_" + str + ".jpg";
    string save_fn   = m_save_dir + "/init_disp_" + str + ".jpg";
    save_disp_data(save_fn_l, disp_l);
    cout << "\ndebug:\tsaving " << save_fn_l << endl;
    save_disp_data(save_fn_r, disp_r);
    cout << "debug:\tsaving " << save_fn_r << endl;
    save_disp_data(save_fn, disp);
    cout << "debug:\tsaving " << save_fn << endl;

# if 0
    int ranges = m_stereo_fl->get_disp_range();
    int w = m_stereo_fl->get_w();
    int h = m_stereo_fl->get_h();
    save_fn_l = m_save_dir + "/best_mcost_l_" + str + ".txt";
    save_float_vec_data(save_fn_l, w, h, m_stereo_fl->get_best_mcost_l());
    save_fn_r = m_save_dir + "/best_mcost_r_" + str + ".txt";
    save_float_vec_data(save_fn_r, w, h, m_stereo_fl->get_best_mcost_r());
    cout << "debug:\tsaving " << save_fn_l  << '\t' << save_fn_r << endl;
    save_fn_l = m_save_dir + "/best_prior_l_" + str + ".txt";
    save_float_vec_data(save_fn_l, w, h, m_stereo_fl->get_best_prior_l());
    save_fn_r = m_save_dir + "/best_prior_r_" + str + ".txt";
    save_float_vec_data(save_fn_r, w, h, m_stereo_fl->get_best_prior_r());
    cout << "debug:\tsaving " << save_fn_l  << '\t' << save_fn_r << endl;
    save_fn_l = m_save_dir + "/best_disp_l_" + str + ".txt";
    save_float_vec_data(save_fn_l, w, h, m_stereo_fl->get_disp_l());
    save_fn_r = m_save_dir + "/best_disp_r_" + str + ".txt";
    save_float_vec_data(save_fn_r, w, h, m_stereo_fl->get_disp_r());
    cout << "debug:\tsaving " << save_fn_l  << '\t' << save_fn_r << endl;

    save_fn_l = m_save_dir + "/best_k_l_" + str + ".txt";
    save_int_vec_data(save_fn_l, w, h, m_stereo_fl->get_bestk_l());
    save_fn_r = m_save_dir + "/best_k_r_" + str + ".txt";
    save_int_vec_data(save_fn_r, w, h, m_stereo_fl->get_bestk_r());
    cout << "debug:\tsaving " << save_fn_l  << '\t' << save_fn_r << endl;

    //    for(int i = 0; i < ranges; ++i) {
    //        save_fn_l = m_save_dir + "/mcost_l_" + str + "_" + int2string(i) + ".txt";
    //        save_float_vec_data(save_fn_l, w, h, m_stereo_fl->get_mcost_l(i));
    //        save_fn_r = m_save_dir + "/mcost_r_" + str + "_" + int2string(i) + ".txt";
    //        save_float_vec_data(save_fn_r, w, h, m_stereo_fl->get_mcost_r(i));
    //        cout << "debug:\tsaving " << save_fn_l << "\t" << save_fn_r << endl;
    //    }
# endif
}

void Debugger::save_prop_infos(const int level) {
    Mat disp_l, disp_r, disp;
    get_disp_datas(disp_l, disp_r, disp);

    string str = int2string(level);
    string save_fn_l = m_save_dir + "/prop_disp_l_" + str + ".jpg";
    string save_fn_r = m_save_dir + "/prop_disp_r_"  + str + ".jpg";
    string save_fn   = m_save_dir + "/prop_disp_" + str + ".jpg";

    save_disp_data(save_fn_l, disp_l);
    cout << "\ndebug:\tsaving " << save_fn_l << endl;
    save_disp_data(save_fn_r, disp_r);
    cout << "debug:\tsaving " << save_fn_r << endl;
    save_disp_data(save_fn, disp);
    cout << "debug:\tsaving " << save_fn << endl;
}

void Debugger::save_clean_prop_infos(const int level) {
    Mat disp_l, disp_r, disp;
    get_disp_datas(disp_l, disp_r, disp);

    string str = int2string(level);
    string save_fn_l = m_save_dir + "/prop_clean_disp_l_" + str + ".jpg";
    string save_fn_r = m_save_dir + "/prop_clean_disp_r_"  + str + ".jpg";
    string save_fn   = m_save_dir + "/prop_clean_disp_" + str + ".jpg";

    save_disp_data(save_fn_l, disp_l);
    cout << "\ndebug:\tsaving " << save_fn_l << endl;
    save_disp_data(save_fn_r, disp_r);
    cout << "debug:\tsaving " << save_fn_r << endl;
    save_disp_data(save_fn, disp);
    cout << "debug:\tsaving " << save_fn << endl;
}

void Debugger::save_order_check_infos(const int level, const string& savename) {
    Mat disp_l, disp_r, disp;
    get_disp_datas(disp_l, disp_r, disp);

    string str = int2string(level);
    string save_fn   = m_save_dir + "/" + savename;
    save_disp_data(save_fn, disp);
    cout << "\ndebug:\tsaving " << save_fn << endl;
}


void Debugger::save_rematch_infos(const int level) {
    Mat disp_l, disp_r, disp;
    get_disp_datas(disp_l, disp_r, disp);

    string str = int2string(level);
    string save_fn_l = m_save_dir + "/rematch_disp_l_" + str + ".jpg";           //i.e. rematch
    string save_fn_r = m_save_dir + "/rematch_disp_r_"  + str + ".jpg";
    string save_fn   = m_save_dir + "/rematch_disp_" + str + ".jpg";

    save_disp_data(save_fn_l, disp_l);
    cout << "\ndebug:\tsaving " << save_fn_l << endl;
    save_disp_data(save_fn_r, disp_r);
    cout << "debug:\tsaving " << save_fn_r << endl;
    save_disp_data(save_fn, disp);
    cout << "debug:\tsaving " << save_fn << endl;
}

void Debugger::save_upsamling_infos(const int level) {
    Mat disp_l, disp_r, disp;
    get_disp_datas(disp_l, disp_r, disp);

    string str = int2string(level);
    string save_fn_l = m_save_dir + "/upsamling_disp_l_" + str + ".jpg";
    string save_fn_r = m_save_dir + "/upsamling_disp_r_"  + str + ".jpg";
    string save_fn   = m_save_dir + "/upsamling_disp_" + str + ".jpg";

    save_disp_data(save_fn_l, disp_l);
    cout << "\ndebug:\tsaving " << save_fn_l << endl;
    save_disp_data(save_fn_r, disp_r);
    cout << "debug:\tsaving " << save_fn_r << endl;
    save_disp_data(save_fn, disp);
    cout << "debug:\tsaving " << save_fn << endl;
}

void Debugger::save_so_infos(const int level) {
    Mat disp_l, disp_r, disp;
    get_disp_datas(disp_l, disp_r, disp);

    string str = int2string(level);
    string save_fn_l = m_save_dir + "/so_disp_l_" + str + ".jpg";
    string save_fn_r = m_save_dir + "/so_disp_r_"  + str + ".jpg";
    string save_fn   = m_save_dir + "/so_disp_" + str + ".jpg";

    save_disp_data(save_fn_l, disp_l);
    cout << "\ndebug:\tsaving " << save_fn_l << endl;
    save_disp_data(save_fn_r, disp_r);
    cout << "debug:\tsaving " << save_fn_r << endl;
    save_disp_data(save_fn, disp);
    cout << "debug:\tsaving " << save_fn << endl;
}

void Debugger::save_rv_infos(const int level) {
    Mat disp_l, disp_r, disp;
    get_disp_datas(disp_l, disp_r, disp);

    string str = int2string(level);
    string save_fn_l = m_save_dir + "/rv_disp_l_" + str + ".jpg";
    string save_fn_r = m_save_dir + "/rv_disp_r_"  + str + ".jpg";
    string save_fn   = m_save_dir + "/rv_disp_" + str + ".jpg";

    save_disp_data(save_fn_l, disp_l);
    cout << "\ndebug:\tsaving " << save_fn_l << endl;
    save_disp_data(save_fn_r, disp_r);
    cout << "debug:\tsaving " << save_fn_r << endl;
    save_disp_data(save_fn, disp);
    cout << "debug:\tsaving " << save_fn << endl;
}

void Debugger::save_median_infos(const int level) {
    Mat disp_l, disp_r, disp;
    get_disp_datas(disp_l, disp_r, disp);

    string str = int2string(level);
    string save_fn_l = m_save_dir + "/mf_disp_l_" + str + ".jpg";          //i.e. median filter
    string save_fn_r = m_save_dir + "/mf_disp_r_"  + str + ".jpg";
    string save_fn   = m_save_dir + "/mf_disp_" + str + ".jpg";

    save_disp_data(save_fn_l, disp_l);
    cout << "\ndebug:\tsaving " << save_fn_l << endl;
    save_disp_data(save_fn_r, disp_r);
    cout << "debug:\tsaving " << save_fn_r << endl;
    save_disp_data(save_fn, disp);
    cout << "debug:\tsaving " << save_fn << endl;
}

void Debugger::save_final_infos(const int level) {
    Mat disp_l, disp_r, disp;
    get_disp_datas(disp_l, disp_r, disp);

    string str = int2string(level);
    string save_fn_l = m_save_dir + "/final_disp_l_" + str + ".jpg";
    string save_fn_r = m_save_dir + "/final_disp_r_"  + str + ".jpg";
    string save_fn   = m_save_dir + "/final_disp_" + str + ".jpg";

    save_disp_data(save_fn_l, disp_l);    cout << "\ndebug:\tsaving " << save_fn_l << endl;
    save_disp_data(save_fn_r, disp_r);    cout << "debug:\tsaving " << save_fn_r << endl;
    save_disp_data(save_fn, disp);    cout << "debug:\tsaving " << save_fn << endl;

    int w = m_stereo_fl->get_w();
    int h = m_stereo_fl->get_h();
    vector<float>& disp_vec_l = m_stereo_fl->get_disp_l();
    vector<float>& disp_vec_r = m_stereo_fl->get_disp_r();
    vector<float>& disp_vec = m_stereo_fl->get_disp();

    save_fn_l = m_save_dir + "/final_disp_l_" + str + ".txt"; save_float_vec_data(save_fn_l, w, h, disp_vec_l); cout << "debug:\tsaving " << save_fn_l << endl;
    save_fn_r = m_save_dir + "/final_disp_r_" + str + ".txt"; save_float_vec_data(save_fn_r, w, h, disp_vec_r); cout << "debug:\tsaving " << save_fn_r << endl;
    save_fn = m_save_dir + "/final_disp_" + str + ".txt"; save_float_vec_data(save_fn, w, h, disp_vec); cout << "debug:\tsaving " << save_fn << endl;

    Mat mask_l = m_stereo_fl->get_mat_mask_l();
    Mat mask_r = m_stereo_fl->get_mat_mask_r();
    int wnd_sz = m_stereo_fl->get_wnd_size();
    Mat erode_mask_l =  qing_erode_image(mask_l, 2*wnd_sz);
    Mat erode_mask_r =  qing_erode_image(mask_r, 2*wnd_sz);
    Mat erode_disp_l, erode_disp_r, erode_disp;
    disp_l.copyTo(erode_disp_l, erode_mask_l);
    disp_r.copyTo(erode_disp_r, erode_mask_r);
    disp.copyTo(erode_disp, erode_mask_l);

    save_fn_l = m_save_dir + "/erode_final_disp_l_" + str + ".jpg"; save_disp_data(save_fn_l, erode_disp_l);    cout << "debug:\tsaving " << save_fn_l << endl;
    save_fn_r = m_save_dir + "/erode_final_disp_r_" + str + ".jpg"; save_disp_data(save_fn_r, erode_disp_r);    cout << "debug:\tsaving " << save_fn_r << endl;
    save_fn = m_save_dir + "/erode_final_disp_" + str + ".jpg"; save_disp_data(save_fn, erode_disp);    cout << "debug:\tsaving " << save_fn << endl;

    int total = w * h;
    vector<float> erode_disp_vec_l(total);
    vector<float> erode_disp_vec_r(total);
    vector<float> erode_disp_vec(total);
    memcpy(&erode_disp_vec_l.front(), erode_disp_l.data, sizeof(float)*total );
    memcpy(&erode_disp_vec_r.front(), erode_disp_r.data, sizeof(float)*total );
    memcpy(&erode_disp_vec.front(), erode_disp.data, sizeof(float)*total);

    save_fn_l = m_save_dir + "/erode_final_disp_l_" + str + ".txt"; save_float_vec_data(save_fn_l, w, h, erode_disp_vec_l); cout << "debug:\tsaving " << save_fn_l << endl;
    save_fn_r = m_save_dir + "/erode_final_disp_r_" + str + ".txt"; save_float_vec_data(save_fn_r, w, h, erode_disp_vec_r); cout << "debug:\tsaving " << save_fn_r << endl;
    save_fn = m_save_dir + "/erode_final_disp_" + str + ".txt"; save_float_vec_data(save_fn, w, h, erode_disp_vec); cout << "debug:\tsaving " << save_fn << endl;
}

void Debugger::save_subpixel_infos(const int level) {

    string str = int2string(level);
    string save_fn = m_save_dir + "/subpixel_final_disp_" + str + ".txt";

    int w = m_stereo_fl->get_w();
    int h = m_stereo_fl->get_h();
    vector<float>& disp = m_stereo_fl->get_disp();
    save_float_vec_data(save_fn, w, h, disp);
    cout << "\ndebug:\tsaving " << save_fn << endl;
}

//by 2017.05.30
//check by disparity
void Debugger::fast_check_sgbm(const string sgbmname) {

//    StereoSGBM sgbm;
//    sgbm.preFilterCap = 63;
//    sgbm.SADWindowSize = 3;
//
//    int cn = 1;
//
//    sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
//    sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
//    sgbm.minDisparity = 0;                                              //0
//    sgbm.numberOfDisparities = (m_stereo_fl->get_disp_range() + 1)*16;                              //total search disparity : 480 * 16
//    sgbm.uniquenessRatio = 10;
//    sgbm.speckleWindowSize = 100;
//    sgbm.speckleRange = 32;
//    sgbm.disp12MaxDiff = 1;
//    sgbm.fullDP = true;
//
//    Mat mat_gray_l = m_stereo_fl->get_mat_gray_l();
//    Mat mat_gray_r = m_stereo_fl->get_mat_gray_r();
//    Mat mat_mask_l = m_stereo_fl->get_mat_mask_l();
//    Mat mat_disp;
//    sgbm(mat_gray_l, mat_gray_r, mat_disp);
//
//    Mat show_disp, show_disp_with_mask, true_disp;
//    mat_disp.convertTo(true_disp, CV_32FC1, 1.f/16);
//    mat_disp.convertTo(show_disp, CV_8U, 255.0/(sgbm.numberOfDisparities));
//    show_disp.copyTo(show_disp_with_mask, mat_mask_l);
//
//    double maxVal, minVal;
//    minMaxLoc(show_disp, &minVal, &maxVal);
//
//    string jpgname = sgbmname + ".jpg";
//    imwrite(jpgname, show_disp_with_mask );
//    cout << "debug:\tsaving " + jpgname << endl;
//    string plyname = sgbmname + ".ply";
//    fast_check_disp_by_depth(plyname, true_disp.ptr<float>(0));
}

void Debugger::fast_check_disp_by_depth(const string filename, float * mdisp) {
    Mat mat_view_l = m_stereo_fl->get_mat_view_l().clone();
    Mat mat_mask_l = m_stereo_fl->get_mat_mask_l().clone();
    cvtColor(mat_view_l, mat_view_l, CV_BGR2RGB);

    unsigned char * pmsk = (unsigned char *)(mat_mask_l.ptr<unsigned char>(0));
    unsigned char * pclr = (unsigned char *)(mat_view_l.ptr<unsigned char>(0));
    double * qmtx = (double *)m_qmatrix.ptr<double>(0);

    float * pdsp = new float[m_total];
    //vector<float>& mdisp = m_stereo_fl->get_disp(); //m_stereo_fl->get_disp_l();
    memcpy(pdsp, mdisp, sizeof(float)*m_total);

    for(int i = 0; i < m_total; ++i) {
        if(255 != pmsk[i] || 0 == pdsp[i]) continue;
        pdsp[i] += m_crop_d;
    }

    vector<Vec3f> points, colors;
    qing_disp_2_depth(points, colors, pdsp, pmsk, pclr, qmtx, m_crop_l, m_w, m_h );

    qing_write_point_color_ply(m_save_dir + "/" + filename, points, colors);
    cout << "debug:\tsaving " << filename << "..\t" << points.size() << " points. " << endl;
}

//check by disparity histogram: draw histogram
void Debugger::fast_check_by_histogram(const string histname, int disp_range, int y_min, int y_max) {
    Mat mat_disp = Mat::zeros(m_h, m_w, CV_32FC1);
    qing_vec_2_img<float>(m_stereo_fl->get_disp_l(), mat_disp);
    Mat mat_mask = m_stereo_fl->get_mat_mask_l();

    if(0 == disp_range) disp_range = m_stereo_fl->get_disp_range();
    if(0 == (y_min * y_max)) { y_min = 0; y_max = m_h - 1; }

    int hist_size = disp_range+1;
    int hist_w = disp_range+1;
    int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/hist_size );
    float range[] = {0, (float)(disp_range+1)};
    const float * hist_range = { range };
    bool uniform = true, accumulate = false;

    Mat disp_hist;
    calcHist(&mat_disp, 1, 0, mat_mask, disp_hist, 1, &hist_size, &hist_range, uniform, accumulate);
    cout << "histogram size: " << disp_hist.size() << ", bin_w = " << bin_w << endl;
    float * p_disp_hist = (float *)disp_hist.ptr<float>(0);

    Mat hist_img( hist_h, hist_w, CV_8UC1, Scalar(0) );
    for(int i = 1; i < hist_size; ++i) {
        line(hist_img, Point( bin_w*(i-1), hist_h - p_disp_hist[(i-1)] ), Point(bin_w*(i), hist_h - p_disp_hist[(i)] ), Scalar(255), 2, 8, 0 );
    }

    imwrite(m_save_dir + "/" + histname, hist_img);
    cout << "saving " << histname << endl;
}

//check by disparity difference:
void Debugger::fast_check_by_diff(const string diffname, const int diff_thresh) {

    Mat mat_diff_disp(m_h, m_w, CV_8UC1, Scalar(0));
    Mat mat_mask_l = m_stereo_fl->get_mat_mask_l();

    unsigned char * diff = (unsigned char *)mat_diff_disp.ptr<unsigned char>(0);
    unsigned char * pmsk = (unsigned char *)(mat_mask_l.ptr<unsigned char>(0));
    vector<float>&  disp = m_stereo_fl->get_disp();
    int maxval = 0;

    int idx = 0;
    for(int y = 0; y < m_h; ++y) {
        for(int x = 1; x < m_w; ++x) {
            if(255!=pmsk[++idx]) continue;
            if(disp[idx] && disp[idx-1] && (abs(disp[idx] - disp[idx-1]) > diff_thresh)) {
                diff[idx] = abs(disp[idx] - disp[idx-1]);
                maxval = max((int)diff[idx], maxval);
            } //adjacent pixels exceeds 1 disparity

        }
    }

    if(0 != maxval) {
        mat_diff_disp.convertTo(mat_diff_disp, CV_8UC1, 255/maxval);
    }
    imwrite(m_save_dir + "/" + diffname, mat_diff_disp);
    cout << "debug:\tsaving " << diffname << "..\tmaxval = " << maxval << endl;
}

void Debugger::compare_init_final_disp(const int level) {
    string lvlstr = qing_int_2_string(level);
    string init_disp_name = m_save_dir + "/init_disp_l_" + lvlstr + ".jpg";
    string final_disp_name = m_save_dir + "/final_disp_l_" + lvlstr + ".jpg";
    Mat init_disp = imread(init_disp_name, 0);
    Mat final_disp = imread(final_disp_name, 0);
    Mat diff_disp;

    qing_subtract_image(diff_disp, init_disp, final_disp);

    string diff_disp_name = m_save_dir + "/diff_init_final_l_" + lvlstr + ".jpg";
    imwrite(diff_disp_name, diff_disp);
    cout << "debug:\tsaving " << diff_disp_name << endl;
}

void Debugger::save_disparity(const string& prefix, const string& lvl, const string& suffix) {
    string savename;

    Mat uimg(m_h, m_w, CV_8UC1, Scalar(0));
    qing_float_vec_2_uchar_img(m_stereo_fl->get_disp_l(), m_scale, uimg);
    savename = m_save_dir + "/" + prefix + "_l_" + lvl + suffix;
    imwrite(savename, uimg);
//  cout << "debug:\tsaving " << savename << endl;
    qing_float_vec_2_uchar_img(m_stereo_fl->get_disp_r(), m_scale, uimg);
    savename = m_save_dir + "/" + prefix + "_r_" + lvl + suffix;
    imwrite(savename, uimg);
//  cout << "debug:\tsaving " << savename << endl;
}

void Debugger::save_disparity_with_outliers(const string& prefix, const string& lvl, const string& suffix, const char& c) {

    Scalar color(0,0,0);
    switch (c){
        case 'r': color.val[2] = 125; break;
        case 'g': color.val[1] = 125; break;
        case 'b': color.val[0] = 125; break;
        default: break;
    }

    string savename;

    vector<uchar>& mask_l = m_stereo_fl->get_mask_l();
    vector<uchar>& mask_r = m_stereo_fl->get_mask_r();
    vector<uchar>& outliers_l = m_stereo_fl->get_outliers_l();
    vector<uchar>& outliers_r = m_stereo_fl->get_outliers_r();
    vector<float>& disp_l = m_stereo_fl->get_disp_l();
    vector<float>& disp_r = m_stereo_fl->get_disp_r();

    Mat validness = Mat::zeros(m_h, m_w, CV_8UC3);
    uchar * ptr = (uchar *)validness.ptr<uchar>(0);

    for(int i = 0; i < m_total; ++i) {
        if(0==mask_l[i]) continue;
        if(0==outliers_l[i])  {
            ptr[i * 3 + 0] = disp_l[i] * m_scale;
            ptr[i * 3 + 1] = ptr[i * 3 + 0];
            ptr[i * 3 + 2] = ptr[i * 3 + 0];
        }
        if(1==outliers_l[i]) {
            ptr[i * 3 + 0] = color.val[0];
            ptr[i * 3 + 1] = color.val[1];
            ptr[i * 3 + 2] = color.val[2];
        }
    }
    savename = m_save_dir + "/" + prefix + "_l_" + lvl + suffix;
    imwrite(savename, validness);
//    cout << "debug:\tsaving " << savename << endl;

    validness = Mat::zeros(m_h, m_w, CV_8UC3);
    ptr = (uchar *)validness.ptr<uchar>(0);
    for(int i = 0; i < m_total; ++i) {
        if(0==mask_r[i]) continue;
        if(0==outliers_r[i])  {
            ptr[i * 3 + 0] = disp_r[i] * m_scale;
            ptr[i * 3 + 1] = ptr[i * 3 + 0];
            ptr[i * 3 + 2] = ptr[i * 3 + 0];
        }
        if(1==outliers_r[i]) {
            ptr[i * 3 + 0] = color.val[0];
            ptr[i * 3 + 1] = color.val[1];
            ptr[i * 3 + 2] = color.val[2];
        }
    }
    savename = m_save_dir + "/" + prefix + "_r_" + lvl + suffix;
    imwrite(savename, validness);
//    cout << "debug:\tsaving " << savename << endl;

#if 0
    Mat uimg = Mat::zeros(m_h,m_w, CV_8UC1);
    mask_l = m_stereo_fl->get_mask_l();
    qing_vec_2_img(mask_l, uimg);
    imshow("debug-mask-l", uimg);
    mask_r = m_stereo_fl->get_mask_r();
    qing_vec_2_img(mask_r, uimg);
    imshow("debug-mask-r", uimg);
    waitKey(0);
    destroyAllWindows();
#endif
}
