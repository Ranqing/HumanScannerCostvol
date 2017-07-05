#include "scanner.h"
#include "stereo_flow.h"
#include "debug.h"

#include "../../Qing/qing_ply.h"
#include "../../Qing/qing_timer.h"

bool HumanBodyScanner::init()
{
    if( false == qing_check_file_suffix(m_stereo_file, ".info")) {
        cerr << "invalid stereo info file format..." << endl;
        return false;
    }

    qing_cwd();

    fstream fin(m_stereo_file.c_str(), ios::in);
    if( false == fin.is_open() ) {
        cerr << "failed to open " << m_stereo_file << endl;
        return false;
    }

    fin >> m_stereo_name;
    fin >> m_stereo_idx;
    fin >> m_camL >> m_camR;
    fin >> m_frame_name;
    fin >> m_img_nameL >> m_img_nameR;
    fin >> m_msk_nameL >> m_msk_nameR;
    fin >> m_crop_pointL.x >> m_crop_pointL.y;
    fin >> m_crop_pointR.x >> m_crop_pointR.y;
    fin >> m_size.width >> m_size.height;
    fin >> m_max_disp;
    fin >> m_min_disp;

    cout << m_max_disp << '\t' << '~' << '\t' << m_min_disp << endl;

    m_qmatrix = Mat::zeros(4, 4, CV_64FC1);
    double * ptr = (double *)m_qmatrix.ptr<double>(0);
    for(int i = 0; i < 16; ++i) {
        fin >> ptr[i];
    }

    m_out_dir = "./" + m_frame_name;
    qing_create_dir(m_out_dir);
    m_out_dir = m_out_dir + "/" + m_stereo_name;
    qing_create_dir(m_out_dir);

#if DEBUG
    qing_cwd();
    cout << "out dir: "     << m_out_dir << endl;
    cout << "crop pointL: " << m_crop_pointL << endl;
    cout << "crop pointR: " << m_crop_pointR << endl;
    cout << "image size: "  << m_size << endl;
    cout << "disp range:"   << m_min_disp << " ~ " << m_max_disp << endl;
    cout << "qmatrix: "     << m_qmatrix  << endl;
    cout << m_img_nameL     << endl;
    cout << m_img_nameR     << endl;
    cout << m_msk_nameL     << endl;
    cout << m_msk_nameR     << endl;

    m_debugger = new Debugger(m_out_dir);
#endif

    fin.close();

    //load and crop images
    m_img_nameL = m_img_folder +  m_img_nameL;
    m_img_nameR = m_img_folder +  m_img_nameR;
    m_msk_nameL = m_msk_folder +  m_msk_nameL;
    m_msk_nameR = m_msk_folder +  m_msk_nameR;

    load_and_crop_images();
    cout << "scanner initialization done..." << endl;
    return true;
}

void HumanBodyScanner::load_and_crop_images()
{
    Mat full_imgL, full_imgR;
    Mat full_mskL, full_mskR;

    //load images
    if(false == qing_load_image(m_img_nameL, CV_LOAD_IMAGE_UNCHANGED, full_imgL)) {
        cerr << "failed to open " << m_img_nameL << endl;
        exit(-1);
    }
    if(false == qing_load_image(m_img_nameR, CV_LOAD_IMAGE_UNCHANGED, full_imgR)) {
        cerr << "failed to open " << m_img_nameR << endl;
        exit(-1);
    }

    //load masks
    if(false == qing_load_image(m_msk_nameL, CV_LOAD_IMAGE_GRAYSCALE, full_mskL) ||
            false == qing_load_image(m_msk_nameR, CV_LOAD_IMAGE_GRAYSCALE, full_mskR)) {
        cout << "no masks here..." << endl;
        m_mskL = Mat(m_size, CV_8UC1, cv::Scalar(255));
        m_mskR = Mat(m_size, CV_8UC1, cv::Scalar(255));
    }
    else {
        threshold(full_mskL, full_mskL, 128, 255, THRESH_BINARY);
        threshold(full_mskR, full_mskR, 128, 255, THRESH_BINARY);
        m_mskL = Mat::zeros(m_size, CV_8UC1);
        m_mskR = Mat::zeros(m_size, CV_8UC1);
        m_mskL = full_mskL(Rect(m_crop_pointL, m_size)).clone();
        m_mskR = full_mskR(Rect(m_crop_pointR, m_size)).clone();
    }

    //crop images. saving pixels with masks
# if WITH_MASK
    full_imgL(Rect(m_crop_pointL, m_size)).copyTo(m_imgL, m_mskL);
    full_imgR(Rect(m_crop_pointR, m_size)).copyTo(m_imgR, m_mskR);
# else
    full_imgL(Rect(m_crop_pointL, m_size)).copyTo(m_imgL);
    full_imgR(Rect(m_crop_pointR, m_size)).copyTo(m_imgR);
# endif
}

void HumanBodyScanner::build_stereo_pyramid()
{
    cout << "\nstereo pyramid building start." << endl;

    m_max_levels = MAX_LEVELS;
    m_stereo_pyramid = new StereoFlow * [m_max_levels];

    Mat t_imgL = m_imgL.clone();
    Mat t_imgR = m_imgR.clone();
    Mat t_mskL = m_mskL.clone();
    Mat t_mskR = m_mskR.clone();

    int t_max_disp = m_max_disp;
    int t_min_disp = m_min_disp;
    int t_wnd_sz;
    float t_disp_scale = (m_max_disp <= 255) ? (255/m_max_disp) : min((255.f/m_max_disp), 0.5f);

    for(int p = 0; p < m_max_levels; ++p) {
        int w = t_imgL.size().width;
        int h = t_imgR.size().height;

        if(max(w,h) > 1000) t_wnd_sz = QING_WND_SIZE + 4;
        else t_wnd_sz = QING_WND_SIZE;

        if( w < MIN_IMG_SIZE || h < MIN_IMG_SIZE || t_max_disp < MIN_DISP_VALUE) {
            m_max_levels = p;
            break;
        }

        cout << "\tPyramid " << p << ": "<< t_min_disp << " ~ " << t_max_disp << ", scale = " << t_disp_scale << ", wnd = " << t_wnd_sz << '\t';
        cout << "initialization of stereo flow..." << endl;
        if(0>p) {
            cout << "\tdon't consider the original image." << endl;
            m_stereo_pyramid[p] = NULL;
        }
        else{
            m_stereo_pyramid[p] = new StereoFlow(t_imgL, t_imgR, t_mskL, t_mskR, t_max_disp, t_min_disp, t_disp_scale);
            m_stereo_pyramid[p]->set_wnd_size(t_wnd_sz);
            m_stereo_pyramid[p]->calc_mean_images();
        }

        //down-sample
        t_max_disp = t_max_disp * 0.5;
        t_min_disp = t_min_disp * 0.5;
        t_disp_scale = min(t_disp_scale * 2, 1.f);

#if DEBUG
        string savefn;
        string lvlstr = int2string(p);

        savefn = m_out_dir + "/crop_imgL_" + lvlstr + ".jpg";            imwrite(savefn, t_imgL);
        savefn = m_out_dir + "/crop_imgR_" + lvlstr + ".jpg";            imwrite(savefn, t_imgR);
        savefn = m_out_dir + "/crop_mskL_" + lvlstr + ".jpg";            qing_save_image(savefn, t_mskL);
        savefn = m_out_dir + "/crop_mskR_" + lvlstr + ".jpg";            qing_save_image(savefn, t_mskR);
        savefn = m_out_dir + "/mean_imgL_" + lvlstr + ".jpg";            qing_save_image(savefn, m_stereo_pyramid[p]->get_mat_mean_l(), 255);
        savefn = m_out_dir + "/mean_imgR_" + lvlstr + ".jpg";            qing_save_image(savefn, m_stereo_pyramid[p]->get_mat_mean_r(), 255);
#endif
        //guassian filter then choose half cols
        pyrDown(t_imgL, t_imgL);
        pyrDown(t_imgR, t_imgR);
        pyrDown(t_mskL, t_mskL);        qing_threshold_msk(t_mskL, t_mskL, 128, 255);
        pyrDown(t_mskR, t_mskR);        qing_threshold_msk(t_mskR, t_mskR, 128, 255);
    }

    cout << "stereo pyramid building done. max levels = " << m_max_levels << endl;
}

void HumanBodyScanner::build_stereo_costvol(const int end) {
    cout << "\ncost volume pyramid building start." << endl;
    double duration ;
    for(int p = m_max_levels - 1; p >= end; --p) {
        duration = getTickCount();
        printf("\tlevel = %d\t", p);
        m_stereo_pyramid[p]->matching_cost();
        printf( "\tmatching cost volume computation: %.2lf s\n", ((double)(getTickCount())-duration)/getTickFrequency() );   // the elapsed time in sec

    }
    cout << "cost volume pyramid building finished." << endl;
}

void HumanBodyScanner::match() {
    QingTimer timer;

    int end = 1, times, cnt;
    string lvlstr, savename;

    build_stereo_pyramid();
    build_stereo_costvol(end);

    for(int lvl = m_max_levels - 1; lvl >= end; --lvl) {

        cout << "\nlevel " << lvl << " disparity calculation start:\n";
        lvlstr = qing_int_2_string(lvl);

        m_stereo_pyramid[lvl]->calc_support_region();     //cross-based support region, initialization
        if( m_max_levels - 1 == lvl)
            m_stereo_pyramid[lvl]->calc_init_disparity_from_cost() ;
        else
            m_stereo_pyramid[lvl]->calc_init_disparity_from_cost(m_stereo_pyramid[lvl+1]->get_bestk_l(), m_stereo_pyramid[lvl+1]->get_bestk_r(),
                                                                 m_stereo_pyramid[lvl+1]->get_w(), m_stereo_pyramid[lvl+1]->get_h());

        m_debugger->set_data_source(m_stereo_pyramid[lvl]);
        m_debugger->save_disparity("mcost_aggr_disp", lvlstr, ".jpg");

        m_stereo_pyramid[lvl]->local_smoothness_check();   m_debugger->save_disparity_with_outliers("sc_mcost_aggr_disp", lvlstr, ".jpg", 'r');
        m_stereo_pyramid[lvl]->ordering_check();           m_debugger->save_disparity_with_outliers("oc_sc_mcost_aggr_disp", lvlstr, ".jpg", 'g');
        m_stereo_pyramid[lvl]->cross_check();              m_debugger->save_disparity_with_outliers("cc_oc_sc_mcost_aggr_disp", lvlstr, ".jpg", 'b');

        m_stereo_pyramid[lvl]->rematch_using_rbf();
        m_debugger->save_disparity("rematch_mcost_aggr_disp", lvlstr, ".jpg");


        m_stereo_pyramid[lvl]->median_filter();
        m_debugger->save_disparity("median_rematch_mcost_aggr_disp", lvlstr, ".jpg");
        cout << "level " << lvl << " disparity calculation end.\n";
        continue;

#if DEBUG
        if(lvl == end) {
            m_debugger->set_triangulate_info((1.0f/(1<<lvl)), m_crop_pointL, m_crop_pointR, m_qmatrix);
            m_debugger->fast_check_disp_by_depth("qing_check_depth_" + qing_int_2_string(lvl) + ".ply", &(m_stereo_pyramid[lvl]->get_disp_l()).front());
        }
#endif
    }

    //subpixel enhancement
    savename = "qing_check_depth_sub_" + lvlstr + ".ply";
    m_stereo_pyramid[end]->subpixel_enhancement();      m_debugger->fast_check_disp_by_depth(savename, &(m_stereo_pyramid[end]->get_disp_l()).front());
//    savename = "qing_check_depth_sub_approx_" + lvlstr + ".ply";
//    m_stereo_pyramid[end]->subpixel_approximation();    m_debugger->fast_check_disp_by_depth(savename, &(m_stereo_pyramid[end]->get_disp_l()).front());

# if 0
    //subpixel refinement using beeler formula
    cnt = 0; times = 30;
    m_stereo_pyramid[end]->trans_costvol_for_beeler();  cout << endl << "applying beeler's subpixel refinement..." << endl;
    float dstep = 0.5f, thresh = 2.0f;
    while((cnt++)<times) {
        if(cnt >= 30) {
            dstep *= 0.5f; thresh = 1.0f;
        }
        cout << cnt << "\titerations: dstep = " << dstep << "\tthresh = " << thresh << "\t" ;
        m_stereo_pyramid[end]->beeler_subpixel_refinement(dstep, thresh * 1.0);
        savename = "qing_check_depth_beeler_" + qing_int_2_string(cnt) + "th.ply";
        m_debugger->fast_check_disp_by_depth(savename, &(m_stereo_pyramid[end]->get_disp_l()).front());
    }
    cout << "beeler's subpixel refinement done.." << endl;
# endif

    copy_disp_from_stereo();
}

void HumanBodyScanner::copy_disp_from_stereo() {
    cout << "\n\tcopy disparity from stereo pyramid..." << endl;

    vector<float>& disp_vec = m_stereo_pyramid[0]->get_disp_l();
    m_disp  = Mat::zeros(m_size, CV_32FC1);
    qing_vec_2_img<float>(disp_vec, m_disp);

# if 0
    Mat disp_img(m_size, CV_8UC1), small_disp;
    float scale = m_stereo_pyramid[0]->get_scale();
    m_disp.convertTo(disp_img, CV_8UC1, scale);

    Size dsize = Size(0.25 * m_size.width, 0.25 * m_size.height);
    resize(disp_img, small_disp, dsize);
    imshow("disp", small_disp);
    waitKey(0);
    destroyAllWindows();
#endif
}

void HumanBodyScanner::disp_2_depth(const Mat &dsp, const Mat &msk, const Mat &img, vector<Vec3f> &points, vector<Vec3f> &colors) {
    float * ptr_dsp = (float *)dsp.ptr<float>(0);
    uchar * ptr_msk = (uchar *)msk.ptr<uchar>(0);
    uchar * ptr_rgb = (uchar *)img.ptr<uchar>(0);

    double * qmtx = (double *)m_qmatrix.ptr<double>(0);

    int w = m_size.width;
    int h = m_size.height;

    points.reserve(w*h);
    colors.reserve(w*h);

    for(int y = 0; y < h; ++y) {
        for(int x = 0; x < w; ++x) {
            int index = y * w + x;
            if( 0 == ptr_msk[index] || 0 == ptr_dsp[index] ) continue;
            else {
                double uvd1[4], xyzw[4] ;
                uvd1[0] = x + m_crop_pointL.x;
                uvd1[1] = y + m_crop_pointL.y;
                uvd1[2] = ptr_dsp[index];
                uvd1[3] = 1.0;

                xyzw[0] = qmtx[ 0] * uvd1[0] + qmtx[ 1] * uvd1[1] + qmtx[ 2] * uvd1[2] + qmtx[ 3] * uvd1[3];
                xyzw[1] = qmtx[ 4] * uvd1[0] + qmtx[ 5] * uvd1[1] + qmtx[ 6] * uvd1[2] + qmtx[ 7] * uvd1[3];
                xyzw[2] = qmtx[ 8] * uvd1[0] + qmtx[ 9] * uvd1[1] + qmtx[10] * uvd1[2] + qmtx[11] * uvd1[3];
                xyzw[3] = qmtx[12] * uvd1[0] + qmtx[13] * uvd1[1] + qmtx[14] * uvd1[2] + qmtx[15] * uvd1[3];

                points.push_back( Vec3f(xyzw[0]/xyzw[3], xyzw[1]/xyzw[3], xyzw[2]/xyzw[3]) );
                colors.push_back( Vec3f(ptr_rgb[3*index + 0], ptr_rgb[3*index + 1], ptr_rgb[3*index + 2]) );
# if 0
                cout << uvd1[0] << ' ' << uvd1[1] << ' ' << uvd1[2] << ' ' << uvd1[3] << endl;
                cout << m_qmatrix << endl;
                cout << xyzw[0] << ' ' << xyzw[1] << ' ' << xyzw[2] << ' ' << xyzw[3] << endl;
                cout << points[points.size() - 1] << "\tgetchar(): " << getchar() << endl;
# endif
            }
        }
    }
}

void HumanBodyScanner::triangulate() {
    cout << "\n triangulate 3d points from disparity results..." << endl;

    int w = m_size.width;
    int h = m_size.height;

    Mat erode_msk  = qing_erode_image(m_mskL, 20);

    float * ptr_disp = (float *)m_disp.ptr<float>(0);
    uchar * ptr_msk  = (uchar *)erode_msk.ptr<uchar>(0);

    //preparing disparity
    float offset = m_crop_pointL.x - m_crop_pointR.x;
    for(int y = 0, index = 0; y < h; ++y) {
        for(int x = 0; x < w; ++x) {
            if( 255 == ptr_msk[index] && 0 != (int)ptr_disp[index] )  ptr_disp[index] += offset;
            else  ptr_disp[index] = 0.f;
            index ++;
        }
    }
    vector<Vec3f> points(0);
    vector<Vec3f> colors(0);

    disp_2_depth(m_disp, erode_msk, m_imgL, points, colors);
    string savefn = m_out_dir + "/" + m_frame_name + "_pointcloud_" + m_stereo_name + ".ply";
    qing_write_point_color_ply(savefn, points, colors);
    cout << "\nsave " << savefn << " done. " << points.size() << " Points." << endl;
}
