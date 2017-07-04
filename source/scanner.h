#ifndef SCANNER_H
#define SCANNER_H

#include "common.h"

class StereoFlow;
class Debugger;                                 //save debugging infos

class HumanBodyScanner
{
public:
    HumanBodyScanner();
    HumanBodyScanner(const string& img_folder, const string& msk_folder, const string& info_folder, const string& frame_name, const string stereofn);
    ~HumanBodyScanner();

    bool init();                                //initialization
    void load_and_crop_images();
    void build_stereo_pyramid();
    void build_stereo_costvol(const int end = 0);

    void match();                               //stereo matching
    void copy_disp_from_stereo();
    void disp_2_depth(const Mat& dsp, const Mat& msk, const Mat& img, vector<Vec3f>& points, vector<Vec3f>& colors);
    void triangulate();                         //disparity to depth then to 3D points


private:
    string m_img_folder;                        //image folder
    string m_msk_folder;                        //mask folder
    string m_info_folder;                       //stereo info folder
    string m_frame_name;                        //frame name
    string m_stereo_file;                       //stereo info file
    string m_stereo_name;                       //stereo name
    string m_camL, m_camR;                      //camera names
    string m_img_nameL, m_img_nameR;            //image names
    string m_msk_nameL, m_msk_nameR;            //mask names

    int m_stereo_idx;                           //stereo idx
    int m_max_levels;                           //layers num
    int m_max_disp, m_min_disp;                 //disp range

    string m_out_dir;                           //out directory

    Point2i m_crop_pointL, m_crop_pointR;       //crop points
    Mat m_imgL, m_imgR;                         //rgb   0 ~ 255
    Mat m_mskL, m_mskR;                         //uchar 0 ~ 255
    Mat m_dispL, m_dispR, m_disp;               //disparity result
    Size m_size;                                //image size
    Mat m_qmatrix;                              //qmatrix to triangulate points

    StereoFlow ** m_stereo_pyramid;             //layered stereo matcher

#if DEBUG
    Debugger * m_debugger;
#endif
};

inline HumanBodyScanner::HumanBodyScanner() {

}

inline HumanBodyScanner::HumanBodyScanner(const string &img_folder, const string &msk_folder, const string &info_folder, const string &frame_name, const string stereofn):
    m_img_folder(img_folder), m_msk_folder(msk_folder), m_info_folder(info_folder), m_frame_name(frame_name)
{
    m_stereo_file = info_folder + frame_name + "/" + stereofn;
    m_img_folder = m_img_folder + frame_name + "/";
    m_msk_folder = m_msk_folder + frame_name + "/";

    cout << "image folder : " << m_img_folder  << endl;
    cout << "mask folder : "  << m_msk_folder  << endl;
    cout << "info folder : "  << m_info_folder << endl;
    cout << "frame name : "   << m_frame_name  << endl;
    cout << "stereo file : "  << m_stereo_file << endl;

}

inline HumanBodyScanner::~HumanBodyScanner() {

}

#endif // SCANNER_H
