#include "scanner.h"

int main(int argc, char * argv[])
{
//    string img_folder  = "../../../HumanDatas_20161224/Humans_frame/";
//    string msk_folder  = "../../../HumanDatas_20161224/Humans_mask/" ;
//    string stereo_folder = "../../../HumanDatas_20161224/Humans_stereo/";

    string img_folder  = "/Users/Qing/Data/20161224/Human_frame/";
    string msk_folder  = "/Users/Qing/Data/20161224/Human_mask/" ;
    string stereo_folder = "/Users/Qing/Data/20161224/Human_stereo/";

    cout << "usage: " << argv[0] << " FRM_0176 stereo_A01A02.info" << endl;           //stereo

    if(argc != 3)
    {
        cerr << "invalid arguments.." << endl;
        return -1;
    }
    string frame_name = argv[1];
    string stereo_fn  = argv[2];

    HumanBodyScanner * scanner = new HumanBodyScanner(img_folder, msk_folder, stereo_folder, frame_name, stereo_fn);
    if(scanner->init())
    {
        double duration = (double)(cv::getTickCount());

        scanner->match();
        scanner->triangulate();

        duration = (double)(cv::getTickCount()) - duration;
        duration /= cv::getTickFrequency();                       //the elapsed time in sec
        printf( "\n--------------------------------------------------------\n" );
        printf( "Total Time: %.2lf s\n", duration );
        printf( "--------------------------------------------------------\n" );
    }
    return 1;
}
