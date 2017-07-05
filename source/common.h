#ifndef COMMON_H
#define COMMON_H

#include "../../Qing/qing_string.h"
#include "../../Qing/qing_dir.h"
#include "../../Qing/qing_image.h"

#define DEBUG 1
#define MIN_IMG_SIZE 160        //图像金字塔分辨率最小值
#define MIN_DISP_VALUE 5        //视差最小值

#define DISP_STEP 1             //步长
#define DISP_OCCLUSION 1        //遮挡
#define DISP_MISMATCH  2        //误匹配
#define DISP_TOLERANCE 0
#define REGION_VOTE_TIMES 3

#define MAX_LEVELS 4

#define K2DISP(maxd,mind,k)		(((mind)*(1-(k)) + (maxd)*(k))/((maxd)- (mind)))    //离散级数到视差值的换算, d(k) = (dmin*(1-k) + dmax*(k)) / (dmax-dmin)
#define DISP2K(maxd,mind,d)		((d) - (mind)/((maxd)-(mind)))                      //视差值换算到离散级数

//enum CCType{zncc = 1, cen, tad} ;           //CG: CEN + GRD; ZC: ZNCC + CEN;
//enum CAType{box = 1, gf, bf} ;              //Guidian, Biletaral, Non-linear, Segment-Tree
//enum PPType{wm = 1, sg, np};                //Weight-Median Filter

#endif // COMMON_H
