#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

PGM_IMG contrast_enhancement_g(PGM_IMG img_in, int *hist, float *cdf_d, int *lut, unsigned char *out)
{
    PGM_IMG result;
    //int hist[256];
    
    //memset(hist, 0, 256);

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    mempcpy(result.img, out, img_in.h*img_in.w*sizeof(unsigned char));
    

    //histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    //histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256, cdf_d, lut);
    return result;
}
