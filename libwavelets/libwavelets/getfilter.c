#include <stdio.h>
#include "bio.h"

void getfilter(p_bio_xd, p_bio_skip_xd,
               p_bio_xd_char, p_bio_skip_xd_char, Filterlength, flag)

    void (**p_bio_xd)();
void (**p_bio_skip_xd)();
void (**p_bio_xd_char)();
void (**p_bio_skip_xd_char)();
int flag;
int Filterlength;
{

    if (flag == 3) {/* 2d  */
        switch (Filterlength) {
        case 1:
            *p_bio_xd = bioD1_2d;
            *p_bio_skip_xd = bioD1_2d;
            *p_bio_xd_char = bioD1_2d;
            *p_bio_skip_xd_char = bioD1_2d;
            break;
        case 3:
            *p_bio_xd = bioD3_2d;
            *p_bio_skip_xd = bioD3_2d;
            *p_bio_xd_char = bioD3_2d;
            *p_bio_skip_xd_char = bioD3_2d;
            break;
        case 5:
            *p_bio_xd = bioD5_2d;
            *p_bio_skip_xd = bioD5_2d;
            *p_bio_xd_char = bioD5_2d;
            *p_bio_skip_xd_char = bioD5_2d;
            break;
        case 7:
            *p_bio_xd = bioD7_2d;
            *p_bio_skip_xd = bioD7_2d;
            *p_bio_xd_char = bioD7_2d;
            *p_bio_skip_xd_char = bioD7_2d;
            break;
        case 9:
            *p_bio_xd = bioD9_2d;
            *p_bio_skip_xd = bioD9_2d;
            *p_bio_xd_char = bioD9_2d;
            *p_bio_skip_xd_char = bioD9_2d;
            break;
        case -1:
            *p_bio_xd = bioR1_2d;
            *p_bio_skip_xd = bioR1_2d;
            *p_bio_xd_char = bioR1_2d;
            *p_bio_skip_xd_char = bioR1_2d;
            break;
        case -3:
            *p_bio_xd = bioR3_2d;
            *p_bio_skip_xd = bioR3_2d;
            *p_bio_xd_char = bioR3_2d;
            *p_bio_skip_xd_char = bioR3_2d;
            break;
        case -5:
            *p_bio_xd = bioR5_2d;
            *p_bio_skip_xd = bioR5_2d;
            *p_bio_xd_char = bioR5_2d;
            *p_bio_skip_xd_char = bioR5_2d;
            break;
        case -7:
            *p_bio_xd = bioR7_2d;
            *p_bio_skip_xd = bioR7_2d;
            *p_bio_xd_char = bioR7_2d;
            *p_bio_skip_xd_char = bioR7_2d;
            break;
        case -9:
            *p_bio_xd = bioR9_2d;
            *p_bio_skip_xd = bioR9_2d;
            *p_bio_xd_char = bioR9_2d;
            *p_bio_skip_xd_char = bioR9_2d;
            break;
        default:
            printf("Error: Non-valid filterlength\n");
            break;
        }
    }

    if (flag == 2) {/* 2dy  */
        switch (Filterlength) {
        case 1:
            *p_bio_xd = bioD1_2d_y;
            *p_bio_skip_xd = bioD1_2d_y;
            *p_bio_xd_char = bioD1_2d_y;
            *p_bio_skip_xd_char = bioD1_2d_y;
            break;
        case 3:
            *p_bio_xd = bioD3_2d_y;
            *p_bio_skip_xd = bioD3_2d_y;
            *p_bio_xd_char = bioD3_2d_y;
            *p_bio_skip_xd_char = bioD3_2d_y;
            break;
        case 5:
            *p_bio_xd = bioD5_2d_y;
            *p_bio_skip_xd = bioD5_2d_y;
            *p_bio_xd_char = bioD5_2d_y;
            *p_bio_skip_xd_char = bioD5_2d_y;
            break;
        case 7:
            *p_bio_xd = bioD7_2d_y;
            *p_bio_skip_xd = bioD7_2d_y;
            *p_bio_xd_char = bioD7_2d_y;
            *p_bio_skip_xd_char = bioD7_2d_y;
            break;
        case 9:
            *p_bio_xd = bioD9_2d_y;
            *p_bio_skip_xd = bioD9_2d_y;
            *p_bio_xd_char = bioD9_2d_y;
            *p_bio_skip_xd_char = bioD9_2d_y;
            break;
        case -1:
            *p_bio_xd = bioR1_2d_y;
            *p_bio_skip_xd = bioR1_2d_y;
            *p_bio_xd_char = bioR1_2d_y;
            *p_bio_skip_xd_char = bioR1_2d_y;
            break;
        case -3:
            *p_bio_xd = bioR3_2d_y;
            *p_bio_skip_xd = bioR3_2d_y;
            *p_bio_xd_char = bioR3_2d_y;
            *p_bio_skip_xd_char = bioR3_2d_y;
            break;
        case -5:
            *p_bio_xd = bioR5_2d_y;
            *p_bio_skip_xd = bioR5_2d_y;
            *p_bio_xd_char = bioR5_2d_y;
            *p_bio_skip_xd_char = bioR5_2d_y;
            break;
        case -7:
            *p_bio_xd = bioR7_2d_y;
            *p_bio_skip_xd = bioR7_2d_y;
            *p_bio_xd_char = bioR7_2d_y;
            *p_bio_skip_xd_char = bioR7_2d_y;
            break;
        case -9:
            *p_bio_xd = bioR9_2d_y;
            *p_bio_skip_xd = bioR9_2d_y;
            *p_bio_xd_char = bioR9_2d_y;
            *p_bio_skip_xd_char = bioR9_2d_y;
            break;
        default:
            printf("Error: Non-valid filterlength\n");
            break;
        }
    }

    if (flag == 7) /*3d */
        {

        switch (Filterlength) {
        case 1:
            *p_bio_xd = bioD1_3d;
            *p_bio_skip_xd = bioD1_3d;
            *p_bio_xd_char = bioD1_3d;
            *p_bio_skip_xd_char = bioD1_3d;
            break;
        case 3:
            *p_bio_xd = bioD3_3d;
            *p_bio_skip_xd = bioD3_3d;
            *p_bio_xd_char = bioD3_3d;
            *p_bio_skip_xd_char = bioD3_3d;
            break;
        case 5:
            *p_bio_xd = bioD5_3d;
            *p_bio_skip_xd = bioD5_3d;
            *p_bio_xd_char = bioD5_3d;
            *p_bio_skip_xd_char = bioD5_3d;
            break;
        case 7:
            *p_bio_xd = bioD7_3d;
            *p_bio_skip_xd = bioD7_3d;
            *p_bio_xd_char = bioD7_3d;
            *p_bio_skip_xd_char = bioD7_3d;
            break;
        case 9:
            *p_bio_xd = bioD9_3d;
            *p_bio_skip_xd = bioD9_3d;
            *p_bio_xd_char = bioD9_3d;
            *p_bio_skip_xd_char = bioD9_3d;
            break;
        case -1:
            *p_bio_xd = bioR1_3d;
            *p_bio_skip_xd = bioR1_3d;
            *p_bio_xd_char = bioR1_3d;
            *p_bio_skip_xd_char = bioR1_3d;
            break;
        case -3:
            *p_bio_xd = bioR3_3d;
            *p_bio_skip_xd = bioR3_3d;
            *p_bio_xd_char = bioR3_3d;
            *p_bio_skip_xd_char = bioR3_3d;
            break;
        case -5:
            *p_bio_xd = bioR5_3d;
            *p_bio_skip_xd = bioR5_3d;
            *p_bio_xd_char = bioR5_3d;
            *p_bio_skip_xd_char = bioR5_3d;
            break;
        case -7:
            *p_bio_xd = bioR7_3d;
            *p_bio_skip_xd = bioR7_3d;
            *p_bio_xd_char = bioR7_3d;
            *p_bio_skip_xd_char = bioR7_3d;
            break;
        case -9:
            *p_bio_xd = bioR9_3d;
            *p_bio_skip_xd = bioR9_3d;
            *p_bio_xd_char = bioR9_3d;
            *p_bio_skip_xd_char = bioR9_3d;
            break;
        default:
            printf("Error: Non-valid filterlength\n");
            break;
        }
    }
}
