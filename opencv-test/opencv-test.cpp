#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui.hpp>	
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <omp.h>


#define PI 3.14159265358979323846

using namespace cv;
using namespace std;

const int GAUSSIAN_RADIUS = 3;
typedef struct {
    double value[3];
    double weight[3];
}Weights;

int myMin(int a, int b) {
    return a < b ? a : b;
}
int myMax(int a, int b) {
    return a > b ? a : b;
}

void displayResult(Mat image) {
    //Define names of the windows
    String window_name = "Image Blurred with various technique";

    // Create windows with above names
    namedWindow(window_name);

    // Show our images inside the created windows.
    imshow(window_name, image);

    waitKey(0); // Wait for any keystroke in the window
    destroyAllWindows(); // Destroy all opened windows
}

Mat gaussian_blur(IplImage* tmp, double r) {

    IplImage* result = cvCloneImage(tmp);

    int h = tmp->height;
    int w = tmp->width;
    //printf("h=%d, w=%d", h, w);

    double rs = ceil(r * 2.57);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            Weights weights;
            for (int iy = i - rs; iy < i + rs + 1; iy++) {
                for (int ix = j - rs; ix < j + rs + 1; ix++) {
                    int x = myMin(w - 1, myMax(0, ix));
                    int y = myMin(h - 1, myMax(0, iy));
                    double dsq = (ix - j) * (ix - j) + (iy - i) * (iy - i);
                    double wght = exp(-dsq / (2 * r * r)) / (PI * 2 * r * r);

                    CvScalar channels = cvGet2D(tmp, y, x);

                    for (int c = 0; c < 3; c++) {
                        weights.value[c] += channels.val[c] * wght;
                        weights.weight[c] += wght;
                    }
                }
            }

            CvScalar resultingChannels = cvGet2D(result, i, j);
            for (int c = 0; c < 3; c++) {
                resultingChannels.val[c] = round(weights.value[c] / weights.weight[c]);
                weights.value[c] = 0.0;
                weights.weight[c] = 0.0;
            }
            cvSet2D(result, i, j, resultingChannels);
        }
    }
    return cvarrToMat(result);
}

Mat gaussian_blur_parallel(IplImage* tmp, double r) {
    
    IplImage* result = cvCloneImage(tmp);
    
    int h = tmp -> height;
    int w = tmp -> width;
    //printf("h=%d, w=%d", h, w);

    double rs = ceil(r * 2.57);
    omp_set_num_threads(3);
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            Weights weights;
            for (int iy = i - rs; iy < i + rs + 1; iy++) {
                for (int ix = j - rs; ix < j + rs + 1; ix++) {
                    int x = myMin(w - 1, myMax(0, ix));
                    int y = myMin(h - 1, myMax(0, iy));
                    double dsq = (ix - j) * (ix - j) + (iy - i) * (iy - i);
                    double wght = exp(-dsq / (2 * r * r)) / (PI * 2 * r * r);

                    CvScalar channels = cvGet2D(tmp, y, x);

                    for (int c = 0; c < 3; c++) {
                        weights.value[c] += channels.val[c] * wght;
                        weights.weight[c] += wght;
                    }
                }
            }

            CvScalar resultingChannels = cvGet2D(result, i, j);
            for (int c = 0; c < 3; c++) {
                resultingChannels.val[c] = round(weights.value[c] / weights.weight[c]);
                weights.value[c] = 0.0;
                weights.weight[c] = 0.0;
            }
            cvSet2D(result, i, j, resultingChannels);
        }
    }
    return cvarrToMat(result);
}


Mat sequentialProgram(Mat ori) {
    // Image Ori 
    Mat img_ori = ori;

    // Image 1 Bluring
    Mat img1_blurred;
    GaussianBlur(ori, img1_blurred, Size(3, 3), 0);

    // Image 2 Bluring
    Mat img2_blurred;
    GaussianBlur(ori, img2_blurred, Size(7, 7), 0);

    // Image 3 Bluring
    Mat img3_blurred;
    GaussianBlur(ori, img3_blurred, Size(9, 9), 0);

    // Get image size and combine into new image
    Size sz1 = img_ori.size();
    Size sz2 = img1_blurred.size();
    Size sz3 = img2_blurred.size();
    Size sz4 = img3_blurred.size();
    Mat image((sz1.height), (sz1.width + sz2.width + sz3.width), CV_8UC3);
    Mat mostLeft(image, Rect(0, 0, sz1.width, sz1.height)); // x, y, width, height
    /*img_ori.copyTo(mostLeft);
    Mat middle(image, Rect(sz1.width, 0, sz2.width, sz2.height));
    img1_blurred.copyTo(middle);
    Mat botoomright(image, Rect(sz1.width + sz2.width, 0, sz3.width, sz3.height));*/
    img3_blurred.copyTo(mostLeft);

    return image;
}


int main(int argc, char* argv[])
{

    string path = "Resources/Image/ultrasoundbaby2.png";

    // Read the image file
    Mat ori = imread(path);
    Mat result;

    Mat img = imread(path, IMREAD_GRAYSCALE);
    IplImage copy = img;
    IplImage* tmp = &copy;
    //cvError(&tmp, &tmp, 0, 2);


    // Check for failure
    if (ori.empty())
    {
        cout << "Could not open or find the image." << endl;
        return -1;
    }

    // Get image size
    Size sz1 = ori.size();
    Mat image((sz1.height + sz1.height), (sz1.width + sz1.width), CV_8UC3);
    double start_time1 = omp_get_wtime();
    gaussian_blur(tmp, GAUSSIAN_RADIUS);
    double end_time1 = omp_get_wtime();
    double start_time2 = omp_get_wtime();
    result = gaussian_blur_parallel(tmp, GAUSSIAN_RADIUS);
    double end_time2 = omp_get_wtime();
    double start_time3 = omp_get_wtime();
    sequentialProgram(ori);
    double end_time3 = omp_get_wtime();
    
    printf("====================================================\n");
    printf("|   Image Bluring Execution Time Evaluation        |\n");
    printf("====================================================\n");
    printf("| Original Gaussian Blur| %f seconds |\n", end_time1 - start_time1);    
    printf("| OpenMp                | %f seconds |\n", end_time2 - start_time2);
    printf("| OpenCV                |  %f seconds |\n", end_time3 - start_time3);
    printf("====================================================\n");
  
    displayResult(result);
    return 0;
}