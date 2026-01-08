#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>

wchar_t* projectPath;

Mat preprocessImageALPR(const Mat& img)
{
    CV_Assert(!img.empty());

    Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = img.clone();
    }

    //blackhat morphological operation
    Mat rectKern = getStructuringElement(MORPH_RECT, Size(13, 5));
    Mat blackhat;
    morphologyEx(gray, blackhat, MORPH_BLACKHAT, rectKern);

    //find light regions
    Mat squareKern = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat light;
    morphologyEx(gray, light, MORPH_CLOSE, squareKern);
    threshold(light, light, 0, 255, THRESH_BINARY | THRESH_OTSU);

    //sobel gradient
    Mat gradX;
    Sobel(blackhat, gradX, CV_32F, 1, 0, -1);
    gradX = abs(gradX);

    double minVal, maxVal;
    minMaxLoc(gradX, &minVal, &maxVal);

    gradX = 255.0 * (gradX - minVal) / (maxVal - minVal);
    gradX.convertTo(gradX, CV_8U);

    //blur
    GaussianBlur(gradX, gradX, Size(5, 5), 0);
    morphologyEx(gradX, gradX, MORPH_CLOSE, rectKern);

    Mat thresh;
    threshold(gradX, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

    //erode and dilate
    erode(thresh, thresh, Mat(), Point(-1, -1), 2);
    dilate(thresh, thresh, Mat(), Point(-1, -1), 2);

    //AND
    bitwise_and(thresh, thresh, thresh, light);
    dilate(thresh, thresh, Mat(), Point(-1, -1), 2);
    erode(thresh, thresh, Mat(), Point(-1, -1), 1);

    return thresh;
}


Rect findPlate(const Mat& edgeImg)
{
    std::vector<std::vector<Point>> contours;
    findContours(edgeImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    const double targetAspect = 4.7;
    double bestError = 999999.0;
    Rect bestRect;

    for (auto& contour : contours)
    {
        Rect r = boundingRect(contour);
        double area = r.width * r.height;

        if (area < 2000)
            continue;

        double aspect = (double)r.width / r.height;

        double error = fabs(aspect - targetAspect);

        if (error < bestError)
        {
            bestError = error;
            bestRect = r;
        }
    }

    return bestRect;
}

int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

    for (int i = 1; i <= 100; i++)
    {
        char fname[256];
        sprintf(fname, "DataSet/%03d.jpg", i);

        Mat img = cv::imread(fname);
        if (img.empty())
            continue;

        std::cout << "Preprocessing image " << fname << std::endl;

        Mat processed = preprocessImageALPR(img);

        char outName[256];
        sprintf(outName, "ProcessedImages/%03d_processed.jpg", i);
        imwrite(outName, processed);

        std::cout << "Identifying plate " << outName << std::endl;

        Rect plateRect = findPlate(processed);
        char outNamePlate[256];
        sprintf(outNamePlate, "Plates/%03d_plate.jpg", i);

        if (plateRect.area() > 0)
        {
            Mat plate = img(plateRect);
            imwrite(outNamePlate, plate);
        }
    }

	return 0;
}