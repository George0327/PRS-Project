#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

using namespace cv::dnn;

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


std::vector<Rect> findPlateCandidates(const Mat& edgeImg)
{
    std::vector<std::vector<Point>> contours;
    findContours(edgeImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    std::vector<Rect> candidates;

    for (auto& contour : contours)
    {
        Rect r = boundingRect(contour);

        double area = r.area();
        double aspect = (double)r.width / r.height;

        //std::cout << "aspect: " << aspect << std::endl;
        //std::cout << "area: " << area << std::endl;

        if (area > 2000 &&
            aspect > 3 && aspect < 7)
        {
            candidates.push_back(r);
        }
    }

    return candidates;
}

bool hasEnoughEdges(const Mat& plate)
{
    Mat gray, edges;
    cvtColor(plate, gray, COLOR_BGR2GRAY);
    Canny(gray, edges, 100, 200);

    double edgeRatio =
        (double)countNonZero(edges) / (edges.rows * edges.cols);

    return edgeRatio > 0.05;
}

std::vector<Mat> segmentCharacters(const Mat& plate)
{
    Mat gray;

    // 1. Convert to grayscale
    if (plate.channels() == 3) {
        cvtColor(plate, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = plate.clone();
    }

    // 2. Improve contrast slightly
    Mat blurred;
    GaussianBlur(gray, blurred, Size(3, 3), 0);

    // 3. Binarize for contour detection only
    Mat binary;
    threshold(blurred, binary, 0, 255,
        THRESH_BINARY_INV | THRESH_OTSU);

    // 4. Remove small noise
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_OPEN, kernel);

    // 5. Find character contours
    std::vector<std::vector<Point>> contours;
    findContours(binary.clone(), contours,
        RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    std::vector<Rect> charRects;

    for (auto& c : contours)
    {
        Rect r = boundingRect(c);

        // character size constraints (relative to plate)
        if (r.height > plate.rows * 0.4 &&
            r.height < plate.rows * 0.95 &&
            r.width  > plate.cols * 0.02 &&
            r.width < plate.cols * 0.25)  // Increased for wider chars like W, M
        {
            charRects.push_back(r);
        }
    }

    // 6. Sort left-to-right
    std::sort(charRects.begin(), charRects.end(),
        [](const Rect& a, const Rect& b)
        {
            return a.x < b.x;
        });

    // 7. Extract characters
    std::vector<Mat> characters;
    for (auto& r : charRects)
    {
        Mat chGray = gray(r).clone();
        resize(chGray, chGray, Size(32, 32));
        
        Mat chBW;
        threshold(chGray, chBW, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
        
        // small open to clean noise
        Mat k = getStructuringElement(MORPH_RECT, Size(2, 2));
        morphologyEx(chBW, chBW, MORPH_OPEN, k);

        characters.push_back(chBW);
    }

    return characters;
}

// Extract 30 features
std::vector<float> extractCharacterFeatures(const Mat& charImg)
{
    Mat gray;
    if (charImg.channels() == 3) {
        cvtColor(charImg, gray, COLOR_BGR2GRAY);
    } else {
        gray = charImg.clone();
    }

    // Ensure standard size
    resize(gray, gray, Size(32, 32));

    // Binary inverted, OTSU
    Mat binary;
    threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    std::vector<float> features;
    features.reserve(30);

    // 1) Hu moments (7) 
    Moments m = moments(binary, true);
    double hu[7];
    HuMoments(m, hu);
    for (int i = 0; i < 7; ++i) {
        if (hu[i] != 0.0) {
            float v = static_cast<float>(-std::copysign(1.0, hu[i]) * std::log10(std::abs(hu[i]) + 1e-10));
            features.push_back(v);
        } else {
            features.push_back(0.0f);
        }
    }

    // 2) Quadrant densities (4)
    int midX = binary.cols / 2;
    int midY = binary.rows / 2;
    float quarterPixels = (binary.rows * binary.cols) / 4.0f;
    features.push_back(static_cast<float>(countNonZero(binary(Rect(0, 0, midX, midY))) / quarterPixels));
    features.push_back(static_cast<float>(countNonZero(binary(Rect(midX, 0, binary.cols - midX, midY))) / quarterPixels));
    features.push_back(static_cast<float>(countNonZero(binary(Rect(0, midY, midX, binary.rows - midY))) / quarterPixels));
    features.push_back(static_cast<float>(countNonZero(binary(Rect(midX, midY, binary.cols - midX, binary.rows - midY))) / quarterPixels));

    // 3) Horizontal projections (8)
    for (int i = 0; i < 8; ++i) {
        int startRow = i * binary.rows / 8;
        int endRow = (i + 1) * binary.rows / 8;
        Mat roi = binary.rowRange(startRow, endRow);
        features.push_back(static_cast<float>(countNonZero(roi)) / static_cast<float>(roi.total()));
    }

    // 4) Vertical projections (8)
    for (int i = 0; i < 8; ++i) {
        int startCol = i * binary.cols / 8;
        int endCol = (i + 1) * binary.cols / 8;
        Mat roi = binary.colRange(startCol, endCol);
        features.push_back(static_cast<float>(countNonZero(roi)) / static_cast<float>(roi.total()));
    }

    // 5) Bounding box features (2)
    std::vector<Point> pts;
    findNonZero(binary, pts);
    if (!pts.empty()) {
        Rect bbox = boundingRect(pts);
        features.push_back(static_cast<float>(bbox.width) / static_cast<float>(bbox.height + 1));
        features.push_back(static_cast<float>(bbox.area()) / static_cast<float>(binary.total()));
    } else {
        features.push_back(1.0f);
        features.push_back(0.0f);
    }

    // 6) Contour count (1)
    std::vector<std::vector<Point>> contours;
    findContours(binary.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    features.push_back(static_cast<float>(contours.size()) / 10.0f);

    return features;
}

// Load classifier and label mapping
static cv::dnn::Net classifier;
static std::vector<char> charLabels;

void loadClassifierAndLabels()
{
    classifier = readNetFromONNX("mlp_classifier.onnx");
    if (classifier.empty()) {
        std::cout << "Warning: mlp_classifier.onnx not found" << std::endl;
    }

    // load label map
    charLabels.clear();
    std::ifstream in("label_mapping.txt");
    if (in.is_open()) {
        int idx;
        char ch;
        while (in >> idx >> ch) {
            if ((int)charLabels.size() <= idx) {
                charLabels.resize(idx + 1, '?');
            }
            charLabels[idx] = ch;
        }
        in.close();
    }

    if (charLabels.empty()) {
        // Default charset (no I or O)
        std::string chars = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ";
        charLabels.assign(chars.begin(), chars.end());
    }
}

// Recognize a single character
char recognizeChar(const Mat& charImg)
{
    if (classifier.empty() || charLabels.empty()) {
        return '?';
    }

    std::vector<float> feats = extractCharacterFeatures(charImg);
    if (feats.size() != 30) {
        return '?';
    }

    Mat input(1, 30, CV_32F, feats.data());
    classifier.setInput(input);
    Mat output = classifier.forward();

    int classIdx = 0;
    float maxVal = output.at<float>(0, 0);
    for (int i = 1; i < output.cols; ++i) {
        float val = output.at<float>(0, i);
        if (val > maxVal) {
            maxVal = val;
            classIdx = i;
        }
    }

    if (classIdx >= 0 && classIdx < (int)charLabels.size()) {
        return charLabels[classIdx];
    }
    return '?';
}

// Recognize plate text from segmented characters
std::string recognizePlate(const std::vector<Mat>& chars)
{
    std::string text;
    for (const auto& ch : chars) {
        text.push_back(recognizeChar(ch));
    }
    return text;
}

void extractCharactersFromDataset()
{
    const std::string inputFolder = "DataSet";
    const std::string preprocessedFolder = "PreprocessedImages";
    const std::string charactersFolder = "Characters";

    CreateDirectoryA(preprocessedFolder.c_str(), NULL);
    CreateDirectoryA(charactersFolder.c_str(), NULL);

    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA("DataSet/*.jpg", &fd);
    if (hFind == INVALID_HANDLE_VALUE) {
        std::cout << "No dataset images found in DataSet" << std::endl;
        return;
    }

    int totalChars = 0;
    std::cout << "\n=== License Plate Recognition Results ===" << std::endl;

    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        std::string name = fd.cFileName;
        size_t dot = name.find_last_of('.');
        std::string stem = (dot != std::string::npos) ? name.substr(0, dot) : name;

        std::string path = inputFolder + "/" + name;
        Mat img = imread(path);
        if (img.empty()) continue;

        // Preprocess and detect plate candidates
        Mat processed = preprocessImageALPR(img);
        imwrite(preprocessedFolder + "/" + stem + "_proc.jpg", processed);
        std::vector<Rect> candidates = findPlateCandidates(processed);

        if (candidates.empty()) {
            std::cout << stem << ": no plate detected" << std::endl;
            continue;
        }

        int plateIdx = 0;
        for (const auto& r : candidates) {
            Mat plateROI = img(r);
            std::vector<Mat> chars = segmentCharacters(plateROI);

            if (chars.empty()) {
                plateIdx++;
                continue;
            }

            // Recognize plate text
            std::string plateText = recognizePlate(chars);

            // Save segmented characters
            for (size_t j = 0; j < chars.size(); ++j) {
                std::string out = charactersFolder + "/" + stem + "_p" + std::to_string(plateIdx) + "_c" + std::to_string(j) + ".jpg";
                imwrite(out, chars[j]);
                totalChars++;
            }

            // Print recognized text
            std::cout << stem << ": " << plateText << std::endl;

            plateIdx++;
        }
    } while (FindNextFileA(hFind, &fd));
    FindClose(hFind);

    std::cout << "\nTotal segmented characters: " << totalChars << std::endl;
}

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    
    std::cout << "Loading classifier..." << std::endl;
    loadClassifierAndLabels();
    
    std::cout << "Processing dataset..." << std::endl;
    extractCharactersFromDataset();
    
    std::cout << "\nDone." << std::endl;
    return 0;
}
