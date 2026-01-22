#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <direct.h>
#include <io.h>
#include <windows.h>

using namespace cv::ml;
using namespace cv::dnn;

wchar_t* projectPath;

// Global classifier and label mapping
Ptr<ANN_MLP> mlpClassifier;      // OpenCV MLP (fallback)
Net onnxNet;                      // ONNX model via DNN module
bool useOnnx = false;             // Flag to use ONNX model
std::vector<char> labelToChar;    // Maps numeric labels to characters
std::map<char, int> charToLabel;  // Maps characters to numeric labels

// ============== FEATURE EXTRACTION ==============

// Extract Hu Moments (7 moments - invariant to translation, scale, rotation)
std::vector<double> extractHuMoments(const Mat& img)
{
    Mat gray, binary;
    
    // Convert to grayscale if needed
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    
    // Resize to standard size
    Mat resized;
    resize(gray, resized, Size(32, 32));
    
    // Binarize
    threshold(resized, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    
    // Calculate moments
    Moments moments = cv::moments(binary, true);
    
    // Calculate Hu Moments
    double huMoments[7];
    HuMoments(moments, huMoments);
    
    // Log transform for better numerical stability
    std::vector<double> features(7);
    for (int i = 0; i < 7; i++) {
        features[i] = -1.0 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]) + 1e-10);
    }
    
    return features;
}

// Extract additional features for better discrimination
std::vector<double> extractFeatures(const Mat& img)
{
    Mat gray, binary;
    
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    
    // Resize to standard size
    Mat resized;
    resize(gray, resized, Size(32, 32));
    
    // Binarize
    threshold(resized, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    
    std::vector<double> features;
    
    // 1. Hu Moments (7 features)
    Moments moments = cv::moments(binary, true);
    double huMoments[7];
    HuMoments(moments, huMoments);
    for (int i = 0; i < 7; i++) {
        features.push_back(-1.0 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]) + 1e-10));
    }
    
    // 2. Pixel density in quadrants (4 features)
    int midX = binary.cols / 2;
    int midY = binary.rows / 2;
    double totalPixels = binary.rows * binary.cols / 4.0;
    
    features.push_back(countNonZero(binary(Rect(0, 0, midX, midY))) / totalPixels);
    features.push_back(countNonZero(binary(Rect(midX, 0, midX, midY))) / totalPixels);
    features.push_back(countNonZero(binary(Rect(0, midY, midX, midY))) / totalPixels);
    features.push_back(countNonZero(binary(Rect(midX, midY, midX, midY))) / totalPixels);
    
    // 3. Horizontal and vertical projections (16 features - 8 each)
    for (int i = 0; i < 8; i++) {
        int startRow = i * binary.rows / 8;
        int endRow = (i + 1) * binary.rows / 8;
        Mat row = binary(Rect(0, startRow, binary.cols, endRow - startRow));
        features.push_back(countNonZero(row) / (double)(row.rows * row.cols));
    }
    
    for (int i = 0; i < 8; i++) {
        int startCol = i * binary.cols / 8;
        int endCol = (i + 1) * binary.cols / 8;
        Mat col = binary(Rect(startCol, 0, endCol - startCol, binary.rows));
        features.push_back(countNonZero(col) / (double)(col.rows * col.cols));
    }
    
    // 4. Aspect ratio of bounding box of character
    std::vector<Point> points;
    findNonZero(binary, points);
    if (!points.empty()) {
        Rect boundingBox = boundingRect(points);
        features.push_back((double)boundingBox.width / (boundingBox.height + 1));
        features.push_back((double)boundingBox.width * boundingBox.height / (binary.rows * binary.cols));
    } else {
        features.push_back(1.0);
        features.push_back(0.0);
    }
    
    // 5. Number of contours (1 feature)
    std::vector<std::vector<Point>> contours;
    findContours(binary.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    features.push_back(contours.size() / 10.0); // Normalized
    
    // Total: 7 + 4 + 16 + 2 + 1 = 30 features
    return features;
}

// ============== HELPER FUNCTIONS ==============

bool directoryExists(const std::string& path) {
    DWORD attrib = GetFileAttributesA(path.c_str());
    return (attrib != INVALID_FILE_ATTRIBUTES && (attrib & FILE_ATTRIBUTE_DIRECTORY));
}

bool fileExists(const std::string& path) {
    DWORD attrib = GetFileAttributesA(path.c_str());
    return (attrib != INVALID_FILE_ATTRIBUTES && !(attrib & FILE_ATTRIBUTE_DIRECTORY));
}

std::vector<std::string> getFilesInDirectory(const std::string& directory, const std::string& pattern = "*.*") {
    std::vector<std::string> files;
    WIN32_FIND_DATAA findData;
    std::string searchPath = directory + "/" + pattern;
    
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                files.push_back(directory + "/" + findData.cFileName);
            }
        } while (FindNextFileA(hFind, &findData));
        FindClose(hFind);
    }
    return files;
}

std::string getFileExtension(const std::string& filepath) {
    size_t pos = filepath.rfind('.');
    if (pos != std::string::npos) {
        std::string ext = filepath.substr(pos);
        // Convert to lowercase
        for (char& c : ext) c = tolower(c);
        return ext;
    }
    return "";
}

std::string getFileName(const std::string& filepath) {
    size_t pos = filepath.find_last_of("/\\");
    if (pos != std::string::npos) {
        return filepath.substr(pos + 1);
    }
    return filepath;
}

// ============== TRAINING DATA LOADING ==============

void loadTrainingData(const std::string& datasetPath, Mat& trainData, Mat& labels)
{
    std::vector<std::vector<double>> allFeatures;
    std::vector<int> allLabels;
    
    labelToChar.clear();
    charToLabel.clear();
    
    // Define the characters we're looking for (Romanian plates don't use I, O to avoid confusion with 1, 0)
    std::string validChars = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ";
    
    int labelIdx = 0;
    
    for (char c : validChars) {
        std::string charFolder = datasetPath + "/" + std::string(1, c);
        
        if (!directoryExists(charFolder)) {
            std::cout << "Warning: Folder not found for character '" << c << "'" << std::endl;
            continue;
        }
        
        labelToChar.push_back(c);
        charToLabel[c] = labelIdx;
        
        int count = 0;
        std::vector<std::string> files = getFilesInDirectory(charFolder);
        
        for (const auto& filepath : files) {
            std::string ext = getFileExtension(filepath);
            if (ext == ".jpg" || ext == ".png" || ext == ".bmp") {
                
                Mat img = imread(filepath, IMREAD_GRAYSCALE);
                if (img.empty()) continue;
                
                std::vector<double> features = extractFeatures(img);
                allFeatures.push_back(features);
                allLabels.push_back(labelIdx);
                count++;
                
                // Limit samples per class for balance (optional)
                if (count >= 500) break;
            }
        }
        
        std::cout << "Loaded " << count << " samples for character '" << c << "' (label " << labelIdx << ")" << std::endl;
        labelIdx++;
    }
    
    if (allFeatures.empty()) {
        std::cout << "Error: No training data found!" << std::endl;
        return;
    }
    
    // Convert to Mat
    int numSamples = (int)allFeatures.size();
    int numFeatures = (int)allFeatures[0].size();
    
    trainData = Mat(numSamples, numFeatures, CV_32F);
    labels = Mat(numSamples, 1, CV_32S);
    
    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            trainData.at<float>(i, j) = (float)allFeatures[i][j];
        }
        labels.at<int>(i, 0) = allLabels[i];
    }
    
    std::cout << "\nTotal: " << numSamples << " samples, " << numFeatures << " features, " 
              << labelToChar.size() << " classes" << std::endl;
}

// ============== MLP CLASSIFIER ==============

void trainMLPClassifier(const Mat& trainData, const Mat& labels, int numClasses)
{
    std::cout << "\nTraining MLP Classifier..." << std::endl;
    
    // Convert labels to one-hot encoding for MLP
    Mat trainLabels = Mat::zeros(labels.rows, numClasses, CV_32F);
    for (int i = 0; i < labels.rows; i++) {
        trainLabels.at<float>(i, labels.at<int>(i, 0)) = 1.0f;
    }
    
    // Create MLP
    mlpClassifier = ANN_MLP::create();
    
    // Define network architecture
    int numFeatures = trainData.cols;
    Mat layers = (Mat_<int>(1, 4) << numFeatures, 128, 64, numClasses);
    mlpClassifier->setLayerSizes(layers);
    
    // Set activation function (Sigmoid)
    mlpClassifier->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
    
    // Set training method
    mlpClassifier->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
    
    // Set termination criteria
    mlpClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5000, 1e-6));
    
    // Train
    Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainLabels);
    
    std::cout << "Training with " << trainData.rows << " samples..." << std::endl;
    mlpClassifier->train(tData);
    
    std::cout << "Training complete!" << std::endl;
}

void saveClassifier(const std::string& modelPath, const std::string& labelPath)
{
    mlpClassifier->save(modelPath);
    
    // Save label mapping
    std::ofstream file(labelPath);
    for (size_t i = 0; i < labelToChar.size(); i++) {
        file << i << " " << labelToChar[i] << std::endl;
    }
    file.close();
    
    std::cout << "Classifier saved to " << modelPath << std::endl;
    std::cout << "Label mapping saved to " << labelPath << std::endl;
}

bool loadClassifier(const std::string& modelPath, const std::string& labelPath)
{
    if (!fileExists(modelPath) || !fileExists(labelPath)) {
        std::cout << "Classifier files not found. Please train first." << std::endl;
        return false;
    }
    
    mlpClassifier = ANN_MLP::load(modelPath);
    useOnnx = false;
    
    // Load label mapping
    labelToChar.clear();
    charToLabel.clear();
    
    std::ifstream file(labelPath);
    int idx;
    char c;
    while (file >> idx >> c) {
        if (idx >= (int)labelToChar.size()) {
            labelToChar.resize(idx + 1);
        }
        labelToChar[idx] = c;
        charToLabel[c] = idx;
    }
    file.close();
    
    std::cout << "OpenCV MLP Classifier loaded successfully. " << labelToChar.size() << " classes." << std::endl;
    return true;
}

bool loadOnnxClassifier(const std::string& onnxPath, const std::string& labelPath)
{
    if (!fileExists(onnxPath) || !fileExists(labelPath)) {
        std::cout << "ONNX classifier files not found." << std::endl;
        return false;
    }
    
    try {
        onnxNet = readNetFromONNX(onnxPath);
        onnxNet.setPreferableBackend(DNN_BACKEND_OPENCV);
        onnxNet.setPreferableTarget(DNN_TARGET_CPU);
        useOnnx = true;
    }
    catch (const cv::Exception& e) {
        std::cout << "Error loading ONNX model: " << e.what() << std::endl;
        return false;
    }
    
    // Load label mapping
    labelToChar.clear();
    charToLabel.clear();
    
    std::ifstream file(labelPath);
    int idx;
    char c;
    while (file >> idx >> c) {
        if (idx >= (int)labelToChar.size()) {
            labelToChar.resize(idx + 1);
        }
        labelToChar[idx] = c;
        charToLabel[c] = idx;
    }
    file.close();
    
    std::cout << "ONNX Classifier loaded successfully. " << labelToChar.size() << " classes." << std::endl;
    return true;
}

// ============== CHARACTER RECOGNITION ==============

char recognizeCharacter(const Mat& charImg)
{
    std::vector<double> features = extractFeatures(charImg);
    
    Mat featureMat(1, (int)features.size(), CV_32F);
    for (size_t i = 0; i < features.size(); i++) {
        featureMat.at<float>(0, (int)i) = (float)features[i];
    }
    
    int predictedLabel = -1;
    
    if (useOnnx && !onnxNet.empty()) {
        // Use ONNX model via DNN module
        Mat blob = blobFromImage(featureMat, 1.0, Size(), Scalar(), false, false);
        blob = featureMat.clone(); // For 1D input, use feature vector directly
        
        onnxNet.setInput(featureMat);
        Mat output = onnxNet.forward();
        
        // Find class with highest score
        Point maxLoc;
        minMaxLoc(output, nullptr, nullptr, nullptr, &maxLoc);
        predictedLabel = maxLoc.x;
    }
    else if (!mlpClassifier.empty()) {
        // Use OpenCV MLP
        Mat output;
        mlpClassifier->predict(featureMat, output);
        
        Point maxLoc;
        minMaxLoc(output, nullptr, nullptr, nullptr, &maxLoc);
        predictedLabel = maxLoc.x;
    }
    else {
        return '?';
    }
    
    if (predictedLabel >= 0 && predictedLabel < (int)labelToChar.size()) {
        return labelToChar[predictedLabel];
    }
    
    return '?';
}

// ============== PLATE PREPROCESSING ==============

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

    // 1. Convert to grayscale (handle both BGR and grayscale input)
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

    // 7. Extract GRAYSCALE characters (not binary!)
    //    Let extractFeatures handle binarization consistently
    std::vector<Mat> characters;
    for (auto& r : charRects)
    {
        Mat ch = gray(r).clone();  // Extract from grayscale, not binary
        resize(ch, ch, Size(32, 32));
        characters.push_back(ch);
    }

    return characters;
}

// ============== PLATE DETECTION AND RECOGNITION ==============

// Correct plate rotation using minimum area rectangle
Mat correctPlateRotation(const Mat& plate)
{
    Mat gray, binary;
    
    if (plate.channels() == 3) {
        cvtColor(plate, gray, COLOR_BGR2GRAY);
    } else {
        gray = plate.clone();
    }
    
    threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    
    std::vector<Point> points;
    findNonZero(binary, points);
    
    if (points.size() < 5) return plate;
    
    RotatedRect minRect = minAreaRect(points);
    double angle = minRect.angle;
    
    // Correct angle
    if (angle < -45) angle += 90;
    if (angle > 45) angle -= 90;
    
    // Only correct if angle is significant but not too large
    if (abs(angle) > 1 && abs(angle) < 15) {
        Point2f center(plate.cols / 2.0f, plate.rows / 2.0f);
        Mat rotMatrix = getRotationMatrix2D(center, angle, 1.0);
        Mat rotated;
        warpAffine(plate, rotated, rotMatrix, plate.size(), INTER_LINEAR, BORDER_REPLICATE);
        return rotated;
    }
    
    return plate;
}

// Validate Romanian plate format: LL-NN-LLL or L-NN-LLL or L-NNN-LLL
bool isValidRomanianPlate(const std::string& plate)
{
    // Remove any spaces or dashes for validation
    std::string clean;
    for (char c : plate) {
        if (c != ' ' && c != '-') {
            clean += c;
        }
    }
    
    int len = (int)clean.length();
    
    // Valid lengths: 6, 7, or 8 characters
    if (len < 6 || len > 8) return false;
    
    // Count letters and digits
    int letterCount = 0, digitCount = 0;
    for (char c : clean) {
        if (isalpha(c)) letterCount++;
        else if (isdigit(c)) digitCount++;
    }
    
    // Romanian plates: 4-5 letters, 2-3 digits
    if (letterCount < 4 || letterCount > 5) return false;
    if (digitCount < 2 || digitCount > 3) return false;
    
    return true;
}

// Format plate string according to Romanian format
std::string formatPlateString(const std::string& chars)
{
    if (chars.length() < 6) return chars;
    
    // Determine format based on length and character positions
    std::string result;
    int len = (int)chars.length();
    
    // Check if first char is 'B' (Bucharest)
    if (chars[0] == 'B' && len >= 6) {
        // Bucharest: B-NN-LLL or B-NNN-LLL
        result = chars.substr(0, 1) + "-";
        
        // Find where digits start
        int digitStart = 1;
        while (digitStart < len && !isdigit(chars[digitStart])) digitStart++;
        
        // Find where letters resume
        int letterStart = digitStart;
        while (letterStart < len && isdigit(chars[letterStart])) letterStart++;
        
        if (digitStart < len && letterStart <= len) {
            result += chars.substr(digitStart, letterStart - digitStart) + "-";
            result += chars.substr(letterStart);
        } else {
            result = chars; // Fallback
        }
    } else {
        // Other counties: LL-NN-LLL
        // First two are county code
        int digitStart = 0;
        while (digitStart < len && !isdigit(chars[digitStart])) digitStart++;
        
        if (digitStart >= 1 && digitStart <= 2) {
            result = chars.substr(0, digitStart) + "-";
            
            int letterStart = digitStart;
            while (letterStart < len && isdigit(chars[letterStart])) letterStart++;
            
            result += chars.substr(digitStart, letterStart - digitStart) + "-";
            result += chars.substr(letterStart);
        } else {
            result = chars;
        }
    }
    
    return result;
}

// Main plate recognition function
std::string recognizePlate(const Mat& img, bool showSteps = false)
{
    if (img.empty()) {
        return "ERROR: Empty image";
    }
    
    Mat display;
    if (showSteps) {
        display = img.clone();
    }
    
    // Step 1: Preprocess image for plate detection
    Mat preprocessed = preprocessImageALPR(img);
    
    if (showSteps) {
        imshow("1. Preprocessed", preprocessed);
    }
    
    // Step 2: Find plate candidates
    std::vector<Rect> candidates = findPlateCandidates(preprocessed);
    
    std::cout << "Found " << candidates.size() << " plate candidate(s)" << std::endl;
    
    std::string bestPlate;
    int maxChars = 0;
    
    for (size_t i = 0; i < candidates.size(); i++) {
        Rect r = candidates[i];
        
        // Expand rect slightly for better character capture
        int expandX = r.width * 0.05;
        int expandY = r.height * 0.1;
        r.x = max(0, r.x - expandX);
        r.y = max(0, r.y - expandY);
        r.width = min(img.cols - r.x, r.width + 2 * expandX);
        r.height = min(img.rows - r.y, r.height + 2 * expandY);
        
        Mat plateROI = img(r).clone();
        
        // Check if plate has enough detail
        if (!hasEnoughEdges(plateROI)) {
            continue;
        }
        
        if (showSteps) {
            rectangle(display, r, Scalar(0, 255, 0), 2);
        }
        
        // Step 3: Correct rotation
        Mat correctedPlate = correctPlateRotation(plateROI);
        
        if (showSteps) {
            imshow("2. Plate ROI " + std::to_string(i), correctedPlate);
        }
        
        // Step 4: Segment characters
        std::vector<Mat> chars = segmentCharacters(correctedPlate);
        
        std::cout << "Candidate " << i << ": Found " << chars.size() << " characters" << std::endl;
        
        if (chars.size() < 6 || chars.size() > 9) {
            continue; // Romanian plates have 6-8 characters (with possible county separator)
        }
        
        // Step 5: Recognize each character
        std::string plateText;
        for (size_t j = 0; j < chars.size(); j++) {
            char c = recognizeCharacter(chars[j]);
            plateText += c;
            
            if (showSteps) {
                imshow("Char " + std::to_string(j), chars[j]);
            }
        }
        
        std::cout << "Raw recognition: " << plateText << std::endl;
        
        // Keep the best candidate (most characters that form valid plate)
        if ((int)chars.size() > maxChars) {
            maxChars = (int)chars.size();
            bestPlate = plateText;
        }
    }
    
    if (showSteps && !display.empty()) {
        imshow("Detected Plates", display);
    }
    
    if (bestPlate.empty()) {
        return "NO PLATE DETECTED";
    }
    
    // Format the plate string
    std::string formatted = formatPlateString(bestPlate);
    
    return formatted;
}

// ============== TEST FUNCTIONS ==============

void testSingleImage(const std::string& imagePath)
{
    Mat img = imread(imagePath);
    if (img.empty()) {
        std::cout << "Error: Could not load image " << imagePath << std::endl;
        return;
    }
    
    std::cout << "\n=== Processing: " << imagePath << " ===" << std::endl;
    
    std::string result = recognizePlate(img, true);
    
    std::cout << "\n*** RECOGNIZED PLATE: " << result << " ***" << std::endl;
    
    // Draw result on image
    Mat display = img.clone();
    putText(display, result, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 3);
    imshow("Result", display);
    
    waitKey(0);
    destroyAllWindows();
}

void testAllImages(const std::string& datasetPath)
{
    std::cout << "\n=== Testing all images in " << datasetPath << " ===" << std::endl;
    
    int count = 0;
    std::vector<std::string> files = getFilesInDirectory(datasetPath);
    
    for (const auto& filepath : files) {
        std::string ext = getFileExtension(filepath);
        if (ext == ".jpg" || ext == ".png" || ext == ".bmp" || ext == ".webp") {
            Mat img = imread(filepath);
            if (img.empty()) continue;
            
            std::cout << "\n[" << ++count << "] " << getFileName(filepath) << std::endl;
            
            std::string result = recognizePlate(img, false);
            std::cout << "    Result: " << result << std::endl;
            
            // Show image with result
            Mat display = img.clone();
            double scale = min(800.0 / display.cols, 600.0 / display.rows);
            if (scale < 1.0) {
                resize(display, display, Size(), scale, scale);
            }
            putText(display, result, Point(10, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
            imshow("Test Result", display);
            
            int key = waitKey(0);
            if (key == 27) break; // ESC to stop
        }
    }
    
    destroyAllWindows();
    std::cout << "\nTested " << count << " images." << std::endl;
}

// ============== MAIN ==============

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);
    
    // Convert projectPath to string
    char path[MAX_PATH];
    wcstombs(path, projectPath, MAX_PATH);
    std::string basePath(path);
    
    std::string modelPath = basePath + "/mlp_classifier.xml";
    std::string onnxPath = basePath + "/mlp_classifier.onnx";
    std::string labelPath = basePath + "/label_mapping.txt";
    std::string trainingPath = basePath + "/TrainingData/CNN letter Dataset";
    std::string datasetPath = basePath + "/DataSet";

    int choice;
    
    while (true) {
        std::cout << "\n===========================================" << std::endl;
        std::cout << "Romanian License Plate Recognition System" << std::endl;
        std::cout << "===========================================" << std::endl;
        std::cout << "1. Train MLP classifier (OpenCV - slow)" << std::endl;
        std::cout << "2. Load OpenCV MLP classifier (.xml)" << std::endl;
        std::cout << "3. Load ONNX classifier (PyTorch trained)" << std::endl;
        std::cout << "4. Recognize plate from single image" << std::endl;
        std::cout << "5. Test all images in DataSet" << std::endl;
        std::cout << "6. Test single character recognition" << std::endl;
        std::cout << "0. Exit" << std::endl;
        std::cout << "Enter choice: ";
        std::cin >> choice;

        switch (choice) {
        case 1: {
            std::cout << "\nLoading training data from: " << trainingPath << std::endl;
            
            Mat trainData, labels;
            loadTrainingData(trainingPath, trainData, labels);
            
            if (trainData.empty()) {
                std::cout << "Error: No training data loaded!" << std::endl;
                break;
            }
            
            trainMLPClassifier(trainData, labels, (int)labelToChar.size());
            saveClassifier(modelPath, labelPath);
            
            // Test accuracy on training set
            int correct = 0;
            for (int i = 0; i < trainData.rows; i++) {
                Mat sample = trainData.row(i);
                Mat output;
                mlpClassifier->predict(sample, output);
                Point maxLoc;
                minMaxLoc(output, nullptr, nullptr, nullptr, &maxLoc);
                if (maxLoc.x == labels.at<int>(i, 0)) correct++;
            }
            std::cout << "Training accuracy: " << (100.0 * correct / trainData.rows) << "%" << std::endl;
            break;
        }
        
        case 2: {
            loadClassifier(modelPath, labelPath);
            break;
        }
        
        case 3: {
            loadOnnxClassifier(onnxPath, labelPath);
            break;
        }
        
        case 4: {
            if (!useOnnx && mlpClassifier.empty()) {
                std::cout << "No classifier loaded. Loading ONNX model..." << std::endl;
                if (!loadOnnxClassifier(onnxPath, labelPath)) {
                    std::cout << "Trying OpenCV MLP..." << std::endl;
                    if (!loadClassifier(modelPath, labelPath)) {
                        break;
                    }
                }
            }
            
            std::cout << "Enter image path (or filename in DataSet folder): ";
            std::string imgPath;
            std::cin >> imgPath;
            
            // Check if it's just a filename
            if (imgPath.find('/') == std::string::npos && 
                imgPath.find('\\') == std::string::npos) {
                imgPath = datasetPath + "/" + imgPath;
            }
            
            testSingleImage(imgPath);
            break;
        }
        
        case 5: {
            if (!useOnnx && mlpClassifier.empty()) {
                std::cout << "No classifier loaded. Loading ONNX model..." << std::endl;
                if (!loadOnnxClassifier(onnxPath, labelPath)) {
                    std::cout << "Trying OpenCV MLP..." << std::endl;
                    if (!loadClassifier(modelPath, labelPath)) {
                        break;
                    }
                }
            }
            
            testAllImages(datasetPath);
            break;
        }
        
        case 6: {
            if (!useOnnx && mlpClassifier.empty()) {
                std::cout << "No classifier loaded. Loading ONNX model..." << std::endl;
                if (!loadOnnxClassifier(onnxPath, labelPath)) {
                    std::cout << "Trying OpenCV MLP..." << std::endl;
                    if (!loadClassifier(modelPath, labelPath)) {
                        break;
                    }
                }
            }
            
            std::cout << "Enter character image path: ";
            std::string charPath;
            std::cin >> charPath;
            
            Mat charImg = imread(charPath, IMREAD_GRAYSCALE);
            if (charImg.empty()) {
                std::cout << "Error: Could not load image" << std::endl;
                break;
            }
            
            char result = recognizeCharacter(charImg);
            std::cout << "Recognized character: " << result << std::endl;
            
            imshow("Character", charImg);
            waitKey(0);
            destroyAllWindows();
            break;
        }
        
        case 0:
            std::cout << "Exiting..." << std::endl;
            return 0;
            
        default:
            std::cout << "Invalid choice!" << std::endl;
        }
    }

    return 0;
}