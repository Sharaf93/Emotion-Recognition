#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>

using std::cout;
using std::endl;
using namespace std;
using namespace cv;

const int ROWS = 2059;
const int ROWS2 = 8237;
const int COLS = 16;
const int BUFFSIZE = 80;

int main() {

	// Testing Data
	Mat testingDataMat = Mat(2059, 16, CV_32FC1, float(0));
	float testLabels[2059];
	int array[ROWS][COLS];
	char buff[BUFFSIZE];
	std::ifstream file("dataTest.csv");
	std::string line;
	int col = 0;
	int row = 0;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string result;
		while (std::getline(iss, result, ',')) {
			array[row][col] = ceil(atof(result.c_str()));
			col = col + 1;
		}
		row = row + 1;
		col = 0;
	}
	for (int i = 0; i < 2059; i++) {
		for (int j = 0; j < 16; j++) {
			testingDataMat.at<float>(i, j) = array[i][j];
			if (j == 0) {
				testLabels[i] = array[i][j];
			}
		}
	}
	Mat testingLabels(2059, 1, CV_32FC1, testLabels);

// Print Test matrix values
//	  for (int i=0;i<100;i++){
//			for(int j=0;j<16;j++){
//				cout << testingDataMat.at<float>(i,j) << "   ";
//			}
//			cout << "\n";
//	   }

// Training Data
	Mat trainingDataMat = Mat(8237, 16, CV_32FC1, float(0));
	float trainLabels[8237];
	int array2[ROWS2][COLS];
	char buff2[BUFFSIZE];
	std::ifstream file2("dataTrain.csv");
	std::string line2;
	int col2 = 0;
	int row2 = 0;
	while (std::getline(file2, line2)) {
		std::istringstream iss(line2);
		std::string result;
		while (std::getline(iss, result, ',')) {
			array2[row2][col2] = atoi(result.c_str());
			col2 = col2 + 1;
		}
		row2 = row2 + 1;
		col2 = 0;
	}
	for (int i = 0; i < 8237; i++) {
		for (int j = 0; j < 16; j++) {
			trainingDataMat.at<float>(i, j) = array2[i][j];
			if (j == 0) {
				trainLabels[i] = array2[i][j];
			}
		}
	}
	Mat trainingLabels(8237, 1, CV_32FC1, trainLabels);

//	 Print train matrix values
//		  for (int i=0;i<8237;i++){
////				for(int j=0;j<16;j++){
////					cout << trainingDataMat.at<float>(i,j) << "   ";
////				}
////			  cout << trainLabels[i];
//				cout << "\n";
//		   }


// Train Naive Bayes Classifier
    NormalBayesClassifier classifer;
    classifer.train(trainingDataMat, trainingLabels, Mat(), Mat());


    Mat test_sample;
    float response;
    float misClassificationCount = 0 ;
    // extract a row from the testing matrix
    for (int i = 0; i < testingDataMat.rows; i++)
	{
	   test_sample = testingDataMat.row(i);
	   // run decision tree prediction

	   response = classifer.predict(test_sample);
	   cout << response << " ";

	   if ( abs(response - testingLabels.at<float>(i,0)) > 0.00001 )
		   misClassificationCount++;


	}

    cout << endl << "The misclassification error rate is: " << misClassificationCount/testingDataMat.rows << endl;
	return 0;
}
