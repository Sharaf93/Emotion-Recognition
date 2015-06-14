//#include "opencv2/highgui/highgui.hpp"
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//int main(int argc, char* argv[])
//{
//	VideoCapture cap("video.avi"); // open the video file for reading
//
//    if ( !cap.isOpened() )  // if not success, exit program
//    {
//         cout << "Cannot open the video file" << endl;
//         return -1;
//    }
//
//    //cap.set(CV_CAP_PROP_POS_MSEC, 300); //start the video at 300ms
//
//    double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
//
//     cout << "Frame per seconds : " << fps << endl;
//
//    namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
//
//    while(1)
//    {
//        Mat frame;
//
//        bool bSuccess = cap.read(frame); // read a new frame from video
//
//        if (!bSuccess) //if not success, break loop
//        {
//                        cout << "Cannot read the frame from video file" << endl;
//                       break;
//        }
//
//        imshow("MyVideo", frame); //show the frame in "MyVideo" window
//
//        if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
//       {
//                cout << "esc key is pressed by user" << endl;
//                break;
//       }
//    }
//
//    return 0;
//
//}



#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

#include <cv.h>
#include <highgui.h>

#include <iostream>
#include <string>
#include "opencv2/highgui/highgui.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "face_x.h"

using namespace std;

const string kModelFileName = "model.xml.gz";
const string kAlt2 = "haarcascade_frontalface_alt2.xml";
const string path = "images/";
string kTestImage = "test.jpg";
Mat landmarksTotal = Mat(14, 102, CV_32FC1, float(0));

void AlignImage(const FaceX & face_x) {
	for (int i = 0; i < 15; i++) {
		string s = static_cast<ostringstream*>(&(ostringstream() << i))->str();
		kTestImage = path + s + ".jpg";
		cv::Mat image = cv::imread(kTestImage);
		cv::Mat gray_image;
		cv::cvtColor(image, gray_image, CV_BGR2GRAY);
		cv::CascadeClassifier cc(kAlt2);
		if (cc.empty()) {
			cout << "Cannot open model file " << kAlt2
					<< " for OpenCV face detector!" << endl;
			return;
		}
		int counter = 0;
		vector<cv::Rect> faces;
		cc.detectMultiScale(gray_image, faces);
		cv::Rect face = faces[0];
		cv::rectangle(image, face, cv::Scalar(0, 0, 255), 2);
		vector<cv::Point2d> landmarks = face_x.Alignment(gray_image, face);
		int x = 1;
		string v;
		for (cv::Point2d landmark : landmarks) {
			v = static_cast<ostringstream*>(&(ostringstream() << x))->str();
			landmarksTotal.at<float>(i, counter) = landmark.x;
			landmarksTotal.at<float>(i, counter + 1) = landmark.y;
//			if (counter == 0) {

//			putText(image, v, landmark, FONT_HERSHEY_COMPLEX_SMALL, 0.4,
//					cv::Scalar(0, 255, 0), 1, CV_AA);

			cv::circle(image, landmark, 1, cv::Scalar(0, 255, 0), 2);

//			}
			counter += 2;
			x++;
		}
		cout << landmarksTotal << endl;
		cv::imshow("Alignment result", image);
		cv::waitKey();
	}
}

void Tracking(const FaceX & face_x) {
	{
		VideoCapture stream1(0); //0 is the id of video device.0 if you have only one camera.

		if (!stream1.isOpened()) { //check if video device has been initialised
			cout << "cannot open camera";
		}

		//unconditional loop
		while (true) {
			Mat cameraFrame;
			stream1.read(cameraFrame);
			cv::Mat image = cameraFrame;
			cv::Mat gray_image;
			cv::cvtColor(image, gray_image, CV_BGR2GRAY);
			cv::CascadeClassifier cc(kAlt2);
			if (cc.empty()) {
				cout << "Cannot open model file " << kAlt2
						<< " for OpenCV face detector!" << endl;
				return;
			}
			vector<cv::Rect> faces;
			cc.detectMultiScale(gray_image, faces);

			for (cv::Rect face : faces) {
				cv::rectangle(image, face, cv::Scalar(0, 0, 255), 2);
				vector<cv::Point2d> landmarks = face_x.Alignment(gray_image,
						face);
				for (cv::Point2d landmark : landmarks) {
					cout << landmark;
					cv::circle(image, landmark, 1, cv::Scalar(0, 255, 0), 2);
				}
			}
			cv::imshow("Alignment result", image);
			cv::waitKey();

			if (waitKey(30) >= 0)
				break;
		}
	}
//	cout << "Press \"r\" to re-initialize the face location." << endl;
//	cv::Mat frame;
//	cv::Mat img;
//	cv::VideoCapture vc(0);
//	vc >> frame;
//	cv::CascadeClassifier cc(kAlt2);
//	cv::vector<cv::Point2d> landmarks(face_x.landmarks_count());
//
//	for (;;)
//	{
//		vc >> frame;
//		cv::cvtColor(frame, img, cv::COLOR_BGR2GRAY);
//		cv::imshow("test", img);
//
//		cv::vector<cv::Point2d> original_landmarks = landmarks;
//		landmarks = face_x.Alignment(img, landmarks);
//
//		for (int i = 0; i < landmarks.size(); ++i)
//		{
//			landmarks[i].x = (landmarks[i].x + original_landmarks[i].x) / 2;
//			landmarks[i].y = (landmarks[i].y + original_landmarks[i].y) / 2;
//		}
//
//		for (cv::Point2d p : landmarks)
//		{
//			cv::circle(frame, p, 1, cv::Scalar(0, 255, 0), 2);
//		}
//
//		cv::imshow("\"r\" to re-initialize, \"q\" to exit", frame);
//		int key = cv::waitKey(10);
//		if (key == 'q')
//			break;
//		else if (key == 'r')
//		{
//			vector<cv::Rect> faces;
//			cc.detectMultiScale(img, faces);
//			if (!faces.empty())
//			{
//				landmarks = face_x.Alignment(img, faces[0]);
//			}
//		}
//	}
}

int main() {
	try {
		FaceX face_x(kModelFileName);

		cout << "Choice: " << endl;
		cout << "1. Align " << kTestImage
				<< " in the current working directory." << endl;
		cout << "2. Align video from web camera." << endl;
		cout << "Please select one [1/2]: ";
		int choice;
		cin >> choice;
		switch (choice) {
		case 1:
			AlignImage(face_x);
			break;
		case 2:
			Tracking(face_x);
			break;
		}
	} catch (const runtime_error& e) {
		cerr << e.what() << endl;
	}
}


