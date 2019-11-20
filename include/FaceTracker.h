#ifndef __FACETRACKER_H
#define __FACETRACKER_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/face.hpp>

#include <String>
#include <list>

#include "TrackingData.h"

using namespace cv;
using namespace cv::face;
using namespace std;

class FaceTracker
{
	public:	
		FaceTracker();
        virtual ~FaceTracker();

        bool detectAndShow(Mat& frame);
        void reset();

        TrackingData& getTrackingData();

	private:
        //marker color
        float H, S, V;
        float thresholdH, thresholdS, thresholdV;

        cv::CascadeClassifier face_cascade;
        cv::CascadeClassifier eyes_cascade;

//        Ptr<SimpleBlobDetector> marker_detector;
    
    // 创建Facemark类的对象
        Ptr<Facemark> facemark = FacemarkLBF::create();
    
        vector< vector<Point2f> > landmarks;

        Rect savedFacePosition;
        std::vector<KeyPoint> savedKeypoints;
    
        bool hasFoundFace;
        bool face_rest_captured;

        TrackingData face_rest_data;
        TrackingData face_prev_data;
        TrackingData face_move_data;
};

#endif 	/* FACETRACKER_H */
