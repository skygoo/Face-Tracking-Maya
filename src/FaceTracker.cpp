#include "FaceTracker.h"

#include <iostream>

#define ROUND_STICKER_MARKER 1
#define MAX_DIST 20

using namespace std;

float distance(cv::Point2f p1, cv::Point2f p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;

    return sqrt(dx * dx + dy * dy);
}


FaceTracker::FaceTracker() {
    String face_cascade_name = "/Users/sky/CProject/Demo/Res/haarcascade_frontalface_alt2.xml";
    hasFoundFace = false;

    // Load cascade xml files
    if( !face_cascade.load( face_cascade_name ) ) perror("--(!)Error loading face cascade\n");
    
    facemark->loadModel("/Users/sky/CProject/Demo/Res/lbfmodel.yaml");
    
    savedFacePosition = Rect();

    face_rest_captured = false;
}

FaceTracker::~FaceTracker() {
}

bool keySortSmallX(KeyPoint k1, KeyPoint k2) {
    return k1.pt.x < k2.pt.x;
}

bool keySortSmallY(KeyPoint k1, KeyPoint k2) {
    return k1.pt.y < k2.pt.y;
}

bool FaceTracker::detectAndShow(Mat& frame) {
    
    // ###########
    // DETECT FACE
    // ###########
    std::vector<Rect> faces;
    Mat frame_gray = frame;
    
//    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    
//    face_cascade.detectMultiScale( frame_gray, faces, 1.2, 10, 0|CASCADE_SCALE_IMAGE, Size(200, 200));
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, 0, Size(160, 120), Size(320, 240));
    // 将视频帧转换至灰度图, 因为Face Detector的输入是灰度图
    // ############
    // FILTER IMAGE
    // ############
    if (!faces.empty()) {
        savedFacePosition = faces[0];
        hasFoundFace = true;
    } else if (!hasFoundFace) {
        return false;
    }
    
    bool success = facemark->fit(frame,faces,landmarks);
    
    // Uniform size
    float det_width = 300;
    float det_height = 500;


    // ##############
    // DETECT MARKERS
    // ##############

    // Find markers in faceROI
    std::vector<KeyPoint> keypoints;
    
    if(success&&landmarks.size()>0){
        keypoints.push_back(KeyPoint(landmarks[0].at(19), 1.f));
        keypoints.push_back(KeyPoint(landmarks[0].at(21), 1.f));
        keypoints.push_back(KeyPoint(landmarks[0].at(22), 1.f));
        keypoints.push_back(KeyPoint(landmarks[0].at(24), 1.f));
        keypoints.push_back(KeyPoint(landmarks[0].at(41), 1.f));
        keypoints.push_back(KeyPoint(landmarks[0].at(46), 1.f));
        keypoints.push_back(KeyPoint(landmarks[0].at(31), 1.f));
        keypoints.push_back(KeyPoint(landmarks[0].at(35), 1.f));
        keypoints.push_back(KeyPoint(landmarks[0].at(48), 1.f));
        keypoints.push_back(KeyPoint(landmarks[0].at(51), 1.f));
        keypoints.push_back(KeyPoint(landmarks[0].at(54), 1.f));
        keypoints.push_back(KeyPoint(landmarks[0].at(57), 1.f));
    }else{
        return false;
    }
    
    
    // Sort keypoints to correct order
//    std::sort(keypoints.begin(), keypoints.end(), keySortSmallY);
//    std::sort(keypoints.begin()+1, keypoints.begin()+5, keySortSmallX);
//    std::sort(keypoints.begin()+5, keypoints.begin()+7, keySortSmallX);
//    std::sort(keypoints.begin()+7, keypoints.begin()+10, keySortSmallX);
//    std::sort(keypoints.begin()+10, keypoints.end(), keySortSmallX);
//    std::sort(keypoints.begin()+11, keypoints.begin()+13, keySortSmallY);
    
    // Weight keypoints with saved ones
    if (!savedKeypoints.empty()) {
        for (int i = 0; i < keypoints.size(); i++) {
            keypoints[i].pt.x = keypoints[i].pt.x * 0.7 + savedKeypoints[i].pt.x * 0.3;
            keypoints[i].pt.y = keypoints[i].pt.y * 0.7 + savedKeypoints[i].pt.y * 0.3;
        }
    }
    
    // Save keypoints
    savedKeypoints.clear();
    for (int i = 0; i < MARKER_COUNT; i++) {
        savedKeypoints.push_back(keypoints[i]);
        circle(frame, keypoints[i].pt, 2, Scalar(255, 0, 255));
    }
    
//    drawFacemarks(frame, landmarks[0]);
    
    //which keypoint is which?
    int keyIndex = 0;
    TrackingData face_data;

    //forehead
    face_data.markers[FOREHEAD] = keypoints[keyIndex++].pt;

    //brows
    face_data.markers[LEFTOUTERBROW] = keypoints[keyIndex++].pt;
    face_data.markers[LEFTINNERBROW] = keypoints[keyIndex++].pt;
    face_data.markers[RIGHTINNERBROW] = keypoints[keyIndex++].pt;
    face_data.markers[RIGHTOUTERBROW] = keypoints[keyIndex++].pt;

    //cheeks
    face_data.markers[LEFTCHEEK] = keypoints[keyIndex++].pt;
    face_data.markers[RIGHTCHEEK] = keypoints[keyIndex++].pt;

    //nose
    face_data.markers[LEFTNOSE] = keypoints[keyIndex++].pt;
    face_data.markers[NOSE] = keypoints[keyIndex++].pt;
    face_data.markers[RIGHTNOSE] = keypoints[keyIndex++].pt;

    //mouth
    face_data.markers[LEFTMOUTH] = keypoints[keyIndex++].pt;
    face_data.markers[UPPERLIP] = keypoints[keyIndex++].pt;
    face_data.markers[LOWERLIP] = keypoints[keyIndex++].pt;
    face_data.markers[RIGHTMOUTH] = keypoints[keyIndex++].pt;

    //if no rest face, save this one, with forehead at (0,0)
    if (!face_rest_captured) {
        for (int i = 0; i < MARKER_COUNT; i++) {
            face_rest_data.markers[i] = face_data.markers[i] - face_data.markers[FOREHEAD];
            face_prev_data.markers[i] = face_data.markers[i];
        }
        face_rest_captured = true;
        return true;
    }

    //if to big difference from previous frame, skip
    for (int i = 0; i < MARKER_COUNT; i++) {
        if (distance(face_prev_data.markers[i], face_data.markers[i]) > 100) {
            return false;
        }
        face_prev_data.markers[i] = face_data.markers[i];
    }

    //normalized forehead to nose vector in rest state
    float rest_dist = distance(face_rest_data.markers[FOREHEAD], face_rest_data.markers[NOSE]);
    Point2f fn_v_r = (face_rest_data.markers[FOREHEAD] - face_rest_data.markers[NOSE])/rest_dist;

    //normalied forehead to nose vector in current state
    float curr_dist = distance(face_data.markers[FOREHEAD], face_data.markers[NOSE]);
    Point2f fn_v_c = (face_data.markers[FOREHEAD] - face_data.markers[NOSE])/curr_dist;

//    line(frame, face_data.markers[FOREHEAD], face_data.markers[NOSE], Scalar(255,255,255));
//    line(frame, face_rest_data.markers[FOREHEAD] + face_data.markers[FOREHEAD], face_rest_data.markers[NOSE] + face_data.markers[FOREHEAD], Scalar(0,255,255));

    //face tilt
    float cosTheta = fn_v_r.x*fn_v_c.x + fn_v_r.y*fn_v_c.y;
    float sinTheta = sqrt(1.0f - cosTheta*cosTheta);
    if(fn_v_c.x < fn_v_r.x) sinTheta = -sinTheta;

    float trans_scale = curr_dist/rest_dist;

    //rotate and get difference to modified rest data
    for (int i = 0; i < MARKER_COUNT; i++) {
        Point2f rest = face_rest_data.markers[i] * trans_scale;
        rest = Point2f(rest.x*cosTheta - rest.y*sinTheta, rest.x*sinTheta + rest.y*cosTheta);

        Point2f curr = face_data.markers[i] - face_data.markers[FOREHEAD];

        face_move_data.markers[i] = (curr - rest)/curr_dist;

        //draw
        Point2f forehead = face_data.markers[FOREHEAD];
        Point2f pt1 = rest + forehead;
        Point2f pt2 = curr + forehead;
//        line(frame, pt1, pt2, Scalar(0,255,0));
//        circle(frame, pt1, 2, Scalar(255,0,0));
//        circle(frame, pt2, 2, Scalar(255,255,0));
    }

    return true;
}

void FaceTracker::reset() {
    face_rest_captured = false;
    savedFacePosition = Rect();
    hasFoundFace = false;
    savedKeypoints.clear();
    face_move_data = TrackingData();
    face_rest_data = TrackingData();
}

TrackingData& FaceTracker::getTrackingData() {
    return face_move_data;
}
