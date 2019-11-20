//OpenCV webcam example from http://opencv-srf.blogspot.se/2011/09/capturing-images-videos.html

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <queue>
#include <ctime>
#include <string>

#include "MayaSocket.h"
#include "FaceTracker.h"
#include "TrackingData.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    int fc = 0;
    cout << "test1-------" << endl;
    MayaSocket maya;
    FaceTracker tracker;

    VideoCapture cap(0);

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }
//    namedWindow("Webcam",WINDOW_AUTOSIZE);
//    namedWindow("Filtered",WINDOW_AUTOSIZE);
    cout << "test2" << endl;
    // Setup FPS calculation
    int framecount = 0;
    std::clock_t start;
    double duration;
    start = std::clock();

    cout << "test" << endl;
    bool record = false;

    float time = 0;
    
    cout << "test" << endl;
    // ------------------------------------------------------
    // Main loop
    // ------------------------------------------------------
    while (true) {
        //handle input
        char key = waitKey(1);
        if (key == 27) {
            cout << "esc key is pressed by user" << endl;
            break;
        } else if (key == 'r') {
            tracker.reset();
            cout << "tracking reset" << endl;
        } else if (key == 'p') {
            record = !record;
            if(record)
                cout << "recording ON" << endl;
            else
                cout << "recording OFF" << endl;
        }
        
        fc++;
        
        //record frame
        Mat frame;
        if (!cap.read(frame)) {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }
        
//        frame =  imread("/Users/sky/CProject/Face-Tracking-Maya/rigreference.jpg", IMREAD_COLOR);

        
        flip(frame, frame, 1);

        //track markers
        bool new_data_available = tracker.detectAndShow(frame);
        imshow("Webcam", frame );
        //send to maya
        if (new_data_available && record) {
            printf("frame: %i\n",fc);
            TrackingData &data = tracker.getTrackingData();
            float currTime = std::clock()/(float) CLOCKS_PER_SEC;
            data.timeStep = currTime - time;
            time = currTime;
            maya.send(data);
        }

        // Calculate FPS
        framecount++;
        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        if (duration > 1.0) {
            printf("FPS: %i\n", framecount);
            start = std::clock();
            framecount = 0;
        }
        
    }
    
    return 0;
}
