#include <iostream>
#include <iomanip>
#include <cstdio>
#include <string>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "face.h"

#ifdef _EiC
#define WIN32
#endif

using namespace cv;
using namespace std;

String face_cascade_name = "haarcascades/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

bool rotationEnabled = true;
bool blurEnabled = true;
bool faceDetected = false;
double extractionTime;

double scale = 3;

double timeToWait;


bool showPointsEnabled = true;

void drawFacePoints (Mat& img, const Face face);



int main()
{
    CvCapture* capture;
    Mat img;
    int deviceId = CV_CAP_ANY;
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    VideoCapture cap(deviceId);

    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }


    double nextTilt=0;
    Point nextRotCenter;
    double t;

    capture = cvCaptureFromCAM( CV_CAP_ANY );
    if( capture )
    {
        while( true )
        {
          img = cvQueryFrame( capture );

          if( !img.empty() ) {
            Mat draftImg;
            img.copyTo(draftImg);

            //Face detection and points extraction.
            t = (double)cvGetTickCount(); //To count extraction time.
            Face face (img, face_cascade, scale);


            //If autorotation for tracking
            if (nextTilt != 0 && rotationEnabled)
            {
                face.extractCharacteristicPoints(scale,nextTilt,nextRotCenter);
            }
            else
                face.extractCharacteristicPoints();

            t = (double)cvGetTickCount() - t;
            extractionTime = t/((double)cvGetTickFrequency()*1000.0); //in ms

            //Only for autorotation
            if (rotationEnabled)
            {
                if (face.faceFound)
                {
                    nextTilt = -(face.tilt);
                    nextRotCenter = face.cleye; //We center rotation on left eye.
                }else
                    nextTilt = 0;
            }

            if (showPointsEnabled)
                drawFacePoints (draftImg, face);

            cv::imshow( "Captured from Cam", draftImg );
            cv::imshow("Preprocessed Image", face.processedImg);
          }

          else
            { printf(" --(!) No captured frame -- Break!"); }

          int c = waitKey(10);
          if( (char)c == 'c' ) { break; }
        }
    }
    return 0;
}

void drawFacePoints (Mat& img, const Face face)
{
    double scale = face.scale;

    //Lips
    Face::rescaledCircle(img,face.llip, 1, scale, CV_RGB(255,170,120),-1,CV_AA);
    Face::rescaledCircle(img,face.rlip, 1, scale, CV_RGB(255,170,120),-1,CV_AA);
    Face::rescaledCircle(img,face.ulip, 1, scale, CV_RGB(255,170,120),-1,CV_AA);
    Face::rescaledCircle(img,face.dlip, 1, scale, CV_RGB(255,170,120),-1,CV_AA);

    //Eyebrows
    for (int i = 0; i< 4; i++)
    {
        //Left
        Face::rescaledCircle(img,face.leb[i], 1, scale,CV_RGB(255,0,0),-1,CV_AA);

        //Right
        Face::rescaledCircle(img,face.reb[i], 1, scale,CV_RGB(255,0,0),-1,CV_AA);
    }


    //Eyes
    //Center Points
    Face::rescaledCircle(img,face.creye, 1, scale,CV_RGB(255,255,255),-1,CV_AA);
    Face::rescaledCircle(img,face.cleye, 1, scale,CV_RGB(255,255,255),-1,CV_AA);


    //Nosestrills
    Face::rescaledCircle(img,face.lnstrl, 1, scale, CV_RGB(0,255,0),-1,CV_AA);
    Face::rescaledCircle(img,face.rnstrl, 1, scale, CV_RGB(0,255,0),-1,CV_AA);



    //Tilt Angle
    // Drawing tilted lines on face limits
    Point uline1, uline2, dline1, dline2, lline1, lline2, rline1, rline2;
    Point puline1, puline2;
    Point p1,p2;

    Rect fr; //Face frame
    Face::transformPoint(face.location, p1, face.rMat);
    fr.x = p1.x;
    fr.y = p1.y;
    fr.width = face.size.width;
    fr.height = face.size.height;


    lline1.x = fr.x;
    lline2.x = lline1.x - 10;
    rline1.x = fr.x + fr.width - 1;
    rline2.x = rline1.x + 10;
    lline1.y = lline2.y = rline1.y = rline2.y = fr.y + std::max(0, (fr.height/2) - 1);

    uline1.x = uline2.x =  dline1.x = dline2.x = fr.x + std::max(0, (fr.width/2) - 1);
    uline1.y =  fr. y;
    uline2.y = uline1.y - 10;
    dline1.y =  fr. y + fr.height -1;
    dline2.y = dline1.y + 10;

    puline1 = uline1;
    puline2 = uline2;


    Face::transformPoint(uline1 , uline1, face.irMat);
    Face::transformPoint(uline2 , uline2, face.irMat);
    Face::transformPoint(dline1 , dline1, face.irMat);
    Face::transformPoint(dline2 , dline2, face.irMat);
    Face::transformPoint(lline1 , lline1, face.irMat);
    Face::transformPoint(lline2 , lline2, face.irMat);
    Face::transformPoint(rline1 , rline1, face.irMat);
    Face::transformPoint(rline2 , rline2, face.irMat);



    //Scaling to big img
    Face::rescaledLine(img, puline1, puline2, scale, CV_RGB(255,255, 255), 3, CV_AA);
    //Face::rescaledLine(img2, rotCenter, puline1, scale, CV_RGB(255,255, 255), 1, CV_AA);
    //Face::rescaledLine(img2, rotCenter, uline1, scale, CV_RGB(255,255, 255), 1, CV_AA);
    Face::rescaledLine(img, uline1, uline2, scale, CV_RGB(255,170,120), 3, CV_AA);
    Face::rescaledLine(img, dline1, dline2, scale, CV_RGB(255,170,120), 3, CV_AA);
    Face::rescaledLine(img, lline1, lline2, scale, CV_RGB(255,170,120), 3, CV_AA);
    Face::rescaledLine(img, rline1, rline2, scale, CV_RGB(255,170,120), 3, CV_AA);


    // Some Textual Info
    std::stringstream ss;
    ss << "Face tilt: " << setiosflags(ios::fixed) << setprecision(2) << face.tilt << " deg.";
    putText(img, ss.str(), Point (0,15), FONT_HERSHEY_PLAIN, 1, CV_RGB(255,255, 255), 1,CV_AA, false);

    ss.str("");
    ss << "Img. Proc. Time: " << setiosflags(ios::fixed) << setprecision(2) << extractionTime << " ms.";
    putText(img, ss.str(), Point (0,30), FONT_HERSHEY_PLAIN, 1, CV_RGB(255,255, 255), 1,CV_AA, false);

    ss.str("");
    ss << "Auto-rotation: " << (rotationEnabled? "ON": "OFF");
    putText(img, ss.str(), Point (0,45), FONT_HERSHEY_PLAIN, 1, CV_RGB(255,255, 255), 1,CV_AA, false);
}
