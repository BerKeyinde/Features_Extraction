#include <iostream>
#include <iomanip>
#include <cstdio>
#include <string>
#include <math.h>
#include <fstream>
#include <sstream>

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

Mat image;
bool showPointsEnabled = true;

void drawFacePoints (Mat& img, const Face face);
float computeDistance (Point p1, Point p2);
void populate(Mat& img, const Face face, string label);

float input[12][11];
Point face_p[12];

String inputName = "C:\\Users\\mcahi\\Downloads\\Compressed\\faces94\\faces94\\set.txt";

int main()
{
    Mat img;
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    if ( inputName.empty()) {
        cerr << "ERROR: Could not load image or image database.\n"
                "It should be a text file containing the list of the image filenames to be processed - one per line"
        << endl;
        return -1;
    }
    std::ifstream file(inputName.c_str(), ifstream::in);

            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */

    if( !file )
    {
        cerr << "ERROR: Could not open file \"" << inputName.c_str() << "\"."<< endl;
        cerr << "Usage: faceImitator [--cascade=\"<cascade_path>\"]\n"
                "   [--scale=<image scale>]\n"
                "   [filename|camera_index]\n" ;

        return -1;
    }


    double nextTilt=0;
    Point nextRotCenter;
    double t;
    string line, path, classlabel;
    char separator = ';';

    while ( getline(file, line) ) {
            stringstream liness(line);
            getline(liness, path, separator);
            getline(liness, classlabel);

            if ( !path.empty() && !classlabel.empty() )
                img = imread( path, 1 );

            Mat draftImg;
            img.copyTo(draftImg);
                   //Face detection and points extraction.
            t = (double)cvGetTickCount(); //To count extraction time.
            Face face (img, face_cascade, scale);

            //If autorotation for tracking
            if (nextTilt != 0 && rotationEnabled) {
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
               } else
                  nextTilt = 0;
            }

            if (showPointsEnabled)
               drawFacePoints (draftImg, face);

            populate(draftImg, face, classlabel);

            imshow( "Captured from Cam", draftImg );
            imshow("Preprocessed Image", face.processedImg);
//            waitKey(0);

            int c = waitKey(10);
            if( (char)c == 'c' ) { break; }

    }


//    cvDestroyWindow( "Captured from Cam");
//    cvDestroyWindow( "Preprocessed Image");


    return 0;
}


void drawFacePoints (Mat& img, const Face face)
{
    double scale = face.scale;

    //Lips
    Face::rescaledCircle(img,face.llip, 1, scale, CV_RGB(255,170,120),-1,CV_AA);
    Face::rescaledCircle(img,face.rlip, 1, scale, CV_RGB(255,170,120),-1,CV_AA);

    //Eyebrows
    for (int i = 0; i <= 1; i++)
    {
        //Left
        Face::rescaledCircle(img,face.leb[i], 1, scale,CV_RGB(255,0,0),-1,CV_AA);

        //Right
        Face::rescaledCircle(img,face.reb[i], 1, scale,CV_RGB(255,0,0),-1,CV_AA);
    }


    //Eyes
    //All points
    for (int i = 0; i <= 1; i++) {
        Face::rescaledCircle(img, face.lefcps[i], 1, scale, CV_RGB(255,255,255), -1,CV_AA);
        Face::rescaledCircle(img, face.refcps[i], 1, scale, CV_RGB(255,255,255), -1,CV_AA);
    }

    //Nosestrills
    Face::rescaledCircle(img,face.lnstrl, 1, scale, CV_RGB(0,255,0),-1,CV_AA);
    Face::rescaledCircle(img,face.rnstrl, 1, scale, CV_RGB(0,255,0),-1,CV_AA);


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

float computeDistance(Point p1, Point p2)
{
    Point diff = p1 - p2;
    float ans = cv::sqrt(diff.x*diff.x + diff.y*diff.y);

    return ans;
}

void populate(Mat& img, const Face face, string label)
{
    double scale = face.scale;
    face_p[0] = face.rlip; face_p[1] = face.llip; face_p[2] = face.leb[0]; face_p[3] = face.leb[1];
    face_p[4] = face.reb[0]; face_p[5] = face.reb[1]; face_p[6] = face.lefcps[0]; face_p[7] = face.lefcps[1];
    face_p[8] = face.refcps[0]; face_p[9] = face.refcps[1]; face_p[10] = face.lnstrl; face_p[11] = face.rnstrl;

    for (int i = 0; i < sizeof input/sizeof input[0]; i++) {
        for (int j = 0; j < i; j++) {
            input[i][j] = computeDistance(face_p[i], face_p[j]);
            //Face::rescaledLine(img, face_p[i], face_p[j], scale, CV_RGB(0,0,255),1,CV_AA);
        }
        for (int k = i; k < sizeof face_p/sizeof *face_p; k++) {
            input[i][k] = computeDistance(face_p[i], face_p[k + 1]);
            //Face::rescaledLine(img, face_p[i], face_p[k + 1], scale, CV_RGB(0,0,255),1,CV_AA);
        }
    }

    for (int i = 0; i < sizeof input/sizeof input[0]; i++) {
        for (int j = 0; j < sizeof input[0]/sizeof(int); j++) {
            cout << input[i][j]; cout << ", "; }
        cout << label;
        cout << " " << endl;
        cout << " " << endl;
    }
}
