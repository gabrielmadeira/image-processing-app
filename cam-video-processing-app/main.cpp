#include <QCoreApplication>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    int camera = 0;
    VideoCapture cap;

    if(!cap.open(camera))
       return 0;

    int b, e, fh, fv, g, n, o, r, s, t, v, z;

    b = 0;
    cout << "b : Blurring (Gaussian)\n";
    cout << "c : Color (no processing)\n";
    e = 0;
    cout << "e : Edges (Canny)\n";
    fh = 0;
    cout << "f : Flip the video horizontally\n";
    fv = 0;
    cout << "d : Flip the video vertically\n";
    g = 0;
    cout << "g : Grayscale\n";
    n = 0;
    cout << "n : Negative\n";
    o = 0;
    cout << "o : Contrast enhancement\n";
    r = 0;
    cout << "r : Brightness enhancement\n";
    s = 0;
    cout << "s : Gradient (Sobel)\n";
    t = 0;
    cout << "t : Rotate the video frames\n";
    v = 0;
    cout << "v : Toggle video recording\n";
    z = 0;
    cout << "z : Toogle resize frame to 1/4\n\n\n\n\n\n";

    int delay = 1;

    Mat frame(480, 645, CV_8UC1, Scalar(255, 255, 255)), mod(480, 645, CV_8UC1, Scalar(255, 255, 255));
    string modifiedTitle = "Modified";
    imshow("Original", frame);
    imshow(modifiedTitle, mod);

    int slider = 0;
    int slider_max = 255;
    createTrackbar( "Value", modifiedTitle, &slider, slider_max);

    Size S = Size((int) cap.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
                      (int) cap.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter outputVideo;

    for(;;)
    {

         cap >> frame;
         if( frame.empty() ) break; // end of video stream

         imshow("Original", frame);
         mod = frame.clone();

         //resize
         if (z)
             resize(mod, mod, Size(), 0.5, 0.5);

         //flip
         if (fh)
             flip(mod, mod, 1);
         if (fv)
             flip(mod, mod, 0);

         if (n)
            bitwise_not(mod, mod);

         //rotate
         if (t)
            rotate(mod, mod, ROTATE_90_CLOCKWISE);

         // gray
         if (g)
             cvtColor(mod, mod, COLOR_BGR2GRAY);

         //blur
         if (b)
            GaussianBlur(mod, mod, Size(1+(slider*2),1+(slider*2)),0);

         //contrast
         if (o)
            mod.convertTo(mod, -1, 1+(slider/5), 0);

         // brightness
         if (r)
             mod.convertTo(mod, -1, 1, slider);

         //canny
         if (e)
            Canny(mod, mod,10*(slider+1/10),slider*3);

         //sobel
         if (s) {
            Mat aux;
            Sobel(mod, aux, CV_32F, 1, 0, 1+(slider*2));
            mod = aux.clone();
            double minVal, maxVal;
            minMaxLoc(aux, &minVal, &maxVal); //find minimum and maximum intensities
            aux.convertTo(mod, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
         }

         if (v) { //record
             if (mod.type()==CV_8UC1)
                 cvtColor(mod, mod, COLOR_GRAY2RGB);
             outputVideo.write(mod);
         }

         imshow(modifiedTitle, mod);

         int key = waitKey(delay);

         if (key == 98) //blur
             b = !b;

         if(key == 99) { //color
             g = 0;
             s = 0;
             e = 0;
         }

         if(key == 101) { //edge canny
             if(!e) {
                 g = 1;
                 s = 0;
             }else{
                 g = 0;
             }
             e = !e;
         }

         if(key == 102) { //flip
            fh = !fh;
         }
         if(key == 100) {
            fv = !fv;
         }

         if(key == 103) { //gray
             g = !g;
             e = 0;
             s = 0;
         }

         if(key == 110) //negative
             n = !n;

         if(key == 111) { //contrast
            o = !o;
         }

         if(key == 114) { //brightness
            r = !r;
         }

         if(key == 115) { //sobel
            if(!s) {
                g = 1;
                e = 0;
            }else{
                g = 0;
            }
            s = !s;
         }

         if(key == 116) { //rotate
            if (!v)
                t = !t;
         }

         if(key == 118) { //record
            if (v) {
                v = 0;
                outputVideo.release();
            }
            else {
               v = 1;
               outputVideo = VideoWriter("record.mp4", VideoWriter::fourcc('M','J','P','G'), 25, S);
            }
         }

         if(key == 122) { //resize
            if (!v)
                z = !z;
         }

         if(key == 27 ) break; // stop capturing by pressing ESC
    }
    cap.release();  // release the VideoCapture object
    outputVideo.release();
    return 0;

    return a.exec();
}
