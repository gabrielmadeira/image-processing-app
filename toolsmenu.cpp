#include "toolsmenu.h"
#include "ui_toolsmenu.h"
#include <opencv2/opencv.hpp>
#include <QDebug>

using namespace cv;
using namespace std;

Mat editImage;
string editImageTitle = "Edit Image";

Mat originalImage;
string originalImageTitle = "Original Image";

Mat histogramImage;
string histogramImageTitle = "Histogram";

int maxTones = 256;
int nTones = maxTones;
vector<bool> tones(maxTones, false);

vector<long long int> histogram(maxTones, 0);

ToolsMenu::ToolsMenu(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::ToolsMenu)
{
    ui->setupUi(this);
}

ToolsMenu::~ToolsMenu()
{
    delete ui;
}

void ToolsMenu::on_btn_image_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Choose"), "", tr("Images (*.jpg *.jpeg)"));
    if (QString::compare(filename, QString()) != 0)
    {
        originalImage = imread(filename.toStdString(),1);
        editImage = originalImage.clone();

//        namedWindow(editImageTitle, WINDOW_NORMAL);
//        namedWindow(originalImageTitle, WINDOW_NORMAL);

        showOriginalImage();
        showEditImage();
    }
}

void ToolsMenu::showOriginalImage()
{
//    float WIDTH = 500;
//    float HEIGHT = editImage.rows * (WIDTH/editImage.cols);
//    resizeWindow(originalImageTitle, WIDTH, HEIGHT);
    imshow(originalImageTitle, originalImage);
}
void ToolsMenu::showEditImage()
{
//    float WIDTH = 500;
//    float HEIGHT = editImage.rows * (WIDTH/editImage.cols);
//    resizeWindow(editImageTitle, WIDTH, HEIGHT);
    imshow(editImageTitle, editImage);
}

void ToolsMenu::on_btn_flip_horizontally_clicked()
{
    if(editImage.empty())
        return;

    int dif = editImage.cols-1;
    for (int r = 0; r < editImage.rows; r++)
    {
        dif = editImage.cols-1;
        for (int c = 0; c < (editImage.cols/2); c++)
        {
                Vec3b buf = editImage.at<Vec3b>(r, c);
                editImage.at<Vec3b>(r, c) = editImage.at<Vec3b>(r, dif);
                editImage.at<Vec3b>(r, dif) = buf;
                dif--;
        }
    }
    showEditImage();
}


void ToolsMenu::on_btn_flip_vertically_clicked()
{
    if(editImage.empty())
        return;

    int dif;
    for (int c = 0; c < editImage.cols; c++)
    {
        dif = editImage.rows-1;
        for (int r = 0; r < (editImage.rows/2); r++)
        {
                Vec3b buf = editImage.at<Vec3b>(r, c);
                editImage.at<Vec3b>(r, c) = editImage.at<Vec3b>(dif, c);
                editImage.at<Vec3b>(dif, c) = buf;
                dif--;
        }
    }

    showEditImage();
}


void ToolsMenu::on_btn_luminance_clicked()
{
    if(editImage.empty())
        return;

    fill_n(tones.begin(), 256, false);
    nTones = 0;

    for (int r = 0; r < editImage.rows; r++)
    {
        for (int c = 0; c < (editImage.cols); c++)
        {
            float lum = (editImage.at<Vec3b>(r, c)[0] * 0.114)
                    +(editImage.at<Vec3b>(r, c)[1] * 0.587)
                    +(editImage.at<Vec3b>(r, c)[2] * 0.299);

            if(!tones[(int)lum]){
                tones[(int)lum] = true;
                nTones++;
            }
            editImage.at<Vec3b>(r, c)[0] = lum;
            editImage.at<Vec3b>(r, c)[1] = lum;
            editImage.at<Vec3b>(r, c)[2] = lum;
        }
    }
    showEditImage();
}


void ToolsMenu::on_btn_quantize_clicked()
{
    on_btn_luminance_clicked();
    int nColors = ui->spin_colors->text().toInt();
    if(nTones <= nColors)
        return;

    fill_n(tones.begin(), 256, false);
    nTones = 0;

    float div = 255/(float)nColors;
    for (int r = 0; r < editImage.rows; r++)
    {
        for (int c = 0; c < (editImage.cols); c++)
        {
            int value = editImage.at<Vec3b>(r, c)[0];


            int mult = 0;
            while(div*(float)(mult+1) < value)
                mult++;
            int newValue = (float)mult * div;

            if(!tones[newValue]){
                tones[newValue] = true;
                nTones++;
            }

            editImage.at<Vec3b>(r, c) = Vec3b(newValue, newValue, newValue);
        }
    }
    showEditImage();
}


void ToolsMenu::on_btn_copy_clicked()
{
    if(originalImage.empty())
        return;

    editImage = originalImage.clone();
    showEditImage();
}

void ToolsMenu::on_btn_save_clicked()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("Choose"), "", tr("Images (*.jpg *.jpeg)"));
    if (QString::compare(filename, QString()) != 0)
        imwrite(filename.toStdString(),editImage);
}

vector<long long int> ToolsMenu::calcHistogram(Mat channel)
{
    vector<long long int> hist(maxTones, 0);
    for (int r = 0; r < channel.rows; r++)
        for (int c = 0; c < (channel.cols); c++)
        {
            int value = channel.at<uint8_t>(r, c);
            hist[value]++;
        }

    return hist;
}

void ToolsMenu::on_btn_histogram_clicked()
{
    if(editImage.empty())
        return;
    on_btn_luminance_clicked();

    vector<Mat> channels;
    split(editImage, channels);
    histogram = calcHistogram(channels[0]);

    long long int max_qtd = 0;
    for (int i=0; i<maxTones; i++)
        if(histogram[i] > max_qtd)
            max_qtd = histogram[i];

    int nRows = 200;
    int nCols = maxTones;
    int nBGColor = 255;
    histogramImage = Mat(nRows,nCols,CV_8U,nBGColor);

    for (int c = 0; c < nCols; c++)
    {
        long long int row = ((nRows*histogram[c])/max_qtd);
        for (int r = 0; r < row; r++)
        {
            histogramImage.at<uint8_t>(nRows-r-1, c)= 0;
        }
    }

    int nBorder = 6;
    int borderType = BORDER_CONSTANT;
    int top, bottom, left, right;
    RNG rng(12345);
    top = nBorder; bottom = top;
    left = nBorder; right = left;

    Scalar value( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
    copyMakeBorder(histogramImage, histogramImage, top, bottom, left, right, borderType, value);

    imshow(histogramImageTitle, histogramImage);
}


void ToolsMenu::on_btn_negative_clicked()
{
    if(editImage.empty())
        return;

    for (int r = 0; r < editImage.rows; r++)
    {
        for (int c = 0; c < (editImage.cols); c++)
        {
            editImage.at<Vec3b>(r, c)[0] = 255-editImage.at<Vec3b>(r, c)[0];
            editImage.at<Vec3b>(r, c)[1] = 255-editImage.at<Vec3b>(r, c)[1];
            editImage.at<Vec3b>(r, c)[2] = 255-editImage.at<Vec3b>(r, c)[2];
        }
    }
    showEditImage();
}


void ToolsMenu::on_btn_brightness_clicked()
{
    if(editImage.empty())
        return;

    int nBrightness = ui->spin_brightness->text().toInt();

    for (int r = 0; r < editImage.rows; r++)
    {
        for (int c = 0; c < (editImage.cols); c++)
        {
            editImage.at<Vec3b>(r, c)[0] = max(0, min(editImage.at<Vec3b>(r, c)[0] + nBrightness, 255));
            editImage.at<Vec3b>(r, c)[1] = max(0, min(editImage.at<Vec3b>(r, c)[1] + nBrightness, 255));
            editImage.at<Vec3b>(r, c)[2] = max(0, min(editImage.at<Vec3b>(r, c)[2] + nBrightness, 255));
        }
    }
    showEditImage();
}


void ToolsMenu::on_btn_contrast_clicked()
{
    if(editImage.empty())
        return;

    float nContrast = ui->spin_contrast->text().toFloat();

    for (int r = 0; r < editImage.rows; r++)
    {
        for (int c = 0; c < (editImage.cols); c++)
        {
            editImage.at<Vec3b>(r, c)[0] = max(0, min(int(editImage.at<Vec3b>(r, c)[0] * nContrast), 255));
            editImage.at<Vec3b>(r, c)[1] = max(0, min(int(editImage.at<Vec3b>(r, c)[1] * nContrast), 255));
            editImage.at<Vec3b>(r, c)[2] = max(0, min(int(editImage.at<Vec3b>(r, c)[2] * nContrast), 255));
        }
    }
    showEditImage();
}

vector<double> ToolsMenu::calcCumulativeHistogram(vector<long long int> histogram, Mat image)
{
    vector<double> histCum(maxTones, 0);
    double alpha = 255.0f/double(image.rows*image.cols);
    histCum[0] = alpha * histogram[0];

    for(int i=1; i<256; i++)
        histCum[i] = histCum[i-1] + alpha * double(histogram[i]);
    return histCum;
}

void ToolsMenu::on_btn_equalize_histogram_clicked()
{
    if(editImage.empty())
        return;

    Mat lumImage = Mat(editImage.rows,editImage.cols,CV_8U);
    for (int r = 0; r < editImage.rows; r++)
        for (int c = 0; c < (editImage.cols); c++)
        {
            float lum = (editImage.at<Vec3b>(r, c)[0] * 0.114f)
                    +(editImage.at<Vec3b>(r, c)[1] * 0.587f)
                    +(editImage.at<Vec3b>(r, c)[2] * 0.299f);
            lumImage.at<uint8_t>(r, c) = lum;
        }

    histogram = calcHistogram(lumImage);

    vector<double> histCum = calcCumulativeHistogram(histogram, editImage);


    for (int r = 0; r < editImage.rows; r++)
        for (int c = 0; c < (editImage.cols); c++)
            for (int color = 0; color < 3; color++)
                editImage.at<Vec3b>(r, c)[color] = round(histCum[editImage.at<Vec3b>(r, c)[color]]);

    showEditImage();

}

void ToolsMenu::on_btn_histogram_match_clicked()
{
    if(editImage.empty())
        return;
    QString filename = QFileDialog::getOpenFileName(this, tr("Choose"), "", tr("Images (*.jpg *.jpeg)"));
    if (QString::compare(filename, QString()) != 0)
    {
        on_btn_luminance_clicked();
        Mat target_image = imread(filename.toStdString(), IMREAD_GRAYSCALE);
        vector<long long int> hist_target_image = calcHistogram(target_image);
        vector<double> histCumTargetImage = calcCumulativeHistogram(hist_target_image, target_image);
        imshow("Target Image", target_image);

        Mat channel[3];
        split(editImage, channel);
        histogram = calcHistogram(channel[0]);
        vector<double> histCum = calcCumulativeHistogram(histogram, editImage);

        vector<double> histMatch(maxTones, 0);
        for(int shade_level=0; shade_level<maxTones; shade_level++){
            double dif = NULL;
            int bestMatch = NULL;
            for(int shade_level_target=0; shade_level_target<maxTones; shade_level_target++){
                if((bestMatch == NULL) || (abs(histCum[shade_level]-histCumTargetImage[shade_level_target])<dif))
                {
                    bestMatch = shade_level_target;
                    dif = abs(histCum[shade_level]-histCumTargetImage[shade_level_target]);
                }
            }
            histMatch[shade_level] = histCumTargetImage[bestMatch];
        }
        for (int r = 0; r < editImage.rows; r++)
            for (int c = 0; c < (editImage.cols); c++)
            {
                int value = round(histMatch[editImage.at<Vec3b>(r, c)[0]]);
                editImage.at<Vec3b>(r, c) = Vec3b(value, value, value);
            }
        showEditImage();
    }
}


void ToolsMenu::on_btn_zoomout_clicked()
{
    if(editImage.empty())
        return;

    int nSx = ui->spin_zoomout_x->text().toInt();
    int nSy = ui->spin_zoomout_y->text().toInt();
    if(editImage.rows < nSy || editImage.cols < nSx)
        return;
    int nRows = editImage.rows/nSy;
    int nCols = editImage.cols/nSx;
    int nPixelsScale = nSy*nSx;
    Mat outputImage = Mat(nRows,nCols,CV_8UC3);
    for (int r = 0; r < nRows; r++)
        for (int c = 0; c < nCols; c++)
        {
            int sumB = 0, sumG = 0, sumR = 0;
            for (int ri = r*nSy; ri < (r*nSy)+nSy; ri++)
                for (int ci = c*nSx; ci < (c*nSx)+nSx; ci++)
                {
                    sumB += editImage.at<Vec3b>(ri, ci)[0];
                    sumG += editImage.at<Vec3b>(ri, ci)[1];
                    sumR += editImage.at<Vec3b>(ri, ci)[2];
                }

            int B = sumB/nPixelsScale;
            int G = sumG/nPixelsScale;
            int R = sumR/nPixelsScale;

            outputImage.at<Vec3b>(r, c) = Vec3b(B, G, R);
        }

    editImage = outputImage.clone();
    showEditImage();
}



void ToolsMenu::on_btn_zoomin_clicked()
{
    if(editImage.empty())
        return;
    int nRows = editImage.rows * 2 - 1;
    int nCols = editImage.cols * 2 - 1;
    Mat outputImage = Mat(nRows,nCols,CV_8UC3);


    for (int r = 0; r < nRows; r+=2)
    {
        outputImage.at<Vec3b>(r, 0) = editImage.at<Vec3b>(r/2, 0);
        for (int c = 1; c < (nCols-1); c+=2)
        {
            int B, G, R;
            B = (editImage.at<Vec3b>(r/2, (c+1)/2)[0] + editImage.at<Vec3b>(r/2, (c-1)/2)[0])/2;
            G = (editImage.at<Vec3b>(r/2, (c+1)/2)[1] + editImage.at<Vec3b>(r/2, (c-1)/2)[1])/2;
            R = (editImage.at<Vec3b>(r/2, (c+1)/2)[2] + editImage.at<Vec3b>(r/2, (c-1)/2)[2])/2;

            outputImage.at<Vec3b>(r, c+1) = editImage.at<Vec3b>(r/2, (c+1)/2);
            outputImage.at<Vec3b>(r, c) = Vec3b(B, G, R);
        }
    }
    for (int r = 1; r < (nRows-1); r+=2)
        for (int c = 0; c < nCols; c++)
        {
            int B, G, R;
            B = (outputImage.at<Vec3b>(r-1, c)[0]+outputImage.at<Vec3b>(r+1, c)[0])/2;
            G = (outputImage.at<Vec3b>(r-1, c)[1]+outputImage.at<Vec3b>(r+1, c)[1])/2;
            R = (outputImage.at<Vec3b>(r-1, c)[2]+outputImage.at<Vec3b>(r+1, c)[2])/2;
            outputImage.at<Vec3b>(r, c) = Vec3b(B, G, R);
        }

    editImage = outputImage.clone();
    showEditImage();
}

void ToolsMenu::rotate(int direction)
{
    int nRows = editImage.cols;
    int nCols = editImage.rows;
    Mat outputImage = Mat(nRows,nCols,CV_8UC3);

    for (int r = 0; r < nRows; r++)
        for (int c = 0; c < nCols; c++)
        {
            int nRow, nCol;
            if(direction == 0)
            {
                nRow = r;
                nCol = nCols-c-1;
            }
            else
            {
                nRow = nRows-r-1;
                nCol = c;
            }
            outputImage.at<Vec3b>(r, c) = editImage.at<Vec3b>(nCol, nRow);
        }

    editImage = outputImage.clone();
    showEditImage();
}

void ToolsMenu::on_btn_rotate90p_clicked()
{
    rotate(0);
}


void ToolsMenu::on_btn_rotate90n_clicked()
{
    rotate(1);
}


void ToolsMenu::on_btn_convolution_clicked()
{

    bool bGrayBG = ui->check_graybg->isChecked();
    float nAddValue = 0;
    if(bGrayBG)
        nAddValue = 127;

    float a = ui->spin_kernel_a->text().toFloat();
    float b = ui->spin_kernel_b->text().toFloat();
    float c = ui->spin_kernel_c->text().toFloat();
    float d = ui->spin_kernel_d->text().toFloat();
    float e = ui->spin_kernel_e->text().toFloat();
    float f = ui->spin_kernel_f->text().toFloat();
    float g = ui->spin_kernel_g->text().toFloat();
    float h = ui->spin_kernel_h->text().toFloat();
    float i = ui->spin_kernel_i->text().toFloat();

    Mat outputImage = Mat(editImage.rows,editImage.cols,CV_8UC3,1);

    for (int row = 1; row < editImage.rows-1; row++)
        for (int col = 1; col < editImage.cols-1; col++)
            for (int channel = 0; channel < 3; channel++)
            {
                float A = editImage.at<Vec3b>(row-1, col-1)[channel];
                float B = editImage.at<Vec3b>(row-1, col)[channel];
                float C = editImage.at<Vec3b>(row-1, col+1)[channel];
                float D = editImage.at<Vec3b>(row, col-1)[channel];
                float E = editImage.at<Vec3b>(row, col)[channel];
                float F = editImage.at<Vec3b>(row, col+1)[channel];
                float G = editImage.at<Vec3b>(row+1, col-1)[channel];
                float H = editImage.at<Vec3b>(row+1, col)[channel];
                float I = editImage.at<Vec3b>(row+1, col+1)[channel];

                float convE = max(0.0f, min(255.0f, i*A + h*B + g*C + f*D + e*E + d*F + c*G + b*H + a*I + nAddValue));
                outputImage.at<Vec3b>(row, col)[channel] = convE;
            }
    editImage = outputImage.clone();
    showEditImage();
}

void ToolsMenu::setKernel(float a, float b, float c, float d, float e, float f, float g, float h, float i)
{
    ui->spin_kernel_a->setValue(a);
    ui->spin_kernel_b->setValue(b);
    ui->spin_kernel_c->setValue(c);
    ui->spin_kernel_d->setValue(d);
    ui->spin_kernel_e->setValue(e);
    ui->spin_kernel_f->setValue(f);
    ui->spin_kernel_g->setValue(g);
    ui->spin_kernel_h->setValue(h);
    ui->spin_kernel_i->setValue(i);
}

void ToolsMenu::on_btn_kernel_gaussian_clicked()
{
    setKernel(0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625);
}


void ToolsMenu::on_btn_kernel_laplacian_clicked()
{
    on_btn_luminance_clicked();
    setKernel(0,-1,0,-1,4,-1,0,-1,0);
}


void ToolsMenu::on_btn_kernel_hpass_clicked()
{
    on_btn_luminance_clicked();
    setKernel(-1,-1,-1,-1,8,-1,-1,-1,-1);
}


void ToolsMenu::on_btn_kernel_prewitthx_clicked()
{
    on_btn_luminance_clicked();
    setKernel(-1,0,1,-1,0,1,-1,0,1);
}


void ToolsMenu::on_btn_kernel_prewitthy_clicked()
{
    on_btn_luminance_clicked();
    setKernel(-1,-1,-1,0,0,0,1,1,1);
}


void ToolsMenu::on_btn_kernel_sobelhx_clicked()
{
    on_btn_luminance_clicked();
    setKernel(-1,0,1,-2,0,2,-1,0,1);
}


void ToolsMenu::on_btn_kernel_sobelhy_clicked()
{
    on_btn_luminance_clicked();
    setKernel(-1,-2,-1,0,0,0,1,2,1);
}

