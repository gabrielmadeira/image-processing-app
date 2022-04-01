#ifndef TOOLSMENU_H
#define TOOLSMENU_H

#include <QMainWindow>
#include <QFileDialog>
#include <opencv2/opencv.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { class ToolsMenu; }
QT_END_NAMESPACE

class ToolsMenu : public QMainWindow
{
    Q_OBJECT

public:
    ToolsMenu(QWidget *parent = nullptr);
    ~ToolsMenu();

private slots:
    void on_btn_image_clicked();

    void showOriginalImage();

    void showEditImage();

    void on_btn_flip_horizontally_clicked();

    void on_btn_flip_vertically_clicked();

    void on_btn_luminance_clicked();

    void on_btn_quantize_clicked();

    void on_btn_copy_clicked();

    void on_btn_save_clicked();

    std::vector<long long int> calcHistogram(cv::Mat channel);

    void on_btn_histogram_clicked();

    void on_btn_negative_clicked();

    void on_btn_brightness_clicked();

    void on_btn_contrast_clicked();

    std::vector<double> ToolsMenu::calcCumulativeHistogram(std::vector<long long int> histogram, cv::Mat image);

    void on_btn_histogram_match_clicked();

    void on_btn_equalize_histogram_clicked();

    void on_btn_zoomout_clicked();

    void on_btn_zoomin_clicked();

    void rotate(int direction);

    void on_btn_rotate90p_clicked();

    void on_btn_rotate90n_clicked();

    void on_btn_convolution_clicked();

    void setKernel(float a, float b, float c, float d, float e, float f, float g, float h, float i);

    void on_btn_kernel_gaussian_clicked();

    void on_btn_kernel_laplacian_clicked();

    void on_btn_kernel_hpass_clicked();

    void on_btn_kernel_prewitthx_clicked();

    void on_btn_kernel_prewitthy_clicked();

    void on_btn_kernel_sobelhx_clicked();

    void on_btn_kernel_sobelhy_clicked();

private:
    Ui::ToolsMenu *ui;
};
#endif // TOOLSMENU_H
