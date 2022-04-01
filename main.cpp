#include "toolsmenu.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ToolsMenu w;
    w.show();
    return a.exec();
}
