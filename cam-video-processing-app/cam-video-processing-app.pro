QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        main.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target


win32:CONFIG(release, debug|release): LIBS += -LE:/opencv/build/x64/vc15/lib/ -lopencv_world455
else:win32:CONFIG(debug, debug|release): LIBS += -LE:/opencv/build/x64/vc15/lib/ -lopencv_world455d
else:unix: LIBS += -LE:/opencv/build/x64/vc15/lib/ -lopencv_world455

INCLUDEPATH += E:/opencv/build/include
DEPENDPATH += E:/opencv/build/include
