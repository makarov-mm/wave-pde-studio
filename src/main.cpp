#include <QApplication>
#include <QStyleFactory>
#include <QPalette>
#include <QColor>
#include "MainWindow.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    app.setStyle(QStyleFactory::create("Fusion"));
    QPalette dark;
    dark.setColor(QPalette::Window,          QColor(30, 31, 34));
    dark.setColor(QPalette::WindowText,      QColor(220, 220, 220));
    dark.setColor(QPalette::Base,            QColor(24, 25, 28));
    dark.setColor(QPalette::AlternateBase,   QColor(37, 38, 42));
    dark.setColor(QPalette::Text,            QColor(220, 220, 220));
    dark.setColor(QPalette::Button,          QColor(45, 46, 50));
    dark.setColor(QPalette::ButtonText,      QColor(230, 230, 230));
    dark.setColor(QPalette::Highlight,       QColor(64, 132, 214));
    dark.setColor(QPalette::HighlightedText, Qt::white);
    app.setPalette(dark);

    MainWindow w;
    w.show();
    return app.exec();
}
