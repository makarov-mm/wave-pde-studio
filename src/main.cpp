#include "MainWindow.h"

#include <QApplication>
#include <QColor>
#include <QPalette>

namespace {

QPalette darkPalette() {
    QPalette palette;
    palette.setColor(QPalette::Window,          QColor(30, 31, 34));
    palette.setColor(QPalette::WindowText,      QColor(220, 220, 220));
    palette.setColor(QPalette::Base,            QColor(24, 25, 28));
    palette.setColor(QPalette::AlternateBase,   QColor(37, 38, 42));
    palette.setColor(QPalette::Text,            QColor(220, 220, 220));
    palette.setColor(QPalette::Button,          QColor(45, 46, 50));
    palette.setColor(QPalette::ButtonText,      QColor(230, 230, 230));
    palette.setColor(QPalette::Highlight,       QColor(64, 132, 214));
    palette.setColor(QPalette::HighlightedText, Qt::white);
    return palette;
}

} // namespace

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QApplication::setStyle("Fusion");
    QApplication::setPalette(darkPalette());

    MainWindow window;
    window.show();
    return QApplication::exec();
}
