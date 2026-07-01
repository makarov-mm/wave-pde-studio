#include "SimWidget.h"
#include <QPainter>
#include <QMouseEvent>

SimWidget::SimWidget(QWidget* parent) : QWidget(parent) {
    setMinimumSize(320, 320);
    setCursor(Qt::CrossCursor);
    setAttribute(Qt::WA_OpaquePaintEvent, true);
}

void SimWidget::setImage(const QImage& img) {
    m_image = img;
    update();
}

void SimWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), Qt::black);
    if (!m_image.isNull()) {
        // Scale the (grid-resolution) field image to fill the widget.
        p.drawImage(rect(), m_image, m_image.rect());
    }
}

void SimWidget::emitInject(const QPointF& pos) {
    if (width() <= 0 || height() <= 0) return;
    float nx = (float)(pos.x() / width());
    float ny = (float)(pos.y() / height());
    nx = std::min(1.0f, std::max(0.0f, nx));
    ny = std::min(1.0f, std::max(0.0f, ny));
    emit injectAt(nx, ny);
}

void SimWidget::mousePressEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton) {
        m_drawing = true;
        emitInject(e->position());
    }
}

void SimWidget::mouseMoveEvent(QMouseEvent* e) {
    if (m_drawing)
        emitInject(e->position());
}

void SimWidget::mouseReleaseEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton)
        m_drawing = false;
}
