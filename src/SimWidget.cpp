#include "SimWidget.h"
#include <QPainter>
#include <QMouseEvent>
#include <algorithm>

SimWidget::SimWidget(QWidget* parent) : QWidget(parent) {
    setMinimumSize(320, 320);
    setCursor(Qt::CrossCursor);
    setAttribute(Qt::WA_OpaquePaintEvent, true);
    setContextMenuPolicy(Qt::NoContextMenu);   // right button is the wall brush
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

QPointF SimWidget::normalized(const QPointF& pos) const {
    float nx = (width()  > 0) ? (float)(pos.x() / width())  : 0.0f;
    float ny = (height() > 0) ? (float)(pos.y() / height()) : 0.0f;
    nx = std::min(1.0f, std::max(0.0f, nx));
    ny = std::min(1.0f, std::max(0.0f, ny));
    return QPointF(nx, ny);
}

void SimWidget::mousePressEvent(QMouseEvent* e) {
    QPointF n = normalized(e->position());
    if (e->button() == Qt::LeftButton) {
        m_drawing = true;
        emit injectAt((float)n.x(), (float)n.y());
    } else if (e->button() == Qt::RightButton) {
        m_wallDrawing = true;
        m_wallErase   = e->modifiers().testFlag(Qt::ShiftModifier);
        emit wallAt((float)n.x(), (float)n.y(), m_wallErase);
    }
}

void SimWidget::mouseMoveEvent(QMouseEvent* e) {
    QPointF n = normalized(e->position());
    if (m_drawing)
        emit injectAt((float)n.x(), (float)n.y());
    if (m_wallDrawing)
        emit wallAt((float)n.x(), (float)n.y(), m_wallErase);
}

void SimWidget::mouseReleaseEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton)  m_drawing     = false;
    if (e->button() == Qt::RightButton) m_wallDrawing = false;
}
