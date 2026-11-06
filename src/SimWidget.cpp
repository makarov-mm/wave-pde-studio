#include "SimWidget.h"

#include <QMouseEvent>
#include <QPainter>

#include <algorithm>
#include <utility>

SimWidget::SimWidget(QWidget* parent) : QWidget(parent) {
    setMinimumSize(320, 320);
    setCursor(Qt::CrossCursor);
    setAttribute(Qt::WA_OpaquePaintEvent, true);
    setContextMenuPolicy(Qt::NoContextMenu);   // the right button is the wall brush
}

void SimWidget::setImage(QImage image) {
    m_image = std::move(image);
    update();
}

QRect SimWidget::imageRect() const {
    if (m_image.isNull())
        return rect();
    QRect target(QPoint(0, 0), m_image.size().scaled(size(), Qt::KeepAspectRatio));
    target.moveCenter(rect().center());
    return target;
}

void SimWidget::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);
    if (!m_image.isNull())
        painter.drawImage(imageRect(), m_image);   // grid resolution scaled up, square cells stay square
}

QPointF SimWidget::normalized(const QPointF& position) const {
    const QRect target = imageRect();
    if (target.isEmpty())
        return {0.0, 0.0};
    return { std::clamp((position.x() - target.x()) / target.width(), 0.0, 1.0),
             std::clamp((position.y() - target.y()) / target.height(), 0.0, 1.0) };
}

void SimWidget::mousePressEvent(QMouseEvent* event) {
    const QPointF n = normalized(event->position());
    if (event->button() == Qt::LeftButton) {
        m_injecting = true;
        emit injectAt(static_cast<float>(n.x()), static_cast<float>(n.y()));
    } else if (event->button() == Qt::RightButton) {
        m_paintingWall = true;
        m_erasingWall  = event->modifiers().testFlag(Qt::ShiftModifier);
        emit wallAt(static_cast<float>(n.x()), static_cast<float>(n.y()), m_erasingWall);
    }
}

void SimWidget::mouseMoveEvent(QMouseEvent* event) {
    const QPointF n = normalized(event->position());
    if (m_injecting)
        emit injectAt(static_cast<float>(n.x()), static_cast<float>(n.y()));
    if (m_paintingWall)
        emit wallAt(static_cast<float>(n.x()), static_cast<float>(n.y()), m_erasingWall);
}

void SimWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton)
        m_injecting = false;
    else if (event->button() == Qt::RightButton)
        m_paintingWall = false;
}
