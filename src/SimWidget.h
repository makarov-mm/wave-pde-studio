#pragma once
#include <QWidget>
#include <QImage>

// Displays the field (scaled to the widget) and reports where the user clicks or
// drags, as normalized [0,1] coordinates, so the MainWindow can inject a pulse.
class SimWidget : public QWidget {
    Q_OBJECT
public:
    explicit SimWidget(QWidget* parent = nullptr);
    void setImage(const QImage& img);

signals:
    void injectAt(float nx, float ny);   // normalized coordinates in [0,1]

protected:
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;

private:
    void emitInject(const QPointF& pos);
    QImage m_image;
    bool   m_drawing = false;
};
