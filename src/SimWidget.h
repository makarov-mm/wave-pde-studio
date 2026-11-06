#pragma once
#include <QImage>
#include <QWidget>

// Displays the field, scaled to the widget with its aspect ratio preserved, and
// reports where the user clicks or drags as normalized [0, 1] grid coordinates:
//   * left-drag            -> injectAt()                 (add a pulse / seed)
//   * right-drag           -> wallAt(erase = false)      (paint an obstacle)
//   * Shift + right-drag   -> wallAt(erase = true)       (erase obstacles)
class SimWidget : public QWidget {
    Q_OBJECT
public:
    explicit SimWidget(QWidget* parent = nullptr);
    void setImage(QImage image);

signals:
    void injectAt(float nx, float ny);
    void wallAt(float nx, float ny, bool erase);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    QRect   imageRect() const;                       // where the image is drawn
    QPointF normalized(const QPointF& position) const;

    QImage m_image;
    bool   m_injecting    = false;   // left button held
    bool   m_paintingWall = false;   // right button held
    bool   m_erasingWall  = false;   // Shift was held when the right button went down
};
