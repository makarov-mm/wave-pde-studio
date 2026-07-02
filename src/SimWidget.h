#pragma once
#include <QWidget>
#include <QImage>

// Displays the field (scaled to the widget) and reports where the user clicks
// or drags, as normalized [0,1] coordinates:
//   * left-drag            -> injectAt()  (add a pulse / seed)
//   * right-drag           -> wallAt(erase = false)  (paint an obstacle)
//   * Shift + right-drag   -> wallAt(erase = true)   (erase obstacles)
class SimWidget : public QWidget {
    Q_OBJECT
public:
    explicit SimWidget(QWidget* parent = nullptr);
    void setImage(const QImage& img);

signals:
    void injectAt(float nx, float ny);              // normalized [0,1] coords
    void wallAt(float nx, float ny, bool erase);    // normalized [0,1] coords

protected:
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;

private:
    QPointF normalized(const QPointF& pos) const;
    QImage m_image;
    bool   m_drawing     = false;   // left button held
    bool   m_wallDrawing = false;   // right button held
    bool   m_wallErase   = false;
};
