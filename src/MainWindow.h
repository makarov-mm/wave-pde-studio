#pragma once
#include <QMainWindow>
#include "Types.h"
#include "Simulator.h"

class SimWidget;
class QComboBox;
class QSlider;
class QSpinBox;
class QLabel;
class QPushButton;
class QTimer;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void onTick();                 // advance + render one frame
    void onEquationChanged();
    void onGridChanged();
    void onControlsChanged();
    void onBackendChanged();
    void onPlayPause();
    void onReset();
    void onBenchmark();
    void onInject(float nx, float ny);

private:
    Backend currentBackend() const;
    void    renderNow();           // repaint current field without stepping

    SimParams  m_params;
    Simulator  m_sim;
    QTimer*    m_timer = nullptr;
    bool       m_playing = true;

    SimWidget*   m_canvas       = nullptr;
    QComboBox*   m_eqCombo      = nullptr;
    QComboBox*   m_gridCombo    = nullptr;
    QSlider*     m_speedSlider  = nullptr;
    QSlider*     m_diffSlider   = nullptr;
    QSlider*     m_dampSlider   = nullptr;
    QComboBox*   m_colorCombo   = nullptr;
    QComboBox*   m_backendCombo = nullptr;
    QSpinBox*    m_substepsSpin = nullptr;
    QPushButton* m_playButton   = nullptr;
    QLabel*      m_timingLabel  = nullptr;
    QLabel*      m_benchLabel   = nullptr;

    bool m_suppress = false;
};
