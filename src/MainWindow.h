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
    void onSceneChanged();
    void onGridChanged();
    void onControlsChanged();
    void onGsPresetChanged();
    void onGsParamEdited();        // manual feed/kill edit -> preset = Custom
    void onBackendChanged();
    void onPlayPause();
    void onReset();
    void onBenchmark();
    void onExportPng();
    void onInject(float nx, float ny);
    void onWall(float nx, float ny, bool erase);

private:
    Backend currentBackend() const;
    void    renderNow();           // repaint current field without stepping
    void    updateEnabled();       // enable only the sliders the equation uses
    void    updateGsLabels();

    SimParams  m_params;
    Simulator  m_sim;
    QTimer*    m_timer   = nullptr;
    bool       m_playing = true;
    double     m_emaMs   = -1.0;   // smoothed frame step cost

    SimWidget*   m_canvas        = nullptr;
    QComboBox*   m_eqCombo       = nullptr;
    QComboBox*   m_sceneCombo    = nullptr;
    QComboBox*   m_gridCombo     = nullptr;
    QComboBox*   m_boundaryCombo = nullptr;
    QSlider*     m_speedSlider   = nullptr;
    QSlider*     m_diffSlider    = nullptr;
    QSlider*     m_dampSlider    = nullptr;
    QComboBox*   m_gsPresetCombo = nullptr;
    QSlider*     m_feedSlider    = nullptr;
    QSlider*     m_killSlider    = nullptr;
    QLabel*      m_feedLabel     = nullptr;
    QLabel*      m_killLabel     = nullptr;
    QSlider*     m_brushSlider   = nullptr;
    QSlider*     m_ampSlider     = nullptr;
    QComboBox*   m_colorCombo    = nullptr;
    QComboBox*   m_backendCombo  = nullptr;
    QSpinBox*    m_substepsSpin  = nullptr;
    QPushButton* m_playButton    = nullptr;
    QLabel*      m_timingLabel   = nullptr;
    QLabel*      m_benchLabel    = nullptr;

    bool m_suppress = false;
};
