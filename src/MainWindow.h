#pragma once
#include "Simulator.h"
#include "Types.h"

#include <QMainWindow>
#include <QString>

#include <exception>
#include <optional>

class QComboBox;
class QLabel;
class QPushButton;
class QSlider;
class QSpinBox;
class SimWidget;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

private:
    QWidget* createControlPanel();
    void     connectSignals();

    void onTick();                     // advance + render one frame
    void onEquationChanged();
    void onSceneChanged();
    void onGridChanged();
    void onControlsChanged();          // pull slider/combo values into m_params
    void onGrayScottPresetChanged();
    void onGrayScottParamEdited();     // manual feed/kill edit -> pattern = Custom
    void onPlayPause();
    void onReset();
    void onBenchmark();
    void onExportPng();
    void onInject(float nx, float ny);
    void onWall(float nx, float ny, bool erase);

    Backend currentBackend() const;
    float   brushRadiusCells() const;
    void    resetSimulation();         // rebuild the field from m_params and show it
    void    renderNow();               // show the current field without stepping
    void    resetTiming();             // restart the moving average (new workload)
    void    updateControlStates();     // enable only the controls the equation uses
    void    updateGrayScottLabels();
    void    fallBackToCpu(const std::exception& cudaError);

    SimParams m_params;
    Simulator m_sim;
    QString   m_cudaIssue;                    // why CUDA can't be used; empty if it can
    bool      m_playing = true;
    std::optional<double> m_stepMsAverage;    // moving average of step() wall time

    SimWidget*   m_canvas        = nullptr;
    QComboBox*   m_eqCombo       = nullptr;
    QComboBox*   m_sceneCombo    = nullptr;
    QComboBox*   m_gridCombo     = nullptr;
    QComboBox*   m_boundaryCombo = nullptr;
    QSlider*     m_speedSlider   = nullptr;
    QSlider*     m_diffSlider    = nullptr;
    QSlider*     m_dampSlider    = nullptr;
    QComboBox*   m_patternCombo  = nullptr;
    QSlider*     m_feedSlider    = nullptr;
    QSlider*     m_killSlider    = nullptr;
    QLabel*      m_feedLabel     = nullptr;
    QLabel*      m_killLabel     = nullptr;
    QSlider*     m_brushSlider   = nullptr;
    QSlider*     m_ampSlider     = nullptr;
    QComboBox*   m_paletteCombo  = nullptr;
    QComboBox*   m_backendCombo  = nullptr;
    QSpinBox*    m_substepsSpin  = nullptr;
    QPushButton* m_playButton    = nullptr;
    QPushButton* m_resetButton   = nullptr;
    QPushButton* m_benchButton   = nullptr;
    QPushButton* m_exportButton  = nullptr;
    QLabel*      m_timingLabel   = nullptr;
    QLabel*      m_benchLabel    = nullptr;
};
