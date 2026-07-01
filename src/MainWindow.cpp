#include "MainWindow.h"
#include "SimWidget.h"

#include <QWidget>
#include <QComboBox>
#include <QSlider>
#include <QSpinBox>
#include <QLabel>
#include <QPushButton>
#include <QGroupBox>
#include <QFormLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFrame>
#include <QTimer>
#include <QList>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    bool cuda = Simulator::cudaSupported();
    setWindowTitle(cuda ? "Wave / Heat PDE Studio — CPU vs CUDA"
                        : "Wave / Heat PDE Studio — CPU only");

    m_suppress = true;

    // ---- Canvas ------------------------------------------------------------
    m_canvas = new SimWidget(this);

    // ---- Controls ----------------------------------------------------------
    m_eqCombo = new QComboBox;
    m_eqCombo->addItem("Wave equation");
    m_eqCombo->addItem("Heat equation");

    m_gridCombo = new QComboBox;
    m_gridCombo->addItem("256 x 256", 256);
    m_gridCombo->addItem("512 x 512", 512);
    m_gridCombo->addItem("1024 x 1024", 1024);
    m_gridCombo->setCurrentIndex(1);

    m_speedSlider = new QSlider(Qt::Horizontal);
    m_speedSlider->setRange(5, 70);        // waveSpeed = value / 100
    m_speedSlider->setValue(50);

    m_diffSlider = new QSlider(Qt::Horizontal);
    m_diffSlider->setRange(1, 24);         // diffusion = value / 100
    m_diffSlider->setValue(20);
    m_diffSlider->setEnabled(false);

    m_dampSlider = new QSlider(Qt::Horizontal);
    m_dampSlider->setRange(0, 50);         // damping = value / 10000
    m_dampSlider->setValue(5);

    m_colorCombo = new QComboBox;
    m_colorCombo->addItem("Thermal / diverging");
    m_colorCombo->addItem("Ocean");
    m_colorCombo->addItem("Grayscale");

    m_substepsSpin = new QSpinBox;
    m_substepsSpin->setRange(1, 50);
    m_substepsSpin->setValue(m_params.substeps);

    m_backendCombo = new QComboBox;
    m_backendCombo->addItem("CPU (single-threaded)", (int)Backend::CpuSingle);
    m_backendCombo->addItem("CPU (multi-threaded)",  (int)Backend::CpuMulti);
    if (cuda) m_backendCombo->addItem("CUDA (GPU)", (int)Backend::Cuda);
    int cudaIdx = m_backendCombo->findData((int)Backend::Cuda);
    if (cudaIdx >= 0) m_backendCombo->setCurrentIndex(cudaIdx);

    auto* form = new QFormLayout;
    form->addRow("Equation",   m_eqCombo);
    form->addRow("Grid",       m_gridCombo);
    form->addRow("Wave speed", m_speedSlider);
    form->addRow("Diffusion",  m_diffSlider);
    form->addRow("Damping",    m_dampSlider);
    form->addRow("Palette",    m_colorCombo);
    form->addRow("Steps/frame",m_substepsSpin);
    form->addRow("Backend",    m_backendCombo);
    auto* paramsBox = new QGroupBox("Parameters");
    paramsBox->setLayout(form);

    m_playButton      = new QPushButton("Pause");
    auto* resetButton = new QPushButton("Reset");
    auto* benchButton = new QPushButton("Benchmark all backends");

    m_timingLabel = new QLabel("—");
    m_timingLabel->setWordWrap(true);
    m_benchLabel  = new QLabel();
    m_benchLabel->setWordWrap(true);
    m_benchLabel->setTextFormat(Qt::RichText);

    auto* statusBox    = new QGroupBox("Timing");
    auto* statusLayout = new QVBoxLayout;
    statusLayout->addWidget(m_timingLabel);
    auto* line = new QFrame; line->setFrameShape(QFrame::HLine);
    statusLayout->addWidget(line);
    statusLayout->addWidget(m_benchLabel);
    statusBox->setLayout(statusLayout);

    auto* hint = new QLabel("Click / drag on the canvas to add a pulse");
    hint->setStyleSheet("color: #888;");
    hint->setWordWrap(true);

    auto* panelLayout = new QVBoxLayout;
    panelLayout->addWidget(paramsBox);
    panelLayout->addWidget(m_playButton);
    panelLayout->addWidget(resetButton);
    panelLayout->addWidget(benchButton);
    panelLayout->addWidget(statusBox);
    panelLayout->addStretch();
    panelLayout->addWidget(hint);

    auto* panel = new QWidget;
    panel->setLayout(panelLayout);
    panel->setFixedWidth(300);

    auto* central       = new QWidget;
    auto* centralLayout = new QHBoxLayout(central);
    centralLayout->setContentsMargins(0, 0, 0, 0);
    centralLayout->addWidget(panel);
    centralLayout->addWidget(m_canvas, 1);
    setCentralWidget(central);
    resize(1200, 820);

    // ---- Wiring ------------------------------------------------------------
    connect(m_eqCombo,      &QComboBox::currentIndexChanged, this, &MainWindow::onEquationChanged);
    connect(m_gridCombo,    &QComboBox::currentIndexChanged, this, &MainWindow::onGridChanged);
    connect(m_colorCombo,   &QComboBox::currentIndexChanged, this, &MainWindow::onControlsChanged);
    connect(m_backendCombo, &QComboBox::currentIndexChanged, this, &MainWindow::onBackendChanged);
    connect(m_substepsSpin, &QSpinBox::valueChanged,         this, &MainWindow::onControlsChanged);
    connect(m_speedSlider,  &QSlider::valueChanged,          this, &MainWindow::onControlsChanged);
    connect(m_diffSlider,   &QSlider::valueChanged,          this, &MainWindow::onControlsChanged);
    connect(m_dampSlider,   &QSlider::valueChanged,          this, &MainWindow::onControlsChanged);
    connect(m_playButton,   &QPushButton::clicked,           this, &MainWindow::onPlayPause);
    connect(resetButton,    &QPushButton::clicked,           this, &MainWindow::onReset);
    connect(benchButton,    &QPushButton::clicked,           this, &MainWindow::onBenchmark);
    connect(m_canvas,       &SimWidget::injectAt,            this, &MainWindow::onInject);

    m_suppress = false;

    // Initialize state and start the animation loop (~60 fps).
    m_sim.reset(m_params);
    renderNow();
    m_timer = new QTimer(this);
    connect(m_timer, &QTimer::timeout, this, &MainWindow::onTick);
    m_timer->start(16);
}

Backend MainWindow::currentBackend() const {
    return static_cast<Backend>(m_backendCombo->currentData().toInt());
}

void MainWindow::renderNow() {
    m_canvas->setImage(m_sim.toImage(m_params));
}

void MainWindow::onTick() {
    if (!m_playing) return;
    StepResult r = m_sim.step(m_params, currentBackend());
    renderNow();
    double perStep = r.milliseconds / std::max(1, m_params.substeps);
    m_timingLabel->setText(QString("%1\n%2 ms / %3 steps  (%4 ms/step)")
                               .arg(r.backendName)
                               .arg(QString::number(r.milliseconds, 'f', 2))
                               .arg(m_params.substeps)
                               .arg(QString::number(perStep, 'f', 3)));
}

void MainWindow::onControlsChanged() {
    if (m_suppress) return;
    m_params.waveSpeed   = m_speedSlider->value() / 100.0f;
    m_params.diffusion   = m_diffSlider->value()  / 100.0f;
    m_params.damping     = m_dampSlider->value()  / 10000.0f;
    m_params.colorScheme = m_colorCombo->currentIndex();
    m_params.substeps    = m_substepsSpin->value();
    if (!m_playing) renderNow();
}

void MainWindow::onEquationChanged() {
    m_params.equation = (m_eqCombo->currentIndex() == 1) ? Equation::Heat : Equation::Wave;
    bool wave = (m_params.equation == Equation::Wave);
    m_speedSlider->setEnabled(wave);
    m_dampSlider->setEnabled(wave);
    m_diffSlider->setEnabled(!wave);
    m_sim.reset(m_params);
    renderNow();
}

void MainWindow::onGridChanged() {
    m_params.gridW = m_params.gridH = m_gridCombo->currentData().toInt();
    m_sim.reset(m_params);
    renderNow();
}

void MainWindow::onBackendChanged() {
    // No state change; the next tick simply uses the new backend.
}

void MainWindow::onPlayPause() {
    m_playing = !m_playing;
    m_playButton->setText(m_playing ? "Pause" : "Play");
}

void MainWindow::onReset() {
    m_sim.reset(m_params);
    renderNow();
}

void MainWindow::onInject(float nx, float ny) {
    float radius    = m_params.gridW * 0.025f;
    float amplitude = (m_params.equation == Equation::Wave) ? 1.0f : 1.0f;
    m_sim.inject(m_params, nx, ny, radius, amplitude);
    if (!m_playing) renderNow();
}

void MainWindow::onBenchmark() {
    QList<Backend> backends { Backend::CpuSingle, Backend::CpuMulti };
    if (Simulator::cudaSupported())
        backends << Backend::Cuda;

    const int benchSteps = 100;
    int savedSub = m_params.substeps;
    m_params.substeps = benchSteps;   // advance a fixed batch per timed run

    QString text = QString("<b>Benchmark</b> (%1x%2, %3 steps)<br/>")
                       .arg(m_params.gridW).arg(m_params.gridH).arg(benchSteps);

    double ref = -1.0;
    m_sim.saveState();
    for (Backend b : backends) {
        m_sim.restoreState();                 // every backend starts from the same field
        StepResult r = m_sim.step(m_params, b);
        double perStep = r.milliseconds / benchSteps;
        if (b == Backend::CpuSingle) ref = perStep;
        double sp = (ref > 0.0) ? ref / perStep : 1.0;
        text += QString("%1: <b>%2 ms/step</b> (%3x)<br/>")
                    .arg(r.backendName)
                    .arg(QString::number(perStep, 'f', 4))
                    .arg(QString::number(sp, 'f', 1));
    }
    m_sim.restoreState();
    m_params.substeps = savedSub;

    m_benchLabel->setText(text);
    renderNow();
}
