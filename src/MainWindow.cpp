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
#include <QFileDialog>
#include <QDateTime>
#include <QShortcut>
#include <QKeySequence>
#include <algorithm>

namespace {

// Gray-Scott feed/kill presets, in slider units (value / 10000).
struct GsPreset { const char* name; int feed; int kill; };
const GsPreset kGsPresets[] = {
    { "Custom",  0,   0   },
    { "Coral",   545, 620 },
    { "Mitosis", 367, 649 },
    { "Worms",   780, 610 },
    { "Spots",   300, 620 },
    { "Chaos",   260, 510 },
};

// A slider with a small live value label to its right.
QWidget* withValueLabel(QSlider* slider, QLabel*& label) {
    label = new QLabel;
    label->setMinimumWidth(44);
    label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    auto* row    = new QWidget;
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(slider, 1);
    layout->addWidget(label);
    return row;
}

} // namespace

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
    m_eqCombo->addItem("Gray\u2013Scott reaction\u2013diffusion");

    m_sceneCombo = new QComboBox;
    m_sceneCombo->addItem("Empty");
    m_sceneCombo->addItem("Double slit");
    m_sceneCombo->addItem("Ring cavity");
    m_sceneCombo->addItem("Pillar lattice");

    m_gridCombo = new QComboBox;
    m_gridCombo->addItem("256 x 256", 256);
    m_gridCombo->addItem("512 x 512", 512);
    m_gridCombo->addItem("1024 x 1024", 1024);
    m_gridCombo->setCurrentIndex(1);

    m_boundaryCombo = new QComboBox;
    m_boundaryCombo->addItem("Fixed (Dirichlet)");
    m_boundaryCombo->addItem("Reflective (Neumann)");
    m_boundaryCombo->addItem("Periodic (wrap)");

    m_speedSlider = new QSlider(Qt::Horizontal);
    m_speedSlider->setRange(5, 70);        // waveSpeed = value / 100
    m_speedSlider->setValue(50);

    m_diffSlider = new QSlider(Qt::Horizontal);
    m_diffSlider->setRange(1, 24);         // diffusion = value / 100
    m_diffSlider->setValue(20);

    m_dampSlider = new QSlider(Qt::Horizontal);
    m_dampSlider->setRange(0, 50);         // damping = value / 10000
    m_dampSlider->setValue(5);

    m_gsPresetCombo = new QComboBox;
    for (const auto& pr : kGsPresets)
        m_gsPresetCombo->addItem(pr.name);
    m_gsPresetCombo->setCurrentIndex(1);   // Coral

    m_feedSlider = new QSlider(Qt::Horizontal);
    m_feedSlider->setRange(100, 900);      // feed = value / 10000
    m_feedSlider->setValue(kGsPresets[1].feed);

    m_killSlider = new QSlider(Qt::Horizontal);
    m_killSlider->setRange(400, 750);      // kill = value / 10000
    m_killSlider->setValue(kGsPresets[1].kill);

    m_brushSlider = new QSlider(Qt::Horizontal);
    m_brushSlider->setRange(5, 100);       // brushRadius = value / 1000 (of width)
    m_brushSlider->setValue(25);

    m_ampSlider = new QSlider(Qt::Horizontal);
    m_ampSlider->setRange(10, 200);        // brushAmp = value / 100
    m_ampSlider->setValue(100);

    m_colorCombo = new QComboBox;
    m_colorCombo->addItem("Thermal / diverging");
    m_colorCombo->addItem("Ocean");
    m_colorCombo->addItem("Grayscale");
    m_colorCombo->addItem("Viridis");
    m_colorCombo->addItem("Magma");

    m_substepsSpin = new QSpinBox;
    m_substepsSpin->setRange(1, 50);
    m_substepsSpin->setValue(m_params.substeps);

    m_backendCombo = new QComboBox;
    m_backendCombo->addItem("CPU (single-threaded)", (int)Backend::CpuSingle);
    m_backendCombo->addItem("CPU (multi-threaded)",  (int)Backend::CpuMulti);
    if (cuda) {
        m_backendCombo->addItem("CUDA (GPU)",        (int)Backend::Cuda);
        m_backendCombo->addItem("CUDA (GPU, tiled)", (int)Backend::CudaTiled);
    }
    int cudaIdx = m_backendCombo->findData((int)Backend::Cuda);
    if (cudaIdx >= 0) m_backendCombo->setCurrentIndex(cudaIdx);

    auto* form = new QFormLayout;
    form->addRow("Equation",    m_eqCombo);
    form->addRow("Scene",       m_sceneCombo);
    form->addRow("Grid",        m_gridCombo);
    form->addRow("Boundary",    m_boundaryCombo);
    form->addRow("Wave speed",  m_speedSlider);
    form->addRow("Damping",     m_dampSlider);
    form->addRow("Diffusion",   m_diffSlider);
    form->addRow("Pattern",     m_gsPresetCombo);
    form->addRow("Feed F",      withValueLabel(m_feedSlider, m_feedLabel));
    form->addRow("Kill k",      withValueLabel(m_killSlider, m_killLabel));
    form->addRow("Brush size",  m_brushSlider);
    form->addRow("Brush power", m_ampSlider);
    form->addRow("Palette",     m_colorCombo);
    form->addRow("Steps/frame", m_substepsSpin);
    form->addRow("Backend",     m_backendCombo);
    auto* paramsBox = new QGroupBox("Parameters");
    paramsBox->setLayout(form);

    m_playButton       = new QPushButton("Pause");
    auto* resetButton  = new QPushButton("Reset");
    auto* benchButton  = new QPushButton("Benchmark all backends");
    auto* exportButton = new QPushButton("Export PNG\u2026");

    m_timingLabel = new QLabel("\u2014");
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

    auto* hint = new QLabel("Left-drag: pulse \u00b7 Right-drag: wall \u00b7 "
                            "Shift+Right-drag: erase \u00b7 Space: pause \u00b7 R: reset");
    hint->setStyleSheet("color: #888;");
    hint->setWordWrap(true);

    auto* panelLayout = new QVBoxLayout;
    panelLayout->addWidget(paramsBox);
    panelLayout->addWidget(m_playButton);
    panelLayout->addWidget(resetButton);
    panelLayout->addWidget(benchButton);
    panelLayout->addWidget(exportButton);
    panelLayout->addWidget(statusBox);
    panelLayout->addStretch();
    panelLayout->addWidget(hint);

    auto* panel = new QWidget;
    panel->setLayout(panelLayout);
    panel->setFixedWidth(320);

    auto* central       = new QWidget;
    auto* centralLayout = new QHBoxLayout(central);
    centralLayout->setContentsMargins(0, 0, 0, 0);
    centralLayout->addWidget(panel);
    centralLayout->addWidget(m_canvas, 1);
    setCentralWidget(central);
    resize(1240, 860);

    // ---- Wiring ------------------------------------------------------------
    connect(m_eqCombo,       &QComboBox::currentIndexChanged, this, &MainWindow::onEquationChanged);
    connect(m_sceneCombo,    &QComboBox::currentIndexChanged, this, &MainWindow::onSceneChanged);
    connect(m_gridCombo,     &QComboBox::currentIndexChanged, this, &MainWindow::onGridChanged);
    connect(m_boundaryCombo, &QComboBox::currentIndexChanged, this, &MainWindow::onControlsChanged);
    connect(m_colorCombo,    &QComboBox::currentIndexChanged, this, &MainWindow::onControlsChanged);
    connect(m_backendCombo,  &QComboBox::currentIndexChanged, this, &MainWindow::onBackendChanged);
    connect(m_gsPresetCombo, &QComboBox::currentIndexChanged, this, &MainWindow::onGsPresetChanged);
    connect(m_substepsSpin,  &QSpinBox::valueChanged,         this, &MainWindow::onControlsChanged);
    connect(m_speedSlider,   &QSlider::valueChanged,          this, &MainWindow::onControlsChanged);
    connect(m_diffSlider,    &QSlider::valueChanged,          this, &MainWindow::onControlsChanged);
    connect(m_dampSlider,    &QSlider::valueChanged,          this, &MainWindow::onControlsChanged);
    connect(m_feedSlider,    &QSlider::valueChanged,          this, &MainWindow::onGsParamEdited);
    connect(m_killSlider,    &QSlider::valueChanged,          this, &MainWindow::onGsParamEdited);
    connect(m_brushSlider,   &QSlider::valueChanged,          this, &MainWindow::onControlsChanged);
    connect(m_ampSlider,     &QSlider::valueChanged,          this, &MainWindow::onControlsChanged);
    connect(m_playButton,    &QPushButton::clicked,           this, &MainWindow::onPlayPause);
    connect(resetButton,     &QPushButton::clicked,           this, &MainWindow::onReset);
    connect(benchButton,     &QPushButton::clicked,           this, &MainWindow::onBenchmark);
    connect(exportButton,    &QPushButton::clicked,           this, &MainWindow::onExportPng);
    connect(m_canvas,        &SimWidget::injectAt,            this, &MainWindow::onInject);
    connect(m_canvas,        &SimWidget::wallAt,              this, &MainWindow::onWall);

    new QShortcut(QKeySequence(Qt::Key_Space), this, SLOT(onPlayPause()));
    new QShortcut(QKeySequence(Qt::Key_R),     this, SLOT(onReset()));

    m_suppress = false;

    // Initialize state and start the animation loop (~60 fps).
    onControlsChanged();          // pull slider values into m_params
    updateEnabled();
    updateGsLabels();
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

void MainWindow::updateEnabled() {
    bool wave = (m_params.equation == Equation::Wave);
    bool heat = (m_params.equation == Equation::Heat);
    bool gs   = (m_params.equation == Equation::GrayScott);
    m_speedSlider->setEnabled(wave);
    m_dampSlider->setEnabled(wave);
    m_diffSlider->setEnabled(heat);
    m_gsPresetCombo->setEnabled(gs);
    m_feedSlider->setEnabled(gs);
    m_killSlider->setEnabled(gs);
    m_feedLabel->setEnabled(gs);
    m_killLabel->setEnabled(gs);
}

void MainWindow::updateGsLabels() {
    m_feedLabel->setText(QString::number(m_feedSlider->value() / 10000.0, 'f', 4));
    m_killLabel->setText(QString::number(m_killSlider->value() / 10000.0, 'f', 4));
}

void MainWindow::onTick() {
    if (!m_playing) return;
    StepResult r = m_sim.step(m_params, currentBackend());
    renderNow();

    m_emaMs = (m_emaMs < 0.0) ? r.milliseconds
                              : 0.9 * m_emaMs + 0.1 * r.milliseconds;
    int    sub     = std::max(1, m_params.substeps);
    double perStep = m_emaMs / sub;
    double mcells  = (perStep > 0.0)
        ? (double)m_params.gridW * m_params.gridH / perStep / 1000.0   // Mcells/s
        : 0.0;
    m_timingLabel->setText(QString("%1\n%2 ms / %3 steps\navg %4 ms/step \u00b7 %5 Mcells/s")
                               .arg(r.backendName)
                               .arg(QString::number(r.milliseconds, 'f', 2))
                               .arg(sub)
                               .arg(QString::number(perStep, 'f', 3))
                               .arg(QString::number(mcells, 'f', 0)));
}

void MainWindow::onControlsChanged() {
    if (m_suppress) return;
    m_params.waveSpeed   = m_speedSlider->value() / 100.0f;
    m_params.diffusion   = m_diffSlider->value()  / 100.0f;
    m_params.damping     = m_dampSlider->value()  / 10000.0f;
    m_params.feed        = m_feedSlider->value()  / 10000.0f;
    m_params.kill        = m_killSlider->value()  / 10000.0f;
    m_params.brushRadius = m_brushSlider->value() / 1000.0f;
    m_params.brushAmp    = m_ampSlider->value()   / 100.0f;
    m_params.boundary    = m_boundaryCombo->currentIndex();
    m_params.colorScheme = m_colorCombo->currentIndex();
    m_params.substeps    = m_substepsSpin->value();
    updateGsLabels();
    if (!m_playing) renderNow();
}

void MainWindow::onGsParamEdited() {
    if (m_suppress) return;
    // Manual feed/kill tweaking leaves the named presets: show "Custom".
    m_suppress = true;
    m_gsPresetCombo->setCurrentIndex(0);
    m_suppress = false;
    onControlsChanged();
}

void MainWindow::onGsPresetChanged() {
    if (m_suppress) return;
    int idx = m_gsPresetCombo->currentIndex();
    if (idx <= 0) return;                       // Custom: keep current sliders
    m_suppress = true;
    m_feedSlider->setValue(kGsPresets[idx].feed);
    m_killSlider->setValue(kGsPresets[idx].kill);
    m_suppress = false;
    onControlsChanged();
    // A fresh field shows the new regime much better than a mid-flight switch.
    if (m_params.equation == Equation::GrayScott) {
        m_sim.reset(m_params);
        renderNow();
    }
}

void MainWindow::onEquationChanged() {
    if (m_suppress) return;
    m_params.equation = static_cast<Equation>(m_eqCombo->currentIndex());
    updateEnabled();

    // Gray-Scott evolves slowly per step; give it a higher default step rate.
    m_suppress = true;
    m_substepsSpin->setValue(m_params.equation == Equation::GrayScott ? 10 : 4);
    m_suppress = false;
    m_params.substeps = m_substepsSpin->value();

    m_sim.reset(m_params);
    renderNow();
}

void MainWindow::onSceneChanged() {
    if (m_suppress) return;
    m_params.scene = m_sceneCombo->currentIndex();
    m_sim.reset(m_params);
    renderNow();
}

void MainWindow::onGridChanged() {
    if (m_suppress) return;
    m_params.gridW = m_params.gridH = m_gridCombo->currentData().toInt();
    m_sim.reset(m_params);
    renderNow();
}

void MainWindow::onBackendChanged() {
    m_emaMs = -1.0;   // restart the moving average for the new backend
}

void MainWindow::onPlayPause() {
    m_playing = !m_playing;
    m_playButton->setText(m_playing ? "Pause" : "Play");
}

void MainWindow::onReset() {
    m_sim.reset(m_params);
    renderNow();
}

void MainWindow::onExportPng() {
    QString name = QString("pde-%1.png")
                       .arg(QDateTime::currentDateTime().toString("yyyyMMdd-hhmmss"));
    QString path = QFileDialog::getSaveFileName(this, "Export image", name,
                                                "PNG image (*.png)");
    if (path.isEmpty()) return;
    m_sim.toImage(m_params).save(path, "PNG");
}

void MainWindow::onInject(float nx, float ny) {
    float radius = std::max(2.0f, m_params.gridW * m_params.brushRadius);
    m_sim.inject(m_params, nx, ny, radius, m_params.brushAmp);
    if (!m_playing) renderNow();
}

void MainWindow::onWall(float nx, float ny, bool erase) {
    float radius = std::max(2.0f, m_params.gridW * m_params.brushRadius);
    m_sim.setWall(m_params, nx, ny, radius, erase);
    if (!m_playing) renderNow();
}

void MainWindow::onBenchmark() {
    QList<Backend> backends { Backend::CpuSingle, Backend::CpuMulti };
    if (Simulator::cudaSupported())
        backends << Backend::Cuda << Backend::CudaTiled;

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
