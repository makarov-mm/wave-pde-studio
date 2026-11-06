#include "MainWindow.h"
#include "SimWidget.h"

#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QFileDialog>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QKeySequence>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QScopeGuard>
#include <QShortcut>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace {

constexpr int kFrameIntervalMs = 16;    // ~60 fps
constexpr int kBenchmarkSteps  = 100;

// Sliders hold integers: parameter = slider value / scale.
constexpr float kWaveSpeedScale = 100.0f;
constexpr float kDiffusionScale = 100.0f;
constexpr float kDampingScale   = 10000.0f;
constexpr float kFeedKillScale  = 10000.0f;
constexpr float kBrushSizeScale = 1000.0f;   // fraction of the grid width
constexpr float kBrushAmpScale  = 100.0f;

// Gray-Scott feed/kill presets, in slider units.
struct GrayScottPreset {
    const char* name;
    int         feed;
    int         kill;
};
constexpr std::array kGrayScottPresets{
    GrayScottPreset{"Coral",   545, 620},
    GrayScottPreset{"Mitosis", 367, 649},
    GrayScottPreset{"Worms",   780, 610},
    GrayScottPreset{"Spots",   300, 620},
    GrayScottPreset{"Chaos",   260, 510},
};
constexpr int kCustomPattern = 0;   // the pattern combo lists "Custom" first, then the presets

constexpr bool isCuda(Backend backend) {
    return backend == Backend::Cuda || backend == Backend::CudaTiled;
}

QSlider* makeSlider(int minimum, int maximum, int value) {
    auto* slider = new QSlider(Qt::Horizontal);
    slider->setRange(minimum, maximum);
    slider->setValue(value);
    return slider;
}

template <class Enum>
void addItem(QComboBox* combo, const QString& text, Enum value) {
    combo->addItem(text, std::to_underlying(value));
}

template <class Enum>
Enum currentValue(const QComboBox* combo) {
    return static_cast<Enum>(combo->currentData().toInt());
}

// A slider with a small live value label to its right.
QWidget* withValueLabel(QSlider* slider, QLabel* label) {
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

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), m_cudaIssue(Simulator::cudaUnavailableReason()) {
    setWindowTitle(m_cudaIssue.isEmpty() ? "Wave / Heat PDE Studio — CPU vs CUDA"
                                         : "Wave / Heat PDE Studio — CPU only");

    m_canvas = new SimWidget;

    auto* central       = new QWidget;
    auto* centralLayout = new QHBoxLayout(central);
    centralLayout->setContentsMargins(0, 0, 0, 0);
    centralLayout->addWidget(createControlPanel());
    centralLayout->addWidget(m_canvas, 1);
    setCentralWidget(central);
    resize(1240, 860);

    connectSignals();

    onControlsChanged();
    updateControlStates();
    resetSimulation();

    auto* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::onTick);
    timer->start(kFrameIntervalMs);
}

QWidget* MainWindow::createControlPanel() {
    m_eqCombo = new QComboBox;
    addItem(m_eqCombo, "Wave equation", Equation::Wave);
    addItem(m_eqCombo, "Heat equation", Equation::Heat);
    addItem(m_eqCombo, "Gray–Scott reaction–diffusion", Equation::GrayScott);

    m_sceneCombo = new QComboBox;
    addItem(m_sceneCombo, "Empty", Scene::Empty);
    addItem(m_sceneCombo, "Double slit", Scene::DoubleSlit);
    addItem(m_sceneCombo, "Ring cavity", Scene::RingCavity);
    addItem(m_sceneCombo, "Pillar lattice", Scene::PillarLattice);

    m_gridCombo = new QComboBox;
    for (const int size : {256, 512, 1024})
        m_gridCombo->addItem(QString("%1 x %1").arg(size), size);
    m_gridCombo->setCurrentIndex(m_gridCombo->findData(m_params.gridW));

    m_boundaryCombo = new QComboBox;
    addItem(m_boundaryCombo, "Fixed (Dirichlet)", Boundary::Dirichlet);
    addItem(m_boundaryCombo, "Reflective (Neumann)", Boundary::Neumann);
    addItem(m_boundaryCombo, "Periodic (wrap)", Boundary::Periodic);

    m_speedSlider = makeSlider(5, 70, 50);   // waveSpeed 0.05..0.70 (stable below 1/sqrt(2))
    m_dampSlider  = makeSlider(0, 50, 5);    // damping 0..0.005
    m_diffSlider  = makeSlider(1, 24, 20);   // diffusion 0.01..0.24 (stable below 0.25)

    m_patternCombo = new QComboBox;
    m_patternCombo->addItem("Custom");
    for (const GrayScottPreset& preset : kGrayScottPresets)
        m_patternCombo->addItem(preset.name);
    m_patternCombo->setCurrentIndex(1);      // Coral

    m_feedSlider = makeSlider(100, 900, kGrayScottPresets[0].feed);   // feed 0.01..0.09
    m_killSlider = makeSlider(400, 750, kGrayScottPresets[0].kill);   // kill 0.04..0.075
    m_feedLabel  = new QLabel;
    m_killLabel  = new QLabel;

    m_brushSlider = makeSlider(5, 100, 25);  // brush radius 0.5%..10% of the grid width
    m_ampSlider   = makeSlider(10, 200, 100);

    m_paletteCombo = new QComboBox;
    addItem(m_paletteCombo, "Thermal / diverging", Palette::Thermal);
    addItem(m_paletteCombo, "Ocean", Palette::Ocean);
    addItem(m_paletteCombo, "Grayscale", Palette::Grayscale);
    addItem(m_paletteCombo, "Viridis", Palette::Viridis);
    addItem(m_paletteCombo, "Magma", Palette::Magma);

    m_substepsSpin = new QSpinBox;
    m_substepsSpin->setRange(1, 50);
    m_substepsSpin->setValue(m_params.substeps);

    m_backendCombo = new QComboBox;
    addItem(m_backendCombo, Simulator::backendName(Backend::CpuSingle), Backend::CpuSingle);
    addItem(m_backendCombo, Simulator::backendName(Backend::CpuMulti), Backend::CpuMulti);
    if (m_cudaIssue.isEmpty()) {
        addItem(m_backendCombo, Simulator::backendName(Backend::Cuda), Backend::Cuda);
        addItem(m_backendCombo, Simulator::backendName(Backend::CudaTiled), Backend::CudaTiled);
        m_backendCombo->setCurrentIndex(m_backendCombo->findData(std::to_underlying(Backend::Cuda)));
    } else {
        m_backendCombo->setToolTip("CUDA unavailable: " + m_cudaIssue);
    }

    auto* form = new QFormLayout;
    form->addRow("Equation",    m_eqCombo);
    form->addRow("Scene",       m_sceneCombo);
    form->addRow("Grid",        m_gridCombo);
    form->addRow("Boundary",    m_boundaryCombo);
    form->addRow("Wave speed",  m_speedSlider);
    form->addRow("Damping",     m_dampSlider);
    form->addRow("Diffusion",   m_diffSlider);
    form->addRow("Pattern",     m_patternCombo);
    form->addRow("Feed F",      withValueLabel(m_feedSlider, m_feedLabel));
    form->addRow("Kill k",      withValueLabel(m_killSlider, m_killLabel));
    form->addRow("Brush size",  m_brushSlider);
    form->addRow("Brush power", m_ampSlider);
    form->addRow("Palette",     m_paletteCombo);
    form->addRow("Steps/frame", m_substepsSpin);
    form->addRow("Backend",     m_backendCombo);
    auto* paramsBox = new QGroupBox("Parameters");
    paramsBox->setLayout(form);

    m_playButton   = new QPushButton("Pause");
    m_resetButton  = new QPushButton("Reset");
    m_benchButton  = new QPushButton("Benchmark all backends");
    m_exportButton = new QPushButton("Export PNG…");

    m_timingLabel = new QLabel("—");
    m_timingLabel->setWordWrap(true);
    m_benchLabel = new QLabel;
    m_benchLabel->setWordWrap(true);
    m_benchLabel->setTextFormat(Qt::RichText);
    if (!m_cudaIssue.isEmpty())
        m_benchLabel->setText("<span style='color:#888'>CUDA unavailable: " + m_cudaIssue.toHtmlEscaped() + "</span>");

    auto* separator = new QFrame;
    separator->setFrameShape(QFrame::HLine);
    auto* timingLayout = new QVBoxLayout;
    timingLayout->addWidget(m_timingLabel);
    timingLayout->addWidget(separator);
    timingLayout->addWidget(m_benchLabel);
    auto* timingBox = new QGroupBox("Timing");
    timingBox->setLayout(timingLayout);

    auto* hint = new QLabel("Left-drag: pulse · Right-drag: wall · "
                            "Shift+Right-drag: erase · Space: pause · R: reset");
    hint->setStyleSheet("color: #888;");
    hint->setWordWrap(true);

    auto* panelLayout = new QVBoxLayout;
    panelLayout->addWidget(paramsBox);
    panelLayout->addWidget(m_playButton);
    panelLayout->addWidget(m_resetButton);
    panelLayout->addWidget(m_benchButton);
    panelLayout->addWidget(m_exportButton);
    panelLayout->addWidget(timingBox);
    panelLayout->addStretch();
    panelLayout->addWidget(hint);

    auto* panel = new QWidget;
    panel->setLayout(panelLayout);
    panel->setFixedWidth(320);
    return panel;
}

void MainWindow::connectSignals() {
    connect(m_eqCombo,       &QComboBox::currentIndexChanged, this, &MainWindow::onEquationChanged);
    connect(m_sceneCombo,    &QComboBox::currentIndexChanged, this, &MainWindow::onSceneChanged);
    connect(m_gridCombo,     &QComboBox::currentIndexChanged, this, &MainWindow::onGridChanged);
    connect(m_boundaryCombo, &QComboBox::currentIndexChanged, this, &MainWindow::onControlsChanged);
    connect(m_paletteCombo,  &QComboBox::currentIndexChanged, this, &MainWindow::onControlsChanged);
    connect(m_backendCombo,  &QComboBox::currentIndexChanged, this, &MainWindow::resetTiming);
    connect(m_patternCombo,  &QComboBox::currentIndexChanged, this, &MainWindow::onGrayScottPresetChanged);
    connect(m_substepsSpin,  &QSpinBox::valueChanged,         this, &MainWindow::onControlsChanged);
    connect(m_feedSlider,    &QSlider::valueChanged,          this, &MainWindow::onGrayScottParamEdited);
    connect(m_killSlider,    &QSlider::valueChanged,          this, &MainWindow::onGrayScottParamEdited);
    for (QSlider* slider : {m_speedSlider, m_diffSlider, m_dampSlider, m_brushSlider, m_ampSlider})
        connect(slider, &QSlider::valueChanged, this, &MainWindow::onControlsChanged);

    connect(m_playButton,   &QPushButton::clicked, this, &MainWindow::onPlayPause);
    connect(m_resetButton,  &QPushButton::clicked, this, &MainWindow::onReset);
    connect(m_benchButton,  &QPushButton::clicked, this, &MainWindow::onBenchmark);
    connect(m_exportButton, &QPushButton::clicked, this, &MainWindow::onExportPng);

    connect(m_canvas, &SimWidget::injectAt, this, &MainWindow::onInject);
    connect(m_canvas, &SimWidget::wallAt,   this, &MainWindow::onWall);

    connect(new QShortcut(QKeySequence(Qt::Key_Space), this), &QShortcut::activated, this, &MainWindow::onPlayPause);
    connect(new QShortcut(QKeySequence(Qt::Key_R), this),     &QShortcut::activated, this, &MainWindow::onReset);
}

Backend MainWindow::currentBackend() const {
    return currentValue<Backend>(m_backendCombo);
}

float MainWindow::brushRadiusCells() const {
    return std::max(2.0f, static_cast<float>(m_params.gridW) * m_params.brushRadius);
}

void MainWindow::resetSimulation() {
    m_sim.reset(m_params);
    renderNow();
}

void MainWindow::renderNow() {
    m_canvas->setImage(m_sim.toImage(m_params));
}

void MainWindow::resetTiming() {
    m_stepMsAverage.reset();
}

void MainWindow::updateControlStates() {
    const bool wave      = m_params.equation == Equation::Wave;
    const bool heat      = m_params.equation == Equation::Heat;
    const bool grayScott = m_params.equation == Equation::GrayScott;
    m_speedSlider->setEnabled(wave);
    m_dampSlider->setEnabled(wave);
    m_diffSlider->setEnabled(heat);
    for (QWidget* widget : std::initializer_list<QWidget*>{m_patternCombo, m_feedSlider, m_killSlider, m_feedLabel, m_killLabel})
        widget->setEnabled(grayScott);
}

void MainWindow::updateGrayScottLabels() {
    m_feedLabel->setText(QString::number(m_feedSlider->value() / 10000.0, 'f', 4));
    m_killLabel->setText(QString::number(m_killSlider->value() / 10000.0, 'f', 4));
}

void MainWindow::onTick() {
    if (!m_playing)
        return;

    const Backend backend = currentBackend();
    StepResult result;
    try {
        result = m_sim.step(m_params, backend);
    } catch (const std::exception& error) {
        if (!isCuda(backend))
            throw;   // the CPU back-ends can only fail by running out of memory
        fallBackToCpu(error);
        return;
    }
    renderNow();

    m_stepMsAverage = m_stepMsAverage ? 0.9 * *m_stepMsAverage + 0.1 * result.milliseconds : result.milliseconds;
    const int    steps     = std::max(1, m_params.substeps);
    const double msPerStep = *m_stepMsAverage / steps;
    const double megaCellsPerSecond =
        msPerStep > 0.0 ? static_cast<double>(m_params.gridW) * m_params.gridH / msPerStep / 1000.0 : 0.0;

    m_timingLabel->setText(QString("%1\n%2 ms / %3 steps\navg %4 ms/step · %5 Mcells/s")
                               .arg(Simulator::backendName(result.backend))
                               .arg(result.milliseconds, 0, 'f', 2)
                               .arg(steps)
                               .arg(msPerStep, 0, 'f', 3)
                               .arg(megaCellsPerSecond, 0, 'f', 0));
}

void MainWindow::fallBackToCpu(const std::exception& cudaError) {
    // A failed launch or transfer can leave the CUDA context unusable, so the GPU
    // back-ends are dropped for the rest of the session.
    m_cudaIssue = QString::fromUtf8(cudaError.what());
    {
        const QSignalBlocker blocker(m_backendCombo);
        for (const Backend backend : {Backend::Cuda, Backend::CudaTiled})
            m_backendCombo->removeItem(m_backendCombo->findData(std::to_underlying(backend)));
        m_backendCombo->setCurrentIndex(m_backendCombo->findData(std::to_underlying(Backend::CpuMulti)));
    }
    m_backendCombo->setToolTip("CUDA unavailable: " + m_cudaIssue);
    resetTiming();
    setWindowTitle("Wave / Heat PDE Studio — CPU only");

    QMessageBox::warning(this, "CUDA back-end failed",
                         m_cudaIssue + "\n\nThe simulation continues on the multi-threaded CPU back-end.");
}

void MainWindow::onControlsChanged() {
    const int previousSubsteps = m_params.substeps;

    m_params.waveSpeed   = static_cast<float>(m_speedSlider->value()) / kWaveSpeedScale;
    m_params.diffusion   = static_cast<float>(m_diffSlider->value()) / kDiffusionScale;
    m_params.damping     = static_cast<float>(m_dampSlider->value()) / kDampingScale;
    m_params.feed        = static_cast<float>(m_feedSlider->value()) / kFeedKillScale;
    m_params.kill        = static_cast<float>(m_killSlider->value()) / kFeedKillScale;
    m_params.brushRadius = static_cast<float>(m_brushSlider->value()) / kBrushSizeScale;
    m_params.brushAmp    = static_cast<float>(m_ampSlider->value()) / kBrushAmpScale;
    m_params.boundary    = currentValue<Boundary>(m_boundaryCombo);
    m_params.palette     = currentValue<Palette>(m_paletteCombo);
    m_params.substeps    = m_substepsSpin->value();

    if (m_params.substeps != previousSubsteps)
        resetTiming();
    updateGrayScottLabels();
    if (!m_playing)
        renderNow();
}

void MainWindow::onGrayScottParamEdited() {
    // Hand-tuned feed/kill values no longer match a named pattern.
    {
        const QSignalBlocker blocker(m_patternCombo);
        m_patternCombo->setCurrentIndex(kCustomPattern);
    }
    onControlsChanged();
}

void MainWindow::onGrayScottPresetChanged() {
    const int index = m_patternCombo->currentIndex();
    if (index <= kCustomPattern)
        return;   // Custom: keep the current sliders

    const GrayScottPreset& preset = kGrayScottPresets[static_cast<std::size_t>(index - 1)];
    {
        const QSignalBlocker feedBlocker(m_feedSlider);
        const QSignalBlocker killBlocker(m_killSlider);
        m_feedSlider->setValue(preset.feed);
        m_killSlider->setValue(preset.kill);
    }
    onControlsChanged();

    // A fresh field shows the new regime much better than a mid-flight switch.
    if (m_params.equation == Equation::GrayScott)
        resetSimulation();
}

void MainWindow::onEquationChanged() {
    m_params.equation = currentValue<Equation>(m_eqCombo);
    updateControlStates();

    // Gray-Scott evolves slowly per step, so it gets a higher default step rate.
    {
        const QSignalBlocker blocker(m_substepsSpin);
        m_substepsSpin->setValue(m_params.equation == Equation::GrayScott ? 10 : 4);
    }
    m_params.substeps = m_substepsSpin->value();

    resetTiming();
    resetSimulation();
}

void MainWindow::onSceneChanged() {
    m_params.scene = currentValue<Scene>(m_sceneCombo);
    resetSimulation();
}

void MainWindow::onGridChanged() {
    m_params.gridW = m_params.gridH = m_gridCombo->currentData().toInt();
    resetTiming();
    resetSimulation();
}

void MainWindow::onPlayPause() {
    m_playing = !m_playing;
    m_playButton->setText(m_playing ? "Pause" : "Play");
}

void MainWindow::onReset() {
    resetSimulation();
}

void MainWindow::onExportPng() {
    const QString suggestedName =
        QString("pde-%1.png").arg(QDateTime::currentDateTime().toString("yyyyMMdd-hhmmss"));
    const QString path = QFileDialog::getSaveFileName(this, "Export image", suggestedName, "PNG image (*.png)");
    if (path.isEmpty())
        return;
    if (!m_sim.toImage(m_params).save(path, "PNG"))
        QMessageBox::warning(this, "Export image", QString("Could not save %1.").arg(QDir::toNativeSeparators(path)));
}

void MainWindow::onInject(float nx, float ny) {
    m_sim.inject(m_params, nx, ny, brushRadiusCells(), m_params.brushAmp);
    if (!m_playing)
        renderNow();
}

void MainWindow::onWall(float nx, float ny, bool erase) {
    m_sim.paintWall(m_params, nx, ny, brushRadiusCells(), erase);
    if (!m_playing)
        renderNow();
}

void MainWindow::onBenchmark() {
    std::vector<Backend> backends{Backend::CpuSingle, Backend::CpuMulti};
    if (m_cudaIssue.isEmpty())
        backends.insert(backends.end(), {Backend::Cuda, Backend::CudaTiled});

    SimParams measured = m_params;
    measured.substeps  = kBenchmarkSteps;
    SimParams warmUp   = m_params;
    warmUp.substeps    = 1;

    QGuiApplication::setOverrideCursor(Qt::WaitCursor);
    const auto restoreCursor = qScopeGuard([] { QGuiApplication::restoreOverrideCursor(); });

    QString report = QString("<b>Benchmark</b> (%1x%2, %3 steps)<br/>")
                         .arg(m_params.gridW)
                         .arg(m_params.gridH)
                         .arg(kBenchmarkSteps);

    const Simulator::Snapshot start = m_sim.snapshot();
    std::optional<double> baseline;   // ms/step of the first (single-threaded) back-end
    for (const Backend backend : backends) {
        try {
            // The first use of a back-end pays one-time costs (thread start-up, CUDA
            // context, device allocations): run one step first to keep them out.
            m_sim.restore(start);
            m_sim.step(warmUp, backend);

            m_sim.restore(start);   // every back-end starts from the same field
            const double msPerStep = m_sim.step(measured, backend).milliseconds / kBenchmarkSteps;
            if (!baseline)
                baseline = msPerStep;
            report += QString("%1: <b>%2 ms/step</b> (%3x)<br/>")
                          .arg(Simulator::backendName(backend))
                          .arg(msPerStep, 0, 'f', 4)
                          .arg(*baseline / msPerStep, 0, 'f', 1);
        } catch (const std::exception& error) {
            report += QString("%1: failed (%2)<br/>")
                          .arg(Simulator::backendName(backend), QString::fromUtf8(error.what()).toHtmlEscaped());
        }
    }
    m_sim.restore(start);

    m_benchLabel->setText(report);
    renderNow();
}
