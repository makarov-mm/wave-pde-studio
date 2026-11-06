#pragma once
#include "Types.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

// Maps field values to 0xAARRGGBB pixels. Coloring always runs on the host, so
// every back-end's frame is colored by the same code.
namespace colormap {

// Clamps to [0, 1]; NaN maps to 0.
inline float saturate(float value) {
    return std::fmin(1.0f, std::fmax(0.0f, value));
}

inline std::uint32_t pack(float r, float g, float b) {
    const auto channel = [](float c) { return static_cast<std::uint32_t>(saturate(c) * 255.0f + 0.5f); };
    return 0xFF000000u | (channel(r) << 16) | (channel(g) << 8) | channel(b);
}

// Piecewise-linear lookup over control points given in 0..255.
using ControlPoints = std::array<std::array<float, 3>, 9>;

inline std::uint32_t lookup(float t, const ControlPoints& points) {
    constexpr int n = static_cast<int>(std::tuple_size_v<ControlPoints>);
    t = saturate(t);
    const float f = t * static_cast<float>(n - 1);
    int i = static_cast<int>(f);
    if (i > n - 2) i = n - 2;
    const float s = f - static_cast<float>(i);
    const auto& a = points[static_cast<std::size_t>(i)];
    const auto& b = points[static_cast<std::size_t>(i) + 1];
    return pack((a[0] + (b[0] - a[0]) * s) / 255.0f,
                (a[1] + (b[1] - a[1]) * s) / 255.0f,
                (a[2] + (b[2] - a[2]) * s) / 255.0f);
}

// Piecewise-linear approximations of the matplotlib colormaps.
inline constexpr ControlPoints kViridis{{ { 68,   1,  84}, { 72,  40, 120}, { 62,  74, 137},
                                          { 49, 104, 142}, { 38, 130, 142}, { 31, 158, 137},
                                          { 53, 183, 121}, {109, 205,  89}, {253, 231,  37} }};

inline constexpr ControlPoints kMagma{{ {  0,   0,   4}, { 28,  16,  68}, { 79,  18, 123},
                                        {129,  37, 129}, {181,  54, 122}, {229,  80, 100},
                                        {251, 135,  97}, {254, 194, 135}, {252, 253, 191} }};

// `diverging` means the field is signed (wave crests and troughs); otherwise it is
// non-negative (heat, Gray-Scott).
inline std::uint32_t colorize(float v, bool diverging, Palette palette) {
    if (palette == Palette::Viridis || palette == Palette::Magma) {
        const float t = diverging ? 0.5f + 0.5f * std::fmax(-1.0f, std::fmin(1.0f, v)) : saturate(v);
        return lookup(t, palette == Palette::Viridis ? kViridis : kMagma);
    }

    if (diverging) {
        const float pos = saturate(v);
        const float neg = saturate(-v);
        switch (palette) {
        case Palette::Ocean:        // cyan / orange
            return pack(pos + 0.1f * neg, 0.5f * (pos + neg), neg + 0.1f * pos);
        case Palette::Grayscale: {  // signed grayscale
            const float s = 0.5f + 0.5f * std::fmax(-1.0f, std::fmin(1.0f, v));
            return pack(s, s, s);
        }
        default:                    // thermal: red (+) / blue (-)
            return pack(pos, 0.15f * (pos + neg), neg);
        }
    }

    const float s = saturate(v);
    switch (palette) {
    case Palette::Ocean:
        return pack(s * s, s, 0.4f + 0.6f * std::sqrt(s));
    case Palette::Grayscale:
        return pack(s, s, s);
    default:                        // thermal: black -> red -> yellow -> white
        return pack(std::fmin(1.0f, s * 3.0f),
                    std::fmin(1.0f, std::fmax(0.0f, s * 3.0f - 1.0f)),
                    std::fmin(1.0f, std::fmax(0.0f, s * 3.0f - 2.0f)));
    }
}

} // namespace colormap
