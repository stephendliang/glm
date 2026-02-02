#include "glm_models.h"

// Activation Functions
mx::array silu(mx::array x) {
    return mx::multiply(x, mx::sigmoid(x));
}

mx::array gelu_erf(mx::array x) {
    const float inv_sqrt2 = 0.70710678118f;
    auto erf_arg = mx::multiply(mx::array(inv_sqrt2), x);
    auto one_plus = mx::add(mx::array(1.0f), mx::erf(erf_arg));
    return mx::multiply(mx::multiply(mx::array(0.5f), x), one_plus);
}

// Basic Operations
mx::array fast_linear(mx::array x, const LinearWeights* w) {
    if (w->bias.size() == 0) {
        return mx::matmul(x, w->weight);
    }
    return mx::addmm(w->bias, x, w->weight);
}

mx::array rms_norm(mx::array x, const RMSNormWeights* w, float eps) {
    return mx::fast::rms_norm(x, w->weight, eps);
}

mx::array layer_norm(mx::array x, const LayerNormWeights* w, float eps) {
    return mx::fast::layer_norm(x, w->weight, w->bias, eps);
}
