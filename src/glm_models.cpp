#include "glm_models.h"

// Activation Functions
mx::array silu(mx::array x) {
    return mx::multiply(x, mx::sigmoid(x));
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
