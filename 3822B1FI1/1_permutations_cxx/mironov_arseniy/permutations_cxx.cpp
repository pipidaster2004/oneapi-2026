#include "permutations_cxx.h"
#include <algorithm>

void Permutations(dictionary_t& dictionary) {

    dictionary_t fp;
    for (auto& [base, vec]: dictionary) {
        std::string sorted_base = base;
        std::sort(sorted_base.begin(), sorted_base.end());
        fp[sorted_base].push_back(base);
    }

    for (auto& [base, vec]: dictionary) {
        std::string sorted_base = base;
        std::sort(sorted_base.begin(), sorted_base.end());
        for (auto& el: fp[sorted_base]) {
            if (el != base)
                vec.push_back(el);
        }
    }

    for (auto &[base, vec]: dictionary) {
        std::sort(vec.rbegin(), vec.rend());
    }
}
