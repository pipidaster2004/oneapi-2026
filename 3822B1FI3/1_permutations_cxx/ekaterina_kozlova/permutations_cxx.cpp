#include "permutations_cxx.h"
#include <algorithm>
#include <functional>
#include <iterator>
#include <unordered_map>

void Permutations(dictionary_t& dictionary) {
    using ref_t = std::reference_wrapper<const std::string>;
    std::unordered_map<std::string, std::vector<ref_t>> groups;
    groups.reserve(dictionary.size());

    for (const auto& entry : dictionary) {
        const std::string& word = entry.first;
        std::string sorted = word;
        std::sort(sorted.begin(), sorted.end());
        groups[sorted].push_back(std::cref(word));
    }

    for (auto& kv : groups) {
        auto& group = kv.second;
        if (group.size() <= 1) continue;

        std::sort(group.begin(), group.end(),
            [](const ref_t& a, const ref_t& b) {
                return a.get() > b.get();
            });
    }

    for (auto& entry : dictionary) {
        const std::string& word = entry.first;
        std::vector<std::string>& permutations = entry.second;
        std::string sorted = word;
        std::sort(sorted.begin(), sorted.end());
        auto it = groups.find(sorted);
        if (it == groups.end()) continue;

        const auto& group = it->second;
        if (group.size() <= 1) continue;

        permutations.clear();
        permutations.reserve(group.size() - 1);

        for (const auto& r : group) {
            const std::string& cand = r.get();
            if (cand != word) {
                permutations.push_back(cand);
            }
        }
    }
}
