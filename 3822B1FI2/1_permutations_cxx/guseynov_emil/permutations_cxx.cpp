#include "permutations_cxx.h"
#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& dict) {
    std::unordered_map<std::string, std::vector<std::string>> groups;

    for (auto const& item : dict) {
        const std::string& word = item.first;
        std::string sorted_word = word;
        std::sort(sorted_word.begin(), sorted_word.end());
        
        groups[sorted_word].push_back(word);
    }

    for (auto& item : dict) {
        const std::string& word = item.first;
        std::vector<std::string>& res = item.second;

        std::string sorted_word = word;
        std::sort(sorted_word.begin(), sorted_word.end());

        const auto& candidates = groups[sorted_word];

        for (const auto& variant : candidates) {
            if (variant != word) {
                res.push_back(variant);
            }
        }

        std::sort(res.rbegin(), res.rend());
    }
}