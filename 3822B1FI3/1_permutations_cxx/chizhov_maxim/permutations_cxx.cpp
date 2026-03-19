#include "permutations_cxx.h"
#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& dictionary) {
    std::unordered_map<std::string, std::string> sorted_cache;

    for (const auto& pair : dictionary) {
        std::string sorted = pair.first;
        std::sort(sorted.begin(), sorted.end());
        sorted_cache[pair.first] = sorted;
    }

    std::unordered_map<std::string, std::vector<std::string>> groups;

    for (const auto& pair : dictionary) {
        const std::string& word = pair.first;
        groups[sorted_cache[word]].push_back(word);
    }

    for (auto& pair : dictionary) {
        const std::string& word = pair.first;
        std::vector<std::string>& result = pair.second;

        result.clear();

        const std::vector<std::string>& group = groups[sorted_cache[word]];

        for (size_t i = 0; i < group.size(); i++) {
            if (group[i] != word) {
                result.push_back(group[i]);
            }
        }

        std::sort(result.begin(), result.end(), std::greater<std::string>());
    }
}