#include "permutations_cxx.h"

void Permutations(dictionary_t& dictionary) {
    if (dictionary.empty()) return;
    std::unordered_map<std::string, std::vector<std::string>> groups;
    for (const auto& pair : dictionary) {
        std::string sorted = pair.first;
        std::sort(sorted.begin(), sorted.end());
        groups[sorted].push_back(pair.first);
    }
    for (auto& group : groups) {
        std::sort(group.second.begin(), group.second.end(),
                  std::greater<std::string>());
        for (const auto& word : group.second) {
            auto it = dictionary.find(word);
            if (it != dictionary.end()) {
                it->second.clear();
                it->second.reserve(group.second.size() - 1);
                for (const auto& other : group.second) {
                    if (other != word) {
                        it->second.push_back(other);
                    }
                }
            }
        }
    }
}